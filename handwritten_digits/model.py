"""网络、单数字预处理和本地模型加载；识别时不访问网络。"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "digits"
OUTPUT_ROOT = ROOT / "outputs" / "digits"
MEAN, STD = 0.1307, 0.3081


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 10),
        )

    def forward(self, x):
        # 返回十个原始分数 logits；训练时交给交叉熵，不提前做 softmax。
        return self.classifier(self.features(x))


def prepare_digit(image, polarity="auto"):
    """单个数字裁剪、缩放到20像素并居中到28×28，转成 MNIST 输入。"""
    if polarity not in ("auto", "dark", "light"):
        raise ValueError("polarity 必须是 auto、dark 或 light。")
    # 透明图片先铺白底，避免透明底黑色笔迹变成全黑。
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    gray = np.array(background.convert("L"))
    if min(gray.shape) < 8 or float(gray.max()) - float(gray.min()) < 20:
        raise ValueError("画面为空或对比度不足，请先写一个清晰数字。")
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    invert = polarity == "dark" or (polarity == "auto" and np.median(border) > 127)
    if invert:
        gray = 255 - gray
    gray = np.array(ImageOps.autocontrast(Image.fromarray(gray)))
    ys, xs = np.where(gray > 50)
    if len(xs) < 10:
        raise ValueError("没有找到足够的笔迹，请加粗笔画或重新裁剪。")
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    size = (max(1, round(w * 20 / max(w, h))), max(1, round(h * 20 / max(w, h))))
    crop = np.array(Image.fromarray(crop).resize(size, Image.Resampling.LANCZOS))
    padded = np.zeros((28, 28), np.uint8)
    x, y = (28 - size[0]) // 2, (28 - size[1]) // 2
    padded[y:y + size[1], x:x + size[0]] = crop
    moments = cv2.moments(padded)
    if moments["m00"] > 0:
        shift = np.float32([[1, 0, 13.5 - moments["m10"] / moments["m00"]],
                            [0, 1, 13.5 - moments["m01"] / moments["m00"]]])
        padded = cv2.warpAffine(padded, shift, (28, 28))
    tensor = torch.from_numpy(padded.copy()).float().unsqueeze(0) / 255
    return (tensor - MEAN) / STD, Image.fromarray(padded)


def resolve_model(path=None):
    if path:
        return Path(path).resolve()
    pointer = OUTPUT_ROOT / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("还没有正式训练模型。请先双击“训练手写数字.cmd”。")
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["model_path"])


class Predictor:
    def __init__(self, path=None):
        self.path = resolve_model(path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("模型格式不匹配，请使用本项目训练的 best.pt。")
        self.model = DigitCNN()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.info = {key: value for key, value in payload.items() if key != "state_dict"}

    @torch.inference_mode()
    def predict(self, image, polarity="auto"):
        return self.predict_many([image], polarity)[0]

    @torch.inference_mode()
    def predict_many(self, images, polarity="auto"):
        if not images:
            return []
        prepared = [prepare_digit(image, polarity) for image in images]
        probabilities = self.model(torch.stack([item[0] for item in prepared])).softmax(dim=1)
        results = []
        for probabilities_one, (_, preview) in zip(probabilities, prepared):
            top = probabilities_one.topk(3)
            results.append(({"digit": int(top.indices[0]),
                            "top3": [{"digit": int(index), "score": round(float(score), 6)}
                                     for score, index in zip(top.values, top.indices)],
                            "model_path": str(self.path),
                            "note": "softmax 分数不是经校准的正确率；本模型只分类单个0–9数字。"}, preview))
        return results
