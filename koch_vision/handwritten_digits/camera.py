"""多数字定位、CNN批量分类、按行排序；相机/离线图片共用同一流程。"""

import argparse
import json
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from koch_vision.colors import read_image, write_image
from .model import OUTPUT_ROOT, ROOT, Predictor

CONFIG_PATH = ROOT / "configs" / "digit_camera.json"


def default_config():
    return {"version": 1, "camera_index": 1, "raw_size": [1280, 720],
            "rotation": 0, "mirror": False, "roi": None, "threshold": "adaptive",
            "polarity": "dark", "min_height": 18, "min_area": 35,
            "min_score": 0.7, "thicken": 1, "max_candidates": 80}


def load_settings(path=CONFIG_PATH):
    result = default_config()
    if path.exists():
        result.update(json.loads(path.read_text(encoding="utf-8")))
    return result


def orient(frame, rotation=0, mirror=False):
    if rotation not in (0, 90, 180, 270):
        raise ValueError("旋转角度只支持0、90、180、270度（顺时针）。")
    if rotation:
        code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation]
        frame = cv2.rotate(frame, code)
    return cv2.flip(frame, 1) if mirror else frame.copy()


def arrange_lines(items):
    """以纵向重叠和中心距离分行，再按X排序；保留前导0。"""
    groups = []
    for item in sorted(items, key=lambda value: value["center_px"][1]):
        x, y, w, h = item["bbox_px"]
        cy = item["center_px"][1]
        matching = []
        for index, group in enumerate(groups):
            gy = float(np.median([a["center_px"][1] for a in group]))
            gh = float(np.median([a["bbox_px"][3] for a in group]))
            centers_x = [a["center_px"][0] for a in group]
            if len(group) >= 2 and max(centers_x) - min(centers_x) >= gh:
                # 用当前行的倾斜趋势预测此X处的行中心，避免斜纸的最左/右字符掉行。
                slope, intercept = np.polyfit(centers_x, [a["center_px"][1] for a in group], 1)
                if abs(slope) <= 0.5:
                    gy = float(slope * item["center_px"][0] + intercept)
            overlap = max(0, min(y + h, gy + gh / 2) - max(y, gy - gh / 2))
            if overlap >= 0.35 * min(h, gh) and abs(cy - gy) <= 0.7 * max(h, gh):
                matching.append((abs(cy - gy), index))
        if matching:
            groups[min(matching)[1]].append(item)
        else:
            groups.append([item])
    groups.sort(key=lambda group: float(np.median([x["center_px"][1] for x in group])))
    lines, ordered = [], []
    for line_number, group in enumerate(groups, 1):
        group.sort(key=lambda item: item["bbox_px"][0])
        text = ""
        for item in group:
            text += item["text"]
            item["line"] = line_number
            item["number"] = len(ordered) + 1
            ordered.append(item)
        lines.append({"line": line_number, "text": text, "candidate_count": len(group)})
    return ordered, lines


def segment(frame, config):
    height, width = frame.shape[:2]
    roi = config.get("roi") or [0, 0, width, height]
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("识别区域超出画面，请重新框选。")
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if min(gray.shape) < 32:
        raise ValueError("识别区域太小，宽和高都需要至少32像素。")
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if config["polarity"] == "light":
        gray = 255 - gray
    elif config["polarity"] != "dark":
        raise ValueError("笔迹颜色应为dark或light。")
    if config["threshold"] == "adaptive":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    elif config["threshold"] == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if int(gray.max()) - int(gray.min()) < 20:
            mask[:] = 0
    else:
        raise ValueError("分割方式应为adaptive或otsu。")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area >= max(4, config["min_area"] // 5):
            components.append({"ids": [label], "box": [x, y, w, h], "area": area})
    if len(components) > config["max_candidates"] * 3:
        return [], mask, ["笔画碎片过多，已暂停本帧分类。请先缩小识别区域、排除背景纹理。"]
    # 合并同一数字中上下断开的笔画；不沿水平方向连接相邻数字。
    changed = True
    while changed:
        changed = False
        for i in range(len(components)):
            if changed:
                break
            a = components[i]
            ax, ay, aw, ah = a["box"]
            for j in range(i + 1, len(components)):
                b = components[j]
                bx, by, bw, bh = b["box"]
                overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                gap = max(0, max(ay, by) - min(ay + ah, by + bh))
                if overlap > 0.5 * min(aw, bw) and gap <= 0.25 * max(ah, bh):
                    left, top = min(ax, bx), min(ay, by)
                    a["box"] = [left, top, max(ax + aw, bx + bw) - left, max(ay + ah, by + bh) - top]
                    a["ids"] += b["ids"]
                    a["area"] += b["area"]
                    components.pop(j)
                    changed = True
                    break
    candidates, warnings = [], []
    for comp in components:
        x, y, w, h = comp["box"]
        if h < config["min_height"] or comp["area"] < config["min_area"]:
            continue
        if x <= 1 or y <= 1 or x + w >= mask.shape[1] - 1 or y + h >= mask.shape[0] - 1:
            warnings.append("有笔迹触及识别区边界，已跳过；请扩大选区或移动纸张。")
            continue
        if w * h > mask.size * 0.5 or h > mask.shape[0] * 0.95:
            continue
        isolated = np.uint8(np.isin(labels[y:y + h, x:x + w], comp["ids"])) * 255
        if config["thicken"]:
            isolated = cv2.dilate(isolated, np.ones((2, 2), np.uint8))
        crop = cv2.copyMakeBorder(isolated, 5, 5, 5, 5, cv2.BORDER_CONSTANT)
        candidates.append({"bbox_px": [x + x0, y + y0, w, h],
                           "center_px": [round(x + x0 + w / 2, 1), round(y + y0 + h / 2, 1)],
                           "image": Image.fromarray(crop), "too_wide": w / h > 1.6})
    candidates.sort(key=lambda item: (item["center_px"][1], item["center_px"][0]))
    if len(candidates) > config["max_candidates"]:
        warnings.append(f"候选超过{config['max_candidates']}个，已暂停本帧分类。请缩小选区、排除纹理背景。")
        candidates = []
    return candidates, mask, sorted(set(warnings))


def recognize_scene(raw_frame, predictor, config):
    frame = orient(raw_frame, config["rotation"], config["mirror"])
    candidates, mask, warnings = segment(frame, config)
    usable = [item for item in candidates if not item["too_wide"]]
    classified = iter(predictor.predict_many([item["image"] for item in usable], "light"))
    items = []
    previews = []
    for item in candidates:
        record = {key: value for key, value in item.items() if key not in ("image", "too_wide")}
        if item["too_wide"]:
            record.update(prediction=None, text="?", score=0.0, top3=[], accepted=False, reason="过宽，可能是粘连数字或非数字")
        else:
            result, preview = next(classified)
            score = result["top3"][0]["score"]
            accepted = score >= config["min_score"]
            record.update(prediction=result["digit"], text=str(result["digit"]) if accepted else "?",
                          score=score, top3=result["top3"], accepted=accepted,
                          reason="" if accepted else "模型分数不足，请核对预览")
            previews.append((record, preview))
        items.append(record)
    ordered, lines = arrange_lines(items)
    report = {"digits": ordered, "lines": lines, "text": "\n".join(line["text"] for line in lines),
              "candidate_count": len(ordered), "accepted_count": sum(x["accepted"] for x in ordered),
              "warnings": warnings, "model_path": str(predictor.path),
              "coordinate_frame": "画面旋转、镜像后的像素坐标，左上角为原点",
              "note": "传统图像分割+单数字CNN；低分用?显示，高分也可能出错。未训练粘连数字或场景文字检测模型。"}
    marked = annotate_scene(frame, report, config)
    return frame, marked, mask, report, [(record["number"], preview) for record, preview in previews]


def annotate_scene(frame, report, config):
    marked = frame.copy()
    if config.get("roi"):
        x0, y0, x1, y1 = config["roi"]
        cv2.rectangle(marked, (x0, y0), (x1, y1), (180, 180, 180), 1)
    for item in report["digits"]:
        x, y, w, h = item["bbox_px"]
        color = (80, 220, 100) if item["accepted"] else (0, 170, 255)
        cv2.rectangle(marked, (x, y), (x + w, y + h), color, 2)
        text = f"{item['number']}:{item['text']} {item['score']:.0%}"
        position = (x, max(15, y - 6))
        cv2.putText(marked, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(marked, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return marked


def save_scene(raw, predictor, config, directory=None):
    directory = directory or OUTPUT_ROOT / "camera" / "results"
    directory.mkdir(parents=True, exist_ok=True)
    frame, marked, mask, report, previews = recognize_scene(raw, predictor, config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = directory / stamp
    write_image(str(prefix) + "_raw.png", raw)
    write_image(str(prefix) + "_detected.png", marked)
    write_image(str(prefix) + "_mask.png", mask)
    for number, preview in previews:
        preview.save(str(prefix) + f"_cnn_{number:02}.png")
    report["config_snapshot"] = config.copy()
    report["timestamp"] = stamp
    path = directory / f"{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, path


def main():
    parser = argparse.ArgumentParser(description="从图片识别多个独立手写数字与多行数字串")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", type=type(CONFIG_PATH), default=CONFIG_PATH)
    parser.add_argument("--model")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    report, path = save_scene(read_image(args.image), Predictor(args.model), load_settings(args.config))
    print(report["text"] or "未找到数字")
    print("候选数：", report["candidate_count"], "报告：", path)


if __name__ == "__main__":
    main()
