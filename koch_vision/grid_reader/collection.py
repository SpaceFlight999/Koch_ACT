"""Local, material-separated camera captures awaiting human labels."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

from koch_vision.colors import BASE, fit_sample, write_image

DATASET_ROOT = BASE / "datasets/grid_capture"
MATERIAL_NAMES = {"cube": "小立方体", "paper": "彩色纸片"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    temporary.replace(path)


def collect_frame(reader, raw, metadata, root=DATASET_ROOT):
    """Recognize the clicked frame itself; predictions never become truth labels."""
    material = metadata["material"]
    if material not in MATERIAL_NAMES:
        raise ValueError("未知材料类型")
    report, marked, debug = reader.read(raw)
    report.update(metadata)
    root = Path(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = root / material / stamp
    directory.mkdir(parents=True, exist_ok=False)
    write_image(directory / "raw.png", raw)
    write_image(directory / "marked.png", marked)
    regions = []
    for index, item in enumerate(debug):
        cell_id = item["id"]
        write_image(directory / "regions" / f"{cell_id}_tile.png", item["tile"])
        write_image(directory / "regions" / f"{cell_id}_mask.png", item["mask"])
        for i, preview in enumerate(item["previews"]):
            preview.save(directory / "regions" / f"{cell_id}_digit{i+1}.png")
        h, w = item["tile"].shape[:2]
        try:
            config = getattr(reader, "color_config", None) or {}
            color = fit_sample(item["tile"][int(h*.2):int(h*.8), int(w*.2):int(w*.8)],
                               min_saturation=config.get("sample_min_saturation", 45))
        except ValueError as exc:
            color = {"unavailable_reason": str(exc)}
        cell = report["cells"][index]
        regions.append({"id": cell_id, "quad_px": cell["quad_px"],
                        "predicted_color": cell["color"].get("key"),
                        "predicted_digit": cell["raw_number"], "color_features": color,
                        "manual_color": None, "manual_digit": None})
    record = {"format_version": 1, "capture_id": stamp, "material": material,
              "material_name": MATERIAL_NAMES[material], "saved_at": datetime.now().isoformat(timespec="milliseconds"),
              "frame_size": list(raw.shape[1::-1]), "image_sha256": hashlib.sha256((directory/"raw.png").read_bytes()).hexdigest(),
              "training_status": "needs_manual_annotation", "manual_labels": None,
              "note": "颜色与数字预测仅为候选；人工核对前不得作为训练真值或正式颜色标定。",
              "report": report, "regions": regions}
    write_json(directory / "capture.json", record)
    feedback = {"event": "capture_saved", "capture_id": stamp, "material": material,
                "dataset_root": str(root.resolve()), "directory": str(directory.resolve()),
                "metadata": str((directory/"capture.json").resolve()),
                "training_status": "needs_manual_annotation",
                "total_captures": len(list(root.glob("*/*/capture.json")))}
    write_json(root / "feedback.json", feedback)
    return feedback


def feedback_text(root=DATASET_ROOT):
    root = Path(root)
    record = json.loads((root / "feedback.json").read_text(encoding="utf8"))
    return (f"已在程序中保存{record['total_captures']}张训练素材，最新为"
            f"{MATERIAL_NAMES[record['material']]}。请读取 {root.resolve() / 'feedback.json'}，"
            "核对原图、数字与颜色标签后继续训练。")
