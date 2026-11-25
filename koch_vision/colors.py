"""仅相机的颜色样本标定与二维检测；不导入机器人控制库。"""

import argparse
import json
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from koch_robot.paths import ROOT as BASE
DEFAULT_CONFIG = BASE / "configs" / "camera_colors.json"
COLOR_NAMES = {"yellow": "黄色", "red": "红色", "blue": "蓝色", "green": "绿色", "pink": "粉红色"}


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    data.tofile(str(path))


def save_config(config, path=DEFAULT_CONFIG):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "camera_index": 1, "frame_size": [1280, 720],
            "min_area": 500, "max_area_fraction": 0.2, "roi": None,
            "plane": None, "colors": {key: {"name": value, "samples": []}
                                      for key, value in COLOR_NAMES.items()}}


def fit_sample(patch, min_saturation=45):
    """稳健估计 HSV 包络；色相按圆周计算，正确处理红色的 179/0 跨界。"""
    if patch is None or min(patch.shape[:2]) < 6:
        raise ValueError("采样框至少 6×6 像素，请只框方块表面内部。")
    pixels = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    pixels = pixels[(pixels[:, 1] > min_saturation) & (pixels[:, 2] > 18)]
    if len(pixels) < patch.shape[0] * patch.shape[1] * 0.8:
        raise ValueError("样本太暗或包含过多灰白背景，请选纯色内部并改善光照。")
    angles = pixels[:, 0] * np.pi / 90
    mean = np.mean(np.exp(1j * angles))
    if abs(mean) < 0.9:
        raise ValueError("框内混有多种颜色，请缩小采样区域。")
    hue = float(np.angle(mean) * 90 / np.pi % 180)
    delta = (pixels[:, 0] - hue + 90) % 180 - 90
    bgr = np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).reshape(-1, 3), axis=0)[::-1]
    lab = cv2.cvtColor((bgr.reshape(1, 1, 3).astype(np.float32) / 255),
                       cv2.COLOR_BGR2LAB)[0, 0]
    return {"h": hue, "s": float(np.median(pixels[:, 1])),
            "v": float(np.median(pixels[:, 2])),
            "h_tol": float(max(7, np.quantile(abs(delta), 0.95) + 3)),
            "s_tol": float(max(48, np.quantile(abs(pixels[:, 1] - np.median(pixels[:, 1])), 0.95) + 20)),
            "v_tol": 150.0,
            "lab": [round(float(n), 3) for n in lab],
            "bgr": [round(float(n), 1) for n in bgr]}


def classify_sample(patch, config):
    """复核方块顶面颜色；LAB 接近时再用 HSV 区分红与粉。"""
    try:
        profile = fit_sample(patch, min_saturation=config.get("sample_min_saturation", 45))
    except ValueError:
        return None, None
    query = np.asarray(profile["lab"], dtype=float)
    candidates = []
    for key in config["colors"]:
        for sample in config["colors"][key]["samples"]:
            hue_delta = abs((profile["h"] - sample["h"] + 90) % 180 - 90)
            hsv_distance = ((hue_delta / max(sample["h_tol"], 1)) ** 2
                            + ((profile["s"] - sample["s"]) / max(sample["s_tol"], 1)) ** 2
                            + .1 * ((profile["v"] - sample["v"]) / max(sample["v_tol"], 1)) ** 2)
            distance = (float(np.linalg.norm(query - np.asarray(sample["lab"], dtype=float)))
                        if "lab" in sample else None)
            candidates.append((key, sample, distance, hsv_distance))
    if not candidates:
        return None, None
    # HSV 负责区分红与粉；LAB 负责拒绝明显不属于已知样本的颜色。
    for key, _, lab_distance, hsv_distance in sorted(candidates, key=lambda item: item[3]):
        if hsv_distance >= 1.5:
            break
        key_labs = [item[2] for item in candidates if item[0] == key and item[2] is not None]
        if key_labs and min(key_labs) > 35:
            continue
        return key, lab_distance if lab_distance is not None else min(key_labs, default=None)
    lab_values = [item[2] for item in candidates if item[2] is not None]
    return None, min(lab_values, default=None)


def make_plane(points, width_mm, height_mm):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("需要四个有限坐标：左上、右上、右下、左下。")
    if not np.isfinite([width_mm, height_mm]).all() or min(width_mm, height_mm) <= 0:
        raise ValueError("实测宽、高必须是正数。")
    if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 100:
        raise ValueError("四点应按顺序围成凸四边形，不能重复、交叉或共线。")
    world = np.float32([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]])
    matrix = cv2.getPerspectiveTransform(points, world)
    return {"points_px": points.tolist(), "width_mm": width_mm, "height_mm": height_mm,
            "matrix": matrix.tolist(), "coordinate_frame": "user_reference_plane",
            "note": "只适用于所标定平面，不是机械臂坐标；有高度物体存在视差。"}


def plane_position(center, plane):
    if not plane or cv2.pointPolygonTest(np.float32(plane["points_px"]), tuple(center), False) < 0:
        return None
    value = cv2.perspectiveTransform(np.float32([[center]]), np.asarray(plane["matrix"], dtype=float))[0, 0]
    return [round(float(n), 2) for n in value]


@lru_cache(maxsize=4)
def hsv_lookup(profiles):
    """Cache calibrated HSV decisions; V is quantized to 16 levels for speed."""
    h = np.arange(180, dtype=np.float32)[:, None, None]
    s = np.arange(256, dtype=np.float32)[None, :, None]
    v = (np.arange(16, dtype=np.float32) * 16 + 7.5)[None, None, :]
    best = np.full((180, 256, 16), np.inf, np.float32)
    labels = np.full(best.shape, -1, np.int16)
    for index, hh, ss, vv, ht, st, vt in profiles:
        dh = np.abs((h - hh + 90) % 180 - 90)
        ds = np.abs(s - ss)
        score = (dh / ht) ** 2 + (ds / st) ** 2 + .1 * ((v - vv) / vt) ** 2
        valid = (dh <= ht) & (ds <= st) & (score < 1.5) & (score < best)
        np.copyto(best, score, where=valid)
        np.copyto(labels, index, where=valid)
    return labels


def detect(frame, config):
    height, width = frame.shape[:2]
    if [width, height] != config["frame_size"]:
        raise ValueError(f"图像为 {width}×{height}，配置为 {config['frame_size']}；请重设画幅、工作区及平面标定。")
    x1, y1, x2, y2 = config.get("roi") or [0, 0, width, height]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("工作区超出画面，请重新框选。")
    hsv_u8 = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    keys = list(config["colors"])
    if config.get("fast_lookup", False):
        profiles = tuple((i, *(float(s[k]) for k in ("h", "s", "v", "h_tol", "s_tol", "v_tol")))
                         for i, key in enumerate(keys) for s in config["colors"][key]["samples"])
        labels = hsv_lookup(profiles)[hsv_u8[:, :, 0], hsv_u8[:, :, 1], hsv_u8[:, :, 2] // 16]
        labels[hsv_u8[:, :, 2] < 20] = -1
    else:
        hsv = hsv_u8.astype(np.float32)
        best = np.full(hsv.shape[:2], np.inf, dtype=np.float32)
        labels = np.full(hsv.shape[:2], -1, dtype=np.int16)
        for index, key in enumerate(keys):
            for sample in config["colors"][key]["samples"]:
                dh = np.abs((hsv[:, :, 0] - sample["h"] + 90) % 180 - 90)
                ds = np.abs(hsv[:, :, 1] - sample["s"])
                dv = np.abs(hsv[:, :, 2] - sample["v"])
                score = (dh / sample["h_tol"]) ** 2 + (ds / sample["s_tol"]) ** 2 + 0.1 * (dv / sample["v_tol"]) ** 2
                valid = ((dh <= sample["h_tol"]) & (ds <= sample["s_tol"]) &
                         (hsv[:, :, 2] >= 20) & (score < 1.5) & (score < best))
                best[valid] = score[valid]
                labels[valid] = index
    kernel = np.ones((3, 3), np.uint8)
    min_area = max(config["min_area"], width * height * config.get("min_area_fraction", 0))
    perspective = config.get('perspective_corners')
    projected_paper = perspective and config.get("material") == "paper"
    max_aspect = 2.6 if perspective else 2.0
    detections = []
    for index, key in enumerate(keys):
        mask = np.uint8(labels == index) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour + np.int32([[[x1, y1]]])
            area = cv2.contourArea(contour)
            if not min_area <= area <= width * height * config["max_area_fraction"]:
                continue
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            if min(rw, rh) < 8 or max(rw, rh) / min(rw, rh) > max_aspect:
                continue
            quad = None
            if perspective:
                hull = cv2.convexHull(contour)
                outline = cv2.approxPolyDP(hull, .025 * cv2.arcLength(hull, True), True)
                if projected_paper and len(outline) > 4:
                    outline = cv2.approxPolyDP(hull, .035 * cv2.arcLength(hull, True), True)
                if len(outline) == 4 and area <= 1.15 * cv2.contourArea(outline):
                    quad = outline.reshape(4, 2).astype(np.float32)
            if area / (rw * rh) < .65:
                # A slanted paper can fill its own four-corner outline while
                # leaving much of the enclosing rotated rectangle unfilled.
                if (not projected_paper or quad is None or area < .85 * cv2.contourArea(hull)
                        or area < .8 * cv2.contourArea(quad)):
                    continue
            x, y, w, h = cv2.boundingRect(contour)
            clipped = x <= x1 + 1 or y <= y1 + 1 or x + w >= x2 - 1 or y + h >= y2 - 1
            center = [float(n) for n in rect[0]]
            box = cv2.boxPoints(rect)
            edge = box[1] - box[0]
            angle = float(np.degrees(np.arctan2(edge[1], edge[0])) % 90)
            if quad is not None:
                box = quad
                center = box.mean(axis=0).tolist()
            # 立体方块的侧面会扩大颜色轮廓；用包围框中央的顶面重新核对类别。
            px0, px1 = x + int(w * .22), x + int(w * .78)
            py0, py1 = y + int(h * .22), y + int(h * .78)
            reviewed_key, lab_distance = classify_sample(frame[py0:py1, px0:px1], config)
            detected_key = reviewed_key or key
            detections.append({"color": detected_key, "color_name": config["colors"][detected_key]["name"],
                               "center_px": [round(n, 2) for n in center],
                               "bbox_px": [x, y, w, h], "box_px": box.round(2).tolist(),
                               "silhouette_angle_deg_mod90": None if clipped else round(angle, 2) % 90,
                               "plane_reference_mm": plane_position(center, config.get("plane")) if not clipped else None,
                               "area_px": round(area, 1), "clipped": clipped,
                               "profile_lab_distance": None if lab_distance is None else round(lab_distance, 3),
                               "type": "colored_region", "track_id": None})
    detections.sort(key=lambda item: (item["center_px"][1], item["center_px"][0]))
    for number, item in enumerate(detections, 1):
        item["frame_object_number"] = number
    return detections


def annotate(frame, detections, config):
    result = frame.copy()
    if config.get("roi"):
        x1, y1, x2, y2 = config["roi"]
        cv2.rectangle(result, (x1, y1), (x2, y2), (150, 150, 150), 1)
    for item in detections:
        box = np.int32(item["box_px"])
        center = tuple(np.int32(item["center_px"]))
        color = (0, 170, 255) if item["clipped"] else (100, 255, 150)
        cv2.polylines(result, [box], True, color, 2)
        cv2.drawMarker(result, center, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        text = f"{item['frame_object_number']} {item['color']} ({center[0]},{center[1]})"
        position = (max(0, center[0] - 65), max(18, int(box[:, 1].min()) - 8))
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return result


def save_result(frame, config, directory, detections=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    detections = detect(frame, config) if detections is None else detections
    raw = directory / f"{stamp}_raw.png"
    marked = directory / f"{stamp}_detected.png"
    report = directory / f"{stamp}.json"
    write_image(raw, frame)
    write_image(marked, annotate(frame, detections, config))
    report.write_text(json.dumps({"timestamp": stamp, "raw_image": raw.name,
                                 "config_snapshot": config, "detections": detections,
                                 "note": "颜色区域检测；不判别方块/卡纸、占用、高度或抓取姿态。"},
                                ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="离线检查图片中的颜色、二维位置和轮廓角度")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(BASE / "outputs" / "camera_vision" / "results"))
    args = parser.parse_args()
    frame = read_image(args.image)
    config = load_config(args.config)
    items = detect(frame, config)
    print(json.dumps(items, ensure_ascii=False, indent=2))
    print("已保存：", save_result(frame, config, args.output, items))


if __name__ == "__main__":
    main()
