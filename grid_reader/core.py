"""自动定位3×3卡纸，校正透视，读取底色和手写数字，保留不确定状态。"""

import json
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from camera_vision import detect as detect_colors, load_config, write_image
from handwritten_digits.camera import default_config as digit_settings, orient, segment
from handwritten_digits.model import ROOT

OUTPUT = ROOT / "outputs" / "grid_reader"
CELL_SIZE = 192


def order_quad(points):
    points = np.float32(points).reshape(4, 2)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    # 顺时针：左上、右上、右下、左下。避免sum/diff方法在菱形上重复选点。
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    points = np.roll(points, -start, axis=0)
    if cv2.contourArea(points, oriented=True) < 0:
        points = points[[0, 3, 2, 1]]
    return points


def warp_quad(frame, quad, size=CELL_SIZE):
    target = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    matrix = cv2.getPerspectiveTransform(order_quad(quad), target)
    return cv2.warpPerspective(frame, matrix, (size, size)), matrix


def color_measure(tile):
    h, w = tile.shape[:2]
    pixels = tile[int(h * .12):int(h * .88), int(w * .12):int(w * .88)].reshape(-1, 3)
    bgr = np.median(pixels, axis=0).astype(np.uint8)
    lab = cv2.cvtColor((bgr.reshape(1, 1, 3).astype(np.float32) / 255), cv2.COLOR_BGR2LAB)[0, 0]
    all_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    uniform = float(np.mean(np.linalg.norm(all_lab - lab, axis=1) < 20))
    hh, ss, vv = map(float, cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0])
    if ss < 38:
        name = "白色" if vv > 195 else "黑色" if vv < 48 else "灰色"
    elif vv < 38:
        name = "深色（待确认）"
    elif 7 <= hh < 24 and vv < 175:
        name = "棕色"
    elif (hh >= 160 or hh < 8) and ss < 175 and vv > 135:
        name = "粉红色"
    else:
        name = next((label for lo, hi, label in ((0, 8, "红色"), (8, 23, "橙色"), (23, 38, "黄色"),
                      (38, 50, "黄绿色"), (50, 83, "绿色"), (83, 100, "青色"),
                      (100, 130, "蓝色"), (130, 156, "紫色"), (156, 172, "品红色"), (172, 180, "红色"))
                     if lo <= hh < hi), "未知颜色")
    rgb = bgr[::-1].tolist()
    return {"name": name, "rgb": rgb, "hex": "#" + "".join(f"{n:02X}" for n in rgb),
            "hsv": [round(hh, 1), round(ss, 1), round(vv, 1)], "lab": [round(float(n), 2) for n in lab],
            "background_fraction": round(uniform, 3),
            "reliable": uniform >= .60 and not name.startswith(("深色", "未知")),
            "method": "卡纸内部的稳健底色统计；颜色名为通用HSV分类，RGB保留实际图像测量"}


def palette_distinct(colors, threshold=12):
    groups = []
    for color in colors:
        value = np.asarray(color["lab"])
        if not any(np.linalg.norm(value - old) < threshold for old in groups):
            groups.append(value)
    return len(groups)


def find_quads(frame):
    scale = min(1.0, 960 / max(frame.shape[:2]))
    image = cv2.resize(frame, None, fx=scale, fy=scale)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = []
    for channel in (gray, hsv[:, :, 1]):
        edges = cv2.Canny(cv2.GaussianBlur(channel, (3, 3), 0), 35, 100)
        masks.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    colored = np.uint8((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 35)) * 255
    masks.append(cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)))
    # 分色辅助获得相接卡纸的边界，避免把九种颜色合成一个前景块。
    for hue in range(0, 180, 20):
        mask = np.uint8((np.abs((hsv[:, :, 0].astype(float) - hue + 90) % 180 - 90) < 12)
                       & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 35)) * 255
        masks.append(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))
    quads = []
    min_area = max(180, image.shape[0] * image.shape[1] * .0008)
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not min_area < area < image.shape[0] * image.shape[1] * .94:
                continue
            quad = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            quad = order_quad(quad)
            sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
            if sides.min() < 12 or sides.max() / sides.min() > 3:
                continue
            quad = quad / scale
            center = quad.mean(axis=0)
            duplicate = False
            for old in quads:
                ratio = area / (cv2.contourArea(old) * scale * scale)
                if .6 < ratio < 1.7 and np.linalg.norm(center - old.mean(axis=0)) < np.sqrt(area) / scale * .13:
                    duplicate = True
                    break
            if not duplicate:
                quads.append(quad)
    return sorted(quads, key=cv2.contourArea, reverse=True)[:90]


def cluster_1d(values):
    values = np.asarray(values)
    centers = np.quantile(values, [.15, .5, .85])
    for _ in range(12):
        labels = np.argmin(abs(values[:, None] - centers[None, :]), axis=1)
        if len(set(labels)) < 3:
            return None
        new = np.array([np.mean(values[labels == i]) for i in range(3)])
        if np.max(abs(new - centers)) < .01:
            break
        centers = new
    order = np.argsort(centers)
    return centers[order]


def reading_order(items):
    """Order visible regions by image rows, then left-to-right within each row."""
    if not items:
        return []
    heights = [np.ptp(np.asarray(item["box_px"])[:, 1]) for item in items]
    tolerance = max(8, float(np.median(heights)) * .65)
    rows = []
    for item in sorted(items, key=lambda item: item["center_px"][1]):
        if not rows or item["center_px"][1] - np.median([x["center_px"][1] for x in rows[-1]]) > tolerance:
            rows.append([])
        rows[-1].append(item)
    return [item for row in rows for item in sorted(row, key=lambda item: item["center_px"][0])]


def individual_grid(quads):
    best = None
    areas = np.array([cv2.contourArea(q) for q in quads])
    if len(areas) < 9:
        return None
    tested = []
    for reference in sorted(areas, reverse=True):
        if any(abs(np.log(reference / old)) < .16 for old in tested):
            continue
        tested.append(reference)
        ids = np.flatnonzero((areas > reference * .55) & (areas < reference * 1.65))
        if len(ids) < 9 or len(ids) > 45:
            continue
        selected = [quads[i] for i in ids]
        angles = []
        for q in selected:
            edge = q[1] - q[0]
            angles.append((np.degrees(np.arctan2(edge[1], edge[0])) + 45) % 90 - 45)
        angle = np.radians(np.median(angles))
        matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        centers = np.array([q.mean(axis=0) for q in selected]) @ matrix.T
        xs, ys = cluster_1d(centers[:, 0]), cluster_1d(centers[:, 1])
        if xs is None or ys is None:
            continue
        dx, dy = np.diff(xs), np.diff(ys)
        if min(*dx, *dy) < 12 or max(dx) / min(dx) > 1.7 or max(dy) / min(dy) > 1.7:
            continue
        pitch = min(np.mean(dx), np.mean(dy))
        if not .35 < np.sqrt(np.median(areas[ids])) / pitch < 1.25:
            continue
        ordered, errors, used = [], [], set()
        for y in ys:
            for x in xs:
                distances = np.linalg.norm(centers - [x, y], axis=1)
                index = int(np.argmin(distances))
                if index in used or distances[index] > pitch * .28:
                    break
                used.add(index)
                ordered.append(selected[index])
                errors.append(float(distances[index] / pitch))
            else:
                continue
            break
        if len(ordered) != 9:
            continue
        # 不把嵌套框/相互覆盖区域当成九张独立卡纸。
        if any(cv2.intersectConvexConvex(ordered[i], ordered[j])[0] > min(cv2.contourArea(ordered[i]), cv2.contourArea(ordered[j])) * .20
               for i in range(9) for j in range(i)):
            continue
        score = 1 - float(np.mean(errors))
        if best is None or score > best["score"]:
            best = {"quads": ordered, "score": score, "mode": "nine_individual_cards"}
    return best


def board_grid(frame, quads):
    best = None
    for quad in quads[:20]:
        if cv2.contourArea(quad) < frame.shape[0] * frame.shape[1] * .015:
            continue
        board, matrix = warp_quad(frame, quad, CELL_SIZE * 3)
        tiles, cell_quads, colors = [], [], []
        inverse = np.linalg.inv(matrix)
        for row in range(3):
            for col in range(3):
                tile = board[row * CELL_SIZE:(row + 1) * CELL_SIZE, col * CELL_SIZE:(col + 1) * CELL_SIZE]
                tiles.append(tile)
                colors.append(color_measure(tile))
                p = np.float32([[[col * CELL_SIZE, row * CELL_SIZE], [(col + 1) * CELL_SIZE - 1, row * CELL_SIZE],
                                 [(col + 1) * CELL_SIZE - 1, (row + 1) * CELL_SIZE - 1], [col * CELL_SIZE, (row + 1) * CELL_SIZE - 1]]])
                cell_quads.append(cv2.perspectiveTransform(p, inverse)[0])
        if palette_distinct(colors) < 8 or min(x["background_fraction"] for x in colors) < .60:
            continue
        # 三等分边界两侧必须存在真实颜色跳变，不能仅凭一个外框虚构九格。
        lab = cv2.cvtColor(board.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        borders = []
        for k in (1, 2):
            for part in range(3):
                a, b = int((part + .2) * CELL_SIZE), int((part + .8) * CELL_SIZE)
                x = k * CELL_SIZE
                left = np.median(lab[a:b, x - 14:x - 5], axis=(0, 1))
                right = np.median(lab[a:b, x + 5:x + 14], axis=(0, 1))
                top = np.median(lab[x - 14:x - 5, a:b], axis=(0, 1))
                bottom = np.median(lab[x + 5:x + 14, a:b], axis=(0, 1))
                borders.extend([np.linalg.norm(left - right) > 12, np.linalg.norm(top - bottom) > 12])
        if sum(borders) < 9:
            continue
        score = .65 + .3 * np.mean([x["background_fraction"] for x in colors])
        if best is None or score > best["score"]:
            best = {"quads": cell_quads, "score": float(score), "mode": "continuous_3x3_board"}
    return best


def locate_grid(frame):
    quads = find_quads(frame)
    grid = individual_grid(quads)
    if grid:
        tiles = [warp_quad(frame, q)[0] for q in grid["quads"]]
        colors = [color_measure(tile) for tile in tiles]
        if min(c["background_fraction"] for c in colors) >= .52:
            return grid
    return board_grid(frame, quads)


def extract_tokens(tile, reject_boundary_components=False):
    # 在颜色空间中取底色的暗向残差，避免把蓝/红卡纸本身当成黑色数字。
    lab = cv2.cvtColor(tile.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    margin = max(10, int(min(tile.shape[:2]) * .10))
    interior = lab[margin:-margin, margin:-margin]
    background = np.median(interior.reshape(-1, 3), axis=0)
    lightness = lab[:, :, 0]
    distance = np.linalg.norm(lab - background, axis=2)
    local_l = np.uint8(np.clip(lightness * 2.55, 0, 255))
    blackhat = cv2.morphologyEx(local_l, cv2.MORPH_BLACKHAT, np.ones((31, 31), np.uint8))
    dark = ((background[0] - lightness > 7) & (distance > 10)) | ((blackhat > 22) & (background[0] - lightness > 3))
    if background[0] < 78:
        light = (lightness - background[0] > 22) & (distance > 25)
        if light[margin:-margin, margin:-margin].mean() < .22:
            dark |= light
    if reject_boundary_components:
        # 细笔迹在彩色表面的亮度差较低，用去噪后的局部对比补充候选。
        smooth = cv2.GaussianBlur(lightness, (3, 3), 0)
        local = np.uint8(np.clip(smooth * 2.55, 0, 255))
        fine_ink = cv2.morphologyEx(local, cv2.MORPH_BLACKHAT, np.ones((25, 25), np.uint8))
        dark |= (fine_ink > 6) & (background[0] - smooth > 1.5)
    mask = np.uint8(dark) * 255
    mask[:margin] = mask[-margin:] = 0
    mask[:, :margin] = mask[:, -margin:] = 0
    if reject_boundary_components:
        # 立体方块侧面会形成细长暗/亮带，截在内框边缘时外观很像数字1。
        # 这类连通区域未完整落在内部，不送入只认识0–9的分类器。
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < 10 or x <= margin or y <= margin or x + w >= tile.shape[1] - margin or y + h >= tile.shape[0] - margin:
                mask[labels == label] = 0
    options = digit_settings()
    options.update(threshold="otsu", min_height=20, min_area=24, thicken=0, max_candidates=8)
    tokens, _, warnings = segment(cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR), options)
    # 分割只负责找位置；分类保留笔画灰度，避免二值化填洞/截断改变“3”等字形。
    background_bgr = np.median(tile[margin:-margin, margin:-margin].reshape(-1, 3), axis=0)
    ratios = tile.astype(np.float32) / np.maximum(background_bgr, 12)
    signal = np.uint8(np.clip(1 - np.median(ratios, axis=2), 0, 1) * 255)
    if background[0] < 78:
        light_signal = np.uint8(np.clip((lightness - background[0]) / max(100 - background[0], 1), 0, 1) * 255)
        signal = np.maximum(signal, light_signal)
    support = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    signal[support == 0] = 0
    for token in tokens:
        x, y, w, h = token["bbox_px"]
        x0, y0, x1, y1 = max(0, x - 4), max(0, y - 4), min(tile.shape[1], x + w + 4), min(tile.shape[0], y + h + 4)
        crop = signal[y0:y1, x0:x1]
        if reject_boundary_components and crop.max() > 0:
            crop = np.uint8(crop.astype(float) * (255.0 / crop.max()))
        token["image"] = Image.fromarray(cv2.copyMakeBorder(crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT))
    return tokens, mask, warnings


def extract_block_token(tile, recover_faint=True, material="cube"):
    """从一个彩色立体方块顶面提取中央的单数字，避开侧面和边缘阴影。"""
    h, w = tile.shape[:2]
    # Preserve high-written 9s and left-offset 6 loops; narrow windows clip ink.
    x0, x1 = int(w * .22), int(w * .78)
    y0, y1 = int(h * .10), int(h * .90)
    roi = tile[y0:y1, x0:x1]
    if min(roi.shape[:2]) < 32:
        return None, np.zeros((h, w), np.uint8), ["方块顶面太小"]
    lab = cv2.cvtColor(roi.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    if material == "paper":
        # Paper grain needs smoothing, while the small 8/9 loops need a smaller close.
        lab = cv2.GaussianBlur(lab, (0, 0), .7)
    # 大尺度高斯只估计顶面底色，局部残差能保留蓝/绿底上的低对比笔迹。
    background = cv2.GaussianBlur(lab, (0, 0), 10)
    darkness = background[:, :, 0] - lab[:, :, 0]
    residual = np.linalg.norm(lab - background, axis=2)
    surface = np.uint8(np.median(roi.reshape(-1, 3), axis=0)).reshape(1, 1, 3)
    hue, saturation, _ = cv2.cvtColor(surface, cv2.COLOR_BGR2HSV)[0, 0]
    is_blue = 90 <= hue < 130 and saturation > 80
    relaxed = recover_faint
    dark_threshold = ((1.0 if material == "paper" else 0.0) if is_blue else 1.5) if relaxed else 2.5
    residual_threshold = (3 if is_blue else 5) if relaxed else 9
    mask = np.uint8((darkness > dark_threshold) & (residual > residual_threshold)) * 255
    close_size = 2 if material == "paper" else 5
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    accepted = []
    candidates = []
    for label in range(1, count):
        x, y, cw, ch, area = map(int, stats[label])
        if area < 40 or ch < 14 or cw < 4:
            continue
        # 蓝/黄色顶面的小反光或压缩噪点曾被放大成“1”；数字笔画至少在一个方向
        # 覆盖足够的块面，窄而短的内部斑块不作为候选。
        if cw < w * .10 and ch < h * .16:
            continue
        # 空方块最常见误检是顶面侧棱：贴着左右内框的细长竖线。
        # Wider loops may enter the side margins, but narrow rails must still
        # obey the former central window's side-edge rejection.
        edge = (x0 + x <= int(w * .28) + 2 or x0 + x + cw >= int(w * .72) - 1 or
                y <= 2 or y + ch >= mask.shape[0] - 1)
        central_width = max(0, min(x0 + x + cw, int(w * .72)) - max(x0 + x, int(w * .28)))
        if edge and (central_width / max(1, ch) < .22 or ch / max(1, cw) < .22):
            continue
        accepted.append(label)
        center_x = x + cw / 2
        center_y = y + ch / 2
        center_bonus = max(0, 1 - abs(center_x - mask.shape[1] / 2) / (mask.shape[1] / 2))
        candidates.append((area * 1.5 * (1 + center_bonus) + ch * cw, label))
    if not candidates:
        if relaxed:
            return extract_block_token(tile, recover_faint=False, material=material)
        full = np.zeros((h, w), np.uint8)
        full[y0:y1, x0:x1] = mask
        return None, full, []
    main_label = max(candidates)[1]
    mx, my, mw, mh, _ = map(int, stats[main_label])
    related = [main_label]
    # “3”的中段弧线、“9”的左侧圆圈可能在低对比处断成小连通域；只合并紧邻主笔画的部件。
    # Expand transitively: a weak 8 can break into a chain of adjacent arcs.
    changed = True
    while changed:
        changed = False
        for label in accepted:
            if label in related:
                continue
            x, y, cw, ch, area = map(int, stats[label])
            if (x < mx + mw + 8 and x + cw > mx - 8 and
                    y < my + mh + 6 and y + ch > my - 6):
                if relaxed:
                    right, bottom = max(mx + mw, x + cw), max(my + mh, y + ch)
                    mx, my = min(mx, x), min(my, y)
                    mw, mh = right - mx, bottom - my
                related.append(label)
                changed = bool(relaxed)
    isolated = np.uint8(np.isin(labels, related)) * 255
    ys, xs = np.where(isolated > 0)
    bx0, bx1 = max(0, xs.min() - 3), min(mask.shape[1], xs.max() + 4)
    by0, by1 = max(0, ys.min() - 3), min(mask.shape[0], ys.max() + 4)
    crop = isolated[by0:by1, bx0:bx1]
    # 一次轻微模糊提供 MNIST 训练中常见的灰度边缘，而不是硬二值轮廓。
    crop = cv2.GaussianBlur(crop, (3, 3), .55)
    image = Image.fromarray(cv2.copyMakeBorder(crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT))
    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = mask
    token = {"bbox_px": [x0 + bx0, y0 + by0, bx1 - bx0, by1 - by0],
             "center_px": [x0 + (bx0 + bx1) / 2, y0 + (by0 + by1) / 2],
             "image": image, "too_wide": (bx1 - bx0) / max(1, by1 - by0) > 1.6}
    # Expanded dark masks can join surface texture; retain strict extraction then.
    if relaxed and token["too_wide"]:
        return extract_block_token(tile, recover_faint=False, material=material)
    return token, full, []


def upright_is_decisive(records, predictions):
    """A score above .90 cannot lose by the .10 needed to rotate the board."""
    if not records or any(not r["indices"] or r["ambiguous"] for r in records):
        return False
    scores = [min(predictions[i][0]["top3"][0]["score"] for i in r["indices"])
              for r in records]
    return float(np.mean(scores)) > .90


class GridReader:
    def __init__(self, predictor=None, auto_orientation=True, mode="digits", color_config=None):
        if mode not in ("digits", "blocks", "combined"):
            raise ValueError("mode必须是digits、blocks或combined")
        if mode != "blocks" and predictor is None:
            raise ValueError("数字识别模式需要本地数字模型")
        self.predictor = predictor
        self.auto_orientation = auto_orientation
        self.mode = mode
        self.color_config = (color_config if color_config is not None else
                             load_config(ROOT / "configs/grid_blocks_colors.json")) if mode in ("blocks", "combined") else None

    def read_blocks(self, frame, draw=True):
        """先检测实物颜色区域；九格几何确实成立时才分配g11等编号。"""
        # iPhone虚拟摄像头可能返回不同分辨率；本模式只用像素坐标，全画面重定位。
        # Locate on a small image, but extract handwriting from the original HD frame.
        scale = min(1.0, self.color_config.get("detection_max_side", max(frame.shape[:2])) / max(frame.shape[:2]))
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        frame_config = {**self.color_config, "frame_size": list(small.shape[1::-1]), "roi": None, "plane": None,
                        "min_area": self.color_config["min_area"] * scale * scale}
        items = detect_colors(small, frame_config)
        sx, sy = frame.shape[1] / small.shape[1], frame.shape[0] / small.shape[0]
        for item in items:
            item["box_px"] = (np.asarray(item["box_px"]) * [sx, sy]).tolist()
            item["center_px"] = (np.asarray(item["center_px"]) * [sx, sy]).round(2).tolist()
        quads = [order_quad(item["box_px"]) for item in items]
        grid = individual_grid(quads) if len(quads) == 9 else None
        if grid:
            ordered = []
            remaining = list(range(len(items)))
            for quad in grid["quads"]:
                index = min(remaining, key=lambda i: np.linalg.norm(quads[i].mean(axis=0) - quad.mean(axis=0)))
                ordered.append(items[index])
                remaining.remove(index)
            rows = [ordered[i:i+3] for i in range(0, 9, 3)]
            rows.sort(key=lambda row: np.mean([item["center_px"][1] for item in row]))
            items = [item for row in rows for item in sorted(row, key=lambda item: item["center_px"][0])]
        else:
            items = reading_order(items)
        cells, debug = [], []
        for index, item in enumerate(items):
            quad = order_quad(item["box_px"])
            tile, _ = warp_quad(frame, quad, size=self.color_config.get("digit_tile_size", CELL_SIZE))
            color = color_measure(tile)
            color.update(name=item["color_name"], key=item["color"],
                         method="实物HSV样本匹配；RGB取可见颜色区域内部，立体阴影可能影响测量")
            issues = ["方块触及画面边界，位置仅为可见部分"] if item["clipped"] else []
            if not color["reliable"]:
                issues.append("表面明暗差异或颜色不均，需确认")
            if self.color_config.get("calibration_required"):
                issues.append("当前材料颜色尚未实拍标定")
            cell_id = f"g{index // 3 + 1}{index % 3 + 1}" if grid else f"b{index + 1:02d}"
            cell = {"id": cell_id, "row": index // 3 + 1 if grid else None,
                    "column": index % 3 + 1 if grid else None, "quad_px": quad.round(2).tolist(),
                    "center_px": item["center_px"], "color": color, "number": None, "raw_number": None,
                    "number_score": None, "digit_predictions": [], "number_required": False,
                    "confirmed": not issues, "issues": issues, "clipped": item["clipped"],
                    "silhouette_angle_deg_mod90": item["silhouette_angle_deg_mod90"]}
            cells.append(cell)
            debug.append({"id": cell_id, "tile": tile, "mask": np.zeros(tile.shape[:2], np.uint8), "previews": []})
        complete = bool(grid and all(cell["confirmed"] for cell in cells))
        clipped = sum(cell["clipped"] for cell in cells)
        status = ("九格颜色本帧通过，等待连续帧确认。" if complete else
                  f"方块颜色试验：检测到{len(cells)}个颜色区域，尚未确认完整九格。")
        report = {"mode": "blocks", "number_required": False, "grid_found": bool(grid),
                  "complete": complete, "cells": cells, "detected_count": len(cells), "clipped_count": clipped,
                  "grid_mode": "nine_colored_blocks" if grid else "unassigned_color_regions",
                  "status": status, "warnings": [f"{clipped}个区域触及边界。"] if clipped else [],
                  "config_source": self.color_config.get("source_config", "configs/grid_blocks_colors.json"), "distinct_colors_required": False,
                  "coordinates": "原始相机像素；未成九格时b编号仅为本帧列表序号，不表示格子或跟踪ID。",
                  "note": "仅用可见颜色轮廓测试立体方块，不估计高度、顶面中心或抓取姿态；不执行数字推理。"}
        return report, annotate(frame, report) if draw else frame, debug

    def read(self, frame):
        if self.mode == "blocks":
            return self.read_blocks(frame)
        if self.mode == "combined":
            # Combined mode draws the final number labels once, after inference.
            base, _, debug = self.read_blocks(frame, draw=False)
            base.update(mode="combined", number_required=True, model_path=str(self.predictor.path),
                        note="实物颜色样本与CNN数字联合读取；允许同色重复，不估计三维抓取姿态。")
            if not base["cells"]:
                base["status"] = "等待可辨认的彩色方块与手写数字。"
                return base, frame.copy(), []
            grid = {"quads": [np.float32(c["quad_px"]) for c in base["cells"]],
                    "mode": base["grid_mode"], "score": 0}
            return self.read_numbers(frame, grid, [item["tile"] for item in debug],
                                     [c["color"] for c in base["cells"]], base)
        grid = locate_grid(frame)
        if grid is None:
            return {"grid_found": False, "complete": False, "cells": [], "status": "未找到可信的完整九宫格；等待清晰画面。",
                    "model_path": str(self.predictor.path)}, frame.copy(), []
        raw_tiles = [warp_quad(frame, quad)[0] for quad in grid["quads"]]
        colors = [color_measure(tile) for tile in raw_tiles]
        return self.read_numbers(frame, grid, raw_tiles, colors)

    def read_numbers(self, frame, grid, raw_tiles, colors, base_report=None):
        rotations = (0, 90, 180, 270) if self.auto_orientation else (0,)
        variants = []
        batch = []
        predictions = []
        orientation_shortcut = False
        for rotation in rotations:
            records = []
            for tile in raw_tiles:
                oriented = orient(tile, rotation)
                if base_report:
                    token, mask, warnings = extract_block_token(oriented, material=self.color_config.get("material", "cube"))
                    tokens = [token] if token is not None else []
                    if not tokens and self.color_config.get("material") != "paper":
                        tokens, mask, warnings = extract_tokens(oriented, reject_boundary_components=True)
                else:
                    tokens, mask, warnings = extract_tokens(oriented)
                indices = []
                for token in sorted(tokens, key=lambda item: item["bbox_px"][0]):
                    if not token["too_wide"]:
                        indices.append(len(batch))
                        batch.append(token["image"])
                records.append({"indices": indices, "mask": mask, "warnings": warnings,
                                "ambiguous": any(token["too_wide"] for token in tokens)})
            variants.append((rotation, records))
            if base_report and self.auto_orientation and rotation == 0:
                predictions = self.predictor.predict_many(batch, "light")
                if upright_is_decisive(records, predictions):
                    orientation_shortcut = True
                    break
        predictions.extend(self.predictor.predict_many(batch[len(predictions):], "light"))
        scored = []
        # 方向证据只按实际有笔迹的格子归一化，空白格不应稀释方向差值。
        informative = max(1, sum(any(records[i]["indices"] or records[i]["ambiguous"] for _, records in variants)
                                 for i in range(len(raw_tiles))))
        for rotation, records in variants:
            scores = [min([predictions[i][0]["top3"][0]["score"] for i in record["indices"]], default=0)
                      if not record["ambiguous"] else 0 for record in records]
            scored.append((float(np.sum(scores) / informative), rotation, records))
        scored.sort(key=lambda item: (-item[0], item[1]))
        best = scored[0]
        # 单数字模型对旋转字形也会给出高分；只有明显胜出时才旋转。
        upright = next((item for item in scored if item[1] == 0), best)
        if best[0] - upright[0] < .10:
            best = upright
        orientation_gap = scored[0][0] - scored[1][0] if len(scored) > 1 else 1.0
        orientation_clear = best[1] == 0 or orientation_gap >= .10
        score, rotation, records = best
        cells, debug = [], []
        for index, (quad, tile, color, record) in enumerate(zip(grid["quads"], raw_tiles, colors, records)):
            tokens = [predictions[i][0] for i in record["indices"]]
            margins = [(p["top3"][0]["score"] - p["top3"][1]["score"]) for p in tokens]
            reliable = bool(tokens) and not record["ambiguous"] and all(p["top3"][0]["score"] >= .80 for p in tokens) and min(margins, default=0) >= .20
            raw_text = "".join(str(p["digit"]) for p in tokens)
            issues = list(base_report["cells"][index]["issues"]) if base_report else []
            if not reliable:
                issues.append("数字待确认" if tokens else "未读到数字")
            if not color["reliable"]:
                issues.append("底色待确认")
            if not orientation_clear:
                issues.append("文字方向存在歧义")
            source_sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
            if min(source_sides) < 36:
                issues.append("格子分辨率不足")
            if (quad[:, 0].min() <= 1 or quad[:, 1].min() <= 1 or
                    quad[:, 0].max() >= frame.shape[1] - 2 or quad[:, 1].max() >= frame.shape[0] - 2):
                issues.append("卡纸触及画面边界")
            center = quad.mean(axis=0)
            cells.append({"id": f"g{index // 3 + 1}{index % 3 + 1}", "row": index // 3 + 1, "column": index % 3 + 1,
                          "quad_px": np.round(quad, 2).tolist(), "center_px": np.round(center, 2).tolist(),
                          "color": color, "number": raw_text if reliable and orientation_clear else None, "raw_number": raw_text or None,
                          "number_score": round(min((p["top3"][0]["score"] for p in tokens), default=0), 4),
                          "digit_predictions": tokens, "confirmed": not issues, "issues": issues})
            if base_report:
                original = base_report["cells"][index]
                cells[-1].update({key: original[key] for key in ("id", "row", "column", "center_px", "clipped", "silhouette_angle_deg_mod90")})
                cells[-1]["number_required"] = True
            debug.append({"id": cells[-1]["id"], "tile": tile, "mask": record["mask"],
                          "previews": [predictions[i][1] for i in record["indices"]]})
        numbers = [c["raw_number"] for c in cells if c["raw_number"] is not None and c["number_score"] >= .80]
        duplicates = [n for n, count in Counter(numbers).items() if count > 1]
        distinct_colors = palette_distinct(colors)
        warnings = []
        if duplicates and not base_report:
            warnings.append(f"数字候选出现重复{duplicates}，与九格各不相同的条件冲突，需确认；未强行改写结果。")
        if distinct_colors < 9 and not base_report:
            warnings.append(f"底色仅能可靠区分为{distinct_colors}组，需确认相近颜色/光照。")
        if (duplicates or distinct_colors < 9) and not base_report:
            for cell in cells:
                cell["confirmed"] = False
        grid_found = base_report["grid_found"] if base_report else True
        complete = bool(grid_found and all(cell["confirmed"] for cell in cells))
        report = {**(base_report or {}), "grid_found": grid_found, "complete": complete, "cells": cells, "grid_mode": grid["mode"],
                  "geometry_score": round(grid["score"], 3), "orientation_cw": rotation,
                  "orientation_margin": None if orientation_shortcut else round(orientation_gap, 4),
                  "orientation_search": "upright_bound" if orientation_shortcut else "exhaustive" if len(rotations) > 1 else "disabled",
                  "distinct_color_groups": distinct_colors,
                  "warnings": (base_report.get("warnings", []) if base_report else []) + warnings, "model_path": str(self.predictor.path),
                  "status": ("九格颜色和数字本帧均通过，等待多帧确认。" if complete else
                             "九格已定位，数字或颜色待确认。" if grid_found else f"检测到{len(cells)}个颜色区域，数字逐个检查；未确认完整九格。"),
                  "coordinates": "原始相机画面像素；g11为画面上方左侧，数字方向校正不改变格子编号。",
                  "note": "模型分数和多帧一致不保证真实正确；未执行机器人动作。"}
        if base_report:
            report["coordinates"] = base_report["coordinates"]
        return report, annotate(frame, report), debug


def annotate(frame, report):
    image = frame.copy()
    scale = max(1., max(frame.shape[:2]) / 1280)
    thickness = max(2, round(2 * scale))
    for cell in report.get("cells", []):
        quad = np.int32(cell["quad_px"])
        color = (80, 220, 100) if cell["confirmed"] else (0, 175, 255)
        cv2.polylines(image, [quad], True, color, thickness)
        p = tuple(np.int32(quad[0] + [3 * scale, 24 * scale]))
        key = cell["color"].get("key", "")
        text = f"{cell['id']} {key}" if report.get("mode") == "blocks" else f"{cell['id']} {key} {cell['number'] if cell['number'] is not None else '?'}"
        cv2.putText(image, text, p, cv2.FONT_HERSHEY_SIMPLEX, .65 * scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, p, cv2.FONT_HERSHEY_SIMPLEX, .65 * scale, color, max(1, round(scale)), cv2.LINE_AA)
    return image


class Stability:
    def __init__(self, window=5, required=4):
        self.history = deque(maxlen=window)
        self.required = required

    def update(self, report):
        if not report["grid_found"] or not report["complete"]:
            self.history.clear()
            report.update(stable=False, matching_frames=0)
            return report
        signature = tuple((c["number"], c["color"]["name"]) for c in report["cells"])
        centers = np.asarray([c["center_px"] for c in report["cells"]])
        colors = np.asarray([c["color"]["lab"] for c in report["cells"]])
        size = np.mean([np.sqrt(cv2.contourArea(np.float32(c["quad_px"]))) for c in report["cells"]])
        if self.history:
            previous = self.history[-1]
            if (previous[0] != signature or np.max(np.linalg.norm(centers - previous[1], axis=1)) > size * .18
                    or np.max(np.linalg.norm(colors - previous[2], axis=1)) > 8):
                self.history.clear()
        self.history.append((signature, centers, colors))
        count = int(sum(bool(old[0] == signature and np.max(np.linalg.norm(colors - old[2], axis=1)) <= 8) for old in self.history))
        report.update(stable=bool(count >= self.required), matching_frames=count)
        if report["stable"]:
            report["status"] = "九格读取连续一致。"
        return report


def save_report(frame, report, marked, debug, directory=OUTPUT / "results"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    write_image(directory / f"{stamp}_raw.png", frame)
    write_image(directory / f"{stamp}_detected.png", marked)
    for item in debug:
        write_image(directory / f"{stamp}_{item['id']}_card.png", item["tile"])
        write_image(directory / f"{stamp}_{item['id']}_ink.png", item["mask"])
        for i, preview in enumerate(item["previews"], 1):
            preview.save(directory / f"{stamp}_{item['id']}_digit{i}.png")
    path = directory / f"{stamp}.json"
    path.write_text(json.dumps({"timestamp": stamp, **report}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
