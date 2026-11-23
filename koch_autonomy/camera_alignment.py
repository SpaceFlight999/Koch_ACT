"""Compare the live global camera with fixed, named points in a reference frame."""
from hashlib import file_digest
from itertools import combinations
from pathlib import Path
from queue import Empty
from time import monotonic

import cv2
import numpy as np

from camera_vision import read_image, write_image
from grid_reader.cameras import resolve_named_camera
from koch_project_paths import ROOT, resolve_config_reference

from .reverse import read_json, save_json


DEFAULT_PROFILE = ROOT / 'configs/replay_camera_alignment.json'
PATCH_RADIUS = 16
SEARCH_RADIUS = 100
SOURCE_SEARCH_RADIUS = 65
SOURCE_WIDE_SEARCH_RADIUS = 135
SOURCE_GUIDED_SEARCH_RADIUS = 32
REQUIRED_FRAMES = 4
STABLE_SECONDS = 1.0
SOURCE_ADJUST_HINT = '让四个 S 区托座和上层方块完整入画，避免夹爪遮住托座外沿。'


def holder_edges(gray):
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), .55)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    return np.minimum(90, cv2.magnitude(gx, gy))


def fit_holder_surfaces(reference, current):
    """Prefer matches spread over the fixture over many overlapping front-edge matches."""
    candidates = []
    for indices in combinations(range(len(reference)), 3):
        points = reference[list(indices)]
        if cv2.contourArea(points) < 1800:
            continue
        matrix = cv2.getAffineTransform(points, current[list(indices)])
        if np.linalg.norm(matrix[:, :2] - np.eye(2), ord=2) > .24:
            continue
        residual = np.linalg.norm(cv2.transform(reference[None], matrix)[0] - current, axis=1)
        inliers = residual <= 3
        support = reference[inliers]
        if len(support) >= 5 and np.ptp(support[:, 1]) >= 40:
            area = cv2.contourArea(cv2.convexHull(support))
            candidates.append((len(support), area, -float(residual[inliers].mean()), inliers))
    if not candidates:
        return None, None
    best_count = max(item[0] for item in candidates)
    viable = (item for item in candidates if item[0] >= best_count - 2)
    inliers = max(viable, key=lambda item: (item[1], item[0], item[2]))[3]
    points = np.c_[reference[inliers], np.ones(inliers.sum())]
    matrix = np.linalg.lstsq(points, current[inliers], rcond=None)[0].T
    return matrix, inliers


class CameraAlignment:
    def __init__(self, profile_path):
        self.profile_path = Path(profile_path).resolve()
        self.profile = read_json(self.profile_path)
        self.reference_path = resolve_config_reference(self.profile['reference_image'], self.profile_path).resolve()
        self.reference = read_image(self.reference_path)
        with self.reference_path.open('rb') as stream:
            self.reference_sha256 = file_digest(stream, 'sha256').hexdigest()
        self.size = self.reference.shape[1::-1]
        self.tolerance = self.profile.get('max_shift_px', 6)
        self.source_bounds = self.profile.get('source_bounds')
        stock_path = self.profile.get('stock_reference_image')
        self.uses_holder_surfaces = stock_path is not None
        self.holder_templates = []
        if self.uses_holder_surfaces and not self.source_bounds:
            raise ValueError('托座轮廓定位需要 S 区画面范围。')
        if not 0 < self.tolerance <= 10:
            raise ValueError('对位容差须在 0–10 像素之间。')
        self.anchors = []
        gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(self.reference, cv2.COLOR_BGR2HSV)[:, :, 1]
        if self.uses_holder_surfaces:
            stock = read_image(resolve_config_reference(stock_path, self.profile_path))
            if stock.shape != self.reference.shape:
                raise ValueError('满载和空托座参考图必须同尺寸。')
            stock_gray = cv2.cvtColor(stock, cv2.COLOR_BGR2GRAY)
            empty_edges, stock_edges = holder_edges(gray), holder_edges(stock_gray)
        for anchor in self.profile['anchors']:
            x, y = anchor['xy']
            r = anchor.get('radius', PATCH_RADIUS)
            if not r <= x < self.size[0] - r or not r <= y < self.size[1] - r:
                raise ValueError(f'{anchor["name"]} 太靠近参考图边缘。')
            patch = gray[y-r:y+r+1, x-r:x+r+1].copy()
            if self.uses_holder_surfaces:
                edge = anchor.get('edge', False)
                first, second = (empty_edges, stock_edges) if edge else (gray, stock_gray)
                area = np.s_[y-r:y+r+1, x-r:x+r+1]
                self.holder_templates.append((edge, first[area].copy(), second[area].copy()))
            mask = (np.ones(patch.shape, dtype=np.uint8) if self.source_bounds else
                    cv2.erode(np.uint8(saturation[y-r:y+r+1, x-r:x+r+1] <= 40), np.ones((3, 3), np.uint8)))
            if not self.uses_holder_surfaces and (mask.mean() < .45 or patch[mask != 0].std() < 12):
                raise ValueError(f'{anchor["name"]} 缺少明显纹理，请选固定直角或交叉点。')
            self.anchors.append((anchor['name'], (x, y), patch, mask))
        points = np.float32([xy for _, xy, _, _ in self.anchors])
        min_spacing, min_area = (15, .002) if self.source_bounds else (35, .012)
        if self.uses_holder_surfaces:
            min_spacing = 8
        if (len(points) < 4 or len({name for name, *_ in self.anchors}) != len(points)
                or any(np.linalg.norm(a-b) < min_spacing for a, b in combinations(points, 2))
                or cv2.contourArea(cv2.convexHull(points)) < min_area * np.prod(self.size)):
            raise ValueError('至少需要四个不同且分散的固定对准点。')

    def measure(self, frame):
        if tuple(frame.shape[1::-1]) != self.size:
            raise ValueError(f'相机分辨率须为 {self.size[0]}×{self.size[1]}，与参考画面一致。')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = holder_edges(gray) if self.uses_holder_surfaces else None
        search = SOURCE_SEARCH_RADIUS if self.source_bounds else SEARCH_RADIUS
        if self.uses_holder_surfaces:
            search = 55
        readings = self.match_points(gray, search, edges=edges)
        if self.source_bounds:
            matrix, used = self.fit_source_fixture(readings)
            if matrix is None:
                wide = self.match_points(gray, SOURCE_WIDE_SEARCH_RADIUS, edges=edges)
                offsets = np.array([np.subtract(p['current_xy'], p['reference_xy']) for p in wide
                                    if p['current_xy'] is not None])
                if len(offsets) >= 3:
                    nearby = np.linalg.norm(offsets[:, None] - offsets, axis=2) <= SOURCE_GUIDED_SEARCH_RADIUS
                    cluster = nearby[np.argmax(nearby.sum(axis=1))]
                    if cluster.sum() >= 3:
                        center = tuple(map(int, np.rint(np.median(offsets[cluster], axis=0))))
                        readings = self.match_points(gray, SOURCE_GUIDED_SEARCH_RADIUS, center, edges=edges)
                        matrix, used = self.fit_source_fixture(readings)
            if self.uses_holder_surfaces and matrix is not None:
                # The first pass can miss partly covered rear holders. Use the
                # front fit to look for them again in local neighborhoods.
                guided = self.match_points(gray, 20, edges=edges, prediction=matrix)
                refined, refined_used = self.fit_source_fixture(guided)
                if refined is not None:
                    def coverage(points, names):
                        support = np.float32([p['reference_xy'] for p in points if p['name'] in names])
                        return cv2.contourArea(cv2.convexHull(support))

                    if coverage(guided, refined_used) > coverage(readings, used):
                        matrix, used, readings = refined, refined_used, guided
        if self.uses_holder_surfaces:
            aligned = matrix is not None and all(r['shift_px'] <= self.tolerance for r in readings if r['name'] in used)
        else:
            aligned = all(r['shift_px'] is not None and r['shift_px'] <= self.tolerance for r in readings)
        if aligned and not self.uses_holder_surfaces:
            reference = np.float32([(*r['reference_xy'], 1) for r in readings])
            current = np.float32([r['current_xy'] for r in readings])
            transform = np.linalg.lstsq(reference, current, rcond=None)[0]
            aligned = bool(np.linalg.norm(reference @ transform - current, axis=1).max() <= 3)
        report = {'aligned': aligned, 'camera_name': self.profile['camera_name'],
                  'reference_sha256': self.reference_sha256, 'max_shift_px': self.tolerance, 'points': readings}
        if self.source_bounds:
            report.update(source_ready=matrix is not None, source_inliers=used,
                          source_transform=matrix.tolist() if matrix is not None else None)
            if self.uses_holder_surfaces:
                report['source_matcher'] = 'holder_surfaces'
        return report

    def match_points(self, gray, search_radius, offset=(0, 0), *, edges=None, prediction=None):
        height, width = gray.shape
        readings = []
        for index, (name, (x, y), patch, mask) in enumerate(self.anchors):
            radius = patch.shape[0] // 2
            reach = search_radius + radius
            if prediction is None:
                px, py = x + offset[0], y + offset[1]
            else:
                px, py = map(int, np.rint(prediction @ [x, y, 1]))
            x0, y0 = max(0, px-reach), max(0, py-reach)
            area = np.s_[y0:min(height, py+reach+1), x0:min(width, px+reach+1)]
            if self.uses_holder_surfaces:
                edge, empty, stock = self.holder_templates[index]
                search = (edges if edge else gray)[area]
                response = np.maximum(cv2.matchTemplate(search, empty, cv2.TM_CCOEFF_NORMED),
                                      cv2.matchTemplate(search, stock, cv2.TM_CCOEFF_NORMED))
            else:
                response = cv2.matchTemplate(gray[area], patch, cv2.TM_CCOEFF_NORMED,
                                             mask=None if self.source_bounds else mask)
            response = np.where(np.isfinite(response), np.clip(response, -1, 1), -1)
            _, score, _, (u, v) = cv2.minMaxLoc(response)
            suppression, minimum, margin = (8, .66, .025) if self.uses_holder_surfaces else (10, .68, .03)
            response[max(0, v-suppression):v+suppression+1, max(0, u-suppression):u+suppression+1] = -1
            unique = score >= minimum and score - float(response.max()) >= margin
            cx, cy = x0 + u + radius, y0 + v + radius
            readings.append({'name': name, 'reference_xy': [x, y],
                             'current_xy': [cx, cy] if unique else None,
                             'score': round(score, 3),
                             'shift_px': round(float(np.hypot(cx-x, cy-y)), 2) if unique else None})
        return readings

    def fit_source_fixture(self, readings):
        matches = [p for p in readings if p['current_xy'] is not None]
        minimum = 5 if self.uses_holder_surfaces else 4
        if len(matches) < minimum:
            return None, []
        reference = np.float32([p['reference_xy'] for p in matches])
        current = np.float32([p['current_xy'] for p in matches])
        if self.uses_holder_surfaces:
            matrix, inliers = fit_holder_surfaces(reference, current)
        else:
            matrix, inliers = cv2.estimateAffine2D(reference, current, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if matrix is None:
            return None, []
        valid = inliers.ravel().astype(bool)
        support = reference[valid]
        if (len(support) < minimum or np.ptp(support[:, 0]) < 125 or np.ptp(support[:, 1]) < 40
                or cv2.contourArea(cv2.convexHull(support)) < 2800
                or np.linalg.norm(matrix[:, :2] - np.eye(2), ord=2) > (.24 if self.uses_holder_surfaces else .16)):
            return None, []
        left, top, right, bottom = self.source_bounds
        bounds = np.float32([[(left, top), (right, top), (right, bottom), (left, bottom)]])
        visible = cv2.transform(bounds, matrix)[0]
        if not ((visible >= 4) & (visible < np.array(self.size) - 4)).all():
            return None, []
        return matrix, [p['name'] for p, good in zip(matches, valid) if good]

    def register_source(self, frame, report):
        """Compensate small camera shifts for source color sampling."""
        if self.source_bounds:
            matrix = report['source_transform']
            if matrix is None:
                return None
            return cv2.warpAffine(frame, np.array(matrix), self.size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        points = report['points']
        if any(p['shift_px'] is None or p['shift_px'] > 24 for p in points):
            return None
        if max(p['shift_px'] for p in points) <= 3 and report['aligned']:
            return frame
        reference = np.float32([p['reference_xy'] for p in points])
        current = np.float32([p['current_xy'] for p in points])
        matrix, _ = cv2.estimateAffinePartial2D(current, reference, method=cv2.LMEDS)
        if matrix is None:
            return None
        predicted = cv2.transform(current.reshape(1, -1, 2), matrix)[0]
        scale = np.hypot(matrix[0, 0], matrix[1, 0])
        rotation = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
        if (np.linalg.norm(predicted - reference, axis=1).max() > 2
                or abs(rotation) > 1 or abs(scale - 1) > .02):
            return None
        return cv2.warpAffine(frame, matrix, self.size)


def open_alignment(profile_path=DEFAULT_PROFILE, output=None):
    from tkinter import BOTH, LEFT, RIGHT, X, Canvas, StringVar, Tk, ttk
    from PIL import Image, ImageTk
    from camera_vision_app import CameraStream

    alignment = CameraAlignment(profile_path)
    if output is not None:
        if output.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise ValueError('对位截图路径须以 .png、.jpg 或 .jpeg 结尾。')
        if output.resolve() == alignment.reference_path or output.with_suffix('.json').resolve() == alignment.profile_path:
            raise ValueError('对位截图不能覆盖参考图或对位配置。')
    index, camera = resolve_named_camera(alignment.profile['camera_name'])
    root = Tk()
    root.title('Koch · 全局相机对位')
    root.geometry('1120x790')
    root.minsize(700, 490)
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Microsoft YaHei UI', 11), background='#F3F5F4', foreground='#18392F')
    panel = ttk.Frame(root, padding=16)
    panel.pack(fill=BOTH, expand=True)
    ttk.Label(panel, text='全局相机对位', font=('Microsoft YaHei UI', 20, 'bold')).pack(anchor='w')
    instructions = SOURCE_ADJUST_HINT if alignment.source_bounds else '手调支架，使圆点与参考十字重合。'
    ttk.Label(panel, text=f'{camera}  ·  {instructions}').pack(anchor='w', pady=(3, 10))
    canvas = Canvas(panel, background='#13231F', highlightthickness=0)
    canvas.pack(fill=BOTH, expand=True)
    controls = ttk.Frame(panel)
    controls.pack(fill=X, pady=(10, 0))
    status = StringVar(value='相机启动中…')
    ttk.Label(controls, textvariable=status).pack(side=LEFT)
    ttk.Button(controls, text='关闭（Esc）', command=root.destroy).pack(side=RIGHT)
    mode = StringVar(value='live')
    middle_mode = ('校正预览', 'corrected') if alignment.source_bounds else ('重影', 'overlay')
    for label, value in (('参考画面', 'reference'), middle_mode, ('实时画面', 'live')):
        ttk.Radiobutton(controls, text=label, variable=mode, value=value).pack(side=RIGHT, padx=6)
    detail = StringVar(value='正在查找 S 区托座轮廓…' if alignment.source_bounds else '十字为参考点，圆点为实时位置。')
    ttk.Label(panel, textvariable=detail, wraplength=1030).pack(anchor='w', pady=(6, 0))
    good_frames, stable_since = 0, None
    ready = saved = False
    last_frame, last_report, last_stamp = None, None, None
    stream = CameraStream(index, alignment.size)
    root.bind('<Escape>', lambda _: root.destroy())

    def render(frame, report):
        view = mode.get()
        reference_mode = view == 'reference'
        if reference_mode:
            frame = alignment.reference
        elif view == 'overlay' and frame.shape == alignment.reference.shape:
            frame = cv2.addWeighted(frame, .65, alignment.reference, .35, 0)
        elif view == 'corrected' and report and report['source_ready']:
            frame = alignment.register_source(frame, report)
        scale = min(canvas.winfo_width() / frame.shape[1], canvas.winfo_height() / frame.shape[0])
        if scale <= .01:
            return
        size = tuple(round(v * scale) for v in frame.shape[1::-1])
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        canvas.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), master=root)
        canvas.delete('all')
        left, top = (canvas.winfo_width() - size[0]) / 2, (canvas.winfo_height() - size[1]) / 2
        canvas.create_image(left, top, image=canvas.photo, anchor='nw')
        if alignment.source_bounds:
            x0, y0, x1, y1 = alignment.source_bounds
            area = np.float32([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]])
            fitted = report and report['source_ready']
            if fitted and view == 'live':
                area = cv2.transform(area, np.array(report['source_transform']))
            coords = [(left + x * scale, top + y * scale) for x, y in area[0]]
            canvas.create_polygon(*coords, fill='', width=2, outline='#97F5B7' if fitted else '#6BEBFF')
            return
        for i, (name, (x, y), _, _) in enumerate(alignment.anchors):
            x, y = left + x * scale, top + y * scale
            reading = report['points'][i] if report else {}
            current = reading.get('current_xy') if not reference_mode else None
            if current:
                cx, cy = left + current[0] * scale, top + current[1] * scale
                color = '#97F5B7' if reading['shift_px'] <= alignment.tolerance else '#FFCE75'
                if np.hypot(cx-x, cy-y) >= 7:
                    canvas.create_line(cx, cy, x, y, width=2, fill=color, arrow='last')
                canvas.create_oval(cx-5, cy-5, cx+5, cy+5, width=2, outline=color)
            for coords in ((x-7, y, x+7, y), (x, y-7, x, y+7)):
                canvas.create_line(*coords, fill='#142129', width=5)
                canvas.create_line(*coords, fill='#6BEBFF', width=2)
            label = canvas.create_text(x+8, y-8, text=name, anchor='sw', fill='white',
                                       font=('Microsoft YaHei UI', 9, 'bold'))
            box = canvas.create_rectangle(*canvas.bbox(label), fill='#1A2F38', outline='')
            canvas.tag_lower(box, label)

    def save_frame():
        nonlocal saved
        if not ready or last_stamp is None or monotonic() - last_stamp > .75:
            status.set('锁紧支架，等画面保持对齐后再保存。')
            return
        captured = alignment.register_source(last_frame, last_report) if alignment.source_bounds else last_frame
        write_image(output, captured)
        save_json(output.with_suffix('.json'), {**last_report, 'camera_index': index,
                                               'source_registration_applied': captured is not last_frame})
        saved = True
        root.destroy()

    if output is not None:
        ttk.Button(controls, text='对齐后保存', command=save_frame).pack(side=RIGHT, padx=8)

    def tick():
        nonlocal last_frame, last_report, last_stamp, good_frames, stable_since, ready
        try:
            status.set(stream.messages.get_nowait())
        except Empty:
            pass
        try:
            last_stamp, last_frame = stream.frames.get_nowait()
        except Empty:
            if last_stamp is not None and monotonic() - last_stamp > .75:
                good_frames, stable_since, ready = 0, None, False
                status.set('等待新画面…')
        else:
            try:
                last_report = alignment.measure(last_frame)
                matched = last_report.get('source_ready', last_report['aligned'])
                if matched:
                    stable_since = last_stamp if stable_since is None else stable_since
                    good_frames = min(good_frames + 1, REQUIRED_FRAMES)
                else:
                    good_frames, stable_since = 0, None
                ready = good_frames >= REQUIRED_FRAMES and last_stamp - stable_since >= STABLE_SECONDS
                if alignment.source_bounds:
                    if ready:
                        status.set('S 区可用于任务 · 请锁紧支架')
                        detail.set('程序已自动补偿机位偏移。锁紧后仍显示可用，即可关闭窗口。')
                    elif matched:
                        status.set('已找到 S 区 · 请保持支架静止')
                        detail.set('正在确认画面稳定。')
                    else:
                        status.set('暂未找到 S 区')
                        detail.set(SOURCE_ADJUST_HINT)
                else:
                    status.set('已对齐 · 锁紧后仍对齐即可保存' if ready else '正在对位 · 请让圆点与十字重合')
                    detail.set('    '.join(f'{p["name"]} {p["shift_px"]:g}px' if p['shift_px'] is not None
                                           else f'{p["name"]} 未找到' for p in last_report['points']))
            except ValueError as error:
                good_frames, stable_since, ready, last_report = 0, None, False, None
                status.set(str(error))
        render(last_frame if last_frame is not None else alignment.reference, last_report)
        root.after(80, tick)

    root.after(50, tick)
    try:
        root.mainloop()
    finally:
        stream.stop()
        stream.thread.join(timeout=3)
    return saved if output is not None else ready
