"""Four target papers and eight separately calibrated visible source faces."""
from collections import Counter
import copy
from itertools import combinations
import time

import cv2
import numpy as np

from camera_vision import classify_sample, read_image
from grid_reader.core import GridReader, annotate, warp_quad
from handwritten_digits.camera import orient
from .planner import SLOTS, TARGETS, validate_targets


def corner_order(cells):
    if len(cells) != 4:
        raise ValueError('必须恰好检测到四张彩纸；请框选 T 区，排除其他彩色物体。')
    centers = np.float32([c['center_px'] for c in cells])
    if len(cv2.convexHull(centers)) != 4:
        raise ValueError('四张彩纸未形成可信的四角布局。')
    by_y = sorted(range(4), key=lambda i: centers[i, 1])
    top = sorted(by_y[:2], key=lambda i: centers[i, 0])
    bottom = sorted(by_y[2:], key=lambda i: centers[i, 0])
    order = top + bottom
    size = np.median([np.sqrt(cv2.contourArea(np.float32(c['quad_px']))) for c in cells])
    # Oblique wrist views can put the centers only one card-width apart along x.
    if (min(centers[bottom, 1]) - max(centers[top, 1]) < size * .5 or
            any(centers[b, 0] - centers[a, 0] < size for a, b in (top, bottom))):
        raise ValueError('四角间距或朝向不清晰，不能可靠编号。')
    return order


def find_four_corners(cells):
    """Find a single board-sized group in the rotated full-frame color detections."""
    candidates = []
    eligible = [i for i, cell in enumerate(cells) if not cell['clipped'] and cell['color']['reliable']]
    for indices in combinations(eligible, 4):
        group = [cells[i] for i in indices]
        if len({cell['color']['key'] for cell in group}) != 4:
            continue
        try:
            order = corner_order(group)
        except ValueError:
            continue
        areas = [cv2.contourArea(np.float32(cell['quad_px'])) for cell in group]
        if min(areas) <= 0 or max(areas) > 2.4 * min(areas):
            continue
        size = np.sqrt(np.median(areas))
        top_left, top_right, bottom_left, bottom_right = [np.array(group[i]['center_px']) for i in order]
        top, bottom = top_right - top_left, bottom_right - bottom_left
        left, right = bottom_left - top_left, bottom_right - top_right
        spans = np.linalg.norm([top, bottom, left, right], axis=1) / size
        skew = (np.linalg.norm(top - bottom) + np.linalg.norm(left - right)) / size
        if spans.min() < 1. or spans.max() > 3.1 or skew > .75:
            continue
        candidates.append((float(skew), [indices[i] for i in order]))
    if not candidates:
        raise ValueError(f'全画面检测到 {len(cells)} 个颜色区域，未定位到完整的四角任务卡。')
    candidates.sort()
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .15:
        raise ValueError('全画面有多组相近的四角彩纸，请移开 T 区周围的干扰物。')
    return candidates[0][1]


class TargetReader:
    """Rotate the whole frame before assigning IDs. Ambiguous orientation rejects.

    rotation_cw=None tries four orientations; a calibrated/marker-rectified
    frame can supply 0. A single-paper scan must supply its known scan pose ID.
    """
    def __init__(self, predictor, color_config, rotation_cw=None, target=None, verifier=None, *,
                 quantity_rule='parity', locate_four_corners=False):
        if rotation_cw not in (None, 0, 90, 180, 270):
            raise ValueError('方向须为自动或顺时针 0/90/180/270 度。')
        if target is not None and target not in TARGETS:
            raise ValueError('单纸扫描必须指定已到达的 T1–T4。')
        self.reader = GridReader(predictor, auto_orientation=False, mode='combined', color_config=color_config)
        self.rotation, self.target = rotation_cw, target
        self.verifier = verifier
        self.quantity_rule = quantity_rule
        self.locate_four_corners = locate_four_corners and target is None

    def read(self, frame):
        candidates = []
        for rotation in ((0, 90, 180, 270) if self.rotation is None else (self.rotation,)):
            rotated = orient(frame, rotation)
            base, _, debug = self.reader.read_blocks(rotated, draw=False)
            expected = 1 if self.target else 4
            try:
                if self.locate_four_corners:
                    order = find_four_corners(base['cells'])
                else:
                    if len(base['cells']) != expected:
                        raise ValueError(f'期望 {expected} 张彩纸，检测到 {len(base["cells"])} 个颜色区域。')
                    order = [0] if self.target else corner_order(base['cells'])
            except ValueError as exc:
                candidates.append((0., rotation, {'complete': False, 'cells': [], 'status': str(exc)}, rotated))
                continue
            base['cells'] = [base['cells'][i] for i in order]
            tiles = [debug[i]['tile'] for i in order]
            ids = [self.target] if self.target else TARGETS
            for i, (cell, slot) in enumerate(zip(base['cells'], ids)):
                cell.update(id=slot, row=i // 2 + 1, column=i % 2 + 1)
            base.update(grid_found=True, coordinates='顺时针旋正后的画面像素；T1左上、T2右上、T3左下、T4右下。')
            grid = {'quads': [np.float32(c['quad_px']) for c in base['cells']], 'mode': 'four_corners', 'score': 0.}
            report, _, digits_debug = self.reader.read_numbers(rotated, grid, tiles, [c['color'] for c in base['cells']], base)
            if self.verifier is not None:
                # A paper-tuned model can confidently confuse 3/8. A disagreeing
                # independent model rejects the read; it never overwrites digits.
                for cell, item in zip(report['cells'], digits_debug):
                    checks = self.verifier.predict_many(item['previews'], 'light')
                    alternate = ''.join(str(p[0]['digit']) for p in checks)
                    agreed = bool(checks) and alternate == cell['number'] and all(
                        p[0]['top3'][0]['score'] >= .60 for p in checks)
                    cell['verification_number'] = alternate or None
                    cell['digit_models_agree'] = agreed
                    if not agreed:
                        cell['confirmed'] = False
                        cell['issues'].append('纸面模型与独立数字模型不一致或复核分数不足。')
                report['complete'] = bool(report['complete'] and all(c['confirmed'] for c in report['cells']))
            if not self.target:
                try:
                    validate_targets({c['id']: {'color': c['color'].get('key'), 'number': c['number']}
                                      for c in report['cells']}, self.quantity_rule)
                except ValueError as exc:
                    report['complete'] = False
                    for cell in report['cells']:
                        cell['issues'].append(str(exc))
            # Rejected candidates are not orientation evidence. In particular,
            # a confident primary-model error must not beat a verified rotation.
            report['orientation_primary_score'] = float(np.mean([c['number_score'] for c in report['cells']]))
            score = report['orientation_primary_score'] if report['complete'] else 0.
            candidates.append((score, rotation, report, rotated))
        candidates.sort(key=lambda c: (-c[0], -c[2].get('orientation_primary_score', 0), c[1]))
        score, rotation, report, rotated = candidates[0]
        gap = score - candidates[1][0] if len(candidates) > 1 else None
        issues = list(dict.fromkeys(issue for cell in report['cells'] for issue in cell.get('issues', [])))
        if gap is not None and gap < .10:
            issues.append('方向证据不足，可能存在 6/9 等旋转歧义；需已标定方向或定向标记。')
        targets = {c['id']: {'color': c['color'].get('key'), 'number': c['number']} for c in report['cells']}
        if not self.target and len(targets) == 4:
            try:
                validate_targets(targets, self.quantity_rule)
            except ValueError as exc:
                issues.append(str(exc))
        if self.target and targets and not str(targets[self.target]['number']).isdigit():
            issues.append('单纸数字不完整。')
        complete = bool(report['complete'] and not issues)
        report.update(complete=complete, orientation_cw=rotation, orientation_margin=gap,
                      orientation_source='calibrated' if self.rotation is not None else 'digit_evidence',
                      targets=targets if complete else {}, stable=False,
                      mode='four_targets' if not self.target else 'single_target',
                      independent_digit_check=self.verifier is not None, quantity_rule=self.quantity_rule,
                      status='目标本帧读取通过，等待连续帧确认。' if complete else '; '.join(issues) or report['status'],
                      warnings=issues, coordinates='旋正后的画面像素；编号随整个画面旋正，不沿用原图编号。')
        if issues:
            for cell in report['cells']:
                cell['confirmed'] = False
        return report, annotate(rotated, report)


def validate_quad(value, frame_shape):
    q = np.asarray(value, dtype=np.float32)
    h, w = frame_shape[:2]
    if (q.shape != (4, 2) or not np.isfinite(q).all() or
            not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 100 or
            np.any(q < 0) or np.any(q[:, 0] >= w) or np.any(q[:, 1] >= h)):
        raise ValueError('标定区域须为画面内有效凸四边形，按左上、右上、右下、左下记录。')
    return q


def appearance_difference(tile, reference):
    """Mean Lab pixel difference on the same calibrated visible-face region."""
    a = cv2.cvtColor(tile.astype(np.float32) / 255., cv2.COLOR_BGR2LAB)
    b = cv2.cvtColor(reference.astype(np.float32) / 255., cv2.COLOR_BGR2LAB)
    return float(np.mean(np.linalg.norm(a - b, axis=2)))


class StackReader:
    """Fixed-view stacks with explicit occupied / empty / unknown observations.

    Empty is only available with an independently captured empty scene and
    measured separation thresholds. Unknown colors never imply empty space.
    """
    def __init__(self, config, color_config, region):
        self.config, self.colors = copy.deepcopy(config), color_config
        if region not in ('S', 'T') or config.get('region', region) != region:
            raise ValueError('堆栈区域标定不匹配。')
        self.region = region
        self.slots = SLOTS if region == 'S' else TARGETS
        expected = {f'{s}_L{layer}' for s in self.slots for layer in (1, 2)}
        if set(config.get('faces', {})) != expected:
            raise ValueError(f'{region} 区须分别标定 {region}1_L1 到 {region}4_L2 八块独立可见侧面。')
        if config.get('view_validated') is not True:
            raise ValueError(f'{region} 区侧面与层号尚未完成现场标定。')
        self.empty = None
        if config.get('empty_reference_image'):
            low, high = config.get('empty_max_difference'), config.get('occupied_min_difference')
            if any(type(v) not in (int, float) or not np.isfinite(v) for v in (low, high)) or not 0 <= low < high:
                raise ValueError('空位判据须给出有限的 0≤empty_max_difference<occupied_min_difference。')
            self.empty = read_image(config['empty_reference_image'])
            if list(self.empty.shape[1::-1]) != config['frame_size']:
                raise ValueError('空场景参考分辨率与标定不一致。')

    def read(self, frame):
        if list(frame.shape[1::-1]) != self.config['frame_size']:
            raise ValueError('全局相机分辨率变化，请重新标定。')
        view_verified = False
        if self.config.get('view_markers'):
            check_fixed_view(frame, self.config['view_markers'])
            view_verified = True
        inventory, faces, unknown, issues = {s: [] for s in self.slots}, [], [], []
        occupied_quads = []
        for slot in self.slots:
            states = []
            for layer in (1, 2):
                name = f'{slot}_L{layer}'
                quad = validate_quad(self.config['faces'][name], frame.shape)
                for old in occupied_quads:
                    overlap, _ = cv2.intersectConvexConvex(quad, old)
                    if overlap > 1:
                        raise ValueError('八个侧面采样区域不能重叠，以免重复计数。')
                occupied_quads.append(quad)
                tile, _ = warp_quad(frame, quad, 64)
                color, distance = classify_sample(tile, self.colors)
                if self.colors.get('calibration_required'):
                    color = None
                state = 'occupied' if color is not None else 'unknown'
                difference = None
                if self.empty is not None:
                    reference, _ = warp_quad(self.empty, quad, 64)
                    difference = appearance_difference(tile, reference)
                    if difference <= self.config['empty_max_difference']:
                        state, color = 'empty', None
                    elif difference < self.config['occupied_min_difference']:
                        state, color = 'unknown', None
                states.append(state)
                if state == 'unknown':
                    unknown.append(name)
                if state != 'empty':
                    inventory[slot].append(color)
                faces.append({'id': name, 'state': state, 'color': color, 'lab_distance': distance,
                              'empty_difference': difference, 'quad_px': quad.tolist()})
            if states[0] == 'empty' and states[1] != 'empty':
                issues.append(f'{slot} 上层存在/不明但下层为空，层位或遮挡需复核。')
        complete = not unknown and not issues
        result = {'complete': complete, 'stable': False, 'inventory': inventory,
                'faces': faces, 'unknown': unknown,
                'observed_count': sum(f['state'] == 'occupied' for f in faces),
                'confirmed_empty_count': sum(f['state'] == 'empty' for f in faces),
                'counts_by_color': dict(Counter(c for stack in inventory.values() for c in stack if c is not None)),
                'scope': f'{self.region}_fixed_camera_visible_faces', 'empty_reference_used': self.empty is not None,
                'view_pose_verified': view_verified,
                'issues': issues,
                'status': f'{self.region} 区堆栈本帧读取通过。' if complete else f'层位不可确认：{unknown}; {issues}'}
        if self.region == 'T':
            result['placed'] = copy.deepcopy(inventory)
        return result


class SourceReader(StackReader):
    def __init__(self, config, color_config):
        super().__init__(config, color_config, 'S')


class TargetStackReader(StackReader):
    def __init__(self, config, color_config):
        if not config.get('empty_reference_image'):
            raise ValueError('T 放置后核验必须有当前空场景参考，不能用彩纸颜色冒充方块。')
        super().__init__(config, color_config, 'T')


class StableRead:
    """Temporal agreement only; commit_from_reports also requires verified views.

    Consecutive, fresh captures only; repeated frame IDs never add votes.
    """
    def __init__(self, required=4, max_gap=.6):
        from collections import deque
        if type(required) is not int or required < 2 or type(max_gap) not in (int, float) or not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError('稳定读取至少两帧，最大帧间隔须为正。')
        self.required, self.max_gap = required, max_gap
        self.signature, self.count, self.stamp = None, 0, None
        self.window = deque(maxlen=required)

    def update(self, report, stamp, now=None):
        now = time.monotonic() if now is None else now
        if not np.isfinite(stamp) or not np.isfinite(now):
            raise ValueError('帧时间须为有限数值。')
        if (not report.get('complete') or not 0 <= now - stamp <= self.max_gap or
                self.stamp is not None and stamp <= self.stamp):
            self.signature, self.count = None, 0
            self.window.clear()
        else:
            import json
            signature = json.dumps([report.get('targets'), report.get('inventory'),
                                    report.get('placed'), report.get('orientation_cw'), report.get('scope'),
                                    report.get('empty_reference_used'), report.get('view_pose_verified')], sort_keys=True)
            if signature != self.signature or self.stamp is not None and stamp - self.stamp > self.max_gap:
                self.count = 0
                self.window.clear()
            self.signature = signature
            self.count += 1
            self.window.append(stamp)
        self.stamp = stamp
        return {**report, 'stable': self.count >= self.required, 'matching_frames': self.count,
                'required_frames': self.required,
                'stable_since_monotonic': self.window[0] if self.count >= self.required else None,
                'captured_monotonic': stamp}


def marker_centers(frame, ids):
    if len(ids) != 4 or len(set(ids)) != 4 or any(type(i) is not int or not 0 <= i < 50 for i in ids):
        raise ValueError('须提供四个不同的 0–49 定向标记 ID。')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, found, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(frame)
    found = [] if found is None else found.flatten().tolist()
    if any(found.count(i) != 1 for i in ids):
        raise ValueError('定向标记缺失或重复，不能沿用旧相机变换。')
    source = np.float32([corners[found.index(i)].reshape(4, 2).mean(0) for i in ids])
    return validate_quad(source, frame.shape)


def check_fixed_view(frame, config):
    """Reject view drift, without pretending to correct a 3D policy input."""
    current = marker_centers(frame, config['marker_ids'])
    reference = validate_quad(config['reference_points_px'], frame.shape)
    tolerance = config.get('max_shift_px')
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not 0 < tolerance <= 10:
        raise ValueError('固定视角标记容差须为 0–10 像素内的正数。')
    if float(np.max(np.linalg.norm(current - reference, axis=1))) > tolerance:
        raise ValueError('全局相机/场景基准已偏移，暂停；重新标定，不能沿用旧层位或 ACT 视角。')


def rectify_marked_plane(frame, config):
    """Registration for a single marked plane, NOT a 3D ACT-view correction.

    Four IDs specify TL/TR/BR/BL marker centers. Their physical placement must
    be surveyed in reference_points_px. Missing/duplicate markers fail closed.
    """
    source = marker_centers(frame, config['marker_ids'])
    w, h = config['frame_size']
    target = validate_quad(config['reference_points_px'], (h, w, 3))
    if np.sign(cv2.contourArea(source, oriented=True)) != np.sign(cv2.contourArea(target, oriented=True)):
        raise ValueError('标记顺序或镜像不一致。')
    matrix = cv2.getPerspectiveTransform(source, target)
    result = cv2.warpPerspective(frame, matrix, (w, h))
    # Border pixels are not measurements; expose them for ROI validation.
    visible = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), matrix, (w, h))
    return result, visible, matrix
