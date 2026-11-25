"""Locate and register the global camera's S zone from its holder contours."""
from hashlib import file_digest
from itertools import combinations
from pathlib import Path
from queue import Empty
from time import monotonic

import cv2
import numpy as np

from koch_vision.colors import read_image, write_image
from koch_vision.grid_reader.cameras import resolve_named_camera
from koch_robot.paths import ROOT, resolve_config_reference

from .reverse import read_json, save_json


DEFAULT_PROFILE = ROOT / 'configs/replay_camera_alignment.json'
PATCH_RADIUS = 16
SEARCH_RADIUS = 100
SOURCE_SEARCH_RADIUS = 65
SOURCE_WIDE_SEARCH_RADIUS = 135
SOURCE_GUIDED_SEARCH_RADIUS = 32
REQUIRED_FRAMES = 4
STABLE_SECONDS = 1.0
SOURCE_ADJUST_HINT = '让四个托座的底沿、两侧和最高一层方块完整入画。尽量看清后排下层，避免夹爪遮挡。'
LOCATOR_SCALE = .4
HOLDER_SCALES = (.65, .72, .78, .83, .89, .95, 1., 1.06, 1.12, 1.2)
ROTATED_LOCATOR_SCALE = .25
HOLDER_ROTATIONS = (-20, -15, -10, -5, 5, 10, 15, 20)
ROTATED_HOLDER_SCALES = (.72, .82, .92, 1.02, 1.12, 1.22, 1.32)


def holder_edges(gray):
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), .55)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    return np.minimum(90, cv2.magnitude(gx, gy))


def small_edges(gray, scale=LOCATOR_SCALE):
    return holder_edges(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))


def fit_holder_surfaces(reference, current):
    """Fit the fixed fixture, allowing modest shear without extrapolating it from one rear hit."""
    origin = reference.mean(axis=0)
    centered = reference - origin
    x, y = centered.T
    count = len(x)
    # Two weak constraints keep the linear part close to a rotation and uniform
    # scale when matches cluster along the front edge. Rear support can override them.
    design = np.zeros((count, 2, 6))
    design[:, 0, :3] = np.c_[x, y, np.ones(count)]
    design[:, 1, 3:] = design[:, 0, :3]
    prior = 45 * np.array([[1, 0, 0, 0, -1, 0], [0, 1, 0, 1, 0, 0]])

    def fit(indices):
        system = np.vstack((design[indices].reshape(-1, 6), prior))
        values = np.r_[current[indices].ravel(), 0, 0]
        raw = np.linalg.lstsq(system, values, rcond=None)[0].reshape(2, 3)
        raw[:, 2] -= raw[:, :2] @ origin
        return raw

    best = None
    seen = set()
    for indices in combinations(range(count), 2):
        if np.linalg.norm(reference[indices[0]] - reference[indices[1]]) < 100:
            continue
        matrix = fit(list(indices))
        residual = np.linalg.norm(cv2.transform(reference[None], matrix)[0] - current, axis=1)
        inliers = residual <= 6
        signature = tuple(inliers)
        if inliers.sum() < 5 or signature in seen:
            continue
        seen.add(signature)
        matrix = fit(inliers)
        linear = matrix[:, :2]
        if np.linalg.norm(linear - np.eye(2), ord=2) > .24:
            continue
        residual = np.linalg.norm(cv2.transform(reference[None], matrix)[0] - current, axis=1)
        inliers = residual <= 5
        support = reference[inliers]
        if len(support) < 5 or np.ptp(support[:, 0]) < 125 or np.ptp(support[:, 1]) < 25:
            continue
        shear = np.hypot(linear[0, 0]-linear[1, 1], linear[0, 1]+linear[1, 0])
        area = cv2.contourArea(cv2.convexHull(support))
        score = np.sum(np.maximum(0, 1 - (residual / 5)**2)) + min(area / 8000, 1) - 14 * shear
        if best is None or score > best[0]:
            best = score, matrix, inliers
    return (None, None) if best is None else best[1:]


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
        self.source_visibility_bounds = self.profile.get('source_visibility_bounds', self.source_bounds)
        stock_path = self.profile.get('stock_reference_image')
        self.uses_holder_surfaces = stock_path is not None
        self.holder_templates = []
        self.holder_locator = []
        self.common_rotated_locator = []
        self.rotated_holder_locator = []
        self.source_seed = None
        self.source_seed_radius = 14
        if self.uses_holder_surfaces and not self.source_bounds:
            raise ValueError('托座轮廓定位需要 S 区画面范围。')
        if self.source_bounds and self.source_visibility_bounds:
            x0, y0, x1, y1 = self.source_bounds
            left, top, right, bottom = self.source_visibility_bounds
            if not (0 <= x0 <= left < right <= x1 < self.size[0]
                    and 0 <= y0 <= top < bottom <= y1 < self.size[1]):
                raise ValueError('S 区可见范围须在参考画面的 S 区预览框内。')
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
            regions = self.profile.get('holder_search_regions')
            if regions:
                self.build_holder_locator(regions, gray, stock_gray)
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

    def build_holder_locator(self, regions, *references):
        left = min(rect[0] for rect in regions)
        top = min(rect[1] for rect in regions)
        right = max(rect[2] for rect in regions)
        bottom = max(rect[3] for rect in regions)
        self.locator_origin = np.array([left, top])
        mask = np.zeros((bottom - top, right - left), np.uint8)
        for x0, y0, x1, y1 in regions:
            mask[y0-top:y1-top, x0-left:x1-left] = 1
        for gray in references:
            crop = gray[top:bottom, left:right]
            template = small_edges(crop)
            for scale in HOLDER_SCALES:
                scaled = cv2.resize(template, None, fx=scale, fy=scale)
                scaled_mask = cv2.resize(mask, scaled.shape[::-1], interpolation=cv2.INTER_NEAREST)
                self.holder_locator.append((np.eye(2) * scale, np.zeros(2), scaled, scaled_mask))
            self.build_rotated_templates(crop, mask)

    def build_rotated_templates(self, crop, mask):
        small = cv2.resize(crop, None, fx=ROTATED_LOCATOR_SCALE, fy=ROTATED_LOCATOR_SCALE,
                           interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, small.shape[::-1], interpolation=cv2.INTER_NEAREST)
        h, w = small.shape
        corners = np.float32([[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]])
        for angle in HOLDER_ROTATIONS:
            for scale in ROTATED_HOLDER_SCALES:
                transform = cv2.getRotationMatrix2D((0, 0), angle, scale)
                rotated = cv2.transform(corners, transform)[0]
                origin = np.floor(rotated.min(axis=0))
                shape = tuple((np.ceil(rotated.max(axis=0)) - origin + 1).astype(int))
                transform[:, 2] = -origin
                surface = cv2.warpAffine(small, transform, shape, borderMode=cv2.BORDER_REPLICATE)
                selected = cv2.warpAffine(small_mask, transform, shape, flags=cv2.INTER_NEAREST)
                templates = (self.common_rotated_locator if abs(angle) <= 15 and 1.02 <= scale <= 1.12
                             else self.rotated_holder_locator)
                templates.append((transform[:, :2], transform[:, 2], holder_edges(surface), selected))

    def search_holder_region(self, gray, *, need_location):
        groups = ((self.common_rotated_locator, ROTATED_LOCATOR_SCALE, 22),
                  (self.holder_locator, LOCATOR_SCALE, 24),
                  (self.rotated_holder_locator, ROTATED_LOCATOR_SCALE, 22)) if need_location else (
                  (self.holder_locator, LOCATOR_SCALE, 24),)
        edges_by_scale = {}
        for templates, scale, radius in groups:
            if not templates:
                continue
            if scale not in edges_by_scale:
                edges_by_scale[scale] = small_edges(gray, scale)
            edges = edges_by_scale[scale]
            hits = []
            for linear, template_offset, template, mask in templates:
                response = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED, mask=mask)
                response[~np.isfinite(response)] = -1
                _, score, _, corner = cv2.minMaxLoc(response)
                if score >= (.45 if radius == 22 else .47):
                    offset = (np.asarray(corner) + template_offset) / scale - linear @ self.locator_origin
                    hits.append((score, np.column_stack((linear, offset))))
            for _, matrix in sorted(hits, key=lambda match: match[0], reverse=True)[:32]:
                yield matrix, radius

    def refine_holder_region(self, gray, seed, radius):
        rough = cv2.warpAffine(gray, seed, self.size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        edges = holder_edges(rough)
        readings = self.match_points(rough, radius, edges=edges)
        matrix, used = self.fit_source_fixture(readings)
        if matrix is None:
            return None, [], readings
        guided = self.match_points(rough, 15, edges=edges, prediction=matrix)
        refined, refined_used = self.fit_source_fixture(guided)
        if refined is not None and self.coverage(guided, refined_used) > self.coverage(readings, used):
            matrix, used, readings = refined, refined_used, guided
        matrix = seed @ np.vstack((matrix, [0, 0, 1]))
        if not self.source_visible(matrix):
            return None, [], readings
        for point in readings:
            if point['current_xy'] is not None:
                xy = np.rint(seed @ [*point['current_xy'], 1]).astype(int)
                point['current_xy'] = xy.tolist()
                point['shift_px'] = round(float(np.linalg.norm(xy - point['reference_xy'])), 2)
        return matrix, used, readings

    @staticmethod
    def coverage(points, names):
        support = np.float32([p['reference_xy'] for p in points if p['name'] in names])
        return cv2.contourArea(cv2.convexHull(support))

    def measure(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width, height = frame.shape[1::-1]
        if (width, height) != self.size:
            same_aspect = abs(width / height / (self.size[0] / self.size[1]) - 1) <= .005
            if not self.uses_holder_surfaces:
                raise ValueError(f'相机分辨率须为 {self.size[0]}×{self.size[1]}，与参考画面一致。')
            if not same_aspect or height < 720:
                raise ValueError(f'当前画面 {width}×{height}。请使用 {self.size[0]}×{self.size[1]}'
                                 ' 或同宽高比、至少 720 行的画面。')
            gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA if height > self.size[1] else cv2.INTER_LINEAR)
        report = self.measure_reference_size(gray)
        report['camera_frame_size'] = [width, height]
        report['reference_frame_size'] = list(self.size)
        if self.source_bounds and report.get('source_ready'):
            matrix = self.live_transform(frame, report)
            left, top, right, bottom = self.source_bounds
            outline = np.float32([[(left, top), (right, top), (right, bottom), (left, bottom)]])
            polygon = cv2.transform(outline, matrix)[0]
            if self.profile.get('source_visibility_bounds') and self.source_touches_edge(frame, polygon):
                self.source_seed = None
                report.update(aligned=False, source_ready=False, source_transform=None,
                              source_issue='possible_source_crop')
            else:
                report['source_region_normalized'] = (polygon / [width, height]).tolist()
        return report

    def measure_reference_size(self, gray):
        if self.holder_locator and self.source_seed is not None:
            matrix, used, readings = self.refine_holder_region(gray, self.source_seed, self.source_seed_radius)
            if matrix is not None:
                return self.source_report(readings, matrix, used)
            self.source_seed = None
        edges = holder_edges(gray) if self.uses_holder_surfaces else None
        search = SOURCE_SEARCH_RADIUS if self.source_bounds else SEARCH_RADIUS
        if self.uses_holder_surfaces:
            search = 55
        readings = self.match_points(gray, search, edges=edges)
        if self.source_bounds:
            tracking_seed = None
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
                    if self.coverage(guided, refined_used) > self.coverage(readings, used):
                        matrix, used, readings = refined, refined_used, guided
            if self.holder_locator and (matrix is None or np.linalg.norm(matrix[:, :2] - np.eye(2), ord=2) > .14):
                for seed, radius in self.search_holder_region(gray, need_location=matrix is None):
                    found, found_used, found_readings = self.refine_holder_region(gray, seed, radius)
                    if found is not None:
                        matrix, used, readings = found, found_used, found_readings
                        # Reuse the search view that actually matched. A second
                        # warp through the fitted affine can alter narrow edges
                        # enough to lose them on a strongly tilted fixture.
                        if radius == 22:
                            tracking_seed = seed
                        break
            if self.holder_locator:
                self.source_seed = matrix if tracking_seed is None else tracking_seed
                self.source_seed_radius = 14 if tracking_seed is None else 22
        if self.uses_holder_surfaces:
            return self.source_report(readings, matrix, used)
        aligned = all(r['shift_px'] is not None and r['shift_px'] <= self.tolerance for r in readings)
        if aligned:
            reference = np.float32([(*r['reference_xy'], 1) for r in readings])
            current = np.float32([r['current_xy'] for r in readings])
            transform = np.linalg.lstsq(reference, current, rcond=None)[0]
            aligned = bool(np.linalg.norm(reference @ transform - current, axis=1).max() <= 3)
        report = {'aligned': aligned, 'camera_name': self.profile['camera_name'],
                  'reference_sha256': self.reference_sha256, 'max_shift_px': self.tolerance, 'points': readings}
        if self.source_bounds:
            report.update(source_ready=matrix is not None, source_inliers=used,
                          source_transform=matrix.tolist() if matrix is not None else None)
        return report

    def source_report(self, readings, matrix, used):
        ready = matrix is not None
        aligned = ready and all(p['shift_px'] <= self.tolerance for p in readings if p['name'] in used)
        return {'aligned': aligned, 'camera_name': self.profile['camera_name'],
                'reference_sha256': self.reference_sha256, 'max_shift_px': self.tolerance,
                'points': readings, 'source_ready': ready, 'source_inliers': used,
                'source_transform': matrix.tolist() if ready else None, 'source_matcher': 'holder_surfaces'}

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
            x1, y1 = min(width, px+reach+1), min(height, py+reach+1)
            if x1-x0 < patch.shape[1] or y1-y0 < patch.shape[0]:
                readings.append({'name': name, 'reference_xy': [x, y], 'current_xy': None,
                                 'score': None, 'shift_px': None})
                continue
            area = np.s_[y0:y1, x0:x1]
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
        minimum_depth = 25 if self.uses_holder_surfaces else 40
        if (len(support) < minimum or np.ptp(support[:, 0]) < 125 or np.ptp(support[:, 1]) < minimum_depth
                or cv2.contourArea(cv2.convexHull(support)) < 2800
                or np.linalg.norm(matrix[:, :2] - np.eye(2), ord=2) > (.24 if self.uses_holder_surfaces else .16)):
            return None, []
        if not self.source_visible(matrix):
            return None, []
        return matrix, [p['name'] for p, good in zip(matches, valid) if good]

    def source_visible(self, matrix):
        left, top, right, bottom = self.source_visibility_bounds
        bounds = np.float32([[(left, top), (right, top), (right, bottom), (left, bottom)]])
        visible = cv2.transform(bounds, matrix)[0]
        return ((visible >= 4) & (visible < np.array(self.size) - 4)).all()

    def source_touches_edge(self, frame, corners):
        """Reject colored material cut off near the projected fixture."""
        x0, y0 = np.floor(corners.min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
        h, w = frame.shape[:2]
        edge = 4
        strips = []
        if x0 <= edge:
            strips.append(frame[max(0, y0):min(h, y1), :edge])
        if x1 >= w-edge:
            strips.append(frame[max(0, y0):min(h, y1), -edge:])
        if y0 <= edge:
            strips.append(frame[:edge, max(0, x0):min(w, x1)])
        if y1 >= h-edge:
            strips.append(frame[-edge:, max(0, x0):min(w, x1)])
        for strip in strips:
            if strip.size:
                hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
                if np.count_nonzero((hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 35)) >= 16:
                    return True
        return False

    def register_source(self, frame, report):
        """Map the current view into the calibrated source image coordinates."""
        if self.source_bounds:
            if report['source_transform'] is None:
                return None
            matrix = self.live_transform(frame, report)
            return cv2.warpAffine(frame, matrix, self.size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
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

    def live_transform(self, frame, report):
        # Pixel centers must follow the same mapping as the normalized gray
        # frame used to locate the holders.
        scales = np.asarray(frame.shape[1::-1], dtype=float) / self.size
        matrix = np.asarray(report['source_transform']) * scales[:, None]
        matrix[:, 2] += (scales - 1) / 2
        return matrix


def open_alignment(profile_path=DEFAULT_PROFILE, output=None, *, source_config=None):
    from tkinter import BOTH, LEFT, RIGHT, X, Canvas, StringVar, Tk, ttk
    from PIL import Image, ImageTk
    from koch_vision.color_app import CameraStream
    from .vision import StableRead, load_stack_reader, mark_source

    alignment = CameraAlignment(profile_path)
    reader = load_stack_reader(source_config, 'source') if source_config else None
    stable_read = StableRead()
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
    last_frame, last_report, last_stamp, inventory_report = None, None, None, None
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
            if inventory_report and 'faces' in inventory_report:
                frame = mark_source(frame, inventory_report)
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
                # The live frame is resized above for display. Its original
                # capture size is available in the alignment report.
                ratio = np.asarray(report['camera_frame_size']) / alignment.size
                area = cv2.transform(area, np.array(report['source_transform'])) * ratio
            elif view == 'live':
                area *= np.array(last_frame.shape[1::-1] if last_frame is not None else alignment.size) / alignment.size
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
        nonlocal last_frame, last_report, last_stamp, good_frames, stable_since, ready, inventory_report, stable_read
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
                if reader:
                    registered = alignment.register_source(last_frame, last_report) if matched else None
                    inventory_report = stable_read.update(reader.read(registered, registered=True) if registered is not None
                                                          else {'complete': False}, last_stamp)
                    ready = inventory_report['stable']
                elif matched:
                    stable_since = last_stamp if stable_since is None else stable_since
                    good_frames = min(good_frames + 1, REQUIRED_FRAMES)
                    ready = good_frames >= REQUIRED_FRAMES and last_stamp - stable_since >= STABLE_SECONDS
                else:
                    good_frames, stable_since, ready = 0, None, False
                if alignment.source_bounds:
                    if ready:
                        if reader:
                            status.set('S 区识别通过 · 可关闭此窗口')
                            detail.set(f'已确认 {inventory_report["observed_count"]} 块方块'
                                       f'和 {inventory_report["confirmed_empty_count"]} 个空位。')
                        else:
                            status.set('S 区定位通过')
                            detail.set('程序已自动补偿机位偏移。')
                    elif matched:
                        status.set('已找到 S 区 · 正在确认层位')
                        detail.set(inventory_report['status'] + ' 切到「校正预览」可查看八层识别框。'
                                   if reader else '正在确认画面稳定。')
                    else:
                        status.set('暂未找到 S 区')
                        detail.set(SOURCE_ADJUST_HINT)
                else:
                    status.set('已对齐 · 锁紧后仍对齐即可保存' if ready else '正在对位 · 请让圆点与十字重合')
                    detail.set('    '.join(f'{p["name"]} {p["shift_px"]:g}px' if p['shift_px'] is not None
                                           else f'{p["name"]} 未找到' for p in last_report['points']))
            except ValueError as error:
                good_frames, stable_since, ready, last_report, inventory_report = 0, None, False, None, None
                stable_read = StableRead()
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
