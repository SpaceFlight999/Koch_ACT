"""Locate a calibrated S-area view using fixed scene features outside the blocks."""

import cv2
import numpy as np

from camera_vision import read_image
from .vision import validate_quad


class SourceCaptureView:
    def __init__(self, config, alternate_views):
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher()
        self.views = []
        for view in [config, *alternate_views]:
            frame = read_image(view['reference_image'])
            if list(frame.shape[1::-1]) != config['frame_size'] or set(view['faces']) != set(config['faces']):
                raise ValueError('采集视角参考与原始标定不兼容。')
            for quad in view['faces'].values():
                validate_quad(quad, frame.shape)
            points = np.float32(list(view['faces'].values()))
            mask = np.full(frame.shape[:2], 255, np.uint8)
            h, w = mask.shape
            start = np.maximum(0, np.floor(points.min(axis=(0, 1))).astype(int) - 76)
            end = np.minimum([w, h], np.ceil(points.max(axis=(0, 1))).astype(int) + 76)
            mask[start[1]:end[1], start[0]:end[0]] = 0
            keypoints, descriptors = self.detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
            if descriptors is None or len(descriptors) < 30:
                raise ValueError('相机参考图缺少足够固定特征。')
            self.views.append((view, points, keypoints, descriptors))

    def align(self, frame):
        keypoints, descriptors = self.detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        if descriptors is None or len(descriptors) < 2:
            raise ValueError('画面缺少可对齐的桌面特征。')
        candidates = []
        for view, points, reference_points, reference_descriptors in self.views:
            pairs = [nearest for nearest, second in self.matcher.knnMatch(reference_descriptors, descriptors, k=2)
                     if nearest.distance < .69 * second.distance]
            if len(pairs) < 30:
                continue
            p = np.float32([reference_points[match.queryIdx].pt for match in pairs])
            q = np.float32([keypoints[match.trainIdx].pt for match in pairs])
            matrix, inliers = cv2.findHomography(p, q, cv2.RANSAC, 2.5)
            if matrix is None or int(inliers.sum()) < max(26, .45 * len(pairs)):
                continue
            projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(points.shape)
            if not np.isfinite(projected).all():
                continue
            norms = np.linalg.norm(projected - points, axis=2)
            shift = float(np.median(norms))
            if float(norms.max()) > 24:
                continue  # A large change in viewpoint needs its own calibrated template.
            selected = points if float(norms.max()) <= 1.5 else projected
            try:
                quads = {face: validate_quad(quad, frame.shape).tolist()
                         for face, quad in zip(view['faces'], selected)}
            except ValueError:
                continue
            candidates.append((shift, quads, {'reference_image': view['reference_image'],
                                              'inliers': int(inliers.sum()), 'shift_px': round(shift, 2)}))
        if not candidates:
            raise ValueError('画面与任何已标定的视角对不上。采样框错位时请重试对齐。')
        _, quads, evidence = min(candidates, key=lambda candidate: candidate[0])
        return quads, evidence
