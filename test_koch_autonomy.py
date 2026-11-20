"""Offline regression checks; no serial ports or live cameras."""
import copy
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from koch_autonomy.__main__ import demo
from koch_autonomy.planner import plan_task, validate_inventory, validate_targets
from koch_autonomy.reverse import reverse_samples, features, validate_route_layers, ROOT
from koch_autonomy.vision import SourceReader, TargetStackReader, StableRead, TargetReader, corner_order, rectify_marked_plane
from koch_autonomy.workflow import commands, commit_observed_move, commit_from_reports, simulate
from camera_vision import fit_sample, write_image


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.data = demo()

    def test_parity_and_top_before_bottom_end_to_end(self):
        original = copy.deepcopy(self.data)
        result = simulate(**self.data)
        self.assertEqual(self.data, original)
        self.assertEqual([len(result['placed'][f'T{i}']) for i in range(1, 5)], [1, 2, 1, 2])
        for event in result['events']:
            move = event['move']
            self.assertEqual(len(event['inventory'][move['source']]), move['source_layer'] - 1)
        self.assertFalse(result['hardware_executed'])

    def test_zero_is_even(self):
        self.data['targets']['T2']['number'] = 0
        self.assertEqual(plan_task(**self.data)['required_counts']['T2'], 2)

    def test_inventory_shortage(self):
        self.data['inventory']['S3'] = ['red']
        with self.assertRaisesRegex(ValueError, '不足'):
            plan_task(**self.data)

    def test_unwanted_top_blocks_required_bottom(self):
        self.data['inventory'] = {'S1': ['red', 'pink'], 'S2': ['blue', 'blue'],
                                  'S3': ['yellow', 'pink'], 'S4': ['green', 'green']}
        with self.assertRaisesRegex(ValueError, '缓冲区'):
            plan_task(**self.data)

    def test_unknown_and_floating_source_rejected(self):
        for value in (None, '?', ''):
            with self.subTest(value=value):
                data = copy.deepcopy(self.data['inventory'])
                data['S1'] = [value, 'blue']
                with self.assertRaises(ValueError):
                    validate_inventory(data)

    def test_bad_digits_and_duplicate_targets(self):
        for value in (None, -1, 1.5, True, 'x', '1e2'):
            data = copy.deepcopy(self.data['targets'])
            data['T1']['number'] = value
            with self.assertRaises(ValueError):
                validate_targets(data)
        self.data['targets']['T2']['number'] = 1
        with self.assertRaisesRegex(ValueError, '各异'):
            plan_task(**self.data)

    def test_directed_cost_global_optimum_matches_exhaustive_search(self):
        inventory = {f'S{i}': [c] for i, c in enumerate(['red', 'blue', 'yellow', 'green'], 1)}
        targets = {f'T{i}': {'color': c, 'number': 2 * i - 1}
                   for i, c in enumerate(['red', 'blue', 'yellow', 'green'], 1)}
        nodes = list(inventory) + list(targets)
        rng = np.random.default_rng(2)
        costs = {f'{a}->{b}': int(rng.integers(1, 20)) for a in nodes for b in nodes if a[0] != b[0]}
        result = plan_task(inventory, targets, edge_costs=costs)
        totals = []
        for order in itertools.permutations(range(1, 5)):
            current, total = 'T3', 0
            for i in order:
                total += costs[f'{current}->S{i}'] + costs[f'S{i}->T{i}']
                current = f'T{i}'
            totals.append(total)
        self.assertEqual(result['cost'], min(totals))

    def test_invalid_cost_rejected(self):
        with self.assertRaises(ValueError):
            plan_task(**self.data, edge_costs={})
        positions = {n: [0, 0] for n in list(self.data['inventory']) + list(self.data['targets'])}
        positions['S1'] = [float('nan'), 0]
        with self.assertRaises(ValueError):
            plan_task(**self.data, positions=positions)

    def test_null_json_objects_are_controlled_errors(self):
        targets = copy.deepcopy(self.data['targets'])
        targets['T1'] = None
        with self.assertRaises(ValueError):
            validate_targets(targets)
        positions = {n: [0, 0] for n in list(self.data['inventory']) + list(self.data['targets'])}
        positions['S1'] = None
        with self.assertRaises(ValueError):
            plan_task(**self.data, positions=positions)
        with self.assertRaises(ValueError):
            plan_task(**self.data, edge_costs=[])

    def test_no_inventory_commit_on_timeout_or_failed_pick(self):
        result = plan_task(**self.data)
        move = result['moves'][0]
        placed = {t: [] for t in self.data['targets']}
        with self.assertRaisesRegex(ValueError, '实测'):
            commit_observed_move(self.data['inventory'], placed, move, self.data['inventory'], placed)
        self.assertEqual(self.data, demo())

    def test_scan_order_and_no_gripper_command_during_scan(self):
        steps = commands(plan_task(**self.data))
        scan = [s for s in steps if s['kind'] == 'scan_move']
        self.assertEqual([s['target'] for s in scan], ['T1', 'T2', 'T4', 'T3'])
        self.assertTrue(all(s['gripper'] == 'hold' for s in scan))


def color_config():
    values = {'red': (50, 50, 230), 'blue': (230, 100, 50),
              'yellow': (40, 220, 240), 'green': (80, 180, 50)}
    return {'frame_size': [500, 500], 'min_area': 500, 'max_area_fraction': .2,
            'roi': None, 'plane': None, 'material': 'paper', 'colors': {
                key: {'name': key, 'samples': [fit_sample(np.full((30, 30, 3), value, np.uint8))]}
                for key, value in values.items()}}, values


def paper_image():
    config, colors = color_config()
    frame = np.full((500, 500, 3), 230, np.uint8)
    for (key, color), (x, y) in zip(colors.items(), ((40, 40), (350, 40), (40, 350), (350, 350))):
        frame[y:y+100, x:x+100] = color
    return frame, config


class VisionTests(unittest.TestCase):
    def test_invalid_high_score_rotation_cannot_hide_verified_candidate(self):
        frame, config = paper_image()
        reader = TargetReader(type('Predictor', (), {'path': 'test-only'})(), config)
        calls = []
        def numbers(oriented, grid, tiles, colors, base):
            rotation_index = len(calls)
            calls.append(True)
            for i, cell in enumerate(base['cells']):
                cell.update(number=str(i+1), number_score=.99 if rotation_index == 0 else .90,
                            confirmed=rotation_index == 1)
                if rotation_index != 1:
                    cell['issues'].append('模拟复核拒绝')
            return {**base, 'complete': rotation_index == 1}, oriented, []
        with patch.object(reader.reader, 'read_numbers', side_effect=numbers):
            report, _ = reader.read(frame)
        self.assertTrue(report['complete'], report)
        self.assertEqual(report['orientation_cw'], 90)

    def test_target_ids_follow_whole_frame_rotation(self):
        frame, config = paper_image()
        predictor = type('Predictor', (), {'path': 'test-only'})()
        for k in range(4):
            with self.subTest(k=k):
                reader = TargetReader(predictor, config, rotation_cw=k * 90)
                def numbers(oriented, grid, tiles, colors, base):
                    for cell in base['cells']:
                        cell.update(number=str(list(config['colors']).index(cell['color']['key']) + 1),
                                    number_score=.99, confirmed=True)
                    return {**base, 'complete': True}, oriented, []
                with patch.object(reader.reader, 'read_numbers', side_effect=numbers):
                    report, _ = reader.read(np.ascontiguousarray(np.rot90(frame, k)))
                self.assertTrue(report['complete'], report)
                self.assertEqual([report['targets'][t]['color'] for t in ('T1', 'T2', 'T3', 'T4')],
                                 ['red', 'blue', 'yellow', 'green'])

    def test_automatic_rotation_requires_decisive_evidence(self):
        frame, config = paper_image()
        reader = TargetReader(type('Predictor', (), {'path': 'test-only'})(), config)
        def numbers(oriented, grid, tiles, colors, base):
            for i, cell in enumerate(base['cells']):
                cell.update(number=str(i+1), number_score=.99, confirmed=True)
            return {**base, 'complete': True}, oriented, []
        with patch.object(reader.reader, 'read_numbers', side_effect=numbers):
            report, _ = reader.read(frame)
        self.assertFalse(report['complete'])
        self.assertEqual(report['targets'], {})
        self.assertIn('方向证据不足', report['status'])

    def test_missing_paper_not_fabricated(self):
        frame, config = paper_image()
        frame[350:450, 350:450] = 230
        reader = TargetReader(type('Predictor', (), {'path': 'test-only'})(), config, rotation_cw=0)
        report, _ = reader.read(frame)
        self.assertFalse(report['complete'])
        self.assertEqual(report['targets'], {})

    def test_source_eight_visible_faces_layers(self):
        colors, bgr = color_config()
        image = np.full((200, 400, 3), 230, np.uint8)
        faces = {}
        expected = {}
        for i, color in enumerate(bgr):
            slot = f'S{i+1}'
            expected[slot] = [color, color]
            for layer in (1, 2):
                x, y = 10 + i * 100, 110 if layer == 1 else 10
                image[y:y+60, x:x+60] = bgr[color]
                faces[f'{slot}_L{layer}'] = [[x, y], [x+59, y], [x+59, y+59], [x, y+59]]
        config = {'frame_size': [400, 200], 'view_validated': True, 'faces': faces}
        report = SourceReader(config, colors).read(image)
        self.assertTrue(report['complete'], report)
        self.assertEqual(report['observed_count'], 8)
        self.assertEqual(report['inventory'], expected)
        image[110:170, 10:70] = 230
        report = SourceReader(config, colors).read(image)
        self.assertFalse(report['complete'])
        self.assertIsNone(report['inventory']['S1'][0])
        self.assertEqual(report['observed_count'], 7)

    def test_source_missing_calibration_refused(self):
        with self.assertRaisesRegex(ValueError, '标定'):
            SourceReader({}, {})

    def test_source_overlapping_faces_refused(self):
        colors, bgr = color_config()
        image = np.full((200, 200, 3), bgr['red'], np.uint8)
        faces = {f'S{s}_L{l}': [[10, 10], [80, 10], [80, 80], [10, 80]]
                 for s in range(1, 5) for l in (1, 2)}
        reader = SourceReader({'faces': faces, 'frame_size': [200, 200], 'view_validated': True}, colors)
        with self.assertRaisesRegex(ValueError, '重叠'):
            reader.read(image)

    def test_stability_rejects_duplicates_stale_and_changed_frames(self):
        state = StableRead(required=3)
        report = {'complete': True, 'targets': {'T1': {'color': 'red', 'number': 1}}}
        self.assertFalse(state.update(report, 1., now=1.)['stable'])
        self.assertFalse(state.update(report, 1., now=1.)['stable'])
        self.assertFalse(state.update(report, 1.1, now=1.1)['stable'])
        self.assertFalse(state.update(report, 1.2, now=1.2)['stable'])
        self.assertTrue(state.update(report, 1.3, now=1.3)['stable'])
        self.assertFalse(state.update(report, 1.4, now=3.)['stable'])
        self.assertFalse(state.update({'complete': False}, 3.1, now=3.1)['stable'])

    def test_stability_window_rolls_forward_and_resets(self):
        state = StableRead(required=3)
        report = {'complete': True, 'inventory': {'S1': ['red']}}
        for stamp in (10., 10.1, 10.2, 10.3):
            result = state.update(report, stamp, now=stamp)
        self.assertAlmostEqual(result['stable_since_monotonic'], 10.1)
        self.assertEqual(result['required_frames'], 3)
        reset = state.update(report, 10.3, now=10.3)
        self.assertIsNone(reset['stable_since_monotonic'])
        for stamp in (10.4, 10.5, 10.6):
            result = state.update(report, stamp, now=stamp)
        self.assertTrue(result['stable'])
        self.assertAlmostEqual(result['stable_since_monotonic'], 10.4)

    def test_stability_does_not_reuse_unverified_view_frames(self):
        state = StableRead(required=2)
        report = {'complete': True, 'inventory': {'S1': []}, 'view_pose_verified': False}
        state.update(report, 1., now=1.)
        result = state.update({**report, 'view_pose_verified': True}, 1.1, now=1.1)
        self.assertFalse(result['stable'])

    def test_stability_rejects_invalid_window_settings(self):
        for required, gap in ((True, .6), (2.5, .6), (2, float('nan')), (2, float('inf'))):
            with self.subTest(required=required, gap=gap):
                with self.assertRaises(ValueError):
                    StableRead(required=required, max_gap=gap)

    def test_stability_restarts_on_inventory_or_placed_changes(self):
        for field in ('inventory', 'placed'):
            for sequence in ((['red'], ['red'], []), ([], ['red'], [])):
                with self.subTest(field=field, sequence=sequence):
                    state = StableRead(required=3)
                    for index, stack in enumerate(sequence):
                        stamp = 1. + index * .1
                        result = state.update({'complete': True, field: {'slot': stack}}, stamp, now=stamp)
                    self.assertFalse(result['stable'])
                    self.assertEqual(result['matching_frames'], 1)
                    self.assertIsNone(result['stable_since_monotonic'])

    def test_marker_missing_does_not_reuse_old_transform(self):
        config = {'marker_ids': [0, 1, 2, 3], 'frame_size': [500, 500],
                  'reference_points_px': [[50, 50], [450, 50], [450, 450], [50, 450]]}
        with self.assertRaisesRegex(ValueError, '缺失'):
            rectify_marked_plane(np.full((500, 500, 3), 255, np.uint8), config)

    def test_marked_plane_recovers_reference_and_reports_visibility(self):
        frame = np.full((600, 600, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for i, (x, y) in enumerate(((30, 30), (490, 30), (490, 490), (30, 490))):
            marker = cv2.aruco.generateImageMarker(dictionary, i, 80)
            frame[y:y+80, x:x+80] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        ref = np.float32([[69.5, 69.5], [529.5, 69.5], [529.5, 529.5], [69.5, 529.5]])
        result, valid, matrix = rectify_marked_plane(frame, {'marker_ids': [0, 1, 2, 3],
            'frame_size': [600, 600], 'reference_points_px': ref.tolist()})
        np.testing.assert_allclose(matrix, np.eye(3), atol=.02)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(valid.shape, frame.shape[:2])


class ReverseTests(unittest.TestCase):
    def test_route_and_layer_shape_validated_before_features(self):
        for route, pair in (('', [1, 1]), ('S0_T2', [1, 1]), ('S1_T1', []),
                            ('S1_T1', [True, 1]), ('S1_T1', [1, 3])):
            with self.assertRaises(ValueError):
                validate_route_layers(route, pair)

    def test_reverse_body_only_and_preserve_source_times(self):
        t = np.arange(21) * .05
        q = np.tile(t[:, None], (1, 5)) * 100 + 1000
        elapsed, reverse, original = reverse_samples(t, q, .1, .9)
        np.testing.assert_allclose(reverse[0], 1090)
        np.testing.assert_allclose(reverse[-1], 1010)
        self.assertTrue(np.all(np.diff(elapsed) > 0))
        self.assertTrue(np.all(np.diff(original) < 0))
        self.assertEqual(reverse.shape[1], 5)

    def test_reverse_gap_and_wrap_rejected(self):
        with self.assertRaisesRegex(ValueError, '中断'):
            reverse_samples([0, .05, .8, 1.], np.zeros((4, 5)), 0, 1.)
        q = np.zeros((3, 5))
        q[1, 0] = 4096
        with self.assertRaisesRegex(ValueError, '跳变'):
            reverse_samples([0, .05, .1], q, 0, .1)

    def test_bad_time_and_nonfinite_rejected(self):
        for times in ([0, .1, .1], [0, float('nan'), 1], [0, .2, .1]):
            with self.assertRaises(ValueError):
                reverse_samples(times, np.zeros((3, 5)), 0, .1)

    def test_inverse_condition_keeps_source_and_from_target_distinct(self):
        x = features({'route': 'S2_T4', 'layer_pair': [2, 1]}, [0, 1])
        self.assertEqual(x.shape, (2, 13))
        self.assertEqual(np.flatnonzero(x[0]).tolist(), [1, 7, 10])
        self.assertEqual(x[1, -1], 1)


class StackObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.autonomy_test_', dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.colors, self.bgr = color_config()
        self.empty = np.full((240, 440, 3), 230, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.marker_points = []
        for i, (x, y) in enumerate(((5, 75), (400, 75), (400, 210), (5, 210))):
            marker = cv2.aruco.generateImageMarker(dictionary, i, 24)
            self.empty[y:y+24, x:x+24] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            self.marker_points.append([x+11.5, y+11.5])
        self.empty_path = Path(self.temp.name) / 'empty.png'
        write_image(self.empty_path, self.empty)

    def fixture(self, region, stacks):
        faces = {}
        image = self.empty.copy()
        for i in range(4):
            for layer in (1, 2):
                x, y = 10 + i * 100, 110 if layer == 1 else 10
                faces[f'{region}{i+1}_L{layer}'] = [[x, y], [x+59, y], [x+59, y+59], [x, y+59]]
                if len(stacks[f'{region}{i+1}']) >= layer:
                    image[y:y+60, x:x+60] = self.bgr[stacks[f'{region}{i+1}'][layer-1]]
        config = {'region': region, 'frame_size': [440, 240], 'view_validated': True, 'faces': faces,
                  'empty_reference_image': str(self.empty_path), 'empty_max_difference': 5., 'occupied_min_difference': 15.,
                  'view_markers': {'marker_ids': [0, 1, 2, 3], 'reference_points_px': self.marker_points, 'max_shift_px': 2.}}
        return image, config

    def test_post_pick_and_place_reports_commit_real_image_changes(self):
        data = demo()
        move = plan_task(**data)['moves'][0]
        before = data['inventory']
        after = copy.deepcopy(before)
        placed = {f'T{i}': [] for i in range(1, 5)}
        next_placed = copy.deepcopy(placed)
        next_placed[move['target']].append(after[move['source']].pop())
        source_image, source_config = self.fixture('S', after)
        target_image, target_config = self.fixture('T', next_placed)
        sr = SourceReader(source_config, self.colors).read(source_image)
        tr = TargetStackReader(target_config, self.colors).read(target_image)
        self.assertTrue(sr['complete'], sr)
        self.assertTrue(tr['complete'], tr)
        self.assertEqual(sr['inventory'], after)
        self.assertEqual(tr['placed'], next_placed)
        self.assertEqual(sr['confirmed_empty_count'], 1)
        self.assertEqual(tr['confirmed_empty_count'], 7)
        a, b = StableRead(required=2), StableRead(required=2)
        for stamp in (100.1, 100.2):
            s = a.update(sr, stamp, now=stamp)
            t = b.update(tr, stamp, now=stamp)
        self.assertEqual(commit_from_reports(before, placed, move, s, t, 100., now=100.3), (after, next_placed))
        with self.assertRaisesRegex(ValueError, '新鲜'):
            commit_from_reports(before, placed, move, s, t, 100., now=102.)
        with self.assertRaisesRegex(ValueError, '动作之后'):
            commit_from_reports(before, placed, move, s, t, 100.4, now=100.5)

        # The final frame alone cannot certify an entire post-motion window.
        for source_times, target_times in (((99.9, 100.1), (100.1, 100.2)),
                                           ((100.1, 100.2), (99.9, 100.1)),
                                           ((100., 100.1), (100.1, 100.2))):
            reports = []
            for report, times in ((sr, source_times), (tr, target_times)):
                state = StableRead(required=2)
                for stamp in times:
                    result = state.update(report, stamp, now=stamp)
                reports.append(result)
            with self.subTest(source_times=source_times, target_times=target_times):
                with self.assertRaises(ValueError):
                    commit_from_reports(before, placed, move, *reports, 100., now=100.3)

        for field, value in (('stable_since_monotonic', None), ('stable_since_monotonic', float('nan')),
                             ('stable_since_monotonic', 100.25), ('required_frames', 1),
                             ('matching_frames', 1)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    commit_from_reports(before, placed, move, {**s, field: value}, t, 100., now=100.3)

        # The newest frame is fresh here, but the oldest window frame has expired.
        with self.assertRaises(ValueError):
            commit_from_reports(before, placed, move, s, t, 100., now=100.75, max_age=.6)

    def test_all_six_moves_commit_from_independent_synthetic_images(self):
        data = demo()
        inventory = copy.deepcopy(data['inventory'])
        placed = {t: [] for t in data['targets']}
        plan = plan_task(**data)
        trackers = [StableRead(required=4), StableRead(required=4)]
        self.assertEqual(len(plan['moves']), 6)
        for index, move in enumerate(plan['moves']):
            expected_s, expected_t = copy.deepcopy(inventory), copy.deepcopy(placed)
            expected_t[move['target']].append(expected_s[move['source']].pop())
            source_image, source_config = self.fixture('S', expected_s)
            target_image, target_config = self.fixture('T', expected_t)
            readers = [SourceReader(source_config, self.colors), TargetStackReader(target_config, self.colors)]
            finished = 200. + index
            reports = []
            for tracker, reader, image in zip(trackers, readers, (source_image, target_image)):
                for offset in (.05, .1, .15, .2):
                    stamp = finished + offset
                    report = tracker.update(reader.read(image), stamp, now=stamp)
                reports.append(report)
            inventory, placed = commit_from_reports(inventory, placed, move, *reports, finished, now=finished+.25)
            self.assertEqual((inventory, placed), (expected_s, expected_t))
        self.assertEqual([len(placed[f'T{i}']) for i in range(1, 5)], [1, 2, 1, 2])
        self.assertEqual(sum(map(len, inventory.values())), 2)

    def test_unmatched_changed_surface_is_unknown_not_empty(self):
        empty = {f'S{i}': [] for i in range(1, 5)}
        image, config = self.fixture('S', empty)
        image[110:170, 10:70] = 50
        report = SourceReader(config, self.colors).read(image)
        self.assertFalse(report['complete'])
        self.assertIn('S1_L1', report['unknown'])
        self.assertEqual(report['faces'][0]['state'], 'unknown')

    def test_upper_without_support_rejected(self):
        empty = {f'T{i}': [] for i in range(1, 5)}
        image, config = self.fixture('T', empty)
        image[10:70, 10:70] = self.bgr['blue']
        report = TargetStackReader(config, self.colors).read(image)
        self.assertFalse(report['complete'])
        self.assertIn('下层为空', report['status'])

    def test_target_empty_reference_required_and_mismatched_region_refused(self):
        image, config = self.fixture('T', {f'T{i}': [] for i in range(1, 5)})
        config.pop('empty_reference_image')
        with self.assertRaisesRegex(ValueError, '空场景'):
            TargetStackReader(config, self.colors)
        with self.assertRaisesRegex(ValueError, '区域'):
            SourceReader(config, self.colors)

    def test_invalid_empty_thresholds_refused(self):
        _, config = self.fixture('S', {f'S{i}': [] for i in range(1, 5)})
        for low, high in ((20, 5), (-1, 5), (5, float('nan')), (None, 15)):
            with self.subTest(low=low, high=high):
                config.update(empty_max_difference=low, occupied_min_difference=high)
                with self.assertRaisesRegex(ValueError, '空位判据'):
                    SourceReader(config, self.colors)

    def test_camera_shift_rejected_before_inventory_classification(self):
        image, config = self.fixture('S', demo()['inventory'])
        moved = cv2.warpAffine(image, np.float32([[1, 0, 0], [0, 1, -5]]), (440, 240), borderValue=(230, 230, 230))
        with self.assertRaisesRegex(ValueError, '偏移'):
            SourceReader(config, self.colors).read(moved)

    def test_unverified_view_not_accepted_for_commit(self):
        data = demo()
        move = plan_task(**data)['moves'][0]
        report = {'complete': True, 'empty_reference_used': True, 'view_pose_verified': False,
                  'scope': 'S_fixed_camera_visible_faces'}
        tracker = StableRead(required=2)
        for stamp in (10.1, 10.2):
            report = tracker.update(report, stamp, now=stamp)
        self.assertTrue(report['stable'])  # Agreement is not a calibrated-view guarantee.
        with self.assertRaisesRegex(ValueError, '固定视角'):
            commit_from_reports(data['inventory'], {t: [] for t in data['targets']}, move, report, report, 10., now=10.3)


if __name__ == '__main__':
    unittest.main()
