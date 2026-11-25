import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from koch_vision.colors import read_image
from koch_autonomy import replay_auto
from koch_autonomy import replay_single
from koch_autonomy.replay_bridge import route_is_disabled, toggle_route
from koch_autonomy.__main__ import camera_observe, demo
from koch_autonomy.camera_alignment import CameraAlignment, DEFAULT_PROFILE
from koch_autonomy.planner import plan_task
from koch_robot.paths import ROOT, resolve_config_reference


FIELD_ROOT = Path(os.environ.get('KOCH_FIELD_ROOT', ROOT)).resolve()
CALIBRATION = FIELD_ROOT / 'calibration/koch_legacy/follower/my_follower.json'
SCAN = FIELD_ROOT / 'outputs/koch_autonomy/target_manual_collection/t_motion_20261118_203509_23ab5ee7'
FIELD_PROFILE = FIELD_ROOT / DEFAULT_PROFILE.relative_to(ROOT)
REAR_OCCLUSION_CAPTURE = FIELD_ROOT / 'outputs/koch_replay_auto/20261123_161130_706096/source-before-raw.png'
needs_recorded_scan = unittest.skipUnless(CALIBRATION.is_file() and (SCAN / 'capture.json').is_file(),
                                           '需要现场标定和录制的扫描示范')
needs_alignment_reference = unittest.skipUnless(FIELD_PROFILE.is_file(), '需要现场全局相机参考配置')


class AutoReplayTests(unittest.TestCase):
    def test_route_toggle_persists_across_aliases_and_preserves_episode_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            site = base / 'site.json'
            episode = base / 'datasets/koch_st_layer2_to_layer1/S3-T4/sample'
            exact = episode.relative_to(base / 'datasets').as_posix()
            site.write_text(json.dumps({'excluded_replay_paths': [exact]}), encoding='utf-8')
            selected = ('S3', 'T4', (2, 1))
            self.assertTrue(toggle_route(site, *selected))
            self.assertTrue(route_is_disabled(site, *selected))
            saved = json.loads(site.read_text(encoding='utf-8'))['excluded_replay_paths']
            blocked = {(base / 'datasets' / path).resolve() for path in saved}
            with patch.object(replay_auto, 'route_episodes', return_value=[episode]) as lookup:
                self.assertEqual(replay_auto.eligible_episodes(base / 'datasets', (2, 1), 'S3_T4', blocked), [])
                lookup.assert_not_called()
            self.assertFalse(toggle_route(site, *selected))
            self.assertFalse(route_is_disabled(site, *selected))
            self.assertEqual(json.loads(site.read_text(encoding='utf-8'))['excluded_replay_paths'], [exact])

    def test_direct_counts_aggregate_demand_for_repeated_colors(self):
        stock = {'S1': ['red'], 'S2': [], 'S3': [], 'S4': []}
        targets = {f'T{i}': {'color': 'red', 'number': int(i < 3)} for i in range(1, 5)}
        with self.assertRaisesRegex(ValueError, "库存不足.*'red': 1"):
            plan_task(stock, targets, quantity_rule='direct')
        stock['S2'] = ['red']
        self.assertEqual(len(plan_task(stock, targets, quantity_rule='direct')['moves']), 2)
        with self.assertRaisesRegex(ValueError, '颜色各异'):
            plan_task(stock, targets)

    def test_chain_selects_compatible_episode_instead_of_latest_incompatible_one(self):
        def episode(path, start, end):
            goals = [{name: ticks for name in replay_auto.NAMES} for ticks in (start, end)]
            return SimpleNamespace(path=Path(path), rows=[{'goal_raw': row} for row in goals])

        stock = {'S1': ['red'], 'S2': ['red'], 'S3': [], 'S4': []}
        targets = {f'T{i}': {'color': 'red', 'number': int(i < 3)} for i in range(1, 5)}
        first = episode('first', 1000, 1000)
        compatible = episode('old', 1050, 1100)
        incompatible = episode('new', 1200, 1100)
        with patch.object(replay_auto, 'route_episodes', side_effect=[[first], [compatible, incompatible]]), \
                patch.object(replay_auto, 'read_trajectory', side_effect=lambda item, *_: item):
            _, selected = replay_auto.prepare_chain(ROOT / 'datasets',
                {'inventory': stock, 'targets': targets}, 'direct', None)
        self.assertEqual(selected, (first, compatible))
        with patch.object(replay_auto, 'route_episodes', side_effect=[[first], [incompatible]]), \
                patch.object(replay_auto, 'read_trajectory', side_effect=lambda item, *_: item):
            with self.assertRaisesRegex(ValueError, '第 1 步接第 2 步.*未连接从臂'):
                replay_auto.prepare_chain(ROOT / 'datasets', {'inventory': stock, 'targets': targets}, 'direct', None)

    def test_preview_reads_site_relative_paths_without_connecting(self):
        with tempfile.TemporaryDirectory(prefix='.replay-site-test-', dir=ROOT) as tmp:
            folder = Path(tmp)
            site = folder / 'replay.json'
            site.write_text(json.dumps({'source': {'config': 'field/source.json', 'camera': 2},
                'targets': {'config': 'field/targets.json', 'rotation': 0, 'camera': 1},
                'quantity_rule': 'direct', 'data': 'my-data'}), encoding='utf-8')
            views = []

            def observe(args, **_):
                views.append(args)
                report = dict(complete=True, stable=True)
                if args.kind == 'source':
                    return dict(report, empty_reference_used=True, inventory={
                        'S1': ['red'], 'S2': [], 'S3': [], 'S4': []})
                return dict(report, targets={f'T{i}': {'color': 'red', 'number': int(i == 1)} for i in range(1, 5)})

            with patch.object(replay_auto, 'ROOT', folder), patch.object(replay_auto, 'connect_bus') as connect, \
                    patch.object(replay_auto, 'prepare_chain',
                        return_value=({'moves': [{'source': 'S1', 'route': 'S1_T1'}]},
                                      [SimpleNamespace(path='demo')])) as chain:
                replay_auto.run_auto(site, True, observe, cancelled=lambda: False)
                connect.assert_not_called()
                self.assertEqual(chain.call_args.args[0], folder / 'my-data')
            self.assertEqual([args.config for args in views],
                             [folder / 'field/source.json', folder / 'field/targets.json'])
            self.assertEqual(resolve_config_reference('../calibration.json', views[0].config),
                             folder / 'calibration.json')

    def test_latest_only_uses_newest_recording_after_exclusions(self):
        data = ROOT / 'datasets'
        folder = data / 'koch_st' / 'S1_T1'
        prior, latest, excluded = [folder / str(i) for i in range(3)]
        task = {'inventory': {'S1': ['red'], 'S2': [], 'S3': [], 'S4': []},
                'targets': {f'T{i}': {'color': 'red', 'number': int(i == 1)} for i in range(1, 5)}}
        with (
            patch.object(replay_auto, 'route_episodes', return_value=[prior, latest, excluded]),
            patch.object(replay_auto, 'read_trajectory', side_effect=lambda item, *_: item) as read,
        ):
            _, selected = replay_auto.prepare_chain(data, task, 'direct', None, latest_only=True,
                                                     excluded_paths={excluded.resolve()})
        self.assertEqual(selected, (latest,))
        self.assertEqual(read.call_args.args[0], latest)
        with patch.object(replay_auto, 'route_episodes', return_value=[prior, latest, excluded]):
            with self.assertRaisesRegex(ValueError, '缺少已确认的动作'):
                replay_auto.prepare_chain(data, task, 'direct', None, latest_only=True,
                                          excluded_paths={folder.resolve()})

    def test_automatic_planning_falls_back_only_for_an_incompatible_seam(self):
        home = dict.fromkeys(replay_auto.NAMES, 2000)
        task = {'inventory': {'S1': ['green', 'red'], 'S2': [], 'S3': [], 'S4': []},
                'targets': {f'T{i}': {'color': 'red' if i == 1 else 'green', 'number': int(i <= 2)}
                            for i in range(1, 5)}}

        def episode(name, start, end):
            poses = [dict(home, wrist_roll=position) for position in (start, end)]
            return Mock(path=Path(name), duration=1., endpoint_tolerance=3.,
                        rows=[{'goal_raw': p, 'follower_raw': p} for p in poses])

        first = episode('first', 2000, 2040)
        older = episode('older', 2090, 2000)
        newest = episode('newest', 2135, 2000)
        disabled = episode('disabled', 2040, 2000)
        by_route = {'S1_T1': [first.path], 'S1_T2': [older.path, newest.path, disabled.path]}
        versions = {t.path: t for t in (first, older, newest, disabled)}
        options = dict(home=home, legacy_replay=True, latest_only=True, allow_older_fallback=True,
                       excluded_paths={disabled.path.resolve()})
        with (
            patch.object(replay_auto, 'route_episodes', side_effect=lambda _, pair, route: by_route.get(route, [])),
            patch.object(replay_auto, 'read_trajectory', side_effect=lambda path, *a, **kw: versions[path]),
        ):
            plan, selected = replay_auto.prepare_chain(ROOT / 'datasets', task, 'direct', None, **options)
            self.assertEqual(selected, [first, older])
            self.assertEqual(plan['older_episode_steps'], [2])
            newest.rows[0]['goal_raw']['wrist_roll'] = 2090
            plan, selected = replay_auto.prepare_chain(ROOT / 'datasets', task, 'direct', None, **options)
            self.assertEqual(selected, [first, newest])
            self.assertEqual(plan['older_episode_steps'], [])

    def test_single_replay_reports_feedback_against_the_previous_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            site = folder / 'site.json'
            site.write_text(json.dumps({'follower_port': 'mock', 'follower_calibration': str(CALIBRATION)}),
                            encoding='utf-8')
            episode = folder / 'latest'
            home = dict.fromkeys(replay_auto.NAMES, 2000)
            commanded = dict(home, elbow_flex=2040)
            actual = dict(home, elbow_flex=2020)
            trajectory = SimpleNamespace(rows=[{'goal_raw': home}], meta={'route': 'S3_T4'},
                                         motion_origin=0., settled=False, trim_wait=Mock())
            events = []

            def simulate(*, target_source, stop, on_frame):
                on_frame({'sent': True, 'follower_raw': home, 'goal_raw': commanded})
                trajectory.settled = True
                on_frame({'sent': True, 'follower_raw': actual, 'goal_raw': home})
                self.assertTrue(stop())

            with (
                patch.object(replay_single, 'ROOT', folder),
                patch.object(replay_single, 'select_episode', return_value=episode),
                patch.object(replay_single, 'load_calibration'),
                patch.object(replay_single, 'read_trajectory', return_value=trajectory),
                patch.object(replay_single, 'connect_bus') as connect,
                patch.object(replay_single, 'Teleoperator') as controller,
                patch.object(replay_single, 'monotonic', side_effect=(1., 2.)),
            ):
                controller.return_value.run.side_effect = simulate
                replay_single.run_single(site, 'S3', 'T4', (2, 1), episode, lambda: False,
                                         lambda kind, **payload: events.append((kind, payload)))
                connect.return_value.disconnect.assert_called_once_with(disable_torque=False)
                trajectory.trim_wait.assert_called_once_with(after_settle=True)

            path = Path(events[-1][1]['path'])
            self.assertEqual(events[-1][0], 'single_complete')
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['outcome'], 'trajectory_completed')
            self.assertEqual(report['layer_pair'], [2, 1])
            self.assertEqual(report['samples'][1]['compared_goal_raw'], commanded)
            self.assertAlmostEqual(report['peak_error_degrees']['elbow_flex'], 1.76, places=2)

    @needs_recorded_scan
    def test_scan_and_six_carry_episodes_match_recorded_joint_seams(self):
        cal = replay_auto.load_calibration(CALIBRATION)
        forward, backward, _ = replay_auto.load_scan(SCAN, cal)
        home = backward.rows[-1]['goal_raw']
        tasks = demo()
        plan, selected = replay_auto.prepare_chain(FIELD_ROOT / 'datasets', tasks, 'parity', cal, home=home)
        self.assertEqual(plan['required_counts'], {'T1': 1, 'T2': 2, 'T3': 1, 'T4': 2})
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(item.trimmed_seconds > 0 for item in selected))
        self.assertLessEqual(replay_auto.gap_degrees(home, forward.rows[0]['goal_raw']), .75)
        goals = [home]
        stock = tasks['inventory']
        for move, trajectory in zip(plan['moves'], selected):
            self.assertEqual(stock[move['source']].pop(), move['color'])
            goals.extend(trajectory.rows[i]['goal_raw'] for i in (0, -1))
        goals.append(home)
        self.assertEqual(sum(map(len, stock.values())), 2)
        for i in range(0, len(goals) - 1, 2):
            self.assertLessEqual(replay_auto.gap_degrees(goals[i], goals[i+1]), 8.5)

    @needs_recorded_scan
    def test_scan_camera_failure_still_replays_return_before_reporting_error(self):
        cal = replay_auto.load_calibration(CALIBRATION)
        scan = replay_auto.load_scan(SCAN, cal)
        events = []
        pose = scan[0].rows[0]['goal_raw']
        registers = scan[2]
        follower = SimpleNamespace(read=lambda key, joint, **_: registers[joint][key])

        class SimulatedController:
            def run(self, *, target_source, stop, on_frame, tracking_progress):
                actual = pose
                for step in range(1000):
                    if stop():
                        self.returned_to = actual
                        return
                    actual = target_source(step * .1, actual, actual)
                    on_frame({'sent': True, 'follower_time': step * .1,
                              'follower_raw': actual, 'goal_raw': actual})
                    if 'observe_targets' in events and 'scan_return' not in events:
                        from time import sleep
                        sleep(.001)
                raise AssertionError('扫描未结束')

        driver = SimulatedController()
        with patch.object(replay_auto, 'Teleoperator', return_value=driver):
            with self.assertRaisesRegex(ValueError, '无法识别彩纸'):
                replay_auto.scan_and_return(follower, cal, scan,
                    lambda **_: self.fail_reading(), lambda: False, lambda kind: events.append(kind))
        self.assertEqual(events, ['scan_approach', 'observe_targets', 'scan_return'])
        self.assertEqual(driver.returned_to, scan[1].rows[-1]['goal_raw'])

    @needs_recorded_scan
    def test_auto_uses_global_cards_when_wrist_scan_returns_without_a_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            events = []
            stock = {'S1': ['red'], 'S2': [], 'S3': [], 'S4': []}
            cards = {f'T{i}': {'color': 'red', 'number': int(i == 1)} for i in range(1, 5)}

            def observe(args, **_):
                report = {'complete': True, 'stable': True}
                if args.kind == 'source':
                    return {**report, 'empty_reference_used': True, 'inventory': stock}
                self.assertEqual(args.camera_name, '1080P USB Camera')
                return {**report, 'targets': cards}

            with (
                patch.object(replay_auto, 'ROOT', output),
                patch.object(replay_auto, 'connect_bus'),
                patch.object(replay_auto, 'stable_positions'),
                patch.object(replay_auto, 'prepare_chain', return_value=({'moves': []}, [])),
                patch.object(replay_auto, 'scan_and_return', side_effect=replay_auto.ScanReadError('腕部未到位')),
            ):
                replay_auto.run_auto(FIELD_ROOT / 'replay-auto.site.json', False, observe,
                                     cancelled=lambda: False, on_event=lambda kind, **_: events.append(kind))

            self.assertIn('observe_global_targets', events)
            self.assertEqual(events[-1], 'complete')
            report_path, = output.glob('outputs/koch_replay_auto/*/targets-global.json')
            self.assertEqual(json.loads(report_path.read_text(encoding='utf-8'))['wrist_scan_error'], '腕部未到位')

    @needs_recorded_scan
    def test_scan_uses_vision_at_a_stable_offset_and_returns_over_measured_path(self):
        cal = replay_auto.load_calibration(CALIBRATION)
        original = replay_auto.load_scan(SCAN, cal)
        home = original[0].rows[0]['goal_raw']
        above_t = dict(home, elbow_flex=home['elbow_flex'] + 500)
        observed_limit = above_t['elbow_flex'] - 140
        def trajectory(label, positions):
            rows = [{'command_time': float(i), 'goal_raw': pos, 'follower_raw': pos}
                    for i, pos in enumerate(positions)]
            return replay_auto.ReplayTrajectory.from_rows(SCAN, label, rows, cal, 1., 60.)
        forward = trajectory('scan', (home, above_t))
        forward.align_start = True
        backward = trajectory('return', (above_t, home))
        events = []
        registers = original[2]
        follower = SimpleNamespace(read=lambda key, joint, **_: registers[joint][key])

        class SimulatedController:
            def run(self, *, target_source, stop, on_frame, tracking_progress):
                actual = home
                for step in range(400):
                    if stop():
                        self.final = actual
                        return
                    goals = target_source(step * .1, actual, actual)
                    actual = dict(goals)
                    if 'scan_return' not in events:
                        actual['elbow_flex'] = min(actual['elbow_flex'], observed_limit)
                    on_frame({'sent': True, 'follower_time': step * .1,
                              'follower_raw': actual, 'goal_raw': goals})
                    if 'observe_targets' in events and 'scan_return' not in events:
                        from time import sleep
                        sleep(.001)
                raise AssertionError('扫描未结束')

        def read_cards(**_):
            return {'complete': True, 'stable': True}

        driver = SimulatedController()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(replay_auto, 'Teleoperator', return_value=driver),
        ):
            output = Path(folder) / 'motion.json'
            result = replay_auto.scan_and_return(follower, cal, (forward, backward, registers),
                                                 read_cards, lambda: False, events.append, motion_output=output)
            logged = json.loads(output.read_text(encoding='utf-8'))
        self.assertTrue(result['stable'])
        self.assertEqual(events, ['scan_approach', 'observe_targets', 'scan_return'])
        self.assertEqual({row['phase'] for row in logged['samples']}, {'approach', 'hold', 'return'})
        hold = next(row for row in logged['samples'] if row['phase'] == 'hold')
        self.assertEqual(hold['positions']['elbow_flex'], observed_limit)
        self.assertEqual(hold['goals']['elbow_flex'], above_t['elbow_flex'])
        self.assertGreater(logged['scan_pose_gap_degrees'], 12.)
        self.assertTrue(logged['returned_to_rest'])
        self.assertEqual(driver.final, home)

    @staticmethod
    def fail_reading():
        raise ValueError('无法识别彩纸')

    @needs_alignment_reference
    def test_camera_alignment_checks_offset_and_ignores_reference_paper_color(self):
        alignment = CameraAlignment(FIELD_PROFILE)
        frame = alignment.reference.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame[hsv[:, :, 1] > 45] = (10, 10, 10)
        self.assertTrue(alignment.measure(frame)['aligned'])
        displaced = cv2.warpAffine(frame, np.float32([[1, 0, 14], [0, 1, -8]]), alignment.size)
        delta = alignment.measure(displaced)
        self.assertFalse(delta['aligned'])
        registered = alignment.register_source(displaced, delta)
        self.assertIsNotNone(registered)
        self.assertTrue(alignment.measure(registered)['aligned'])
        tilted = cv2.warpAffine(frame, np.float32([[1., .02, -26], [-.075, 1., 46]]), alignment.size)
        tilted[710:1020, 370:920] = 255  # T cards and grid are outside the S alignment.
        shifted = alignment.measure(tilted)
        self.assertFalse(shifted['aligned'])
        self.assertTrue(shifted['source_ready'])
        self.assertIsNotNone(alignment.register_source(tilted, shifted))
        shifted_left = cv2.warpAffine(frame, np.float32([[1, 0, -103], [0, 1, -20]]), alignment.size)
        wide_report = alignment.measure(shifted_left)
        self.assertTrue(wide_report['source_ready'])
        json.dumps(wide_report, allow_nan=False)
        far = cv2.warpAffine(frame, np.float32([[1, 0, 310], [0, 1, 25]]), alignment.size)
        self.assertIsNotNone(alignment.register_source(far, alignment.measure(far)))
        cropped = cv2.warpAffine(frame, np.float32([[1, 0, 1000], [0, 1, 0]]), alignment.size)
        self.assertIsNone(alignment.register_source(cropped, alignment.measure(cropped)))
        self.assertFalse(alignment.measure(np.zeros_like(frame))['source_ready'])

    @needs_alignment_reference
    def test_holder_registration_works_with_the_grid_lines_covered(self):
        alignment = CameraAlignment(FIELD_PROFILE)
        grid_gap = np.int32([[919, 235], [970, 238], [939, 350], [869, 343]])
        pose = np.float32([[1., .018, -24], [-.044, 1., 27]])
        stock = read_image(resolve_config_reference(alignment.profile['stock_reference_image'], FIELD_PROFILE))
        for baseline in (alignment.reference, stock):
            frame = baseline.copy()
            cv2.fillConvexPoly(frame, grid_gap, (186, 186, 186))
            cv2.rectangle(frame, (1048, 274), (1086, 291), (186, 186, 186), -1)
            frame[700:1030, 370:935] = 255
            moved = cv2.warpAffine(frame, pose, alignment.size)
            report = alignment.measure(moved)
            self.assertTrue(report['source_ready'])
            self.assertEqual(report['source_matcher'], 'holder_surfaces')
            centers = np.float32([[[845, 220], [1030, 280]]])
            projected = cv2.transform(centers, np.float32(report['source_transform']))
            self.assertLess(np.linalg.norm(projected - cv2.transform(centers, pose), axis=2).max(), 5)

    @needs_alignment_reference
    @unittest.skipUnless(REAR_OCCLUSION_CAPTURE.is_file(), '需要后排托座偏移的现场复现原图')
    def test_offset_camera_reads_rear_block_layers(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'source.json'
            camera_observe(SimpleNamespace(kind='source', alignment=FIELD_PROFILE, camera_name='1080P USB Camera',
                plane=None, roi=None, config=FIELD_ROOT / 'configs/autonomy_source.json',
                image=REAR_OCCLUSION_CAPTURE, output=output))
            report = json.loads(output.read_text(encoding='utf-8'))
        self.assertTrue(report['complete'])
        self.assertEqual(report['inventory'], {'S1': ['blue', 'red'], 'S2': ['green', 'yellow'],
                                              'S3': ['red', 'pink'], 'S4': ['blue', 'pink']})

    @needs_alignment_reference
    def test_source_inventory_is_reacquired_after_camera_moves_across_frame(self):
        from koch_autonomy.vision import load_stack_reader
        paths = [FIELD_ROOT / 'outputs/koch_replay_auto' / run / 'source-before-raw.png'
                 for run in ('20261123_171106_450302', '20261123_171223_485306')]
        if not all(path.is_file() for path in paths):
            self.skipTest('需要机位大幅移动前后的现场复现原图')
        alignment = CameraAlignment(FIELD_PROFILE)
        reader = load_stack_reader(FIELD_ROOT / 'configs/autonomy_source.json', 'source')
        expected = {'S1': ['blue', 'red'], 'S2': ['pink', 'yellow'], 'S3': ['red', 'pink'], 'S4': ['blue', 'green']}
        for path in paths:
            frame = read_image(path)
            for _ in range(4):
                pose = alignment.measure(frame)
                self.assertTrue(pose['source_ready'])
                self.assertFalse(pose['aligned'])
                report = reader.read(alignment.register_source(frame, pose), registered=True)
                self.assertTrue(report['complete'])
                self.assertEqual(report['inventory'], expected)

    @needs_alignment_reference
    def test_holder_registration_maps_720p_to_reference_coordinates(self):
        alignment = CameraAlignment(FIELD_PROFILE)
        frame = cv2.resize(alignment.reference, (1280, 720), interpolation=cv2.INTER_AREA)
        report = alignment.measure(frame)
        self.assertTrue(report['source_ready'])
        self.assertEqual(report['camera_frame_size'], [1280, 720])
        self.assertEqual(alignment.register_source(frame, report).shape, alignment.reference.shape)
        region = np.asarray(report['source_region_normalized'])
        self.assertTrue(((region > 0) & (region < 1)).all())

    @needs_alignment_reference
    def test_rear_block_does_not_count_as_the_upper_layer_of_an_empty_front_slot(self):
        from koch_autonomy.vision import SourceReader, load_stack_reader
        path = FIELD_ROOT / 'outputs/koch_replay_auto/20261123_170532_735971_recovery_check/source-after-stop-raw.png'
        if not path.is_file():
            self.skipTest('需要前排空托座与后排方块投影重叠的现场原图')
        alignment = CameraAlignment(FIELD_PROFILE)
        frame = read_image(path)
        registered = alignment.register_source(frame, alignment.measure(frame))
        reader = load_stack_reader(FIELD_ROOT / 'configs/autonomy_source.json', 'source')
        fixed = SourceReader({**reader.config, 'layer_search_offsets': None}, reader.colors)
        self.assertIn('S2_L2', fixed.read(registered)['unknown'])
        self.assertEqual(fixed.read(registered, registered=True)['inventory'],
                         {'S1': ['blue', 'red'], 'S2': [], 'S3': ['red'], 'S4': ['blue']})
        if reader.windows:
            report = reader.read(registered, registered=True)
            self.assertFalse(report['complete'])
            self.assertIn('S2 上层有信号但下层为空', ' '.join(report['issues']))


if __name__ == '__main__':
    unittest.main()
