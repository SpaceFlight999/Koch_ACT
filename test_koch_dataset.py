"""采集与回放离线测试：模拟串口、真实编码小视频，不启用真实硬件。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import collect_koch as cli
import koch_dataset as data
import teleoperate_koch as app
from test_koch_teleop import FakeBus, FakeClock, calibration


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='.dataset_test_', dir=app.ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cal = calibration()
        self.camera = SimpleNamespace(index=2, fps=20, size=(32, 24))
        self.actual = dict.fromkeys(app.NAMES, 2048)

    def recorder(self):
        recorder = data.EpisodeRecorder(self.root, 'S1_T1', self.camera, self.cal, self.cal)
        self.addCleanup(lambda: recorder.finish('aborted') if not recorder.closing.is_set() else None)
        return recorder

    def populate(self, recorder, goal=None):
        for i in range(3):
            t = recorder.origin + .05 * i
            recorder.camera_frame(t, np.full((24, 32, 3), i * 60, dtype=np.uint8))
            recorder.joint_frame({'leader_time': t + .001, 'follower_time': t + .002,
                                  'command_time': t + .003,
                                  'leader_raw': dict(self.actual), 'follower_raw': dict(self.actual),
                                  'goal_raw': dict(goal or self.actual), 'sent': True})

    def episode(self, goal=None):
        recorder = self.recorder()
        self.populate(recorder, goal)
        recorder.finish('review')
        recorder.accept(True)
        return recorder.path

    def rewrite_rows(self, episode, edit):
        rows = data.read_jsonl(episode / 'joints.jsonl')
        edit(rows)
        (episode / 'joints.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')

    def test_exact_16_routes_and_selection(self):
        self.assertEqual(len(set(data.ROUTES)), 16)
        self.assertEqual(data.ROUTES[0], 'S1_T1')
        self.assertEqual(data.ROUTES[-1], 'S4_T4')
        for i, route in enumerate(data.ROUTES, 1):
            self.assertEqual(cli.route_name(str(i)), route)
            self.assertEqual(cli.route_name(route.lower()), route)
        for bad in ('0', '17', 'S1_T5', '../S1_T1'):
            with self.assertRaises(app.TeleopError):
                cli.route_name(bad)

    def test_video_joints_timestamps_and_progress(self):
        episode = self.episode()
        self.assertEqual(data.completed_episodes(self.root, 'S1_T1'), [episode])
        rows = data.read_jsonl(episode / 'joints.jsonl')
        stamps = data.read_jsonl(episode / 'camera_timestamps.jsonl')
        self.assertEqual(len(rows), 3)
        self.assertEqual([s['frame'] for s in stamps], [0, 1, 2])
        self.assertAlmostEqual(rows[1]['command_time'] - stamps[1]['t'], .003)
        self.assertEqual(rows[2]['camera_reference'][0], 2)
        video = cv2.VideoCapture(str(episode / 'camera.avi'))
        frames = []
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frames.append(frame)
        video.release()
        self.assertEqual(len(frames), 3)
        self.assertGreater(frames[-1].mean(), frames[0].mean())

    def test_retake_preserves_previous(self):
        first, second = self.episode(), self.episode()
        self.assertNotEqual(first, second)
        self.assertEqual(len(data.completed_episodes(self.root, 'S1_T1')), 2)
        self.assertTrue((first / 'camera.avi').is_file())

    def test_rejected_aborted_unreviewed_do_not_count(self):
        for status in ('aborted', 'error', 'review'):
            rec = self.recorder()
            self.populate(rec)
            rec.finish(status)
        rec = self.recorder()
        self.populate(rec)
        rec.finish('review')
        rec.accept(False)
        self.assertEqual(data.completed_episodes(self.root, 'S1_T1'), [])

    def test_empty_capture_cannot_be_accepted(self):
        rec = self.recorder()
        with self.assertRaisesRegex(app.TeleopError, '不完整'):
            rec.finish('review')
        self.assertEqual(json.loads((rec.path / 'episode.json').read_text(encoding='utf-8'))['status'], 'error')
        with self.assertRaises(app.TeleopError):
            rec.accept(True)

    def test_writer_failure_is_not_success(self):
        rec = self.recorder()
        with patch.object(rec, 'video', wraps=rec.video) as video:
            video.write.side_effect = OSError('disk full')
            self.populate(rec)
            with self.assertRaisesRegex(app.TeleopError, 'disk full'):
                rec.finish('review')
        self.assertEqual(data.completed_episodes(self.root, 'S1_T1'), [])

    def test_calibration_change_and_incomplete_capture_rejected(self):
        episode = self.episode()
        changed = copy.deepcopy(self.cal)
        changed['homing_offset'][0] += 1
        with self.assertRaisesRegex(app.TeleopError, '校准'):
            data.ReplayTrajectory(episode, changed)
        meta = json.loads((episode / 'episode.json').read_text(encoding='utf-8'))
        meta['status'] = 'aborted'
        data.write_json(episode / 'episode.json', meta)
        with self.assertRaisesRegex(app.TeleopError, '完整记录'):
            data.ReplayTrajectory(episode, self.cal)

    def test_bad_timestamp_and_bad_goals_rejected(self):
        for bad in ('order', 'nan', 'missing', 'range'):
            episode = self.episode()
            def edit(rows):
                if bad == 'order':
                    rows[1]['command_time'] = rows[0]['command_time']
                elif bad == 'nan':
                    rows[1]['command_time'] = float('nan')
                elif bad == 'missing':
                    rows[1]['goal_raw'].pop('gripper')
                else:
                    rows[1]['goal_raw']['gripper'] = 9000
            self.rewrite_rows(episode, edit)
            with self.subTest(bad=bad), self.assertRaises(app.TeleopError):
                data.ReplayTrajectory(episode, self.cal)

    def test_interpolation_and_half_rate(self):
        episode = self.episode()
        self.rewrite_rows(episode, lambda rows: [row['goal_raw'].update(shoulder_pan=2048 + i * 10)
                                               for i, row in enumerate(rows)])
        source = data.ReplayTrajectory(episode, self.cal, rate=.5)
        self.assertAlmostEqual(source.duration, .2)
        self.assertEqual(source(0, self.actual, self.actual)['shoulder_pan'], 2048)
        self.assertEqual(source(.05, self.actual, self.actual)['shoulder_pan'], 2053)
        self.assertEqual(source(.2, self.actual, self.actual)['shoulder_pan'], 2068)

    def test_replay_default_matches_recorded_timing(self):
        source = data.ReplayTrajectory(self.episode(), self.cal)
        self.assertAlmostEqual(source.duration, .1)
        self.assertAlmostEqual(source.rate, 1)

    def test_one_short_interval_does_not_slow_the_whole_recording(self):
        episode = self.episode()
        def edit(rows):
            start = rows[0]['command_time']
            for row, t, position in zip(rows, (0, .001, .101), (2048, 2148, 2158)):
                row['command_time'] = start + t
                row['goal_raw']['wrist_roll'] = position
        self.rewrite_rows(episode, edit)
        source = data.ReplayTrajectory(episode, self.cal, rate=1, max_speed=120)
        self.assertLess(source.duration, .25)
        self.assertGreater(source.duration, .101)
        source(0, self.actual, self.actual)
        self.assertEqual(source(source.times[1], self.actual, self.actual)['wrist_roll'], 2148)
        safe_speed = .9 * app.math.floor(120 / (.229 * 6)) * (.229 * 6)
        for a, b, ta, tb in zip(source.rows, source.rows[1:], source.times, source.times[1:]):
            self.assertLessEqual(abs(b['goal_raw']['wrist_roll'] - a['goal_raw']['wrist_roll'])
                                 * app.TICK_DEGREES / (tb - ta), safe_speed + 1e-6)

    def test_gripper_replay_preserves_closing_goal_and_contact(self):
        goals = dict(self.actual, gripper=2070)
        source = data.ReplayTrajectory(self.episode(goals), self.cal)
        actual = dict(self.actual, gripper=2500)
        # 首帧对齐必须手动完成，不能自动闭合到远处起点。
        with self.assertRaisesRegex(app.TeleopError, '起点'):
            source(0, actual, goals)
        source(0, goals, goals)
        self.assertEqual(source(1, actual, goals)['gripper'], 2070)
        self.assertTrue(source.settled)

    def test_start_alignment_and_whole_turn_shift(self):
        source = data.ReplayTrajectory(self.episode(), self.cal)
        actual = {n: v + 4096 for n, v in self.actual.items()}
        self.assertEqual(source(0, actual, actual), actual)
        source = data.ReplayTrajectory(source.path, self.cal)
        # 一刻度约 0.088°：±113 tick 在 10° 内，±114 tick 超过 10°。
        for delta in (-113, 113):
            source = data.ReplayTrajectory(source.path, self.cal)
            self.assertEqual(source(0, dict(self.actual, wrist_roll=2048 + delta), self.actual), self.actual)
        for delta in (-114, 114):
            source = data.ReplayTrajectory(source.path, self.cal)
            with self.subTest(delta=delta), self.assertRaisesRegex(app.TeleopError, '10°'):
                source(0, dict(self.actual, wrist_roll=2048 + delta), self.actual)

    def test_endpoint_timeout(self):
        source = data.ReplayTrajectory(self.episode(), self.cal)
        source(0, self.actual, self.actual)
        with self.assertRaisesRegex(app.TeleopError, '未到位'):
            source(6, dict(self.actual, shoulder_pan=2200), self.actual)

    def test_failed_send_frames_are_not_replayed(self):
        episode = self.episode()
        self.rewrite_rows(episode, lambda rows: rows[1].update(sent=False, command_time=None))
        source = data.ReplayTrajectory(episode, self.cal)
        self.assertEqual(len(source.rows), 2)

    def test_low_speed_stretches_timeline_instead_of_skipping_waypoints(self):
        episode = self.episode()
        self.rewrite_rows(episode, lambda rows: [row['goal_raw'].update(shoulder_pan=2048 + i * 10)
                                               for i, row in enumerate(rows)])
        source = data.ReplayTrajectory(episode, self.cal, rate=1, max_speed=2)
        self.assertLess(source.rate, .1)
        source(0, self.actual, self.actual)
        self.assertEqual(source(source.times[1] / 2, self.actual, self.actual)['shoulder_pan'], 2053)
        middle = source(source.duration / 2, self.actual, self.actual)
        self.assertEqual(middle['shoulder_pan'], 2058)

    def test_recorded_feedback_gap_holds_then_moves_with_a_bounded_transition(self):
        episode = self.episode()
        def edit(rows):
            start = rows[0]['command_time']
            for row, stamp, position in zip(rows, (0, 1, 1.05), (2048, 2058, 2068)):
                row['command_time'] = start + stamp
                row['goal_raw']['wrist_roll'] = position
        self.rewrite_rows(episode, edit)
        source = data.ReplayTrajectory(episode, self.cal)
        source(0, self.actual, self.actual)
        self.assertEqual(source(.8, self.actual, self.actual)['wrist_roll'], 2048)
        self.assertEqual(source((source.times[1] + source.times[2]) / 2,
                                self.actual, self.actual)['wrist_roll'], 2053)
        self.assertEqual(source(source.duration, self.actual, self.actual)['wrist_roll'], 2068)

    def test_cli_replay_defaults_to_one_times_recorded_speed(self):
        with patch.object(cli, 'resolve_episode', return_value=self.root), \
                patch.object(cli, 'replay') as replay:
            self.assertEqual(cli.main(['--replay', '1']), 0)
        args = replay.call_args.args[0]
        self.assertEqual(args.replay_rate, 1)

    def test_camera_keys_latch_interrupt_even_if_finish_follows(self):
        window = data.PreviewWindow.__new__(data.PreviewWindow)
        window.key = -1
        window._key(SimpleNamespace(keysym='Escape', char='\x1b'))
        window._key(SimpleNamespace(keysym='f', char='f'))
        self.assertEqual(window.key, 27)

    def test_replay_cli_writes_diagnostics_and_preserves_episode(self):
        episode = self.episode()
        original = (episode / 'episode.json').read_bytes()
        bus = MagicMock(port='COM_FAKE')
        control = MagicMock()
        control.period = app.REPLAY_PERIOD
        control.preflight.return_value = (None, dict(self.actual))
        def replay_run(**kwargs):
            kwargs['emit']('回放暂停于 0.050s，保持最后目标及夹持并重读：ID 5 timeout')
            kwargs['emit']('回放通信恢复，从 0.050s 继续。')
            kwargs['stop'].finished = True
        control.run.side_effect = replay_run
        args = SimpleNamespace(follower_calibration='unused.json', speed=None,
                               replay_rate=.5, follower_port='COM_FAKE')
        with patch.object(cli, 'ROOT', self.root), \
                patch.object(cli, 'load_calibration', return_value=self.cal), \
                patch.object(cli, 'connect_bus', return_value=bus), \
                patch.object(cli, 'Teleoperator', return_value=control) as constructor, \
                patch('builtins.input', return_value=''), patch('builtins.print'):
            cli.replay(args, episode)
        logs = list((self.root / 'outputs/koch_replay').glob('*.log'))
        self.assertEqual(len(logs), 1)
        text = logs[0].read_text(encoding='utf-8')
        self.assertIn('ID 5 timeout', text)
        self.assertIn('通信恢复', text)
        self.assertIn('回放已完成', text)
        self.assertEqual((episode / 'episode.json').read_bytes(), original)
        self.assertEqual(constructor.call_args.args[-1], data.REPLAY_DEFAULT_SPEED)
        self.assertEqual(data.REPLAY_DEFAULT_SPEED, 120)
        bus.disconnect.assert_called_once_with(disable_torque=False)


class ControlIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.leader, self.follower = FakeBus('leader'), FakeBus('follower')
        self.cal = calibration()
        self.clock = FakeClock()
        stable = patch.object(app, 'stable_positions', side_effect=lambda bus: bus.sync_read('Present_Position'))
        stable.start()
        self.addCleanup(stable.stop)

    def run_control(self, leader=True, source=None, callback=None, frames=5):
        control = app.Teleoperator(self.leader if leader else None, self.follower,
                                  self.cal if leader else None, self.cal)
        return control.run(stop=lambda: self.clock.sleeps >= frames, clock=self.clock,
                           sleep=self.clock.sleep, emit=lambda _: None,
                           target_source=source, on_frame=callback)

    def assert_released(self):
        self.assertTrue(all(r['Torque_Enable'] == 0 and r['Bus_Watchdog'] == 0
                            and r['Profile_Velocity'] == 0 and r['Profile_Acceleration'] == 11
                            for r in self.follower.regs.values()))
        self.assertEqual(self.leader.calls, [])

    def test_frame_hook_records_observations_and_commands(self):
        rows = []
        self.assertEqual(self.run_control(callback=rows.append), 5)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r['sent'] and r['leader_raw'] and r['follower_raw'] and r['goal_raw'] for r in rows))
        self.assertLess(rows[0]['command_time'], rows[-1]['command_time'])
        self.assert_released()

    def test_observer_failure_releases_and_propagates(self):
        def broken(row):
            raise OSError('disk full')
        with self.assertRaisesRegex(OSError, 'disk full'):
            self.run_control(callback=broken)
        self.assert_released()

    def test_replay_without_leader_and_source_clock_starts_after_enable(self):
        samples = []
        def source(elapsed, actual, previous):
            samples.append(elapsed)
            return dict.fromkeys(app.NAMES, 2048 + round(elapsed * 100))
        self.assertEqual(self.run_control(leader=False, source=source), 5)
        self.assertEqual(samples[:2], [0.0, 0.0])
        self.assertGreater(self.follower.goal['wrist_roll'], 2048)
        self.assert_released()

    def test_start_source_failure_before_writes(self):
        def source(*_):
            raise app.TeleopError('起点不符')
        with self.assertRaisesRegex(app.TeleopError, '起点'):
            self.run_control(leader=False, source=source)
        self.assertEqual(self.follower.calls, [])

    def test_replay_transient_read_failure_freezes_time_and_recovers_without_releasing(self):
        from koch_teleop_bus import FeedbackReadError
        read, samples, held_torque = self.follower.sync_read, [], []
        def flaky(*args, **kwargs):
            if 1 <= self.clock.sleeps < 3:
                held_torque.append(all(r['Torque_Enable'] == 1 for r in self.follower.regs.values()))
                self.clock.now += .2
                raise FeedbackReadError(-3001, 'COM5 ID 5 wrist_roll no status packet')
            return read(*args, **kwargs)
        self.follower.sync_read = flaky
        def source(elapsed, actual, previous):
            samples.append((elapsed, self.clock.now))
            return dict.fromkeys(app.NAMES, 2048 + round(elapsed * 100))
        self.run_control(leader=False, source=source)
        self.assertEqual(held_torque, [True, True])
        resumed = next(i for i, (_, wall) in enumerate(samples) if wall > .4)
        self.assertEqual(samples[resumed][0], samples[resumed - 1][0])
        self.assertTrue(all(b[0] - a[0] <= .051 for a, b in zip(samples, samples[1:])))
        self.assert_released()

    def test_replay_health_packet_loss_also_freezes_clock_until_all_six_recover(self):
        read, samples, health = self.follower.read_joint_status, [], []
        def flaky(name):
            if self.clock.sleeps in (3, 4) and name == 'shoulder_lift':
                raise app.FeedbackReadError(-3002, 'bad health packet')
            health.append((self.clock.sleeps, name))
            return read(name)
        self.follower.read_joint_status = flaky
        def source(elapsed, actual, previous):
            samples.append((elapsed, self.clock.sleeps))
            return dict(actual)
        self.run_control(leader=False, source=source, frames=8)
        resumed = next(t for t, frame in samples if frame == 5)
        self.assertAlmostEqual(resumed, 2 / 60)
        self.assertEqual({name for frame, name in health if frame == 5}, set(app.NAMES))
        self.assert_released()

    def test_replay_uses_60_hz_with_20_hz_health_checks(self):
        control = app.Teleoperator(None, self.follower, None, self.cal)
        checks, rows, messages = [], [], []
        check = control.check_joint
        def health(bus, arm, name, active):
            checks.append(name)
            check(bus, arm, name, active)
        control.check_joint = health
        control.run(stop=lambda: self.clock.sleeps >= 18, clock=self.clock,
                    sleep=self.clock.sleep, emit=messages.append,
                    target_source=lambda t, a, p: dict(a), on_frame=rows.append)
        self.assertAlmostEqual(control.period, 1 / 60)
        self.assertAlmostEqual(self.clock.now, .3)
        self.assertEqual(checks, list(app.NAMES))
        intervals = [b['command_time'] - a['command_time'] for a, b in zip(rows, rows[1:])]
        self.assertTrue(all(abs(dt - 1 / 60) < 1e-6 for dt in intervals))
        self.assert_released()

    def test_replay_reports_actual_update_rate(self):
        control = app.Teleoperator(None, self.follower, None, self.cal)
        messages = []
        control.run(stop=lambda: self.clock.sleeps >= 62, clock=self.clock,
                    sleep=self.clock.sleep, emit=messages.append,
                    target_source=lambda t, a, p: dict(a))
        self.assertTrue(any('目标 60.0 Hz，实际 60.0 Hz' in m for m in messages))
        self.assert_released()

    def test_replay_persistent_communication_loss_is_bounded(self):
        read = self.follower.sync_read
        def broken(*args, **kwargs):
            if self.clock.sleeps:
                raise app.FeedbackReadError(-3001, 'ID 5 lost')
            return read(*args, **kwargs)
        self.follower.sync_read = broken
        control = app.Teleoperator(None, self.follower, None, self.cal)
        with self.assertRaisesRegex(app.TeleopError, '连续 10 秒.*停止回放'):
            control.run(stop=lambda: False, clock=self.clock, sleep=self.clock.sleep,
                        emit=lambda _: None, target_source=lambda t, a, p: dict(a))
        self.assertLess(self.clock.now, 10.2)
        self.assert_released()

    def test_replay_recovery_does_not_hide_hardware_fault(self):
        read, samples = self.follower.sync_read, []
        def broken(*args, **kwargs):
            if self.clock.sleeps == 1:
                raise app.FeedbackReadError(-3001, 'ID 5 lost')
            if self.clock.sleeps >= 2:
                self.follower.regs['wrist_roll']['Hardware_Error_Status'] = 4
            return read(*args, **kwargs)
        self.follower.sync_read = broken
        def source(t, a, p):
            samples.append(t)
            return dict(a)
        with self.assertRaisesRegex(app.TeleopError, 'wrist_roll.Hardware_Error_Status'):
            self.run_control(leader=False, source=source)
        self.assertEqual(samples, [0, 0])
        self.assert_released()

    def test_replay_noncommunication_error_is_not_retried(self):
        read = self.follower.sync_read
        def broken(*args, **kwargs):
            if self.clock.sleeps:
                raise RuntimeError('explicit motor error')
            return read(*args, **kwargs)
        self.follower.sync_read = broken
        with self.assertRaisesRegex(app.TeleopError, '非瞬时通信异常.*explicit motor error'):
            self.run_control(leader=False, source=lambda t, a, p: dict(a))
        self.assertEqual(self.clock.sleeps, 1)
        self.assert_released()

    def test_replay_recovery_does_not_ignore_encoder_jump(self):
        read = self.follower.sync_read
        def broken(*args, **kwargs):
            if self.clock.sleeps == 1:
                raise app.FeedbackReadError(-3001, 'ID 5 lost')
            row = read(*args, **kwargs)
            if self.clock.sleeps >= 2:
                row['wrist_roll'] += 4096
            return row
        self.follower.sync_read = broken
        with self.assertRaisesRegex(app.TeleopError, '跳变'):
            self.run_control(leader=False, source=lambda t, a, p: dict(a))
        self.assert_released()

    def test_replay_stop_remains_responsive_while_waiting_for_feedback(self):
        read = self.follower.sync_read
        def broken(*args, **kwargs):
            if self.clock.sleeps:
                raise app.FeedbackReadError(-3001, 'ID 5 lost')
            return read(*args, **kwargs)
        self.follower.sync_read = broken
        control = app.Teleoperator(None, self.follower, None, self.cal)
        control.run(stop=lambda: self.clock.now >= .1, clock=self.clock, sleep=self.clock.sleep,
                    emit=lambda _: None, target_source=lambda t, a, p: dict(a))
        self.assertLess(self.clock.now, .15)
        self.assert_released()


class SmoothCollectionTests(unittest.TestCase):
    def setUp(self):
        self.leader, self.follower = FakeBus('leader'), FakeBus('follower')
        self.cal = calibration()
        self.clock = FakeClock()
        self.control = app.Teleoperator(self.leader, self.follower, self.cal, self.cal,
                                       app.COLLECTION_SPEED, smooth_collection=True)
        self.rows, self.messages = [], []
        stable = patch.object(app, 'stable_positions', side_effect=lambda bus: bus.sync_read('Present_Position'))
        stable.start()
        self.addCleanup(stable.stop)

    def run_control(self, frames=12, stop=None):
        return self.control.run(stop=stop or (lambda: self.clock.sleeps >= frames),
                                clock=self.clock, sleep=self.clock.sleep,
                                emit=self.messages.append, on_frame=self.rows.append)

    def assert_released(self):
        self.assertTrue(all(r['Torque_Enable'] == 0 and r['Bus_Watchdog'] == 0
                            and r['Profile_Velocity'] == 0 and r['Profile_Acceleration'] == 11
                            for r in self.follower.regs.values()))
        self.assertEqual(self.leader.calls, [])

    def test_double_update_rate_but_health_still_covers_all_six_joints(self):
        checks, goal_checks = [], []
        check, verify = self.control.check_joint, self.control.check_goal_applied
        def health(bus, arm, name, active):
            checks.append((arm, name, self.clock()))
            check(bus, arm, name, active)
        def applied(name, goal):
            goal_checks.append(name)
            verify(name, goal)
        self.control.check_joint, self.control.check_goal_applied = health, applied
        self.assertEqual(self.run_control(), 12)
        self.assertAlmostEqual(self.clock.now, .3)
        for arm in ('leader', 'follower'):
            self.assertEqual([n for a, n, _ in checks if a == arm], list(app.NAMES))
        self.assertEqual(goal_checks, list(app.NAMES))
        self.assert_released()

    def test_isolated_leader_or_follower_spike_does_not_stop_or_enter_dataset(self):
        for bus in (self.leader, self.follower):
            read = bus.sync_read
            def glitch(*a, _read=read, **k):
                row = _read(*a, **k)
                if self.clock.sleeps == 1:
                    row['wrist_roll'] += 4096
                return row
            bus.sync_read = glitch
        self.assertEqual(self.run_control(frames=6), 5)
        self.assertEqual(self.control.discarded_samples, 1)
        self.assertTrue(all(row['leader_raw']['wrist_roll'] == 2048
                            and row['follower_raw']['wrist_roll'] == 2048 for row in self.rows))
        goals = [c[1]['wrist_roll'] for c in self.follower.calls if c[0] == 'sync_goals']
        self.assertEqual(set(goals), {2048})
        self.assert_released()

    def test_persistent_jump_still_stops_and_releases(self):
        read = self.leader.sync_read
        def glitch(*a, **k):
            row = read(*a, **k)
            if self.clock.sleeps:
                row['wrist_roll'] += 4096
            return row
        self.leader.sync_read = glitch
        with self.assertRaisesRegex(app.TeleopError, '位置异常持续'):
            self.run_control(frames=100)
        self.assertLess(self.clock.now, .5)
        self.assertEqual(len(self.rows), 1)
        self.assert_released()

    def test_stop_is_responsive_during_rejected_samples(self):
        read = self.leader.sync_read
        def glitch(*a, **k):
            row = read(*a, **k)
            if self.clock.sleeps:
                row['wrist_roll'] += 4096
            return row
        self.leader.sync_read = glitch
        self.run_control(stop=lambda: self.clock.now >= .1)
        self.assertLessEqual(self.clock.now, .101)
        self.assert_released()

    def test_isolated_send_failure_while_holding_does_not_stop(self):
        read, send = self.leader.sync_read, self.follower.sync_write_goals
        def glitch(*a, **k):
            row = read(*a, **k)
            if self.clock.sleeps == 1:
                row['wrist_roll'] += 4096
            return row
        def lost(goals):
            if self.clock.sleeps == 1:
                raise app.DynamixelCommunicationError(-1001, 'lost hold packet')
            send(goals)
        self.leader.sync_read, self.follower.sync_write_goals = glitch, lost
        self.assertEqual(self.run_control(frames=6), 5)
        self.assert_released()

    def test_startup_speed_ramps_then_reaches_target_at_full_precision(self):
        self.leader.raw['wrist_roll'] += 512
        self.run_control(frames=50)
        goals = [row['goal_raw']['wrist_roll'] for row in self.rows]
        self.assertLessEqual((goals[1] - goals[0]) * app.TICK_DEGREES,
                             34 * app.COLLECTION_PERIOD)
        steps = [(b - a) * app.TICK_DEGREES for a, b in zip(goals, goals[1:])]
        self.assertLessEqual(max(steps), app.COLLECTION_SPEED * app.COLLECTION_PERIOD)
        self.assertGreater(max(steps), 90 * app.COLLECTION_PERIOD)
        self.assertEqual(goals[-1], 2560)
        self.assert_released()

    def test_minimum_speed_still_moves_at_40_hz(self):
        self.control.speed = 2
        self.leader.raw['wrist_roll'] += 20
        self.run_control(frames=50)
        self.assertEqual(self.follower.goal['wrist_roll'], 2068)
        self.assert_released()

    def test_hardware_fault_and_torque_start_jump_are_not_suppressed(self):
        self.follower.jump_on_enable = True
        with self.assertRaisesRegex(app.TeleopError, '启用扭矩后位置发生跳变'):
            self.run_control()
        self.assert_released()

    def test_runtime_hardware_fault_still_stops(self):
        sleep = self.clock.sleep
        def fault(duration):
            sleep(duration)
            self.follower.regs['shoulder_lift']['Hardware_Error_Status'] = 4
        self.clock.sleep = fault
        with self.assertRaisesRegex(app.TeleopError, 'Hardware_Error_Status'):
            self.run_control()
        self.assert_released()

    def test_replay_cannot_accidentally_use_collection_mode(self):
        with self.assertRaisesRegex(app.TeleopError, '独立回放'):
            app.Teleoperator(None, self.follower, None, self.cal, smooth_collection=True)


if __name__ == '__main__':
    unittest.main()
