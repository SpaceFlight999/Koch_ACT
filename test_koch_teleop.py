"""主从遥操作离线测试；不连接真实机械臂。"""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import teleoperate_koch as app
from calibrate_koch import CalibrationError, NAMES
from koch_legacy_calibration import compute_calibration
from koch_teleop_bus import FeedbackReadError, KochTeleopBus
from koch_calibration_bus import DynamixelCommunicationError
from test_calibrate_koch import FakeBus as CalibrationFakeBus


def calibration(inverted=()):
    zero = dict.fromkeys(NAMES, 2048)
    rotated = {n: 2048 + (-1024 if n in inverted else 1024) for n in NAMES}
    return compute_calibration(zero, rotated)


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = 0.0, 0
    def __call__(self):
        return self.now
    def sleep(self, duration):
        self.now += max(duration, 0.001)
        self.sleeps += 1


class FakeBus(CalibrationFakeBus):
    def __init__(self, arm):
        super().__init__(arm, 'COM_LEADER' if arm == 'leader' else 'COM_FOLLOWER')
        self.raw = dict.fromkeys(NAMES, 2048)
        for regs in self.regs.values():
            regs.update(Drive_Mode=0, Homing_Offset=0, Bus_Watchdog=0, Profile_Velocity=0,
                        Profile_Acceleration=11)
        self.goal = dict(self.raw)
        self.failure = None
        self.jump_on_enable = False
        self.simulate_motion = True

    def read_motion(self, register, name):
        return self.goal[name] if register == 'Goal_Position' else self.regs[name][register]

    def read_joint_status(self, name):
        return dict(self.regs[name])

    def write_motion(self, register, name, value):
        if self.arm == 'leader':
            raise AssertionError('不允许写主臂')
        self.calls.append(('motion', register, name, value))
        if self.failure and self.failure(register, name, value):
            raise RuntimeError('injected serial failure')
        if register == 'Goal_Position':
            self.goal[name] = value
            if self.regs[name]['Torque_Enable'] and self.simulate_motion:
                self.raw[name] = value
        else:
            self.regs[name][register] = value
            if register == 'Torque_Enable' and value and self.jump_on_enable:
                self.raw[name] += 4096

    def sync_write_goals(self, goals):
        if self.arm == 'leader':
            raise AssertionError('不允许写主臂')
        self.calls.append(('sync_goals', dict(goals)))
        for name, value in goals.items():
            self.goal[name] = value
            if self.regs[name]['Torque_Enable'] and self.simulate_motion:
                self.raw[name] = value


class TeleopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='.teleop_test_', dir=app.ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.leader, self.follower = FakeBus('leader'), FakeBus('follower')
        self.leader_cal, self.follower_cal = calibration(('elbow_flex',)), calibration(('wrist_flex',))
        self.control = app.Teleoperator(self.leader, self.follower, self.leader_cal, self.follower_cal)
        self.clock = FakeClock()
        stable = patch.object(app, 'stable_positions', side_effect=lambda bus: bus.sync_read('Present_Position'))
        stable.start()
        self.addCleanup(stable.stop)

    def run_control(self, frames=3, **kwargs):
        return self.control.run(stop=lambda: self.clock.sleeps >= frames, clock=self.clock,
                                sleep=self.clock.sleep, emit=lambda _: None, **kwargs)

    def assert_released(self):
        self.assertTrue(all(r['Torque_Enable'] == 0 and r['Bus_Watchdog'] == 0
                            and r['Profile_Velocity'] == 0 and r['Profile_Acceleration'] == 11
                            for r in self.follower.regs.values()))
        self.assertEqual(self.leader.calls, [])

    def test_calibration_file_validation_and_reordering(self):
        path = self.root / 'leader.json'
        path.write_text(json.dumps(self.leader_cal), encoding='utf-8')
        self.assertEqual(app.load_calibration(path), self.leader_cal)
        data = {key: list(reversed(value)) for key, value in self.leader_cal.items()}
        path.write_text(json.dumps(data), encoding='utf-8')
        self.assertEqual(app.load_calibration(path), self.leader_cal)
        for bad in ({}, dict(self.leader_cal, homing_offset=[0] * 6),
                    dict(self.leader_cal, start_pos=[float('nan')] * 6)):
            path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaises(app.TeleopError):
                app.load_calibration(path)
        with self.assertRaises(app.TeleopError):
            app.load_calibration(self.root / 'missing.json')

    def test_mapping_uses_turn_nearest_reference(self):
        raw = dict.fromkeys(NAMES, 2048)
        base = app.map_targets(raw, self.leader_cal, self.follower_cal)
        shifted = app.map_targets(raw, self.leader_cal, self.follower_cal,
                                  {name: value + 4096 for name, value in base.items()})
        for name in NAMES:
            self.assertEqual(shifted[name] - base[name], 4096)

    def test_alignment_ignores_full_turn_offset_between_arms(self):
        for name in NAMES:
            self.follower.raw[name] += 4096
            self.follower.goal[name] += 4096
        self.control.preflight()

    def test_software_angle_limit_crosses_half_turn_but_rejects_two_turns(self):
        raw = dict.fromkeys(NAMES, 2048)
        raw['elbow_flex'] = 4100  # 约 -180.4°，实测启动时就是这个姿态
        app.checked_angles(raw, self.follower_cal)
        raw['elbow_flex'] = -6200  # 约 +725°，明显异常
        with self.assertRaises(app.TeleopError):
            app.checked_angles(raw, self.follower_cal)

    def test_direction_mapping_is_absolute_without_initial_excursion_limit(self):
        for angle in (-90, 0, 90, 150):
            raw = dict.fromkeys(NAMES, 2048)
            raw['elbow_flex'] -= round(angle / app.TICK_DEGREES)
            raw['wrist_flex'] += round(angle / app.TICK_DEGREES)
            goals = app.map_targets(raw, self.leader_cal, self.follower_cal)
            angles = app.checked_angles(goals, self.follower_cal)
            self.assertAlmostEqual(angles['elbow_flex'], angle, delta=.1)
            self.assertAlmostEqual(angles['wrist_flex'], angle, delta=.1)

    def test_gripper_uses_measured_open_close_endpoints(self):
        follower_cal = copy.deepcopy(self.follower_cal)
        # 校准允许的安装误差使两夹爪的实际采样幅度不同。
        zero, rotated = dict.fromkeys(NAMES, 2048), dict.fromkeys(NAMES, 3072)
        rotated['gripper'] = 2850
        follower_cal = compute_calibration(zero, rotated)
        for source, expected in ((2048, 2048), (2560, 2449), (3072, 2850), (3300, 2850), (1900, 2048)):
            raw = dict.fromkeys(NAMES, 2048)
            raw['gripper'] = source
            self.assertEqual(app.map_targets(raw, calibration(), follower_cal)['gripper'], expected)

    def test_slew_limit_handles_long_pause(self):
        old, target = dict.fromkeys(NAMES, 0), dict.fromkeys(NAMES, 10000)
        for dt in (.01, .05, 5):
            result = app.limit_speed(target, old, 20, dt)
            self.assertLessEqual(max(result.values()) * app.TICK_DEGREES, 20 * min(dt, .1))

    def test_short_and_reversed_gripper_files_load_and_map_endpoints(self):
        zero, rotated = dict.fromkeys(NAMES, 2048), dict.fromkeys(NAMES, 3072)
        for leader_end in (2641, 1161):
            leader = compute_calibration(dict(zero, gripper=1901), dict(rotated, gripper=leader_end))
            follower = compute_calibration(dict(zero, gripper=3500), dict(rotated, gripper=3000))
            path = self.root / 'short_gripper.json'
            path.write_text(json.dumps(leader), encoding='utf-8')
            leader = app.load_calibration(path)
            for fraction, target in ((0, 3500), (.5, 3250), (1, 3000)):
                raw = dict(zero, gripper=round(1901 + fraction * (leader_end - 1901)))
                self.assertEqual(app.map_targets(raw, leader, follower)['gripper'], target)

    def test_previous_rounded_gripper_calibration_loads_without_rewriting_file(self):
        zero, rotated = dict.fromkeys(NAMES, 2048), dict.fromkeys(NAMES, 3072)
        zero['gripper'], rotated['gripper'] = 2118, 3142
        old = compute_calibration(zero, rotated)
        old['homing_offset'][-1] = -2048
        path = self.root / 'previous.json'
        original = json.dumps(old)
        path.write_text(original, encoding='utf-8')
        data = app.load_calibration(path)
        self.assertEqual(data['homing_offset'][-1], -2118)
        self.assertEqual(path.read_text(encoding='utf-8'), original)
        old['homing_offset'][-1] = 123
        path.write_text(json.dumps(old), encoding='utf-8')
        with self.assertRaises(app.TeleopError):
            app.load_calibration(path)

    def test_preflight_retries_transient_read_failure(self):
        real, calls = app.snapshot, []

        def flaky(bus):
            calls.append(bus)
            if len(calls) == 1:
                raise RuntimeError('COM5 读取 ID 6 gripper.Bus_Watchdog: 连续 3 次读取失败')
            return real(bus)

        with patch.object(app, 'snapshot', side_effect=flaky):
            self.control.preflight()
        self.assertEqual(len(calls), 3)  # 第一次失败后重试，再读主臂、从臂

    def test_preflight_is_read_only(self):
        self.control.preflight()
        self.assertEqual(self.follower.calls, [])
        self.assertEqual(self.leader.calls, [])

    def test_preflight_rejects_health_baseline_and_alignment_faults(self):
        for register, value in (('Torque_Enable', 1), ('Homing_Offset', 16), ('Drive_Mode', 1),
                                ('Operating_Mode', 3), ('Bus_Watchdog', 50), ('Present_Input_Voltage', 131)):
            original = self.follower.regs['shoulder_pan'].copy()
            self.follower.regs['shoulder_pan'][register] = value
            with self.subTest(register=register), self.assertRaises((app.TeleopError, CalibrationError)):
                self.control.preflight()
            self.follower.regs['shoulder_pan'] = original
        self.leader.raw['shoulder_pan'] += 513
        with self.assertRaises(app.TeleopError):
            self.control.preflight()
        self.assertEqual(self.follower.calls, [])

    def test_preflight_shows_actual_temperature_and_distinguishes_zero_fault(self):
        self.follower.regs['shoulder_lift']['Present_Temperature'] = 56
        with self.assertRaises(app.TeleopError) as failure:
            self.control.preflight()
        message = str(failure.exception)
        self.assertIn('ID 2 shoulder_lift', message)
        self.assertIn('56°C', message)
        self.assertIn('55°C', message)
        self.assertIn('故障码 0x00', message)
        self.assertNotIn('停止校准', message)
        self.assertEqual(self.follower.calls, [])

    def test_preflight_hardware_fault_has_numeric_code_and_temperature(self):
        self.follower.regs['shoulder_lift'].update(Hardware_Error_Status=4, Present_Temperature=40)
        with self.assertRaises(app.TeleopError) as failure:
            self.control.preflight()
        message = str(failure.exception)
        self.assertIn('ID 2 shoulder_lift.Hardware_Error_Status=0x04', message)
        self.assertIn('40°C', message)
        self.assertEqual(self.follower.calls, [])

    def test_start_and_runtime_temperature_checks_share_the_same_boundary(self):
        for temp in (54, 55, 56):
            self.follower.regs['shoulder_lift']['Present_Temperature'] = temp
            if temp < 55:
                self.control.preflight()
                self.control.check_joint(self.follower, 'follower', 'shoulder_lift', False)
            else:
                with self.assertRaisesRegex(app.TeleopError, f'{temp}°C.*55°C'):
                    self.control.preflight()
                with self.assertRaisesRegex(app.TeleopError, f'{temp}°C.*55°C'):
                    self.control.check_joint(self.follower, 'follower', 'shoulder_lift', False)

    def test_start_alignment_accepts_both_directions_up_to_45_degrees(self):
        for delta in (-512, -256, 256, 512):
            with self.subTest(delta=delta):
                self.leader.raw['wrist_roll'] = 2048 + delta
                self.control.preflight()
        for delta in (-513, 513):
            self.leader.raw['wrist_roll'] = 2048 + delta
            with self.subTest(delta=delta), self.assertRaisesRegex(app.TeleopError, '45°.*wrist_roll'):
                self.control.preflight()
        self.assertEqual(self.follower.calls, [])

    def test_large_start_difference_converges_with_existing_speed_limit(self):
        self.leader.raw['wrist_roll'] += 512
        self.run_control(frames=70)
        writes = self.follower.calls
        goals = [call[1]['wrist_roll'] for call in writes if call[0] == 'sync_goals']
        seed = ('motion', 'Goal_Position', 'wrist_roll', 2048)
        self.assertIn(seed, writes)
        self.assertLess(writes.index(('motion', 'Torque_Enable', 'wrist_roll', 1)), writes.index(seed))
        self.assertEqual(goals[-1], 2560)
        self.assertTrue(all(0 <= after - before <= self.control.speed * .05 / app.TICK_DEGREES
                            for before, after in zip([2048] + goals, goals)))
        self.assert_released()

    def test_speed_bounds_allow_faster_following(self):
        self.assertEqual(self.control.speed, app.DEFAULT_SPEED)
        for speed in (2, app.DEFAULT_SPEED, 90, 120):
            app.Teleoperator(self.leader, self.follower, self.leader_cal, self.follower_cal, speed)
        for speed in (1.9, 120.1, float('nan')):
            with self.subTest(speed=speed), self.assertRaises(app.TeleopError):
                app.Teleoperator(self.leader, self.follower, self.leader_cal, self.follower_cal, speed)

    def test_faster_following_reaches_exact_stationary_target(self):
        previous = dict.fromkeys(NAMES, 2048)
        target = dict(previous, wrist_roll=2560)
        slow = app.limit_speed(target, previous, 60, app.PERIOD)
        fast = app.limit_speed(target, previous, app.DEFAULT_SPEED, app.PERIOD)
        self.assertGreater(fast['wrist_roll'], slow['wrist_roll'])
        self.leader.raw['wrist_roll'] = 2560
        messages = []
        self.control.run(stop=lambda: self.clock.sleeps >= 40, clock=self.clock,
                         sleep=self.clock.sleep, emit=messages.append)
        self.assertEqual(self.follower.raw['wrist_roll'], 2560)
        self.assertEqual(self.follower.goal['wrist_roll'], 2560)
        self.assertTrue(any('最大关节' in message and '=0.00°' in message for message in messages))
        self.assert_released()

    def test_goal_tolerance_scales_with_speed(self):
        def tolerance(speed):
            return app.Teleoperator(self.leader, self.follower, self.leader_cal,
                                    self.follower_cal, speed).goal_tolerance()
        self.assertEqual(tolerance(2), app.GOAL_READBACK_TOLERANCE)
        self.assertEqual(tolerance(20), app.GOAL_READBACK_TOLERANCE)
        self.assertGreater(tolerance(90), tolerance(20))
        self.assertGreaterEqual(tolerance(90), 3 * 90 * app.PERIOD / app.TICK_DEGREES)

    def test_loop_starts_slow_then_seeds_goal_after_torque(self):
        self.assertEqual(self.run_control(), 3)
        writes = self.follower.calls
        for name in NAMES:
            slow = writes.index(('motion', 'Profile_Velocity', name, 1))
            enable = writes.index(('motion', 'Torque_Enable', name, 1))
            seed = writes.index(('motion', 'Goal_Position', name, 2048))
            self.assertLess(slow, enable)
            self.assertLess(enable, seed)
            self.assertLess(seed, writes.index(('motion', 'Profile_Velocity', name,
                                                app.math.floor(self.control.speed / (0.229 * 6)))))
        self.assert_released()

    def test_xl430_up_to_thirteen_volts_runs_and_releases(self):
        for reading in (125, 130):
            with self.subTest(reading=reading):
                self.clock = FakeClock()
                for name in ('shoulder_pan', 'shoulder_lift'):
                    self.follower.regs[name]['Present_Input_Voltage'] = reading
                self.assertEqual(self.run_control(), 3)
                self.assert_released()

    def test_runtime_uses_model_specific_voltage_boundaries(self):
        for bus, arm, name, low, high in ((self.follower, 'follower', 'shoulder_pan', 65, 130),
                                         (self.follower, 'follower', 'elbow_flex', 37, 60),
                                         (self.leader, 'leader', 'shoulder_pan', 37, 60)):
            for reading in (low, high):
                with self.subTest(arm=arm, joint=name, reading=reading):
                    bus.regs[name]['Present_Input_Voltage'] = reading
                    self.control.check_joint(bus, arm, name, active=False)
            for reading in (low - 1, high + 1):
                with self.subTest(arm=arm, joint=name, reading=reading), self.assertRaises(app.TeleopError):
                    bus.regs[name]['Present_Input_Voltage'] = reading
                    self.control.check_joint(bus, arm, name, active=False)

    def test_dry_run_never_writes_anything(self):
        self.leader.raw['shoulder_pan'] += 512  # 预览不需要两臂对齐，也不限制启动点附近幅度。
        self.run_control(dry_run=True)
        self.assertEqual(self.follower.calls, [])
        self.assertEqual(self.leader.calls, [])

    def test_keyboard_interrupt_during_loop_releases(self):
        def interrupt(duration):
            raise KeyboardInterrupt
        self.clock.sleep = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_control()
        self.assert_released()

    def test_goal_seed_failure_releases(self):
        self.follower.failure = lambda reg, name, value: reg == 'Goal_Position'
        with self.assertRaises(RuntimeError):
            self.run_control()
        self.assert_released()

    def test_isolated_goal_send_losses_do_not_stop_following(self):
        original = self.follower.sync_write_goals
        frames = []

        def flaky(goals):
            frames.append(dict(goals))
            if len(frames) <= 2:
                raise DynamixelCommunicationError(-1001, 'send failed')
            original(goals)

        self.follower.sync_write_goals = flaky
        self.assertEqual(self.run_control(frames=6), 6)
        self.assertEqual(len(frames), 6)
        self.assert_released()

    def test_three_consecutive_goal_send_failures_stop_and_release(self):
        def dead(goals):
            raise DynamixelCommunicationError(-1001, 'send failed')

        self.follower.sync_write_goals = dead
        with self.assertRaisesRegex(app.TeleopError, '连续 3 个周期'):
            self.run_control(frames=10)
        self.assert_released()

    def test_seed_ack_loss_is_resent_with_same_value_after_torque(self):
        write, attempts = self.follower.write_motion, []

        def lost(register, name, value):
            if register == 'Goal_Position':
                self.assertEqual(self.follower.regs[name]['Torque_Enable'], 1)
                attempts.append((name, value))
                if len(attempts) == 1:
                    raise DynamixelCommunicationError(-3001, 'lost write acknowledgement')
            write(register, name, value)

        self.follower.write_motion = lost
        self.run_control()
        self.assertEqual(attempts[0], ('shoulder_pan', 2048))
        self.assertEqual(attempts[1], attempts[0])
        self.assertIn(('motion', 'Torque_Enable', 'shoulder_pan', 1), self.follower.calls)
        self.assert_released()

    def test_seed_unconfirmed_after_three_attempts_stops_and_releases(self):
        write, attempts = self.follower.write_motion, []

        def lost(register, name, value):
            if register == 'Goal_Position':
                attempts.append((name, value))
                raise DynamixelCommunicationError(-3001, 'lost write acknowledgement')
            write(register, name, value)

        self.follower.write_motion = lost
        with self.assertRaises(DynamixelCommunicationError):
            self.run_control()
        self.assertEqual(attempts, [('shoulder_pan', 2048)] * app.CONFIRM_ATTEMPTS)
        self.assertIn(('motion', 'Torque_Enable', 'shoulder_pan', 1), self.follower.calls)
        self.assert_released()

    def test_transient_read_failure_holds_then_resumes(self):
        original, calls, messages = self.follower.sync_read, [], []

        def flaky(register, normalize=False):
            calls.append(register)
            if len(calls) in (4, 5):
                raise RuntimeError('COM5 读取 ID 6 gripper.Present_Position: 连续 3 次读取失败')
            return original(register, normalize=normalize)

        self.follower.sync_read = flaky
        frames = self.control.run(stop=lambda: self.clock.now >= 0.6, clock=self.clock,
                                  sleep=self.clock.sleep, emit=messages.append)
        self.assertGreater(frames, 3)
        self.assertTrue(any('保持当前目标' in message for message in messages))
        goals = [call for call in self.follower.calls if call[0] == 'sync_goals']
        self.assertGreater(len(goals), frames)
        self.assert_released()

    def test_persistent_read_failure_holds_then_stops(self):
        original, calls = self.follower.sync_read, []

        def broken(register, normalize=False):
            calls.append(register)
            if len(calls) >= 4:
                raise RuntimeError('COM5 读取 ID 6 gripper.Present_Position: 连续 3 次读取失败')
            return original(register, normalize=normalize)

        self.follower.sync_read = broken
        with self.assertRaisesRegex(app.TeleopError, '无法读取关节位置'):
            self.control.run(stop=lambda: False, clock=self.clock, sleep=self.clock.sleep,
                             emit=lambda _: None)
        self.assert_released()

    def test_dead_goal_stream_is_detected_by_goal_readback(self):
        original = self.clock.sleep

        def move(duration):
            original(duration)
            for name in NAMES:
                self.leader.raw[name] += 40

        self.clock.sleep = move
        self.follower.sync_write_goals = lambda goals: self.follower.calls.append(
            ('sync_goals', dict(goals)))
        with self.assertRaisesRegex(app.TeleopError, '目标流未生效'):
            self.run_control(frames=12)
        self.assert_released()

    def test_tracking_failure_stops_and_releases(self):
        self.follower.simulate_motion = False
        original = self.clock.sleep
        def move(duration):
            original(duration)
            self.leader.raw['shoulder_lift'] = 3072
        self.clock.sleep = move
        with self.assertRaisesRegex(app.TeleopError, '跟踪误差'):
            self.run_control(frames=100)
        self.assert_released()

    def test_gripper_contact_allows_sustained_grasp(self):
        self.leader.raw['gripper'] = self.follower.raw['gripper'] = 2560
        self.follower.simulate_motion = False
        original = self.clock.sleep
        def close(duration):
            original(duration)
            self.leader.raw['gripper'] = 2048
        self.clock.sleep = close
        self.assertEqual(self.run_control(frames=120), 120)
        self.assertEqual(self.follower.goal['gripper'], 2048)
        self.assert_released()

    def test_gripper_blocked_while_opening_still_stops(self):
        self.follower.simulate_motion = False
        original = self.clock.sleep
        def open_gripper(duration):
            original(duration)
            self.leader.raw['gripper'] = 3072
        self.clock.sleep = open_gripper
        with self.assertRaisesRegex(app.TeleopError, 'gripper.*跟踪误差'):
            self.run_control(frames=100)
        self.assert_released()

    def test_grasp_survives_small_leader_opening_while_target_still_closes(self):
        self.leader.raw['gripper'] = self.follower.raw['gripper'] = 2560
        self.follower.simulate_motion = False
        original = self.clock.sleep
        def move(duration):
            original(duration)
            self.leader.raw['gripper'] = 2048 + max(0, self.clock.sleeps - 40) * 2
        self.clock.sleep = move
        self.assertEqual(self.run_control(frames=90), 90)
        self.assertLess(self.follower.goal['gripper'], self.follower.raw['gripper'])
        self.assert_released()

    def test_gripper_contact_handles_direction_turns_and_invalid_range(self):
        for inverted in ((), ('gripper',)):
            cal = calibration(inverted)
            sign = -1 if inverted else 1
            for turn in (-4096, 0, 4096):
                closed = 2048 + turn
                with self.subTest(inverted=inverted, turn=turn):
                    self.assertTrue(app.gripper_closing_contact(
                        closed + sign * 512, closed + sign * 100, closed, cal))
                    self.assertFalse(app.gripper_closing_contact(
                        closed + sign * 512, closed + sign * 100, closed + sign * 600, cal))
                    self.assertFalse(app.gripper_closing_contact(
                        closed + sign * 1200, closed + sign * 100, closed, cal))

    def test_gripper_contact_does_not_hide_overtemperature(self):
        self.leader.raw['gripper'] = self.follower.raw['gripper'] = 2560
        self.follower.simulate_motion = False
        original = self.clock.sleep
        def close_and_heat(duration):
            original(duration)
            self.leader.raw['gripper'] = 2048
            if self.clock.sleeps > 40:
                self.follower.regs['gripper']['Present_Temperature'] = 55
        self.clock.sleep = close_and_heat
        with self.assertRaisesRegex(app.TeleopError, 'gripper.*温度'):
            self.run_control(frames=120)
        self.assert_released()

    def test_watchdog_fault_restores_profiles_after_torque_off(self):
        original = self.clock.sleep
        def fault(duration):
            original(duration)
            for regs in self.follower.regs.values():
                regs.update(Bus_Watchdog=255, Profile_Velocity=0, Profile_Acceleration=0)
        self.clock.sleep = fault
        with self.assertRaisesRegex(app.TeleopError, '看门狗'):
            self.run_control()
        self.assert_released()

    def test_failed_torque_release_reports_error_and_keeps_watchdog(self):
        def fail_release(num_retry=0):
            raise RuntimeError('lost serial connection')
        self.follower.disable_torque = fail_release
        with self.assertRaisesRegex(app.TeleopError, '硬件断电'):
            self.run_control()
        self.assertTrue(all(regs['Bus_Watchdog'] == 50 for regs in self.follower.regs.values()))

    def test_cleanup_accepts_lost_ack_only_after_readback_matches(self):
        write = self.follower.write_motion
        def lost_ack(register, name, value):
            write(register, name, value)
            if (register, name, value) == ('Profile_Velocity', 'gripper', 0):
                raise DynamixelCommunicationError(-3001, 'lost write acknowledgement')
        self.follower.write_motion = lost_ack
        self.run_control()
        self.assert_released()

    def test_cleanup_retries_unapplied_config_write_with_torque_off(self):
        write, attempts = self.follower.write_motion, []
        def lost_write(register, name, value):
            if (register, name, value) == ('Profile_Velocity', 'gripper', 0):
                self.assertEqual(self.follower.regs[name]['Torque_Enable'], 0)
                attempts.append(value)
                if len(attempts) == 1:
                    raise DynamixelCommunicationError(-3001, 'lost write')
            write(register, name, value)
        self.follower.write_motion = lost_write
        self.run_control()
        self.assertEqual(len(attempts), 2)
        self.assert_released()

    def test_cleanup_reports_config_failure_separately_and_keeps_restoring(self):
        write, attempts = self.follower.write_motion, []
        def lost_write(register, name, value):
            if (register, name, value) == ('Profile_Velocity', 'gripper', 0):
                attempts.append(value)
                self.follower.regs[name]['Profile_Acceleration'] = 0
                raise DynamixelCommunicationError(-3001, 'lost write')
            write(register, name, value)
        self.follower.write_motion = lost_write
        with self.assertRaisesRegex(app.TeleopError, '扭矩关闭.*配置') as failure:
            self.run_control()
        self.assertNotIn('硬件断电', str(failure.exception))
        self.assertEqual(len(attempts), 3)
        self.assertTrue(all(r['Torque_Enable'] == 0 for r in self.follower.regs.values()))
        self.assertEqual(self.follower.regs['gripper']['Profile_Acceleration'], 11)

    def test_torque_off_ack_loss_is_resolved_by_actual_readback(self):
        disable = self.follower.disable_torque
        def lost_ack(num_retry=0):
            disable(num_retry)
            raise RuntimeError('torque off acknowledgement lost')
        self.follower.disable_torque = lost_ack
        self.run_control()
        self.assert_released()

    def test_original_stop_reason_survives_cleanup_failure(self):
        write, stream = self.follower.write_motion, self.follower.sync_write_goals

        def fail(register, name, value):
            if (register, name, value) == ('Profile_Velocity', 'gripper', 0):
                raise DynamixelCommunicationError(-3001, 'cleanup write lost')
            write(register, name, value)

        def fail_stream(goals):
            if self.clock.sleeps > 0:
                raise RuntimeError('original movement failure')
            stream(goals)

        self.follower.write_motion = fail
        self.follower.sync_write_goals = fail_stream
        with self.assertRaises(app.TeleopError) as failure:
            self.run_control()
        self.assertIn('original movement failure', str(failure.exception))
        self.assertIn('cleanup write lost', str(failure.exception))

    def test_cleanup_stops_retrying_if_torque_turns_on(self):
        write, attempts = self.follower.write_motion, []
        def fail(register, name, value):
            if (register, name, value) == ('Profile_Velocity', 'gripper', 0):
                attempts.append(value)
                self.follower.regs[name]['Torque_Enable'] = 1
                raise DynamixelCommunicationError(-3001, 'write lost')
            write(register, name, value)
        self.follower.write_motion = fail
        with self.assertRaisesRegex(app.TeleopError, '硬件断电'):
            self.run_control()
        self.assertEqual(len(attempts), 1)

    def test_cleanup_never_retries_motor_fault_or_motion_targets(self):
        self.follower.regs['gripper']['Profile_Velocity'] = 14
        write, attempts = self.follower.write_motion, []
        def motor_error(register, name, value):
            attempts.append(register)
            raise RuntimeError('motor hardware error')
        self.follower.write_motion = motor_error
        with self.assertRaisesRegex(RuntimeError, 'motor hardware error'):
            self.control.restore_config('gripper', 'Profile_Velocity', 0)
        self.assertEqual(attempts, ['Profile_Velocity'])
        for register in ('Goal_Position', 'Torque_Enable'):
            with self.assertRaises(ValueError):
                self.control.restore_config('gripper', register, 0)
        self.assertEqual(attempts, ['Profile_Velocity'])

    def test_watchdog_readback_failure_never_enables_torque(self):
        read = self.follower.read_motion
        def bad_watchdog(register, name):
            return 0 if register == 'Bus_Watchdog' else read(register, name)
        self.follower.read_motion = bad_watchdog
        with self.assertRaisesRegex(app.TeleopError, '看门狗配置回读'):
            self.run_control()
        self.assertFalse(any(call[0] == 'motion' and call[1] == 'Torque_Enable' and call[3]
                             for call in self.follower.calls))
        self.assert_released()

    def test_torque_enable_jump_rejected_and_released(self):
        self.follower.jump_on_enable = True
        with self.assertRaises(app.TeleopError):
            self.run_control()
        self.assert_released()

    def test_partial_enable_failure_releases_all_joints(self):
        self.follower.failure = lambda reg, name, value: reg == 'Torque_Enable' and value == 1 and name == 'elbow_flex'
        with self.assertRaises(RuntimeError):
            self.run_control()
        self.assert_released()

    def test_frame_jump_is_rejected_after_start(self):
        original = self.clock.sleep
        def jump(duration):
            original(duration)
            self.leader.raw['shoulder_pan'] += 4096
        self.clock.sleep = jump
        with self.assertRaises(app.TeleopError):
            self.run_control()
        self.assert_released()

    def test_slow_io_rejects_stale_target_and_releases(self):
        original = self.control.check_joint
        def slow(*args):
            original(*args)
            self.clock.now += .3
        self.control.check_joint = slow
        with self.assertRaises(app.TeleopError):
            self.run_control()
        self.assert_released()

    def test_missing_files_and_same_port_fail_before_connection(self):
        with patch.object(app, 'connect_bus') as connect, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(app.main(['--leader-calibration', str(self.root / 'none')]), 1)
            self.assertEqual(app.main(['--leader-port', 'COM5', '--follower-port', 'com5']), 1)
        connect.assert_not_called()

    def test_second_connection_failure_closes_first_port(self):
        with patch.object(app, 'load_calibration', return_value=self.leader_cal), \
                patch.object(app, 'connect_bus', side_effect=[self.leader, RuntimeError('not connected')]), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(app.main(['--check']), 1)
        self.assertFalse(self.leader.is_connected)


class MotionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.packet = MagicMock()
        self.packet.write1ByteTxRx.return_value = (0, 0)
        self.packet.write4ByteTxRx.return_value = (0, 0)
        self.packet.syncWriteTxOnly.return_value = 0
        self.packet.readTxRx.side_effect = lambda port, motor_id, address, width: (
            [0] if width == 1 else [0xF0, 0xFF, 0xFF, 0xFF], 0, 0)
        self.bus = KochTeleopBus('COM_FAKE', {'joint': (1, 1060)}, 'follower')
        self.bus._packet, self.bus._port, self.bus.is_connected = self.packet, MagicMock(), True

    def test_teleop_exhausted_read_retries_are_typed_as_recoverable(self):
        for reply in (([], -3001, 0), ([], -3002, 0), ([0], 0, 0)):
            self.packet.readTxRx.reset_mock()
            self.packet.readTxRx.side_effect = None
            self.packet.readTxRx.return_value = reply
            with patch('koch_teleop_bus.time.sleep'), self.assertRaises(FeedbackReadError):
                self.bus.read('Present_Position', 'joint')
            self.assertEqual(self.packet.readTxRx.call_count, 3)
        self.assertEqual(self.bus.read_failures['joint'], 3)

    def test_teleop_motor_error_and_tx_failure_are_not_recoverable_read_errors(self):
        for reply in (([], 0, 4), ([], 0, 0x83), ([], -1001, 0)):
            self.packet.readTxRx.reset_mock()
            self.packet.readTxRx.side_effect = None
            self.packet.readTxRx.return_value = reply
            with self.assertRaises(RuntimeError) as failure:
                self.bus.read('Present_Position', 'joint')
            self.assertNotIsInstance(failure.exception, FeedbackReadError)
            self.assertEqual(self.packet.readTxRx.call_count, 1)

    def test_fault_diagnostics_reads_alert_only_packet_without_writes(self):
        packet = bytearray(94)
        packet[4], packet[8] = 1, 0x80
        packet[9], packet[15] = 0, 0x20
        packet[89:91], packet[91] = bytes((119, 0)), 48
        self.packet.readTx.return_value = 0
        self.packet.rxPacket.return_value = (packet, 0)
        self.assertEqual(self.bus.read_fault_snapshot('joint'),
                         {'fault': 0x20, 'temperature': 48, 'voltage': 11.9, 'torque': 0})
        packet[8] = 0x84
        with self.assertRaisesRegex(RuntimeError, '应答无效'):
            self.bus.read_fault_snapshot('joint')
        self.packet.write1ByteTxRx.assert_not_called()
        self.packet.write4ByteTxRx.assert_not_called()

    def test_teleop_recovers_from_truncated_sdk_packet_and_crc_without_motion_write(self):
        self.packet.readTxRx.side_effect = [IndexError('short'), ([], 0, 3), ([0, 8, 0, 0], 0, 0)]
        with patch('koch_teleop_bus.time.sleep'):
            self.assertEqual(self.bus.read('Present_Position', 'joint'), 2048)
        self.packet.write1ByteTxRx.assert_not_called()
        self.packet.write4ByteTxRx.assert_not_called()

    def test_joint_status_uses_two_reads_and_decodes_signed_offset(self):
        values = {10: (1, 0), 11: (1, 4), 20: (4, -16),
                  64: (1, 1), 70: (1, 4), 98: (1, 50),
                  144: (2, 125), 146: (1, 54)}
        def read(port, motor_id, address, width):
            data = bytearray(width)
            for offset, (size, value) in values.items():
                if address <= offset and offset + size <= address + width:
                    data[offset-address:offset-address+size] = value.to_bytes(
                        size, 'little', signed=value < 0)
            return list(data), 0, 0
        self.packet.readTxRx.side_effect = read
        status = self.bus.read_joint_status('joint')
        self.assertEqual(status, dict(Drive_Mode=0, Operating_Mode=4, Homing_Offset=-16,
                         Torque_Enable=1, Hardware_Error_Status=4, Bus_Watchdog=50,
                         Present_Input_Voltage=125, Present_Temperature=54))
        self.assertEqual([call.args[2:] for call in self.packet.readTxRx.call_args_list],
                         [(10, 14), (64, 83)])

    def test_joint_status_rejects_truncated_packets(self):
        self.packet.readTxRx.return_value = ([0], 0, 0)
        self.packet.readTxRx.side_effect = None
        with self.assertRaisesRegex(RuntimeError, '长度'):
            self.bus.read_joint_status('joint')

    def test_goal_encoding_and_signed_readback(self):
        self.bus.write_motion('Goal_Position', 'joint', -16)
        self.assertEqual(self.packet.write4ByteTxRx.call_args.args[2:], (116, 0xFFFFFFF0))
        self.assertEqual(self.bus.read_motion('Goal_Position', 'joint'), -16)

    def test_sync_goals_are_sent_as_one_broadcast_packet(self):
        self.bus.motors['second'] = (2, 1060)
        self.bus.sync_write_goals({'joint': -16, 'second': 3026})
        args = self.packet.syncWriteTxOnly.call_args.args
        self.assertEqual(args[:3], (self.bus._port, 116, 4))
        self.assertEqual(args[3], bytes([1]) + (0xFFFFFFF0).to_bytes(4, 'little')
                         + bytes([2]) + (3026).to_bytes(4, 'little'))
        self.assertEqual(args[4], 10)
        self.packet.write4ByteTxRx.assert_not_called()
        self.packet.write1ByteTxRx.assert_not_called()

    def test_sync_goals_require_every_joint_valid_range_and_follower(self):
        self.bus.motors['second'] = (2, 1060)
        for goals in ({'joint': 2048}, {'joint': 9000, 'second': 2048},
                      {'joint': 2048, 'second': 2048, 'third': 2048}):
            with self.subTest(goals=goals), self.assertRaises(ValueError):
                self.bus.sync_write_goals(goals)
        self.bus.arm = 'leader'
        with self.assertRaises(ValueError):
            self.bus.sync_write_goals({'joint': 2048, 'second': 2048})
        self.packet.syncWriteTxOnly.assert_not_called()

    def test_sync_goals_send_failure_is_reported(self):
        self.packet.syncWriteTxOnly.return_value = -1001
        self.packet.getTxRxResult.return_value = 'Failed to write instruction packet'
        with self.assertRaisesRegex(DynamixelCommunicationError, '同步写入'):
            self.bus.sync_write_goals({'joint': 2048})

    def test_leader_motion_is_forbidden_and_calibration_guard_remains(self):
        self.bus.arm = 'leader'
        with self.assertRaises(ValueError):
            self.bus.write_motion('Torque_Enable', 'joint', 1)
        self.bus.arm = 'follower'
        with self.assertRaises(ValueError):
            self.bus.write('Torque_Enable', 'joint', 1)
        self.packet.write1ByteTxRx.assert_not_called()

    def test_zero_profile_only_restored_with_torque_off(self):
        self.packet.readTxRx.side_effect = None
        self.packet.readTxRx.return_value = ([1], 0, 0)
        with self.assertRaises(ValueError):
            self.bus.write_motion('Profile_Velocity', 'joint', 0)
        self.packet.readTxRx.return_value = ([0], 0, 0)
        self.bus.write_motion('Profile_Velocity', 'joint', 0)

    def test_lost_write_ack_is_accepted_only_with_matching_readback_and_no_fault(self):
        for register, value, address, width in (('Torque_Enable', 1, 64, 1),
                                                ('Goal_Position', -16, 116, 4),
                                                ('Profile_Velocity', 14, 112, 4)):
            for status in (-3001, -3002):
                with self.subTest(register=register, status=status):
                    write = getattr(self.packet, f'write{width}ByteTxRx')
                    write.reset_mock()
                    write.return_value = (status, 0)
                    encoded = value & ((1 << (8 * width)) - 1)
                    self.packet.readTxRx.side_effect = lambda p, i, a, w: (
                        list((encoded if a == address else 0).to_bytes(w, 'little')), 0, 0)
                    self.bus.write_motion(register, 'joint', value)
                    write.assert_called_once()

    def test_lost_ack_with_wrong_readback_never_resends_motion(self):
        for register, value, width in (('Torque_Enable', 1, 1), ('Goal_Position', 2048, 4)):
            write = getattr(self.packet, f'write{width}ByteTxRx')
            write.reset_mock()
            write.return_value = (-3001, 0)
            self.packet.readTxRx.side_effect = lambda p, i, a, w: ([0] * w, 0, 0)
            with self.subTest(register=register), self.assertRaisesRegex(DynamixelCommunicationError, '回读'):
                self.bus.write_motion(register, 'joint', value)
            write.assert_called_once()

    def test_matching_readback_does_not_hide_motor_fault(self):
        self.packet.write1ByteTxRx.return_value = (-3001, 0)
        self.packet.readTxRx.side_effect = lambda p, i, a, w: ([4 if a == 70 else 1], 0, 0)
        with self.assertRaisesRegex(RuntimeError, '故障'):
            self.bus.write_motion('Torque_Enable', 'joint', 1)
        self.packet.write1ByteTxRx.assert_called_once()

    def test_torque_crc_retries_only_after_healthy_readback_shows_unapplied(self):
        self.packet.getRxPacketError.return_value = "CRC doesn't match!"
        control = app.Teleoperator(None, self.bus, None, calibration())
        for readback, attempts in ((0, 2), (1, 1)):
            with self.subTest(readback=readback):
                self.packet.write1ByteTxRx.reset_mock()
                self.packet.readTxRx.reset_mock()
                self.packet.write1ByteTxRx.side_effect = [(0, 3), (0, 0)]
                self.packet.readTxRx.side_effect = lambda p, i, a, w: ([readback if a == 64 else 0], 0, 0)
                control.confirmed_write('Torque_Enable', 'joint', 1)
                self.assertEqual(self.packet.write1ByteTxRx.call_count, attempts)
                self.assertEqual([call.args[2] for call in self.packet.readTxRx.call_args_list], [64, 70])

    def test_torque_crc_preserves_fault_stop_and_retry_limit(self):
        control = app.Teleoperator(None, self.bus, None, calibration())
        self.packet.getRxPacketError.return_value = 'motor reported error'
        for error, status, attempts in (
                (0x83, ([0], 0, 0), 1),
                (3, ([], -3001, 0), 1),
                (3, ([4], 0, 0), 1),
                (3, ([0], 0, 0), 3)):
            with self.subTest(error=error, status=status):
                self.packet.write1ByteTxRx.reset_mock()
                self.packet.write1ByteTxRx.return_value = (0, error)
                self.packet.readTxRx.side_effect = lambda p, i, a, w: ([0], 0, 0) if a == 64 else status
                with self.assertRaises(RuntimeError):
                    control.confirmed_write('Torque_Enable', 'joint', 1)
                self.assertEqual(self.packet.write1ByteTxRx.call_count, attempts)

    def test_explicit_write_error_never_uses_readback_to_claim_success(self):
        for comm, error in ((-1001, 0), (0, 4)):
            self.packet.readTxRx.reset_mock()
            self.packet.write1ByteTxRx.return_value = (comm, error)
            with self.subTest(comm=comm, error=error), self.assertRaises(RuntimeError):
                self.bus.write_motion('Torque_Enable', 'joint', 1)
            self.packet.readTxRx.assert_not_called()



class SyncGoalPacketTests(unittest.TestCase):
    def test_broadcast_sync_write_packet_is_well_formed(self):
        from koch_calibration_bus import ValidatedProtocol2PacketHandler

        sent = []

        class Port:
            is_using = False

            def clearPort(self):
                pass

            def writePort(self, packet):
                sent.append(list(packet))
                return len(packet)

        bus = KochTeleopBus('COM_FAKE', {'wrist_flex': (4, 1200), 'wrist_roll': (5, 1200)}, 'follower')
        handler = ValidatedProtocol2PacketHandler()
        bus._packet, bus._port, bus.is_connected = handler, Port(), True
        bus.sync_write_goals({'wrist_flex': 2048, 'wrist_roll': -16})

        packet = sent[0]
        total = (packet[5] | (packet[6] << 8)) + 7
        self.assertEqual(packet[:5], [0xFF, 0xFF, 0xFD, 0x00, 0xFE])
        self.assertEqual(packet[7], 0x83)  # INST_SYNC_WRITE
        self.assertEqual(packet[8:12], [116, 0, 4, 0])
        self.assertEqual(bytes(packet[12:22]), bytes([4]) + (2048).to_bytes(4, 'little')
                         + bytes([5]) + (0xFFFFFFF0).to_bytes(4, 'little'))
        self.assertEqual(len(packet), total)
        self.assertEqual(packet[-2:], list(handler.updateCRC(0, packet, total - 2).to_bytes(2, 'little')))


if __name__ == '__main__':
    unittest.main()
