"""校准回归测试：使用模拟总线，不连接、不驱动真实电机。"""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lerobot.motors import MotorCalibration

import calibrate_koch as app


class FakeBus:
    def __init__(self, arm="follower", port="COM_TEST"):
        self.port, self.arm = port, arm
        self.is_connected = True
        self.calls = []
        self.mismatch = False
        self.stationary_fault = False
        self.positive_reverse = False
        self.raw = {name: 1800 + index * 80 for index, name in enumerate(app.NAMES)}
        self.regs = {}
        for index, name in enumerate(app.NAMES):
            self.regs[name] = dict(zip(app.REGISTERS, (
                app.SPECS[arm]["models"][index], self.raw[name],
                115 if app.SPECS[arm]["models"][index] == 1060 else 50,
                25, 0, 0, 4, 0, 0, 0, 4095,
            )))

    def sign(self, name):
        return -1 if self.regs[name]["Drive_Mode"] == 1 and not self.positive_reverse else 1

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque=True):
        self.calls.append(("disconnect", disable_torque))
        self.is_connected = False

    def read(self, register, name, normalize=False):
        if register == "Present_Position":
            return self.raw[name] + self.sign(name) * self.regs[name]["Homing_Offset"]
        return self.regs[name][register]

    def sync_read(self, register, normalize=False):
        if self.stationary_fault:
            self.raw["shoulder_pan"] += 4096
        return {name: self.read(register, name) for name in app.NAMES}

    def write(self, register, name, value, normalize=False):
        if register.startswith("Goal_") or (register == "Torque_Enable" and value):
            raise AssertionError("校准不得发出运动目标或启用扭矩")
        self.calls.append(("write", register, name, value))
        self.regs[name][register] = value

    def disable_torque(self, num_retry=0):
        self.calls.append(("disable_torque", num_retry))
        for name in app.NAMES:
            self.regs[name]["Torque_Enable"] = 0

    def set_operating_mode(self, mode):
        if mode != "extended_position":
            raise AssertionError(mode)
        for name in app.NAMES:
            self.write("Operating_Mode", name, 4)

    def reset_calibration(self):
        for name in app.NAMES:
            self.write("Homing_Offset", name, 0)
            self.write("Min_Position_Limit", name, 0)
            self.write("Max_Position_Limit", name, 4095)

    def record_ranges_of_motion(self, names):
        if set(names) != set(app.NAMES) - app.FULL_TURN:
            raise AssertionError(names)
        return ({name: 1700 for name in names}, {name: 2600 for name in names})

    def write_calibration(self, calibration):
        self.calls.append(("commit",))
        for name, cal in calibration.items():
            self.write("Homing_Offset", name, cal.homing_offset)
            self.write("Min_Position_Limit", name, cal.range_min)
            self.write("Max_Position_Limit", name, cal.range_max)

    def read_calibration(self):
        result = {name: MotorCalibration(
            id=index, drive_mode=self.regs[name]["Drive_Mode"],
            homing_offset=self.regs[name]["Homing_Offset"],
            range_min=self.regs[name]["Min_Position_Limit"],
            range_max=self.regs[name]["Max_Position_Limit"],
        ) for index, name in enumerate(app.NAMES, 1)}
        if self.mismatch:
            result["elbow_flex"].range_max -= 1
        return result


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parent
        self.directory = tempfile.TemporaryDirectory(prefix=".calibration_test_", dir=workspace)
        if not Path(self.directory.name).resolve().is_relative_to(workspace):
            raise AssertionError("临时测试目录必须位于当前项目内")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "my_follower.json"
        self.console = contextlib.redirect_stdout(io.StringIO())
        self.console.__enter__()
        self.addCleanup(self.console.__exit__, None, None, None)
        self.sleeper = patch.object(app.time, "sleep")
        self.sleeper.start()
        self.addCleanup(self.sleeper.stop)

    def calibrate(self, bus=None, **kwargs):
        bus = bus or FakeBus()
        return app.calibrate_bus(bus, bus.arm, self.output, ask=lambda _: "", **kwargs)

    def test_both_arms_round_trip_without_motion_commands(self):
        for arm in app.SPECS:
            with self.subTest(arm=arm):
                bus = FakeBus(arm)
                self.calibrate(bus)
                result = json.loads(self.output.read_text(encoding="utf-8"))
                self.assertEqual(len(result), 6)
                self.assertEqual(result["elbow_flex"]["drive_mode"], int(arm == "leader"))
                self.assertTrue(all(state["Torque_Enable"] == 0 for state in bus.regs.values()))
                self.assertEqual(bus.calls[-1], ("disable_torque", 2))
                self.assertFalse(any(call[0] == "write" and call[1].startswith("Goal_")
                                     for call in bus.calls))

    def test_homing_measures_reverse_sign_instead_of_guessing(self):
        for positive in (False, True):
            with self.subTest(positive=positive):
                bus = FakeBus("leader")
                bus.positive_reverse = positive
                bus.write("Drive_Mode", "elbow_flex", 1)
                offsets, middle, signs = app.home_at_middle(bus)
                self.assertEqual(signs["elbow_flex"], 1 if positive else -1)
                self.assertEqual(set(middle.values()), {2047})
                self.assertNotEqual(offsets["elbow_flex"], 0)

    def test_unsupported_probe_response_stops(self):
        bus = FakeBus()
        original = bus.write

        def ignore_offset(register, name, value, normalize=False):
            if register != "Homing_Offset":
                original(register, name, value, normalize)

        bus.write = ignore_offset
        with self.assertRaises(app.CalibrationError):
            self.calibrate(bus)
        self.assertFalse(self.output.exists())

    def test_existing_file_backed_up_and_replaced_only_on_success(self):
        self.output.write_text('"old calibration"', encoding="utf-8")
        self.calibrate()
        backup, = self.root.glob("history/*/previous_calibration.json")
        self.assertEqual(backup.read_text(encoding="utf-8"), '"old calibration"')
        self.assertIsInstance(json.loads(self.output.read_text(encoding="utf-8")), dict)

    def test_readback_mismatch_preserves_old_file_and_closes_torque(self):
        self.output.write_text('"old"', encoding="utf-8")
        bus = FakeBus()
        bus.mismatch = True
        with self.assertRaises(app.CalibrationError):
            self.calibrate(bus)
        self.assertEqual(self.output.read_text(encoding="utf-8"), '"old"')
        result, = self.root.glob("history/*/result.json")
        self.assertEqual(json.loads(result.read_text(encoding="utf-8"))["status"], "failed")
        self.assertEqual(bus.calls[-1], ("disable_torque", 2))

    def test_bad_capture_never_commits_register_limits(self):
        bus = FakeBus()

        def bad_record(names):
            return ({name: 100 for name in names}, {name: 9000 for name in names})

        with self.assertRaises(ValueError):
            self.calibrate(bus, record=bad_record)
        self.assertNotIn(("commit",), bus.calls)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list(self.root.glob("history/*/raw_capture.json"))), 1)

    def test_unmoved_joint_is_not_valid_calibration(self):
        def still(names):
            return ({name: 2047 for name in names}, {name: 2047 for name in names})
        with self.assertRaises(app.CalibrationError):
            self.calibrate(record=still)
        self.assertFalse(self.output.exists())

    def test_cancel_before_start_makes_no_register_writes(self):
        bus = FakeBus()
        with self.assertRaises(KeyboardInterrupt):
            app.calibrate_bus(bus, "follower", self.output,
                              ask=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(bus.calls, [])

    def test_cancel_during_capture_leaves_torque_off(self):
        bus = FakeBus()
        def cancel(names):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.calibrate(bus, record=cancel)
        self.assertEqual(bus.calls[-1], ("disable_torque", 2))
        self.assertFalse(self.output.exists())

    def test_preflight_faults_stop_before_writing(self):
        for register, value in (("Torque_Enable", 1), ("Hardware_Error_Status", 4),
                                ("Present_Temperature", 55), ("Present_Input_Voltage", 131),
                                ("Model_Number", 1190), ("Present_Position", 54_324_621)):
            with self.subTest(register=register):
                bus = FakeBus()
                states = app.snapshot(bus)
                states["shoulder_pan"][register] = value
                with self.assertRaises(app.CalibrationError):
                    app.validate_health(states, "follower")
                self.assertEqual(bus.calls, [])

    def test_four_xl330_motors_retain_six_volt_limit(self):
        states = app.snapshot(FakeBus())
        states["gripper"]["Present_Input_Voltage"] = 61
        with self.assertRaises(app.CalibrationError):
            app.validate_health(states, "follower")

    def test_model_specific_software_voltage_boundaries(self):
        for arm, name, low, high in (('follower', 'shoulder_pan', 65, 130),
                                     ('follower', 'elbow_flex', 37, 60),
                                     ('leader', 'shoulder_pan', 37, 60)):
            states = app.snapshot(FakeBus(arm))
            for reading in (low, high):
                with self.subTest(arm=arm, joint=name, reading=reading):
                    states[name]['Present_Input_Voltage'] = reading
                    app.validate_health(states, arm)
            for reading in (low - 1, high + 1):
                with self.subTest(arm=arm, joint=name, reading=reading), self.assertRaises(app.CalibrationError):
                    states[name]['Present_Input_Voltage'] = reading
                    app.validate_health(states, arm)

    def test_stationary_full_turn_jumps_rejected(self):
        bus = FakeBus()
        bus.stationary_fault = True
        with self.assertRaises(app.CalibrationError):
            app.stable_positions(bus, sleep=lambda _: None)

    def test_full_cycle_works_with_firmware_using_positive_reverse_offset(self):
        bus = FakeBus("leader")
        bus.positive_reverse = True
        self.calibrate(bus)
        self.assertTrue(self.output.exists())

    def test_fit_translates_out_of_range_without_clipping_with_measured_sign(self):
        raw = {name: dict(id=index, drive_mode=int(name == "elbow_flex"), homing_offset=100,
                         range_min=0, range_max=4095)
               for index, name in enumerate(app.NAMES, 1)}
        raw["elbow_flex"].update(range_min=1000, range_max=4500)
        old = copy.deepcopy(raw)
        for sign in (1, -1):
            signs = dict.fromkeys(app.NAMES, 1)
            signs["elbow_flex"] = sign
            fitted = app.fit_capture(raw, "leader", dict.fromkeys(app.NAMES, 2047), signs)
            cal = fitted["elbow_flex"]
            shift = cal["range_min"] - 1000
            self.assertEqual(cal["range_max"] - cal["range_min"], 3500)
            self.assertEqual(cal["homing_offset"], 100 + sign * shift)
        self.assertEqual(raw, old)

    def test_atomic_file_write_does_not_leave_temporary_file(self):
        app.write_json(self.output, {"test": "中文"})
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), {"test": "中文"})
        self.assertFalse(self.output.with_name(self.output.name + ".tmp").exists())

    def test_check_mode_releases_port_and_writes_nothing(self):
        bus = FakeBus()
        with patch.object(app, "ROOT", self.root), patch.object(app, "connect_bus", return_value=bus):
            self.assertEqual(app.main(["--arm", "follower", "--check"]), 0)
        self.assertEqual(bus.calls, [("disconnect", False)])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_error_mode_still_releases_port(self):
        bus = FakeBus()
        bus.regs["shoulder_pan"]["Hardware_Error_Status"] = 4
        with patch.object(app, "connect_bus", return_value=bus), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(app.main(["--arm", "follower", "--check"]), 1)
        self.assertFalse(bus.is_connected)
        self.assertEqual(bus.calls, [("disconnect", False)])

    def test_invalid_identity_rejected_before_serial_access(self):
        with patch.object(app, "connect_bus") as connect, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(app.main(["--arm", "follower", "--id", "../bad"]), 1)
        connect.assert_not_called()


class DiscoveryTests(unittest.TestCase):
    def test_single_matching_arm_is_kept_connected(self):
        bus = FakeBus()
        result = app.connect_bus("follower", factory=lambda *_: bus, ports=["COM5"])
        self.assertIs(result, bus)
        self.assertEqual(bus.calls, [])

    def test_multiple_matching_arms_are_closed_without_choosing(self):
        buses = {port: FakeBus(port=port) for port in ("COM5", "COM7")}
        with self.assertRaises(app.CalibrationError):
            app.connect_bus("follower", factory=lambda arm, port: buses[port], ports=buses)
        self.assertTrue(all(not bus.is_connected for bus in buses.values()))
        self.assertTrue(all(bus.calls == [("disconnect", False)] for bus in buses.values()))

    def test_failed_and_interrupted_connections_release_their_port(self):
        for error in (RuntimeError("wrong model"), KeyboardInterrupt()):
            with self.subTest(error=error):
                bus = FakeBus()
                def fail():
                    raise error
                bus.connect = fail
                with self.assertRaises((app.CalibrationError, KeyboardInterrupt)):
                    app.connect_bus("follower", factory=lambda *_: bus, ports=["COM5"])
                self.assertFalse(bus.is_connected)


if __name__ == "__main__":
    unittest.main()
