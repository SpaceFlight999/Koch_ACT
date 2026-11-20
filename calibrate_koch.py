"""Koch 独立校准：默认按视频零位/旋转位/休息位流程，全程不启用扭矩。"""

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from recover_leader_calibration import fit_single_turn

ROOT = Path(__file__).resolve().parent
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LABELS = ("底座旋转", "肩关节", "肘关节", "手腕俯仰", "手腕旋转", "夹爪")
FULL_TURN = {"shoulder_pan", "wrist_roll"}
# 用户指定的 XL430 软件放行上限为 13V；不是厂商额定值，也不改写电机报警阈值。
SOFTWARE_VOLTAGE_LIMITS = {1060: (6.5, 13.0), 1190: (3.7, 6.0), 1200: (3.7, 6.0)}
SPECS = {
    "follower": {"label": "从臂", "models": (1060, 1060, 1200, 1200, 1200, 1200),
                 "types": ("xl430-w250",) * 2 + ("xl330-m288",) * 4},
    "leader": {"label": "主臂", "models": (1190,) * 6, "types": ("xl330-m077",) * 6},
}
REGISTERS = ("Model_Number", "Present_Position", "Present_Input_Voltage", "Present_Temperature",
             "Hardware_Error_Status", "Torque_Enable", "Operating_Mode", "Drive_Mode",
             "Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")


class CalibrationError(RuntimeError):
    pass


def make_bus(arm, port):
    from koch_calibration_bus import KochCalibrationBus

    return KochCalibrationBus(port, {
        name: (index, model)
        for index, (name, model) in enumerate(zip(NAMES, SPECS[arm]["models"]), 1)
    })


def connect_bus(arm, port=None, *, factory=make_bus, ports=None):
    """只用握手识别型号，不写寄存器；多台同型机械臂时拒绝猜选。"""
    if ports is None and port is None:
        from serial.tools.list_ports import comports
        ports = [item.device for item in comports()]
    candidates = [port] if port else list(ports or [])
    matches, errors = [], []
    try:
        for candidate in candidates:
            bus = factory(arm, candidate)
            try:
                bus.connect()
                matches.append(bus)
            except BaseException as exc:
                bus.disconnect(disable_torque=False)
                if not isinstance(exc, Exception):
                    raise
                errors.append(f"{candidate}: {exc}")
        if len(matches) != 1:
            if matches:
                raise CalibrationError("找到多只同类型机械臂，请用 --port 明确指定："
                                       + ", ".join(bus.port for bus in matches))
            raise CalibrationError("未找到符合型号和 ID 1–6 配置的" + SPECS[arm]["label"]
                                   + "。请检查 USB、外部供电，并退出其他串口程序。\n"
                                   + "\n".join(errors))
        return matches[0]
    except BaseException:
        for bus in matches:
            bus.disconnect(disable_torque=False)
        raise


def snapshot(bus):
    return {name: {reg: int(bus.read(reg, name, normalize=False)) for reg in REGISTERS}
            for name in NAMES}


def validate_health(states, arm):
    if set(states) != set(NAMES):
        raise CalibrationError("必须完整读取 6 个关节，不能使用部分数据校准。")
    for index, name in enumerate(NAMES):
        state = states[name]
        if state["Model_Number"] != SPECS[arm]["models"][index]:
            raise CalibrationError(f"{name} 型号不匹配，请核实主臂/从臂和端口。")
        low, high = SOFTWARE_VOLTAGE_LIMITS[state["Model_Number"]]
        voltage = state["Present_Input_Voltage"] / 10
        if not low <= voltage <= high:
            raise CalibrationError(f"{LABELS[index]}电压读数 {voltage:.1f}V，软件允许范围 {low}–{high}V；"
                                   "请先核实供电，程序不能调整外部电源。")
        if state["Hardware_Error_Status"] or state["Present_Temperature"] >= 55:
            raise CalibrationError(f"{LABELS[index]}故障码或温度异常，停止校准。")
        if state["Torque_Enable"] != 0:
            raise CalibrationError("检测到已启用的扭矩。请托稳机械臂，并在原控制程序中释放关节后重试。")
        if abs(state["Present_Position"]) > 8191:
            raise CalibrationError(f"{LABELS[index]}位置读数异常：{state['Present_Position']}。"
                                   "请检查电机/通信，不要取低 12 位掩盖异常。")


def stable_positions(bus, *, sleep=time.sleep):
    samples = [bus.sync_read("Present_Position", normalize=False)]
    for _ in range(2):
        sleep(0.05)
        samples.append(bus.sync_read("Present_Position", normalize=False))
    for name in NAMES:
        values = [int(sample[name]) for sample in samples]
        if max(values) - min(values) > 24 or max(abs(value) for value in values) > 8191:
            raise CalibrationError(f"{name} 静止位置不稳定：{values}。请托稳；若未移动仍异常，先排查硬件。")
    return {name: int(samples[-1][name]) for name in NAMES}


def write_json(path, data):
    """同目录临时文件 + 原子替换，不留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def require_torque_off(bus):
    for name in NAMES:
        if bus.read("Torque_Enable", name, normalize=False) != 0:
            raise CalibrationError(f"{name} 无法确认扭矩关闭；请托稳并使用硬件断电。")


def home_at_middle(bus):
    """用小幅偏移探针确认零点偏移的实际符号，避免反向肘关节校准到错误坐标。"""
    bus.reset_calibration()
    stable_positions(bus)
    offsets, signs = {}, {}
    for name in NAMES:
        before = int(bus.read("Present_Position", name, normalize=False))
        bus.write("Homing_Offset", name, 16, normalize=False)
        after = int(bus.read("Present_Position", name, normalize=False))
        delta = after - before
        if abs(abs(delta) - 16) > 3:
            raise CalibrationError(f"{name} 零点偏移响应异常（{delta}），请保持静止或排查电机。")
        sign = 1 if delta > 0 else -1
        signs[name] = sign
        offset = sign * (2047 - before)
        if not -1_044_479 <= offset <= 1_044_479:
            raise CalibrationError(f"{name} 零点偏移超出寄存器范围。")
        bus.write("Homing_Offset", name, offset, normalize=False)
        offsets[name] = offset
    middle = stable_positions(bus)
    if any(abs(position - 2047) > 24 for position in middle.values()):
        raise CalibrationError("中点回读不匹配，请保持各关节静止后重新校准。")
    return offsets, middle, signs


def fit_capture(raw, arm, middle, signs):
    if set(raw) != set(NAMES):
        raise CalibrationError("校准采集不完整。")
    fitted = {}
    for index, name in enumerate(NAMES, 1):
        cal = raw[name]
        expected_drive = int(arm == "leader" and name == "elbow_flex")
        if cal["id"] != index or cal["drive_mode"] != expected_drive:
            raise CalibrationError(f"{name} 的 ID 或驱动方向错误。")
        if not cal["range_min"] <= middle[name] <= cal["range_max"]:
            raise CalibrationError(f"{name} 行程没有覆盖已采集的中点。")
        if name not in FULL_TURN and cal["range_max"] - cal["range_min"] < 32:
            raise CalibrationError(f"{name} 有效活动不足约 2.8°，请实际摆动该关节后重试。")
        # 复用现有、已测试的坐标平移；不裁剪范围，超过一圈直接拒绝。
        fitted[name] = fit_single_turn(cal)
        # 以本轮探针实测的符号为准，兼容不同固件对反向模式的零点处理。
        shift = fitted[name]["range_min"] - cal["range_min"]
        if signs[name] not in (-1, 1):
            raise CalibrationError(f"{name} 缺少有效零点探针结果。")
        fitted[name]["homing_offset"] = cal["homing_offset"] + signs[name] * shift
        if not -1_044_479 <= fitted[name]["homing_offset"] <= 1_044_479:
            raise CalibrationError(f"{name} 平移后的零点超出硬件范围。")
    return fitted


def calibrate_bus(bus, arm, target, *, ask=input, record=None):
    """连接由调用方管理。只有全部验证通过才替换最终文件；失败留存现场。"""
    from lerobot.motors import MotorCalibration

    target = Path(target)
    states = snapshot(bus)
    validate_health(states, arm)
    print("\n请支撑好机械臂，保持静止。全程不启用扭矩，也不发送运动目标。")
    stable_positions(bus)
    session = target.parent / "history" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    write_json(session / "before_registers.json", states)
    if target.exists():
        shutil.copy2(target, session / "previous_calibration.json")
    print(f"原始寄存器及已有文件已备份：{session}")
    ask("按回车开始重新校准（会修改零点、驱动方向和行程配置）；Ctrl+C 取消：")
    modified = False
    try:
        modified = True
        bus.disable_torque()
        require_torque_off(bus)
        for name in NAMES:
            bus.write("Drive_Mode", name, int(arm == "leader" and name == "elbow_flex"), normalize=False)
        bus.set_operating_mode("extended_position")
        ask("\n把全部关节摆到各自行程的中间位置（夹爪半开），托稳后按回车：")
        validate_health(snapshot(bus), arm)
        offsets, middle, signs = home_at_middle(bus)
        write_json(session / "middle.json", {"positions": middle, "homing_offsets": offsets,
                                            "homing_signs": signs})
        print("\n接下来缓慢摆动：肩关节、肘关节、手腕俯仰、夹爪，分别覆盖可用行程。")
        print("底座和手腕旋转采用官方完整编码器范围，无需强行转满一圈。")
        print("不要强扳限位。全部摆动完成后，按回车结束采集；Ctrl+C 可随时取消。", flush=True)
        measured = [name for name in NAMES if name not in FULL_TURN]
        recorder = record if record is not None else bus.record_ranges_of_motion
        mins, maxes = recorder(measured)
        raw = {name: {"id": index, "drive_mode": int(arm == "leader" and name == "elbow_flex"),
                      "homing_offset": offsets[name],
                      "range_min": 0 if name in FULL_TURN else min(int(mins[name]), middle[name]),
                      "range_max": 4095 if name in FULL_TURN else max(int(maxes[name]), middle[name])}
               for index, name in enumerate(NAMES, 1)}
        write_json(session / "raw_capture.json", raw)
        fitted = fit_capture(raw, arm, middle, signs)
        write_json(session / "proposed_calibration.json", fitted)
        print("采集已结束，请保持机械臂静止，正在写入并回读校准。", flush=True)
        validate_health(snapshot(bus), arm)
        before_write = stable_positions(bus)
        calibration = {name: MotorCalibration(**cal) for name, cal in fitted.items()}
        bus.write_calibration(calibration)
        actual = {name: asdict(cal) for name, cal in bus.read_calibration().items()}
        if actual != fitted:
            raise CalibrationError("电机配置回读与拟保存结果不一致，不保存最终文件。")
        after_write = stable_positions(bus)
        for name in NAMES:
            expected = before_write[name] + signs[name] * (fitted[name]["homing_offset"] - offsets[name])
            if abs(after_write[name] - expected) > 24:
                raise CalibrationError(f"{name} 写入后的坐标变化与实测符号不符，请保持静止并排查电机。")
        require_torque_off(bus)
        bus.disable_torque(num_retry=2)
        require_torque_off(bus)
        modified = False
        write_json(session / "result.json", {"status": "verified", "path": str(target), "port": bus.port})
        # 最终文件是提交点；之后不再执行可能失败的电机写入或日志写入。
        write_json(target, fitted)
        return target
    except BaseException as exc:
        write_json(session / "result.json", {"status": "failed", "error": str(exc), "port": bus.port})
        print(f"\n校准未完成，现场记录保存在：{session}")
        print("最终文件未更新。电机配置可能已改变，旧文件不能直接当作当前有效校准。")
        raise
    finally:
        if modified:
            bus.disable_torque(num_retry=2)
            require_torque_off(bus)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=SPECS, help="follower=从臂，leader=主臂；不填则显示菜单")
    parser.add_argument("--port", help="可选，例如 COM5；不填则按 6 个电机型号自动识别")
    parser.add_argument("--id", help="校准名称，默认 my_follower / my_leader")
    parser.add_argument("--check", action="store_true", help="只读检查，不修改电机或校准文件")
    parser.add_argument("--method", choices=("legacy", "modern"), default="legacy",
                        help="legacy=视频三姿态（默认）；modern=原中点/行程流程")
    args = parser.parse_args(argv)
    bus = None
    try:
        arm = args.arm
        if arm is None:
            choice = input("选择要独立校准的机械臂：1=从臂，2=主臂（回车退出）：").strip()
            if not choice:
                return 0
            arm = {"1": "follower", "2": "leader"}.get(choice)
            if arm is None:
                raise CalibrationError("请输入 1 或 2。")
        identity = args.id or f"my_{arm}"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity):
            raise CalibrationError("--id 只能包含英文字母、数字、下划线或短横线，长度 1–64。")
        directory = f"koch_legacy/{arm}" if args.method == "legacy" else f"koch_{arm}"
        target = ROOT / "calibration" / directory / f"{identity}.json"
        print(f"\n{SPECS[arm]['label']}校准 | 1,000,000 bps | 不需要连接另一只机械臂", flush=True)
        print("流程：" + ("视频三姿态 / 软件方向和偏移" if args.method == "legacy"
                            else "中点及活动范围 / 硬件校准"))
        bus = connect_bus(arm, args.port)
        print(f"已识别 {bus.port}，ID 1–6 型号匹配。")
        states = snapshot(bus)
        for name, label in zip(NAMES, LABELS):
            state = states[name]
            print(f"{label:6}  位置 {state['Present_Position']:9d}  "
                  f"电压 {state['Present_Input_Voltage'] / 10:.1f}V  "
                  f"扭矩 {state['Torque_Enable']}  故障 {state['Hardware_Error_Status']}")
        validate_health(states, arm)
        if args.check:
            stable_positions(bus)
            print(f"只读检查通过，未修改电机。预定校准文件：{target}")
        else:
            if args.method == "legacy":
                from koch_legacy_calibration import calibrate_legacy
                path = calibrate_legacy(bus, arm, target)
                print(f"\n三姿态校准完成：软件零点和方向已验证，扭矩关闭。\n文件：{path}")
                print("此文件用于旧版软件换算，不能传给新版 MotorCalibration 或关节测试工具。")
            else:
                path = calibrate_bus(bus, arm, target)
                print(f"\n校准完成：6 个关节配置回读一致，扭矩关闭。\n文件：{path}")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\n已取消，未完成校准。")
        return 130
    except Exception as exc:
        print(f"\n停止：{exc}", file=sys.stderr)
        return 1
    finally:
        if bus is not None:
            bus.disconnect(disable_torque=False)
            print("串口已释放。")


if __name__ == "__main__":
    raise SystemExit(main())
