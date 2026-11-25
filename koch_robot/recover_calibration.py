"""恢复本次 COM6 校准：保留已知范围，只重采未落盘的手腕俯仰和夹爪。"""

import json
from datetime import datetime

from koch_robot.paths import ROOT


DIRECTORY = ROOT / "calibration/koch_leader"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def fit_single_turn(calibration):
    """平移数值坐标以适配限制寄存器，不截断行程、不改变归一化比例。"""
    cal = dict(calibration)
    low, high = cal["range_min"], cal["range_max"]
    span = high - low
    if not 0 < span <= 4095:
        raise ValueError(f"无效或超过一圈的行程：{low}–{high}；需要重新采集。")
    if cal["drive_mode"] not in (0, 1):
        raise ValueError("只支持本次校准使用的普通/反向驱动模式。")
    if 0 <= low < high <= 4095:
        return cal
    shift = (4095 - span) // 2 - low
    # 在该主臂上用 +16 偏移探针实测：反向模式报告位置减少约 16。
    sign = -1 if cal["drive_mode"] == 1 else 1
    cal["homing_offset"] += sign * shift
    if not -1_044_479 <= cal["homing_offset"] <= 1_044_479:
        raise ValueError("新的零点偏移超出硬件允许范围。")
    cal["range_min"] += shift
    cal["range_max"] += shift
    return cal


def recover_known_ranges(registers):
    result = {}
    for motor_id, name in enumerate(NAMES, 1):
        saved = registers[name]
        result[name] = dict(id=motor_id, drive_mode=saved["Drive_Mode"],
                            homing_offset=saved["Homing_Offset"],
                            range_min=saved["Min_Position_Limit"],
                            range_max=saved["Max_Position_Limit"])
    # 原调用顺序先写 base、shoulder，再到 elbow；错误信息明确拒绝写入 4560。
    # elbow 的最小值 3880 已写入并由 interrupted_registers.json 回读保留。
    if result["elbow_flex"]["range_min"] != 3880:
        raise ValueError("本次故障备份与已知错误不一致，不能自动恢复。")
    result["elbow_flex"]["range_max"] = 4560
    return result


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    from lerobot.motors import MotorCalibration
    from lerobot.teleoperators.koch_leader.config_koch_leader import KochLeaderConfig
    from lerobot.teleoperators.koch_leader.koch_leader import KochLeader

    registers = json.loads((DIRECTORY / "interrupted_registers.json").read_text(encoding="utf-8"))
    raw = recover_known_ranges(registers)
    leader = KochLeader(KochLeaderConfig(port="COM6", id="my_leader", calibration_dir=DIRECTORY))
    if leader.calibration_fpath.exists():
        raise RuntimeError("主臂校准文件已存在，不覆盖已有结果。")
    try:
        leader.bus.connect()
        for name in NAMES:
            voltage = leader.bus.read("Present_Input_Voltage", name) / 10
            error = leader.bus.read("Hardware_Error_Status", name)
            torque = leader.bus.read("Torque_Enable", name)
            if not 3.7 <= voltage <= 6.0 or error or torque:
                raise RuntimeError(f"{name}: 电压/故障/扭矩检查未通过，停止恢复。")
            for register in ("Homing_Offset", "Drive_Mode", "Operating_Mode"):
                if leader.bus.read(register, name, normalize=False) != registers[name][register]:
                    raise RuntimeError(f"{name}: {register} 已改变，不能复用旧行程。")
        print("COM6 verified; all torque OFF. Recording wrist_flex and gripper ONLY.", flush=True)
        print("Move the leader wrist up/down and open/close gripper through safe ranges; press ENTER when done.", flush=True)
        mins, maxes = leader.bus.record_ranges_of_motion(["wrist_flex", "gripper"], display_values=False)
        for name in ("wrist_flex", "gripper"):
            raw[name]["range_min"] = int(mins[name])
            raw[name]["range_max"] = int(maxes[name])

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = DIRECTORY / f"raw_capture_{stamp}.json"
        save_json(raw_path, dict(
            calibration=raw,
            source="base/shoulder readback; elbow min readback + rejected max 4560; wrist/gripper recaptured",
        ))
        print(f"RAW CAPTURE BACKUP: {raw_path}", flush=True)
        fitted = {name: fit_single_turn(cal) for name, cal in raw.items()}
        proposal = DIRECTORY / f"proposed_calibration_{stamp}.json"
        save_json(proposal, fitted)
        print("RANGES AFTER COORDINATE TRANSLATION:", json.dumps(fitted, indent=2), flush=True)

        # 在写入前先完整验证所有关节，禁止边检查边截断或提交不完整数据。
        calibration = {name: MotorCalibration(**cal) for name, cal in fitted.items()}
        leader.bus.write_calibration(calibration)
        if not leader.bus.is_calibrated:
            raise RuntimeError("校准回读不匹配；原始采集已备份，尚未保存最终文件。")
        leader.calibration = calibration
        leader._save_calibration()
        for name in NAMES:
            if leader.bus.read("Torque_Enable", name) != 0:
                raise RuntimeError(f"{name}: 无法确认扭矩关闭。")
        print(f"LEADER CALIBRATION VERIFIED AND SAVED: {leader.calibration_fpath}", flush=True)
    finally:
        leader.bus.disconnect(disable_torque=False)
        print("COM6 closed. This recovery never enables torque or sends motion goals.", flush=True)


if __name__ == "__main__":
    main()
