# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Adapted for this project's standalone, Chinese calibration wizard.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. Local copy: assets/koch_legacy/LICENSE.

"""视频三姿态校准：软件方向/偏移，独立于当前 LeRobot 的硬件校准格式。

算法参照 Hugging Face LeRobot 固定提交 8a64d8268b821a3a8fee202134be1b8c42af9a1c
中的 dynamixel_calibration.py（Apache-2.0）。这是流程移植，不代表该提交是视频录制版本。
"""

import shutil
import webbrowser
from datetime import datetime
from pathlib import Path

from koch_robot.calibration import (CalibrationError, NAMES, ROOT, SPECS, require_torque_off,
                            snapshot, stable_positions, validate_health, write_json)

REFERENCE = '8a64d8268b821a3a8fee202134be1b8c42af9a1c'
POSE_TOLERANCE = 256  # 22.5°；用于识别摆错姿态，不表示实际校准精度。


def compute_calibration(zero, rotated):
    """五个旋转关节沿用旧版 90°公式；夹爪以实测闭合/打开端点校准。"""
    if set(zero) != set(NAMES) or set(rotated) != set(NAMES):
        raise CalibrationError('零位和旋转位必须包含全部 6 个关节。')
    offsets, modes = [], []
    for name in NAMES:
        start, end = zero[name], rotated[name]
        if not isinstance(start, int) or not isinstance(end, int) or max(abs(start), abs(end)) > 8191:
            raise CalibrationError(f'{name} 原始位置异常。')
        delta = end - start
        inverted = int(delta < 0)
        sign = -1 if inverted else 1
        if name == 'gripper':
            # 两个稳定采样各允许 24 tick 波动；端点须可区分，不要求转过 90°。
            if abs(delta) <= 48:
                raise CalibrationError('夹爪闭合/打开采样太接近，无法区分两个端点，请重新采样。')
            if abs(delta) >= 4096:
                raise CalibrationError('夹爪两次采样相差整圈或更多，请检查编码器跳圈并重新采样。')
            modes.append(inverted)
            offsets.append(-sign * start)
            continue
        if abs(abs(delta) - 1024) > POSE_TOLERANCE:
            raise CalibrationError(f'{name} 两姿态变化 {delta} tick，预期约 ±1024（90°）。'
                                   '请按图摆放该关节，不要强扳限位。')
        offset = 1024 - round(sign * end / 1024) * 1024
        if abs(sign * start + offset) > POSE_TOLERANCE or abs(sign * end + offset - 1024) > POSE_TOLERANCE:
            raise CalibrationError(f'{name} 未对齐参考姿态；零位应约为 0°，旋转位应约为 90°。')
        modes.append(inverted)
        offsets.append(offset)
    # 与固定历史源码返回字段一致；这里的 homing_offset 仅用于软件换算。
    return dict(homing_offset=offsets, drive_mode=modes, start_pos=[zero[n] for n in NAMES],
                end_pos=[rotated[n] for n in NAMES], calib_mode=['DEGREE'] * 6,
                motor_names=list(NAMES))


def calibrated_degrees(positions, calibration):
    return {name: ((-1 if mode else 1) * positions[name] + offset) * 360 / 4096
            for name, mode, offset in zip(calibration['motor_names'], calibration['drive_mode'],
                                          calibration['homing_offset'])}


def show_pose(arm, pose):
    path = ROOT / 'assets/koch_legacy' / f'{arm}_{pose}.webp'
    print(f'姿态参考图：{path}', flush=True)
    if not path.exists():
        raise CalibrationError(f'缺少参考姿态图：{path}，请恢复图文件后重试。')
    try:
        if not webbrowser.open(path.as_uri()):
            print('未能自动打开，请手动打开上方参考图。')
    except OSError:
        print('未能自动打开，请手动打开上方参考图。')


def require_software_baseline(bus):
    require_torque_off(bus)
    for name in NAMES:
        for register, expected in (('Drive_Mode', 0), ('Homing_Offset', 0), ('Operating_Mode', 4)):
            actual = bus.read(register, name, normalize=False)
            if actual != expected:
                raise CalibrationError(f'{name}.{register} 回读 {actual}，预期 {expected}；停止保存。')


def calibrate_legacy(bus, arm, target, *, ask=input, show=show_pose):
    target = Path(target)
    states = snapshot(bus)
    validate_health(states, arm)
    stable_positions(bus)
    session = target.parent / 'history' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    write_json(session / 'before_registers.json', states)
    if target.exists():
        shutil.copy2(target, session / 'previous_calibration.json')
    print(f'\n{SPECS[arm]["label"]}：视频三姿态流程，软件方向/零点。备份：{session}')
    print('本流程会清除电机内旧方向和零点，旧的新版校准文件将不再匹配电机。')
    print('全程手动摆动，不启用扭矩；记录夹爪开合端点，不改写电机行程限制。')
    ask('支撑好机械臂，回车开始；Ctrl+C 取消：')
    modified = False
    try:
        modified = True
        bus.disable_torque()
        require_torque_off(bus)
        bus.set_operating_mode('extended_position')
        for name in NAMES:
            bus.write('Drive_Mode', name, 0, normalize=False)
            bus.write('Homing_Offset', name, 0, normalize=False)
        require_software_baseline(bus)
        show(arm, 'zero')
        ask('\n1/3 零位：严格按图摆放（臂伸直水平、夹爪朝上并闭合），托稳后回车：')
        validate_health(snapshot(bus), arm)
        require_software_baseline(bus)
        zero = stable_positions(bus)
        write_json(session / 'zero.json', zero)
        show(arm, 'rotated')
        ask('\n2/3 旋转位：五个臂关节按图相对零位转约 90°；夹爪正常打开即可，'
            '按实际开合行程记录，不要求 90°。托稳后回车：')
        validate_health(snapshot(bus), arm)
        require_software_baseline(bus)
        rotated = stable_positions(bus)
        write_json(session / 'rotated.json', rotated)
        calibration = compute_calibration(zero, rotated)
        write_json(session / 'proposed_calibration.json', calibration)
        show(arm, 'rest')
        ask('\n3/3 休息位：按图手动放稳机械臂，回车完成；Ctrl+C 取消：')
        validate_health(snapshot(bus), arm)
        rest = stable_positions(bus)
        degrees = calibrated_degrees(rest, calibration)
        if any(abs(value) > 270 for value in degrees.values()):
            raise CalibrationError('休息位坐标异常，请检查摆姿或编码器跳圈。')
        write_json(session / 'rest.json', dict(positions=rest, degrees=degrees))
        bus.disable_torque(num_retry=2)
        require_software_baseline(bus)
        modified = False
        write_json(session / 'result.json', dict(status='verified', path=str(target), port=bus.port,
                                                method='legacy', reference=REFERENCE))
        # 最终文件是提交点；软件偏移不写入电机的 Homing_Offset 寄存器。
        write_json(target, calibration)
        return target
    except BaseException as exc:
        write_json(session / 'result.json', dict(status='failed', error=str(exc), port=bus.port))
        print(f'\n本轮未完成，最终文件未更新。现场：{session}')
        print('本轮校准验证尚未完成，请重新完成校准后再启动主从控制；此提示不代表硬件超限。')
        raise
    finally:
        if modified:
            bus.disable_torque(num_retry=2)
            require_torque_off(bus)
