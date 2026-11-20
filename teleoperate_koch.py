"""Koch 主臂实时控制从臂，使用视频三姿态软件校准；空格/Esc/Ctrl+C 停止。"""
import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

from calibrate_koch import (CalibrationError, NAMES, ROOT, SPECS, SOFTWARE_VOLTAGE_LIMITS,
                            connect_bus, snapshot, stable_positions, validate_health)
from koch_legacy_calibration import POSE_TOLERANCE, calibrated_degrees, compute_calibration
from koch_teleop_bus import FeedbackReadError, KochTeleopBus
from koch_calibration_bus import DynamixelCommunicationError
from dynamixel_sdk.robotis_def import COMM_RX_CORRUPT, COMM_RX_TIMEOUT

TICK_DEGREES = 360 / 4096
PERIOD = 0.05
REPLAY_PERIOD = 1 / 60
MAX_FRAME_SECONDS = 0.5
START_ALIGNMENT_DEGREES = 45
MIN_SPEED, MAX_SPEED, DEFAULT_SPEED = 2.0, 120.0, 90.0
COLLECTION_PERIOD = 0.025
COLLECTION_SPEED = 120.0
SAMPLE_HOLD_SECONDS = 0.3
TEMPERATURE_STOP_C = 55
CONFIRM_ATTEMPTS = 3
# 同一个姿态可以落在不同整圈上（扩展位置模式），这里只拦明显异常的量级。
MAX_SOFTWARE_DEGREES = 720.0
# 单帧丢包造成的目标落后：留出 3 帧步长的余量，速度越快容差越大。
GOAL_READBACK_TOLERANCE = 60
# 关节位置读不到时先保持最后一个目标并重试，超过这个时间才停止。
FEEDBACK_HOLD_SECONDS = 10.0


class TeleopError(RuntimeError):
    pass


class PositionSampleError(TeleopError):
    """不能用于运动目标的单帧位置异常。"""


def load_calibration(path):
    path = Path(path)
    if not path.is_file():
        raise TeleopError(f'缺少旧版校准文件：{path}\n请先分别完成“校准主臂”和“校准从臂”，再启动主从控制。')
    data = json.loads(path.read_text(encoding='utf-8'))
    fields = ('homing_offset', 'drive_mode', 'start_pos', 'end_pos', 'calib_mode', 'motor_names')
    if not isinstance(data, dict) or any(not isinstance(data.get(k), list) or len(data[k]) != 6 for k in fields):
        raise TeleopError(f'{path} 不是视频流程的旧版软件校准格式，不能使用新版 MotorCalibration 文件。')
    if set(data['motor_names']) != set(NAMES) or data['calib_mode'] != ['DEGREE'] * 6:
        raise TeleopError(f'{path} 关节名称或校准模式错误。')
    if any(type(v) is not int for key in fields[:4] for v in data[key]):
        raise TeleopError(f'{path} 的校准数值必须为整数。')
    by_name = {name: i for i, name in enumerate(data['motor_names'])}
    data = {key: [data[key][by_name[name]] for name in NAMES] for key in fields}
    expected = compute_calibration(dict(zip(NAMES, data['start_pos'])), dict(zip(NAMES, data['end_pos'])))
    # 兼容修复前已完成的校准：只在内存中把经过旧规则验证的夹爪零点换成实测闭合点。
    i = NAMES.index('gripper')
    start, end = data['start_pos'][i], data['end_pos'][i]
    sign = -1 if expected['drive_mode'][i] else 1
    old_offset = 1024 - round(sign * end / 1024) * 1024
    if (data['homing_offset'][i] == old_offset
            and abs(abs(end - start) - 1024) <= POSE_TOLERANCE
            and abs(sign * start + old_offset) <= POSE_TOLERANCE
            and abs(sign * end + old_offset - 1024) <= POSE_TOLERANCE):
        data['homing_offset'][i] = expected['homing_offset'][i]
    if data != expected:
        raise TeleopError(f'{path} 的零点/方向与原始采样不一致，请重新校准。')
    return data


def checked_angles(raw, calibration):
    if set(raw) != set(NAMES) or any(type(v) is not int or abs(v) > 8191 for v in raw.values()):
        raise TeleopError('关节读数缺失或异常，停止跟随。')
    angles = calibrated_degrees(raw, calibration)
    if any(abs(v) > MAX_SOFTWARE_DEGREES for v in angles.values()):
        raise TeleopError(f'软件坐标超出 ±{MAX_SOFTWARE_DEGREES:g}°；请检查校准或编码器跳圈。')
    return angles


def map_targets(raw_leader, leader_cal, follower_cal, reference=None):
    angles = checked_angles(raw_leader, leader_cal)
    goals = {}
    for i, name in enumerate(NAMES):
        if name == 'gripper':
            # 两只夹爪实测开合幅度可能不同，用实际采样端点对应闭合/打开。
            start, end = leader_cal['start_pos'][i], leader_cal['end_pos'][i]
            fraction = max(0.0, min(1.0, (raw_leader[name] - start) / (end - start)))
            goal = follower_cal['start_pos'][i] + fraction * (follower_cal['end_pos'][i] - follower_cal['start_pos'][i])
        else:
            sign = -1 if follower_cal['drive_mode'][i] else 1
            goal = sign * (angles[name] / TICK_DEGREES - follower_cal['homing_offset'][i])
        if reference is not None:
            # 同一姿态可落在不同整圈：取离参考（上一帧目标/当前位置）最近的一圈，
            # 既不会为了对齐而整圈倒转，也不受编码器整圈计数重置影响。
            goal += 4096 * round((reference[name] - goal) / 4096)
        goals[name] = round(goal)
    checked_angles(goals, follower_cal)
    return goals


def limit_speed(goals, previous, speed, dt, remainder=None):
    step = speed / TICK_DEGREES * min(max(dt, 0.001), 0.1)
    if remainder is not None and step < 1:
        # 40 Hz 下的低速步长可能不足一刻度，跨帧累积小数，避免永远取整为零。
        result = {}
        for name in NAMES:
            available = step + remainder.get(name, 0.0)
            whole = math.floor(available)
            delta = max(-whole, min(whole, goals[name] - previous[name]))
            result[name] = previous[name] + delta
            remainder[name] = available - whole if result[name] != goals[name] else 0.0
        return result
    if remainder is not None:
        remainder.clear()
    # 截断步长，避免取整后超过限速。速度下限保证正常控制周期内步长大于 1 tick。
    step = math.floor(step)
    return {name: previous[name] + max(-step, min(step, goals[name] - previous[name])) for name in NAMES}


def gripper_closing_contact(actual, previous_goal, requested_goal, calibration):
    """闭合方向的跟踪差允许用于夹持；不能据此识别物体或测量夹持力。"""
    i = NAMES.index('gripper')
    closed, opened = calibration['start_pos'][i], calibration['end_pos'][i]
    # 与目标映射保持同一整圈，支持反向安装和扩展位置坐标。
    turn = 4096 * round((previous_goal - (closed + opened) / 2) / 4096)
    closed, opened = closed + turn, opened + turn
    sign = 1 if opened > closed else -1
    position, previous, requested = (
        sign * (value - closed) for value in (actual, previous_goal, requested_goal))
    span = abs(opened - closed)
    # 主臂略微回开但目标仍在实际夹持位置的闭合侧时，仍然在施加闭合动作。
    return (0 <= previous < position <= span
            and 0 <= requested < position)


def make_bus(arm, port):
    return KochTeleopBus(port, {name: (i, model) for i, (name, model) in
                        enumerate(zip(NAMES, SPECS[arm]['models']), 1)}, arm)


def keyboard_stop():
    import msvcrt
    while msvcrt.kbhit():
        if msvcrt.getwch() in (' ', '\x1b', '\x03'):
            return True
    return False


class Teleoperator:
    def __init__(self, leader, follower, leader_cal, follower_cal, speed=DEFAULT_SPEED,
                 *, smooth_collection=False):
        if not math.isfinite(speed) or not MIN_SPEED <= speed <= MAX_SPEED:
            raise TeleopError(f'速度须为 {MIN_SPEED:g}–{MAX_SPEED:g}°/s。')
        if leader is not None and leader.port.casefold() == follower.port.casefold():
            raise TeleopError('主臂和从臂必须使用不同串口。')
        if smooth_collection and leader is None:
            raise TeleopError('流畅采集模式需要主臂，不能用于独立回放。')
        self.leader, self.follower = leader, follower
        self.leader_cal, self.follower_cal = leader_cal, follower_cal
        self.speed = speed
        self.smooth_collection = smooth_collection
        self.period = (REPLAY_PERIOD if leader is None else
                       COLLECTION_PERIOD if smooth_collection else PERIOD)
        self.discarded_samples = 0
        self.write_retries = 0

    def read_with_retry(self, read, attempts=3):
        """只读准备动作的整段重试：偶发丢包不该让启动失败，读到真实异常才停。"""
        for attempt in range(attempts):
            try:
                return read()
            except CalibrationError:
                raise
            except RuntimeError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.05)

    def preflight(self, *, require_alignment=True):
        for arm, bus in (('leader', self.leader), ('follower', self.follower)):
            if bus is None:
                continue
            states = self.read_with_retry(lambda: snapshot(bus))
            for name, state in states.items():
                self.check_motor_health(arm, name, state)
            validate_health(states, arm)
            for name, state in states.items():
                if state['Drive_Mode'] != 0 or state['Homing_Offset'] != 0 or state['Operating_Mode'] != 4:
                    raise TeleopError(f'{arm}.{name} 不是旧版软件校准基线，请重新运行视频校准。')
                if self.read_with_retry(lambda: bus.read_motion('Bus_Watchdog', name)) != 0:
                    raise TeleopError(f'{arm}.{name} 看门狗非零，请先结束其他控制程序。')
        follower = self.read_with_retry(lambda: stable_positions(self.follower))
        checked_angles(follower, self.follower_cal)
        if self.leader is None:
            return None, follower
        leader = self.read_with_retry(lambda: stable_positions(self.leader))
        goals = map_targets(leader, self.leader_cal, self.follower_cal, follower)
        differences = [f'{name}（{abs(goals[name] - follower[name]) * TICK_DEGREES:.1f}°）'
                       for name in NAMES
                       if abs(goals[name] - follower[name]) * TICK_DEGREES > START_ALIGNMENT_DEGREES]
        if require_alignment and differences:
            raise TeleopError(f'启动姿态相差超过 {START_ALIGNMENT_DEGREES}°：' + ', '.join(differences)
                             + '。请将列出的关节调整到允许差值内再启动。')
        return leader, follower

    def check_motor_health(self, arm, name, state):
        temperature, fault = state['Present_Temperature'], state['Hardware_Error_Status']
        prefix = f'{SPECS[arm]["label"]} ID {NAMES.index(name) + 1} {name}'
        if fault:
            raise TeleopError(f'{prefix}.Hardware_Error_Status=0x{fault:02X}（{fault}），'
                             f'温度 {temperature}°C；检测到电机硬件故障，停止控制。')
        if temperature >= TEMPERATURE_STOP_C:
            raise TeleopError(f'{prefix} 温度 {temperature}°C 达到软件温度保护 '
                             f'{TEMPERATURE_STOP_C}°C；硬件故障码 0x00。'
                             f'请卸载负载并冷却到 {TEMPERATURE_STOP_C}°C 以下后再启动。')

    def check_joint(self, bus, arm, name, active):
        model = SPECS[arm]['models'][NAMES.index(name)]
        state = bus.read_joint_status(name)
        voltage = state['Present_Input_Voltage'] / 10
        low, high = SOFTWARE_VOLTAGE_LIMITS[model]
        if not low <= voltage <= high:
            raise TeleopError(f'{arm}.{name} 电压 {voltage:.1f}V 超出软件允许范围 {low}–{high}V。')
        expected_torque = int(arm == 'follower' and active)
        expected_watchdog = 50 if expected_torque else 0
        self.check_motor_health(arm, name, state)
        checks = (('Torque_Enable', expected_torque),
                  ('Drive_Mode', 0), ('Homing_Offset', 0), ('Operating_Mode', 4))
        for reg, expected in checks:
            if state[reg] != expected:
                raise TeleopError(f'{arm}.{name}.{reg} 异常，停止跟随。')
        if state['Bus_Watchdog'] != expected_watchdog:
            raise TeleopError(f'{arm}.{name} 看门狗状态异常。')

    def restore_config(self, name, register, value):
        if register not in ('Bus_Watchdog', 'Profile_Velocity', 'Profile_Acceleration'):
            raise ValueError('退出恢复仅允许看门狗和速度配置')
        for attempt in range(3):
            if self.follower.read('Torque_Enable', name, normalize=False) != 0:
                raise TeleopError('扭矩未关闭，停止恢复配置')
            write_error = None
            try:
                self.follower.write_motion(register, name, value)
            except DynamixelCommunicationError as exc:
                if exc.result not in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT):
                    raise
                write_error = exc
            # 写入可能已执行但 ACK 丢失；只依据实际回读确认，禁止用于运动目标。
            actual = self.follower.read_motion(register, name)
            if actual == value:
                return
            if attempt == 2:
                detail = f'{register} 恢复 3 次后回读 {actual}，预期 {value}'
                if write_error is not None:
                    detail += f'；{write_error}'
                raise TeleopError(detail)

    def confirmed_write(self, register, name, value):
        """一次性写入（目标种子、扭矩、限速、看门狗）：回读确认未生效时重发同值。

        只允许用于幂等状态写入：重复写入同一个值不会产生新的运动，也不会补发
        过期目标。应答丢失但回读一致视为成功；电机故障等其他错误立即上抛。
        """
        for attempt in range(CONFIRM_ATTEMPTS):
            try:
                self.follower.write_motion(register, name, value)
                return
            except DynamixelCommunicationError as exc:
                if exc.result not in (COMM_RX_TIMEOUT, COMM_RX_CORRUPT) or attempt == CONFIRM_ATTEMPTS - 1:
                    raise
                self.write_retries += 1

    def check_goal_applied(self, name, expected):
        actual = self.follower.read_motion('Goal_Position', name)
        if abs(actual - expected) > self.goal_tolerance():
            raise TeleopError(f'{name}.Goal_Position 回读 {actual}，预期 {expected}；'
                              '目标流未生效或已到电机位置限制，停止跟随。')

    def goal_tolerance(self):
        """目标回读容差：速度越快，一帧的步长越大，需容忍更多帧的丢包。"""
        step = self.speed * PERIOD / TICK_DEGREES
        return max(GOAL_READBACK_TOLERANCE, math.ceil(3 * step) + 24)

    def release(self, saved):
        config_errors, skipped, disable_error = [], {}, None
        try:
            self.follower.disable_torque(num_retry=2)
        except Exception as exc:
            disable_error = str(exc)
        for name in NAMES:
            try:
                if self.follower.read('Torque_Enable', name, normalize=False) != 0:
                    raise TeleopError('扭矩仍开启')
            except Exception as exc:
                skipped[name] = str(exc)
                continue
            for register, value in (('Bus_Watchdog', 0), *saved.get(name, {}).items()):
                try:
                    self.restore_config(name, register, value)
                except Exception as exc:
                    config_errors.append(f'{name}.{register}: {exc}')
        # 用最终回读区分“扭矩状态未知”和“仅退出配置未恢复”。
        torque_errors = []
        for name in NAMES:
            try:
                if self.follower.read('Torque_Enable', name, normalize=False) != 0:
                    raise TeleopError('扭矩仍开启')
            except Exception as exc:
                torque_errors.append(f'{name}: {exc}')
        if torque_errors:
            details = torque_errors + config_errors + ([disable_error] if disable_error else [])
            raise TeleopError('无法确认从臂扭矩已关闭，请托稳并使用硬件断电。' + '; '.join(details))
        config_errors.extend(f'{name}: 首次无法确认扭矩关闭，跳过配置恢复；{reason}'
                             for name, reason in skipped.items())
        if config_errors:
            raise TeleopError('已确认从臂 6 个关节扭矩关闭，但部分退出配置未恢复。' + '; '.join(config_errors))

    def run(self, *, dry_run=False, stop=keyboard_stop, clock=time.monotonic, sleep=time.sleep,
            emit=print, on_frame=None, target_source=None):
        if (self.leader is None) != (target_source is not None):
            raise TeleopError('实时跟随需要主臂；独立回放需要轨迹目标源。')
        previous_leader, previous_follower = self.preflight(require_alignment=not dry_run)
        last_goals = dict(previous_follower)
        speed_remainder = {} if self.period < PERIOD else None
        if target_source is not None:
            # 在任何运动配置写入前校验回放起点；目标源不能自动追赶远处起点。
            checked_angles(target_source(0.0, previous_follower, last_goals), self.follower_cal)
        saved = {}
        modified = False
        count, started = 0, clock()
        last_time, bad_since, bad_error = started, {}, {}
        send_failures = send_errors = 0
        try:
            if stop():
                return 0
            if not dry_run:
                for name in NAMES:
                    # 看门狗触发时固件也会清零加速度，因此两项都保存。
                    saved[name] = {reg: self.follower.read_motion(reg, name)
                                   for reg in ('Profile_Velocity', 'Profile_Acceleration')}
                modified = True
                profile = max(1, math.floor(self.speed / (0.229 * 6)))
                # 先用最慢速度并只确认配置：部分固件在扭矩关闭时丢弃目标写入，
                # 若直接按跟随速度启用扭矩，会朝旧的冻结目标跳一下。
                for name in NAMES:
                    if stop():
                        return count
                    self.confirmed_write('Profile_Velocity', name, 1)
                    self.confirmed_write('Bus_Watchdog', name, 50)
                    if self.follower.read_motion('Profile_Velocity', name) != 1:
                        raise TeleopError(f'{name} 限速配置回读不一致。')
                    if self.follower.read_motion('Bus_Watchdog', name) != 50:
                        raise TeleopError(f'{name} 看门狗配置回读不一致。')
                current = self.read_with_retry(lambda: stable_positions(self.follower))
                if any(abs(current[n] - last_goals[n]) > 24 for n in NAMES):
                    raise TeleopError('准备期间从臂已移动，请重新启动。')
                for name in NAMES:
                    if stop():
                        return count
                    self.confirmed_write('Torque_Enable', name, 1)
                after = self.follower.sync_read('Present_Position', normalize=False)
                if any(abs(after[n] - current[n]) > 24 for n in NAMES):
                    raise TeleopError('启用扭矩后位置发生跳变，停止跟随。')
                # 扭矩已启用，目标写入才会生效；把目标对到启用前的实际位置。
                for name in NAMES:
                    if stop():
                        return count
                    self.confirmed_write('Goal_Position', name, current[name])
                last_goals = dict(current)
                for name in NAMES:
                    if stop():
                        return count
                    self.confirmed_write('Profile_Velocity', name, profile)
            last_time = clock()
            stream_started = last_time
            replay_elapsed, last_replay_sample = 0.0, stream_started
            last_command_time = None
            command_intervals = deque(maxlen=120)
            next_health_time, health_index = stream_started, 0
            hold_started = None
            sample_hold_started = None
            skip_jump_check = False
            while not stop():
                frame_start = clock()
                try:
                    leader = (self.leader.sync_read('Present_Position', normalize=False)
                              if self.leader is not None else None)
                    leader_time = clock()
                    follower = self.follower.sync_read('Present_Position', normalize=False)
                    follower_time = clock()
                    checked_angles(follower, self.follower_cal)
                    if target_source is None:
                        goals = map_targets(leader, self.leader_cal, self.follower_cal, last_goals)
                        checked_angles(goals, self.follower_cal)
                    if not skip_jump_check:
                        for name in NAMES:
                            if max(abs(leader[name] - previous_leader[name]) if leader is not None else 0,
                                   abs(follower[name] - previous_follower[name])) > 1024:
                                raise PositionSampleError(f'{name} 出现超过 90° 的单次位置跳变。')
                    skip_jump_check = False
                    health_due = self.period >= PERIOD or follower_time >= next_health_time - 1e-9
                    if health_due:
                        name = NAMES[health_index % 6]
                        if self.leader is not None:
                            self.check_joint(self.leader, 'leader', name, False)
                        self.check_joint(self.follower, 'follower', name, not dry_run)
                        health_index += 1
                        next_health_time = follower_time + PERIOD
                    if target_source is not None:
                        if hold_started is not None:
                            # 位置重新可读不代表其他电机未复位；六关节均健康才允许续播。
                            for joint in NAMES:
                                if stop():
                                    return count
                                if not health_due or joint != name:
                                    self.check_joint(self.follower, 'follower', joint, not dry_run)
                            if clock() - hold_started > FEEDBACK_HOLD_SECONDS:
                                raise TeleopError('回放通信恢复检查超时，停止。')
                            next_replay_elapsed = replay_elapsed
                        else:
                            next_replay_elapsed = replay_elapsed + max(0, follower_time - last_replay_sample)
                        # 在位置与健康检查都通过后才推进目标源；丢包期间不消耗轨迹时间。
                        goals = target_source(next_replay_elapsed, follower, last_goals)
                        checked_angles(goals, self.follower_cal)
                except PositionSampleError as exc:
                    if not self.smooth_collection or dry_run:
                        raise
                    self.discarded_samples += 1
                    if sample_hold_started is None:
                        sample_hold_started = clock()
                        emit(f'位置采样抖动，丢弃异常帧并保持上一目标重读：{exc}')
                    if (clock() - sample_hold_started >= SAMPLE_HOLD_SECONDS
                            or clock() - frame_start > MAX_FRAME_SECONDS):
                        raise TeleopError(f'位置异常持续，无法确认真实姿态，停止：{exc}') from exc
                    if stop():
                        return count
                    # 保持已确认过的目标，不接纳异常读数，也不让夹爪因一帧抖动松开。
                    try:
                        self.follower.sync_write_goals(last_goals)
                    except DynamixelCommunicationError as write_error:
                        send_failures += 1
                        send_errors += 1
                        if send_failures >= 3:
                            raise TeleopError('连续 3 个周期无法向从臂发送保持目标，停止。') from write_error
                    else:
                        send_failures = 0
                    last_time = clock()
                    sleep(max(0, self.period - (clock() - frame_start)))
                    continue
                except TeleopError:
                    raise
                except RuntimeError as exc:
                    if target_source is not None and not isinstance(exc, FeedbackReadError):
                        raise TeleopError(f'回放读取发生非瞬时通信异常，停止：{exc}') from exc
                    # 读不到位置时不能继续发新目标，但也不必立刻松扭矩：保持最后
                    # 一个目标继续重试，链路恢复就接着跟随，超时才停止。
                    if hold_started is None:
                        hold_started = frame_start if target_source is not None else clock()
                        if target_source is not None:
                            last_command_time = None
                            command_intervals.clear()
                            emit(f'回放暂停于 {replay_elapsed:.3f}s，保持最后目标及夹持并重读：{exc}')
                        else:
                            emit(f'位置回读失败，保持当前目标并重试：{exc}')
                    if clock() - hold_started > FEEDBACK_HOLD_SECONDS:
                        raise TeleopError(
                            f'连续 {FEEDBACK_HOLD_SECONDS:g} 秒无法读取关节位置，'
                            f'停止{"回放" if target_source is not None else "跟随"}：{exc}') from exc
                    if not dry_run:
                        if stop():
                            return count
                        if target_source is not None:
                            try:
                                self.follower.sync_write_goals(last_goals)
                            except DynamixelCommunicationError as write_error:
                                send_failures += 1
                                send_errors += 1
                                if send_failures >= 3:
                                    raise TeleopError('回放连续 3 次无法发送保持目标，停止。') from write_error
                            else:
                                send_failures = 0
                        else:
                            try:
                                self.follower.sync_write_goals(last_goals)
                            except Exception:
                                pass
                    last_time = clock()
                    skip_jump_check = not self.smooth_collection and target_source is None
                    sleep(0.02)
                    continue
                if target_source is not None:
                    if hold_started is not None:
                        emit(f'回放通信恢复，暂停 {clock() - hold_started:.3f}s；'
                             f'六关节健康检查通过，从 {replay_elapsed:.3f}s 继续。')
                        last_replay_sample = clock()
                        bad_since.clear()
                        bad_error.clear()
                    else:
                        last_replay_sample = follower_time
                    replay_elapsed = next_replay_elapsed
                hold_started = None
                sample_hold_started = None
                now = clock()
                if now - frame_start > MAX_FRAME_SECONDS or now - last_time > MAX_FRAME_SECONDS:
                    raise TeleopError('通信或控制周期超过 0.5 秒，停止跟随。')
                sent = False
                if not dry_run:
                    for joint in NAMES:
                        error = abs(last_goals[joint] - follower[joint]) * TICK_DEGREES
                        contact = joint == 'gripper' and gripper_closing_contact(
                            follower[joint], last_goals[joint], goals[joint], self.follower_cal)
                        if error > 20 and not contact:
                            if joint not in bad_since:
                                bad_since[joint] = now
                                bad_error[joint] = error
                            elif error < bad_error[joint] - .25:
                                # 起始偏差较大时允许限速追赶；只拦持续不接近目标的受阻。
                                bad_since[joint] = now
                                bad_error[joint] = error
                            if now - bad_since[joint] > 1:
                                raise TeleopError(f'{joint} 持续跟踪误差超过 20°，可能受阻，停止。')
                        else:
                            bad_since.pop(joint, None)
                            bad_error.pop(joint, None)
                    speed = self.speed
                    if self.smooth_collection:
                        # 仅启动前 0.75 秒渐进提速，避免初始姿态差被高速追赶。
                        speed = min(speed, 30 + max(0, speed - 30) *
                                    min(1, (now - stream_started) / .75))
                    goals = limit_speed(goals, last_goals, speed, now - last_time, speed_remainder)
                    if stop():
                        return count
                    if clock() - now > MAX_FRAME_SECONDS:
                        raise TeleopError('目标数据已过期，停止写入。')
                    try:
                        # 一个广播包写全部目标，不等应答：丢一帧只等于少发一次，
                        # 下一帧的新目标会覆盖，绝不重放旧目标。
                        self.follower.sync_write_goals(goals)
                        command_time = clock()
                        if target_source is not None:
                            if last_command_time is not None and command_time > last_command_time:
                                command_intervals.append(command_time - last_command_time)
                            last_command_time = command_time
                    except DynamixelCommunicationError as exc:
                        send_failures += 1
                        send_errors += 1
                        if send_failures >= 3:
                            raise TeleopError('连续 3 个周期无法向从臂发送目标，停止跟随。') from exc
                    else:
                        sent = True
                        send_failures = 0
                        last_goals = goals
                        if health_due:
                            try:
                                self.check_goal_applied(name, goals[name])
                            except TeleopError:
                                raise
                            except RuntimeError:
                                # 单次回读失败不改变控制；持续读不到会进入保持逻辑。
                                pass
                if on_frame is not None and not dry_run:
                    on_frame({'leader_time': leader_time, 'follower_time': follower_time,
                              'command_time': command_time if sent else None,
                              'leader_raw': dict(leader) if leader is not None else None,
                              'follower_raw': dict(follower), 'goal_raw': dict(last_goals),
                              'sent': sent})
                if count % round(1 / self.period) == 0:
                    emit(('预览' if dry_run else '跟随') + ' | ' + ' '.join(
                        f'{n}={v:.0f}°' for n, v in calibrated_degrees(goals, self.follower_cal).items()))
                    if not dry_run:
                        # 显示映射目标与实际位置之差，包含限速造成的落后；夹爪接触单列。
                        requested = (map_targets(leader, self.leader_cal, self.follower_cal, last_goals)
                                     if target_source is None else goals)
                        worst = max((n for n in NAMES if n != 'gripper'),
                                    key=lambda n: abs(requested[n] - follower[n]))
                        emit(f'实际位置误差 | 最大关节 {worst}='
                             f'{abs(requested[worst] - follower[worst]) * TICK_DEGREES:.2f}°'
                             f'，夹爪={abs(requested["gripper"] - follower["gripper"]) * TICK_DEGREES:.2f}°')
                    if target_source is not None and command_intervals:
                        actual_hz = len(command_intervals) / sum(command_intervals)
                        emit(f'回放更新 | 目标 {1 / self.period:.1f} Hz，实际 {actual_hz:.1f} Hz，'
                             f'最大周期 {max(command_intervals) * 1000:.1f} ms（不含通信暂停）')
                previous_leader, previous_follower = leader, follower
                count += 1
                last_time = now
                sleep(max(0, self.period - (clock() - frame_start)))
            return count
        finally:
            if modified:
                original_error = sys.exception()
                try:
                    self.release(saved)
                except Exception as cleanup_error:
                    if original_error is not None:
                        reason = str(original_error) or type(original_error).__name__
                        raise TeleopError(f'原始停止原因：{reason}\n退出清理：{cleanup_error}') from original_error
                    raise
            failures = {f'从臂.{n}': c for n, c in getattr(self.follower, 'read_failures', {}).items()}
            failures.update({f'主臂.{n}': c for n, c in getattr(self.leader, 'read_failures', {}).items()})
            if self.write_retries or send_errors or failures:
                detail = ('，读取失败 ' + '、'.join(f'{n} {c} 次' for n, c in sorted(failures.items()))
                          if failures else '')
                emit(f'本次串口：一次性写入重试 {self.write_retries} 次，'
                     f'目标包发送失败 {send_errors} 次{detail}。')
            if self.discarded_samples:
                emit(f'本次丢弃位置异常帧 {self.discarded_samples} 帧；这些读数未用于目标或采集。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leader-port', help='主臂端口；默认自动识别')
    parser.add_argument('--follower-port', help='从臂端口；默认自动识别')
    parser.add_argument('--leader-calibration', type=Path, default=ROOT / 'calibration/koch_legacy/leader/my_leader.json')
    parser.add_argument('--follower-calibration', type=Path, default=ROOT / 'calibration/koch_legacy/follower/my_follower.json')
    parser.add_argument('--speed', type=float, default=DEFAULT_SPEED,
                        help=f'最大跟随速度，{MIN_SPEED:g}–{MAX_SPEED:g}°/s，默认 {DEFAULT_SPEED:g}')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='只读检查后退出')
    modes.add_argument('--dry-run', action='store_true', help='持续只读预览映射，不启用从臂')
    args = parser.parse_args(argv)
    buses, result = [], 0
    try:
        if not math.isfinite(args.speed) or not MIN_SPEED <= args.speed <= MAX_SPEED:
            raise TeleopError(f'速度须为 {MIN_SPEED:g}–{MAX_SPEED:g}°/s。')
        if args.leader_port and args.follower_port and args.leader_port.casefold() == args.follower_port.casefold():
            raise TeleopError('主臂与从臂不能使用同一个串口。')
        leader_cal = load_calibration(args.leader_calibration)
        follower_cal = load_calibration(args.follower_calibration)
        for arm, port in (('leader', args.leader_port), ('follower', args.follower_port)):
            bus = connect_bus(arm, port, factory=make_bus)
            buses.append(bus)
            print(f'{SPECS[arm]["label"]}：{bus.port}')
        control = Teleoperator(*buses, leader_cal, follower_cal, args.speed)
        control.preflight(require_alignment=not args.dry_run)
        if args.check:
            print('两臂检查通过；未写入配置，也未启用扭矩。')
        else:
            if not args.dry_run:
                print(f'允许启动姿态差值 ≤{START_ALIGNMENT_DEGREES}°；从臂将以最多 {args.speed:g}°/s 逐渐跟随。'
                      '请清空活动空间，主臂保持自由。')
                if args.speed > 60:
                    print('高速跟随：惯性和碰撞风险更大，软件不会替代急停，请留出更大空间。')
                print('空格/Esc/Ctrl+C 停止并释放从臂，停止时需托稳以防下落。')
                input('准备好后按回车开始实时跟随：')
            else:
                print('只读预览：空格/Esc/Ctrl+C 结束，两臂均不启用扭矩。')
            control.run(dry_run=args.dry_run)
            print('主从控制已结束。')
    except (KeyboardInterrupt, EOFError):
        print('\n已停止。')
        result = 130
    except Exception as exc:
        print(f'\n停止：{exc}', file=sys.stderr)
        result = 1
    finally:
        for bus in reversed(buses):
            try:
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                print(f'释放串口失败 {bus.port}：{exc}', file=sys.stderr)
                result = 1
    return result


if __name__ == '__main__':
    raise SystemExit(main())
