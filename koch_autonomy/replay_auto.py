"""Run a whole visual task in one follower Replay session."""
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from datetime import datetime
from functools import lru_cache, partial
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace
import math
import sys

from koch_dataset import (REPLAY_DEFAULT_RATE, REPLAY_DEFAULT_SPEED, REPLAY_START_ALIGNMENT_DEGREES,
                          ReplayTrajectory, read_jsonl)
from teleoperate_koch import (NAMES, TICK_DEGREES, Teleoperator, connect_bus, keyboard_stop,
                              load_calibration, make_bus, stable_positions)

from .planner import SLOTS, TARGETS, validate_inventory, validate_targets
from .replay_bridge import LAYER_DIRS, RATE, SPEED, motion_plan, read_trajectory, route_episodes
from .replay_policy import data_and_exclusions, route_folders
from .reverse import ROOT, read_json, save_json


AUTO_ENDPOINT_TOLERANCE = 1.5
SCAN_SPEED = 75.
SCAN_TIME_SCALE = 2 / 3
SCAN_MINIMUM_PACE = .60
SCAN_READ_SECONDS = 8.


class ScanReadError(ValueError):
    """The wrist scan returned to rest without a verified read."""


def gap_degrees(first, second):
    return max(abs((first[name] - second[name] + 2048) % 4096 - 2048) for name in NAMES) * TICK_DEGREES


def configure_legacy_alignment(trajectory, *, trim=False):
    if trim:
        trajectory.trim_wait(after_settle=True)
    trajectory.align_start = True
    trajectory.alignment_speed = 18.
    trajectory.start_tolerance = 2.
    trajectory.alignment_hold_seconds = .12
    trajectory.use_recorded_endpoint_feedback = True
    return trajectory


def eligible_episodes(data, pair, route, excluded_paths=()):
    if excluded_paths and route_folders(data, pair, route).intersection(excluded_paths):
        return []
    return [episode for episode in route_episodes(data, pair, route)
            if not excluded_paths or (episode.resolve() not in excluded_paths
                                      and episode.parent.resolve() not in excluded_paths)]


def prepare_chain(data, task, quantity_rule, calibration, *, home=None, trim=False,
                  endpoint_tolerance=AUTO_ENDPOINT_TOLERANCE, initial_pose=None, carry_rate=RATE,
                  observed_body=False, legacy_replay=False, latest_only=False, allow_older_fallback=False,
                  excluded_paths=(), load_trajectory=None):
    plan = motion_plan(task, quantity_rule)
    moves = plan['moves']
    load_trajectory = load_trajectory or read_trajectory
    @lru_cache(None)
    def candidates(pair, route, include_older=False):
        episodes = eligible_episodes(data, pair, route, excluded_paths)
        if latest_only and not include_older:
            episodes = episodes[-1:]
        if not episodes:
            return ()
        rate_options = {} if carry_rate == RATE else {'rate': carry_rate}
        if legacy_replay:
            rate_options = {'rate': REPLAY_DEFAULT_RATE, 'speed': REPLAY_DEFAULT_SPEED}
        elif observed_body:
            rate_options['observed_body'] = True
        options = [load_trajectory(episode, route, calibration, **rate_options) for episode in reversed(episodes)]
        if legacy_replay:
            for trajectory in options:
                configure_legacy_alignment(trajectory, trim=trim)
        elif trim or home is not None:
            for trajectory in options:
                trajectory.trim_wait(after_settle=True)
                trajectory.align_start = True
                trajectory.endpoint_tolerance = endpoint_tolerance
                trajectory.follow_feedback = True
                if observed_body:
                    trajectory.start_tolerance = 5.
        return options

    if home is not None:
        stocks = validate_inventory(task['inventory'])
        colors, demand = validate_targets(task['targets'], quantity_rule)
        seam = REPLAY_START_ALIGNMENT_DEGREES - endpoint_tolerance

        @lru_cache(None)
        def solve(heights, remaining, previous, older_budget):
            endpoint = ((initial_pose or home) if previous is None else
                        previous.rows[-1]['follower_raw' if legacy_replay else 'goal_raw'])
            if not any(remaining):
                return () if gap_degrees(endpoint, home) <= seam else None
            options = []
            for i, height in enumerate(heights):
                if not height:
                    continue
                for j, count in enumerate(remaining):
                    if not count or stocks[i][height - 1] != colors[j]:
                        continue
                    source, target = SLOTS[i], TARGETS[j]
                    layer = demand[j] - count + 1
                    pair, route = (height, layer), f'{source}_{target}'
                    for index, trajectory in enumerate(candidates(pair, route, older_budget > 0)):
                        start = trajectory.rows[0]['goal_raw']
                        limit = REPLAY_START_ALIGNMENT_DEGREES if previous is None and initial_pose else seam
                        if previous is not None and legacy_replay:
                            limit = REPLAY_START_ALIGNMENT_DEGREES - previous.endpoint_tolerance
                        if gap_degrees(endpoint, start) <= limit:
                            move = {'source': source, 'source_layer': height, 'target': target,
                                    'target_layer': layer, 'color': colors[j], 'route': route,
                                    'layer_pair': list(pair)}
                            options.append((i, j, move, trajectory, int(latest_only and index > 0)))
            if previous is None and initial_pose is not None:
                # At the resting pose, prefer demonstrations that already match wrist pitch.
                # Correcting this joint against the table can be less reliable than a turn.
                def start_cost(option):
                    start = option[3].rows[0]['goal_raw']
                    pitch = abs((start['wrist_flex'] - endpoint['wrist_flex'] + 2048) % 4096 - 2048)
                    return pitch, gap_degrees(endpoint, start)
                options.sort(key=start_cost)
            for i, j, move, trajectory, older in options:
                next_heights, next_counts = list(heights), list(remaining)
                next_heights[i] -= 1
                next_counts[j] -= 1
                tail = solve(tuple(next_heights), tuple(next_counts), trajectory, older_budget - older)
                if tail is not None:
                    return ((move, trajectory, older), *tail)
            return None

        heights = tuple(map(len, stocks))
        result = solve(heights, demand, None, 0)
        if result is None and latest_only and allow_older_fallback:
            # Minimize the number of moves that need an earlier recording.
            for budget in range(1, len(moves) + 1):
                result = solve(heights, demand, None, budget)
                if result is not None:
                    break
        if result is None:
            raise ValueError('当前摆放符合层序和数量，但可用 Replay 无法在起终点容差内接成一条完整路线。')
        steps = [item[0] for item in result]
        trajectories = [item[1] for item in result]
        return {**plan, 'moves': steps, 'cost': sum(t.duration for t in trajectories),
                'cost_metric': 'replay_duration_seconds', 'optimal_for_supplied_costs': False,
                'older_episode_steps': [step for step, item in enumerate(result, 1) if item[2]],
                'search_states': solve.cache_info().currsize}, trajectories

    choices = []
    for move in moves:
        pair, route = tuple(move['layer_pair']), move['route']
        options = candidates(pair, route)
        if not options:
            raise ValueError(f'缺少已确认的动作：{LAYER_DIRS[pair]}/{route}。')
        choices.append(options)
    if not choices:
        return plan, []
    paths = [(trajectory,) for trajectory in choices[0]]
    for step, options in enumerate(choices[1:], 1):
        next_paths = {}
        nearest = (float('inf'), None, None)
        for path in paths:
            previous = path[-1]
            for candidate in options:
                gap = gap_degrees(previous.rows[-1]['goal_raw'], candidate.rows[0]['goal_raw'])
                if gap < nearest[0]:
                    nearest = gap, previous, candidate
                if gap <= REPLAY_START_ALIGNMENT_DEGREES - (endpoint_tolerance if trim else 0):
                    next_paths.setdefault(candidate.path, (*path, candidate))
        if not next_paths:
            gap, previous, candidate = nearest
            raise ValueError(f'第 {step} 步接第 {step + 1} 步最近仍相差 {gap:.2f}°，'
                             f'超过动作起点容差 {REPLAY_START_ALIGNMENT_DEGREES}°。未连接从臂。\n'
                             f'前段：{previous.path}\n后段：{candidate.path}')
        paths = list(next_paths.values())
    return plan, paths[0]


def ignore_event(kind, **payload):
    pass


class ReplayChain:
    def __init__(self, trajectories, on_event=ignore_event, cancelled=keyboard_stop, *, return_home=None):
        self.trajectories = trajectories
        self.return_home = return_home
        self.on_event, self.cancelled = on_event, cancelled
        self.completed = 0
        self.origin = 0.0
        self.next_start = False
        self.announced = False
        self.aligning_announced = False
        self.returning = False

    def current(self):
        return self.return_home if self.returning else self.trajectories[self.completed]

    def __call__(self, elapsed, actual, previous):
        if self.next_start:
            self.origin = elapsed
            self.next_start = False
        return self.current()(elapsed - self.origin, actual, previous)

    def accept_frame(self, frame):
        trajectory = self.current()
        if not frame['sent']:
            return
        if not self.announced:
            if trajectory.align_start and not self.returning and not self.aligning_announced:
                self.aligning_announced = True
                self.on_event('move_aligning', step=self.completed + 1)
                print(f'Replay {self.completed + 1}/{len(self.trajectories)} 正在对准录制起点。', flush=True)
            if trajectory.align_start and trajectory.motion_origin is None:
                return
            self.announced = True
            self.on_event('move_started', step=self.completed + 1)
        if not trajectory.settled:
            return
        self.next_start = True
        if self.return_home is not None and self.completed + 1 == len(self.trajectories) and not self.returning:
            self.returning = True
            return
        route = self.trajectories[self.completed].meta['route']
        self.returning = False
        self.completed += 1
        self.announced = False
        self.aligning_announced = False
        self.on_event('move_finished', step=self.completed)
        print(f'Replay {self.completed}/{len(self.trajectories)} 已完成：{route}', flush=True)

    def stop(self):
        if self.cancelled():
            raise KeyboardInterrupt('用户中止自动 Replay。')
        return self.completed == len(self.trajectories)


def load_scan(path, calibration):
    capture = read_json(path / 'capture.json')
    if (capture.get('version') != 1 or capture.get('category') != 't_motion'
            or capture.get('path_kind') != 'manual_approach_to_single_overhead_pose'
            or capture.get('joint_units') != 'raw_encoder_ticks' or capture.get('joint_names') != list(NAMES)):
        raise ValueError(f'{path} 缺少受支持的休息位至 T 区扫描记录。')
    registers = capture['joint_registers_at_start']
    fields = {'Drive_Mode', 'Operating_Mode', 'Homing_Offset'}
    if set(registers) != set(NAMES) or any(set(row) != fields for row in registers.values()):
        raise ValueError('扫描示范缺少完整的关节寄存器记录。')
    samples = read_jsonl(path / 'joints.jsonl')
    if len(samples) != capture.get('sample_count') or len(samples) < 3:
        raise ValueError('扫描示范的关节采样数不完整。')
    times = [row['time_s'] for row in samples]
    if not all(type(t) in (int, float) and math.isfinite(t) for t in times) or any(
            not 0 < b - a <= .2 for a, b in zip(times, times[1:])):
        raise ValueError('扫描示范的采样时间不连续。')
    first_hold = next((i for i, row in enumerate(samples) if row['phase'] == 'hold'), 0)
    if first_hold < 2 or any(r['phase'] != 'approach' for r in samples[:first_hold]) or any(
            r['phase'] != 'hold' or gap_degrees(r['positions'], samples[-1]['positions']) > 3
            for r in samples[first_hold:]):
        raise ValueError('扫描示范需要明确到达 T 区且关节稳定的停顿段。')
    if any(abs(b['positions'][n] - a['positions'][n]) * TICK_DEGREES > 12
           for a, b in zip(samples, samples[1:]) for n in NAMES):
        raise ValueError('扫描示范相邻样本位置跳变超过 12°，请重新采集。')
    end_pose = samples[-1]['positions']
    last_moving = max((i for i in range(first_hold) if gap_degrees(samples[i]['positions'], end_pose) > 1.),
                      default=first_hold - 1)
    rows = [{'command_time': (row['time_s'] - times[0]) * SCAN_TIME_SCALE,
             'goal_raw': row['positions'], 'follower_raw': row['positions']}
            for row in samples[:min(first_hold, last_moving + 2)]]
    rows.append({'command_time': rows[-1]['command_time'] + .15,
                 'goal_raw': end_pose, 'follower_raw': end_pose})
    forward = ReplayTrajectory.from_rows(path, '扫描 T 区', rows, calibration, REPLAY_DEFAULT_RATE, SCAN_SPEED)
    forward.trim_wait()
    backward_rows = [{**row, 'command_time': rows[-1]['command_time'] - row['command_time']} for row in reversed(rows)]
    backward = ReplayTrajectory.from_rows(path, '扫描后回休息位', backward_rows, calibration,
                                          REPLAY_DEFAULT_RATE, SCAN_SPEED)
    backward.trim_wait()
    forward.align_start = backward.align_start = True
    forward.endpoint_tolerance = backward.endpoint_tolerance = 2.5
    forward.follow_feedback = backward.follow_feedback = True
    forward.minimum_feedback_pace = backward.minimum_feedback_pace = SCAN_MINIMUM_PACE
    return forward, backward, registers


def read_live(kind, specification, site_dir, quantity_rule, output, observe, *, cancelled=None):
    def file(key):
        return (site_dir / specification[key]).resolve() if specification.get(key) else None

    args = SimpleNamespace(
        kind=kind, config=file('config'), camera=specification.get('camera'),
        camera_name=specification.get('camera_name'), rotation=specification.get('rotation'), target=None,
        plane=file('plane'), roi=specification.get('roi'), seconds=specification.get('seconds', 30),
        quantity_rule=quantity_rule, image=None, show=False, output=output, alignment=file('alignment'),
        excluded_source_region=specification.get('excluded_source_region'),
    )
    report = observe(args, cancelled=cancelled) if cancelled is not None else observe(args)
    if report.get('complete') is not True or report.get('stable') is not True:
        raise ValueError(f'{kind} 未得到完整且稳定的识别结果。')
    return report


def scan_and_return(follower, calibration, scan, observe_targets, cancelled, on_event, *,
                    read_timeout=SCAN_READ_SECONDS + 7, motion_output=None):
    forward, backward, registers = scan
    if any(follower.read(key, joint, normalize=False) != value
           for joint, settings in registers.items() for key, value in settings.items()):
        raise ValueError('从臂寄存器与 T 区扫描录制时不同，停止回放。')
    cancel_observation = Event()
    future = Future()
    read_deadline = None
    returning = complete = timed_out = False
    return_origin = None
    phase = 'approach'
    samples = []
    traveled = []
    rest_goal = backward.rows[-1]['goal_raw']
    scan_goal = forward.rows[-1]['goal_raw']
    still_pose, still_since, holding = None, None, None
    end_started = None
    pose_gap = None
    return_track = backward
    forward.allow_endpoint_hold = True

    def target_source(elapsed, actual, previous):
        nonlocal return_origin
        if returning:
            if return_origin is None:
                return_origin = elapsed
            return return_track(elapsed - return_origin, actual, previous)
        if holding is not None:
            return holding
        return forward(elapsed, actual, previous)

    def on_frame(frame):
        nonlocal read_deadline, complete, timed_out, phase, still_pose, still_since, holding, end_started, pose_gap
        if not frame['sent']:
            return
        actual = frame['follower_raw']
        stamp = frame['follower_time']
        if motion_output is not None:
            samples.append({'phase': phase, 'time_monotonic': stamp,
                            'positions': actual, 'goals': frame['goal_raw']})
        if returning:
            complete = return_track.settled
            return
        if phase == 'approach' and forward.motion_origin is not None and (
                not traveled or stamp - traveled[-1][0] >= .04):
            traveled.append((stamp, actual))
        if forward.at_end:
            if still_pose is None or gap_degrees(still_pose, actual) > 1.5:
                still_pose, still_since = actual, stamp
            if read_deadline is None:
                pose_gap = gap_degrees(scan_goal, actual)
                end_started = stamp if end_started is None else end_started
                if stamp - still_since < .25:
                    if stamp - end_started >= 6:
                        future.set_exception(ValueError('扫描位未能停稳，从臂已返回休息位。'))
                        begin_return(stamp, actual)
                    return
                holding = frame['goal_raw']
                phase = 'hold'
                print('已到 T 区扫描位，停留识别。', flush=True)
                on_event('observe_targets')
                read_deadline = monotonic() + read_timeout
                Thread(target=read_targets, daemon=True, name='replay-target-camera').start()
            elif future.done() or monotonic() >= read_deadline:
                timed_out = not future.done()
                cancel_observation.set()
                begin_return(stamp, actual)

    def begin_return(stamp, actual):
        nonlocal returning, phase, return_track
        if len(traveled) >= 2:
            if stamp > traveled[-1][0]:
                traveled.append((stamp, actual))
            rows = [{'command_time': (stamp - time) * SCAN_TIME_SCALE, 'goal_raw': pose, 'follower_raw': pose}
                    for time, pose in reversed(traveled)]
            rows[-1]['goal_raw'] = rest_goal
            return_track = ReplayTrajectory.from_rows(forward.path, '按实测轨迹回休息位',
                                                      rows, calibration, REPLAY_DEFAULT_RATE, SCAN_SPEED)
            return_track.trim_wait()
            return_track.endpoint_tolerance = backward.endpoint_tolerance
            return_track.follow_feedback = True
            return_track.minimum_feedback_pace = SCAN_MINIMUM_PACE
        phase = 'return'
        print('扫描停留结束，回休息位。', flush=True)
        on_event('scan_return')
        returning = True

    def read_targets():
        try:
            future.set_result(observe_targets(cancelled=cancel_observation.is_set))
        except BaseException as error:
            future.set_exception(error)

    def stop():
        if cancelled():
            raise KeyboardInterrupt('用户中止自动 Replay。')
        return complete

    try:
        print('从臂前往 T 区扫描位。', flush=True)
        on_event('scan_approach')
        Teleoperator(None, follower, None, calibration, SCAN_SPEED).run(
            target_source=target_source, stop=stop, on_frame=on_frame, tracking_progress=True,
        )
    finally:
        cancel_observation.set()
        if motion_output is not None:
            final = samples[-1]['positions'] if samples else None
            save_json(motion_output, {
                'scan_episode': str(forward.path), 'returned_to_rest': complete,
                'rest_goal_raw': rest_goal, 'last_feedback_raw': final,
                'rest_gap_degrees': round(gap_degrees(final, rest_goal), 3) if final else None,
                'scan_pose_gap_degrees': round(pose_gap, 3) if pose_gap is not None else None,
                'samples': samples,
            })
    if not complete:
        raise RuntimeError('扫描示范未回到休息位。')
    if timed_out:
        raise ScanReadError('腕部相机识别超时，从臂已返回休息位。')
    try:
        return future.result()
    except ValueError as error:
        raise ScanReadError(str(error)) from error


def run_auto(site_path, preview, observe, *, on_event=ignore_event, cancelled=None,
             monitor=None, scan_only=False, global_targets_only=False, reviewed_targets=None, carry_rate=RATE,
             observe_only=False):
    if observe_only and not preview:
        raise ValueError('仅采集任务数据请使用识别预览。')
    if not math.isfinite(carry_rate) or not .1 <= carry_rate <= 1:
        raise ValueError('搬运回放倍率须为 0.1–1。')
    site = read_json(site_path)
    legacy_replay = site.get('legacy_replay') is True
    if legacy_replay:
        carry_rate = REPLAY_DEFAULT_RATE
    carry_speed = REPLAY_DEFAULT_SPEED if legacy_replay else SPEED
    site_dir = site_path.resolve().parent
    source, targets = site['source'], site['targets']
    global_targets = site.get('global_targets')
    quantity_rule = site['quantity_rule']
    if not scan_only and not source.get('config'):
        raise ValueError('自动执行需要 S 区八面标定文件和空台面参考。')
    if targets.get('rotation') not in (0, 90, 180, 270) and not targets.get('plane'):
        raise ValueError('目标相机需要现场已确认的 rotation 或定向标记平面。')
    if cancelled is None and not preview and not sys.stdin.isatty():
        raise ValueError('请从 Windows 交互终端启动自动回放，以便用 Esc 或空格停止。')
    cancelled = cancelled or keyboard_stop

    def check_cancel():
        if cancelled():
            raise KeyboardInterrupt('用户中止自动 Replay。')

    calibration_path = site.get('follower_calibration')
    if not preview and not calibration_path:
        raise ValueError('实机回放须在现场配置中填写从臂校准文件。')
    if not preview and (not site.get('follower_port') or site['follower_port'] == 'COM_PORT'):
        raise ValueError('实机回放须在现场配置中填写实际从臂串口。')
    calibration = load_calibration(site_dir / calibration_path) if calibration_path else None
    if site.get('scan_episode') and calibration is None:
        raise ValueError('扫描示范需要配置从臂校准文件。')
    scan = load_scan((site_dir / site['scan_episode']).resolve(), calibration) if site.get('scan_episode') else None
    if scan_only and scan is None:
        raise ValueError('仅扫描需要现场配置中的 scan_episode。')
    if global_targets_only and (not global_targets or scan_only or reviewed_targets is not None):
        raise ValueError('全局读卡需要已配置全局 T 区相机，并单独选择该模式。')
    if reviewed_targets is not None:
        if scan is None:
            raise ValueError('使用已核对的任务卡仍需录制的休息位。')
        checked_cards = read_json(reviewed_targets)
        validate_targets(checked_cards, quantity_rule)
    data, excluded_paths = data_and_exclusions(site_path, site)
    run_dir = (ROOT / 'outputs' / 'koch_replay_auto' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    print(f'自动 Replay 记录：{run_dir}', flush=True)
    def observe_targets(*, cancelled=None, seconds=None):
        settings = targets if seconds is None else {**targets, 'seconds': min(seconds, targets.get('seconds', seconds))}
        return read_live('targets', settings, site_dir, quantity_rule,
                         run_dir / 'targets.json', observe, cancelled=cancelled)

    def observe_global_targets(wrist_error=None):
        on_event('observe_global_targets', wrist_error=wrist_error)
        settings = global_targets
        if (source_before and source.get('camera_name') and source['camera_name'] == global_targets.get('camera_name')
                and not global_targets.get('roi') and not global_targets.get('plane')):
            polygon = source_before.get('alignment', {}).get('source_region_normalized')
            if polygon is not None:
                settings = {**settings, 'excluded_source_region': polygon}
        report = read_live('targets', settings, site_dir, quantity_rule,
                           run_dir / 'targets-global.json', observe, cancelled=cancelled)
        report.update(target_camera_mode='global', wrist_scan_error=wrist_error)
        save_json(run_dir / 'targets-global.json', report)
        return report

    def observe_source(file):
        report = read_live('source', source, site_dir, quantity_rule, run_dir / file, observe, cancelled=cancelled)
        if report.get('empty_reference_used') is not True:
            raise ValueError('S 区未使用同视角空台面参考，无法验证空层。')
        return report

    @contextmanager
    def open_follower():
        on_event('connecting')
        check_cancel()
        follower = connect_bus('follower', site['follower_port'], factory=make_bus)
        try:
            check_cancel()
            yield follower
        finally:
            follower.disconnect(disable_torque=False)

    def execute(follower):
        if reviewed_targets is not None:
            task = {'targets': checked_cards, 'source': 'operator_reviewed'}
            save_json(run_dir / 'targets.json', task)
        elif global_targets_only:
            task = observe_global_targets()
        elif scan is not None and follower is not None:
            try:
                task = scan_and_return(follower, calibration, scan, partial(observe_targets, seconds=SCAN_READ_SECONDS),
                                       cancelled, on_event, motion_output=run_dir / 'scan-motion.json')
            except ScanReadError as error:
                if scan_only or not global_targets:
                    raise
                task = observe_global_targets(str(error))
        else:
            on_event('observe_targets')
            task = observe_targets(cancelled=cancelled)
        on_event('target_report', report=task)
        if scan_only:
            print('四角任务卡已识别，从臂已回到休息位。', flush=True)
            on_event('scan_complete', directory=str(run_dir))
            return None
        task_input = {'inventory': inventory, 'targets': task['targets']}
        if observe_only:
            return task_input
        on_event('preparing_plan')
        home = scan[1].rows[-1]['goal_raw'] if scan else None
        tolerance = AUTO_ENDPOINT_TOLERANCE if legacy_replay or reviewed_targets is None else 2.
        initial_pose = None
        if follower is not None and (reviewed_targets is not None or task.get('target_camera_mode') == 'global'):
            initial_pose = stable_positions(follower)
            if home is not None and not legacy_replay and gap_degrees(initial_pose, home) > REPLAY_START_ALIGNMENT_DEGREES:
                raise ValueError('从臂未处于已录制的休息位范围，停止搬运。')
        plan, trajectories = prepare_chain(data, task_input, quantity_rule, calibration, home=home,
                                           trim=True, endpoint_tolerance=tolerance, initial_pose=initial_pose,
                                           carry_rate=carry_rate, observed_body=not legacy_replay,
                                           legacy_replay=legacy_replay, latest_only=True, allow_older_fallback=True,
                                           excluded_paths=excluded_paths)
        remaining = {slot: list(stack) for slot, stack in inventory.items()}
        for move in plan['moves']:
            remaining[move['source']].pop()
        save_json(run_dir / 'plan.json', {**plan, 'expected_remaining': remaining,
                   'target_camera_mode': task.get('target_camera_mode'),
                   'carry_rate': carry_rate,
                   'carry_speed_degrees_per_second': carry_speed,
                   'episode_selection': ('latest_preferred_compatible_fallback' if plan.get('older_episode_steps')
                                         else 'latest_available'),
                   'carry_joint_source': ('recorded_command' if legacy_replay
                                          else 'recorded_follower_position_with_gripper_goal'),
                   'episodes': [str(tr.path) for tr in trajectories],
                   'trimmed_seconds': [round(getattr(tr, 'trimmed_seconds', 0), 3) for tr in trajectories]})
        on_event('plan', moves=plan['moves'], older_episode_steps=plan.get('older_episode_steps', []))
        check_cancel()
        if preview:
            print(f'已生成 {len(trajectories)} 步计划，未连接从臂。', flush=True)
            on_event('preview')
            return None
        if trajectories:
            return_home = None
            if home is not None and not legacy_replay:
                rows = [{'goal_raw': home, 'follower_raw': home, 'command_time': t} for t in (0., .05)]
                return_home = ReplayTrajectory.from_rows(scan[0].path, '回休息位', rows, calibration, RATE, SPEED)
                return_home.align_start = True
                return_home.endpoint_tolerance = 2.5
            chain = ReplayChain(trajectories, on_event, cancelled, return_home=return_home)
            with nullcontext(follower) if follower is not None else open_follower() as bus:
                with monitor(source, site_dir) if monitor is not None else nullcontext():
                    Teleoperator(None, bus, None, calibration, carry_speed).run(
                        target_source=chain, stop=chain.stop, on_frame=chain.accept_frame,
                    )
            if chain.completed != len(trajectories):
                raise RuntimeError('路线未全部执行。')
        return remaining

    check_cancel()
    source_before = None
    if not scan_only:
        on_event('observe_source')
        source_before = observe_source('source-before.json')
        inventory = source_before['inventory']
        validate_inventory(inventory)
        if not any(inventory.values()):
            print('S 区已空，没有执行动作。', flush=True)
            on_event('empty')
            return
    if preview:
        return execute(None)
    if scan is not None:
        with open_follower() as follower:
            expected = execute(follower)
    else:
        expected = execute(None)
    if scan_only:
        return
    check_cancel()
    on_event('checking_empty')
    remaining = observe_source('source-after.json')['inventory']
    if remaining != expected:
        raise ValueError(f'路线已全部执行，但 S 区实测库存 {remaining} 和计划剩余 {expected} 不符。')
    print(f'回放已播完，S 区剩余库存与计划一致：{remaining}。请核对 T 区放置。', flush=True)
    on_event('complete')
