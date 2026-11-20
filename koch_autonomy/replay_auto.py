"""Run a whole visual task in one follower Replay session."""
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys

from koch_dataset import REPLAY_START_ALIGNMENT_DEGREES
from teleoperate_koch import NAMES, TICK_DEGREES, Teleoperator, connect_bus, keyboard_stop, load_calibration, make_bus

from .planner import validate_inventory
from .replay_bridge import DATA_ROOT, LAYER_DIRS, SPEED, motion_plan, read_trajectory, route_episodes
from .reverse import ROOT, read_json, save_json


def gap_degrees(first, second):
    return max(abs((first[name] - second[name] + 2048) % 4096 - 2048) for name in NAMES) * TICK_DEGREES


def prepare_chain(data, task, quantity_rule, calibration):
    plan = motion_plan(task, quantity_rule)
    moves = plan['moves']
    if len(moves) != sum(map(len, task['inventory'].values())):
        raise ValueError('任务卡需求小于 S 区库存，不能同时满足任务数量并清空 S 区；请调整卡片或库存。')
    choices = []
    for move in moves:
        pair, route = tuple(move['layer_pair']), move['route']
        episodes = route_episodes(data, pair, route)
        if not episodes:
            raise ValueError(f'缺少已确认的动作：{LAYER_DIRS[pair]}/{route}。')
        choices.append([read_trajectory(episode, route, calibration) for episode in reversed(episodes)])

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
                if gap <= REPLAY_START_ALIGNMENT_DEGREES:
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
    def __init__(self, trajectories, on_event=ignore_event, cancelled=keyboard_stop):
        self.trajectories = trajectories
        self.on_event, self.cancelled = on_event, cancelled
        self.completed = 0
        self.origin = 0.0
        self.next_start = False
        self.announced = False

    def __call__(self, elapsed, actual, previous):
        if self.next_start:
            self.origin = elapsed
            self.next_start = False
        return self.trajectories[self.completed](elapsed - self.origin, actual, previous)

    def accept_frame(self, frame):
        trajectory = self.trajectories[self.completed]
        if not frame['sent']:
            return
        if not self.announced:
            self.announced = True
            self.on_event('move_started', step=self.completed + 1)
        if not trajectory.settled:
            return
        self.completed += 1
        self.next_start = True
        self.announced = False
        self.on_event('move_finished', step=self.completed)
        print(f'Replay {self.completed}/{len(self.trajectories)} 已到末帧：{trajectory.meta["route"]}', flush=True)

    def stop(self):
        if self.cancelled():
            raise KeyboardInterrupt('用户中止自动 Replay。')
        return self.completed == len(self.trajectories)


def read_live(kind, specification, site_dir, quantity_rule, output, observe):
    def file(key):
        return (site_dir / specification[key]).resolve() if specification.get(key) else None

    report = observe(SimpleNamespace(
        kind=kind, config=file('config'), camera=specification.get('camera'),
        camera_name=specification.get('camera_name'), rotation=specification.get('rotation'), target=None,
        plane=file('plane'), roi=specification.get('roi'), seconds=specification.get('seconds', 30),
        quantity_rule=quantity_rule, image=None, show=False, output=output,
    ))
    if report.get('complete') is not True or report.get('stable') is not True:
        raise ValueError(f'{kind} 未得到完整且稳定的识别结果。')
    return report


def run_auto(site_path, preview, observe, *, on_event=ignore_event, cancelled=None,
             monitor=None):
    site = read_json(site_path)
    site_dir = site_path.resolve().parent
    source, targets = site['source'], site['targets']
    quantity_rule = site['quantity_rule']
    if not source.get('config'):
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
    data = (site_dir / site['data']).resolve() if 'data' in site else DATA_ROOT
    run_dir = (ROOT / 'outputs' / 'koch_replay_auto' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    print(f'自动 Replay 记录：{run_dir}', flush=True)
    check_cancel()
    on_event('observe_source')
    start = read_live('source', source, site_dir, quantity_rule, run_dir / 'source-before.json', observe)
    if start.get('empty_reference_used') is not True:
        raise ValueError('S 区未使用空台面参考，不能判断清空。')
    inventory = start['inventory']
    if not any(validate_inventory(inventory)):
        print('S 区已空，没有执行动作。', flush=True)
        on_event('empty')
        return
    check_cancel()
    on_event('observe_targets')
    task = read_live('targets', targets, site_dir, quantity_rule, run_dir / 'targets.json', observe)
    on_event('target_report', report=task)
    plan, trajectories = prepare_chain(data, {'inventory': inventory, 'targets': task['targets']},
                                       quantity_rule, calibration)
    save_json(run_dir / 'plan.json', {**plan, 'episodes': [str(trajectory.path) for trajectory in trajectories]})
    on_event('plan', moves=plan['moves'])
    check_cancel()
    if preview:
        print(f'已生成 {len(trajectories)} 步计划，未连接从臂。', flush=True)
        on_event('preview')
        return
    on_event('connecting')
    chain = ReplayChain(trajectories, on_event, cancelled)
    with monitor(source, site_dir) if monitor is not None else nullcontext():
        check_cancel()
        follower = connect_bus('follower', site['follower_port'], factory=make_bus)
        try:
            check_cancel()
            Teleoperator(None, follower, None, calibration, SPEED).run(
                target_source=chain, stop=chain.stop, on_frame=chain.accept_frame,
            )
        finally:
            follower.disconnect(disable_torque=False)
    if chain.completed != len(trajectories):
        raise RuntimeError('路线未全部执行。')
    check_cancel()
    on_event('checking_empty')
    end = read_live('source', source, site_dir, quantity_rule, run_dir / 'source-after.json', observe)
    remaining = end['inventory']
    if end.get('empty_reference_used') is not True or any(validate_inventory(remaining)):
        raise ValueError(f'路线已全部执行，但没有确认 S 区清空：{remaining}；不自动重复执行。')
    print(f'任务完成：{chain.completed} 条 Replay 已播完，S 区已确认清空。', flush=True)
    on_event('complete')
