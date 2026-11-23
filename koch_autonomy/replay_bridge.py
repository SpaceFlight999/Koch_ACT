"""Resolve reviewed vision decisions to the project's existing joint Replay."""
from collections import defaultdict
from hashlib import file_digest
from pathlib import Path
from types import SimpleNamespace
import sys

from .planner import SLOTS, TARGETS, plan_task
from .reverse import ROOT, read_json, save_json

DATA_ROOT = ROOT / 'datasets'
LAYER_DIRS = {
    (1, 1): 'koch_st',
    (1, 2): 'koch_st_layer1_to_layer2',
    (2, 1): 'koch_st_layer2_to_layer1',
    (2, 2): 'koch_st_layer2_to_layer2',
}
RATE, SPEED = .5, 60


def fingerprint(episode):
    result = {}
    for name in ('episode.json', 'joints.jsonl'):
        with (episode / name).open('rb') as stream:
            result[name] = file_digest(stream, 'sha256').hexdigest()
    return result


def read_trajectory(episode, route, calibration=None, *, rate=RATE, speed=SPEED, observed_body=False):
    from koch_dataset import ReplayTrajectory
    from teleoperate_koch import TICK_DEGREES

    meta = read_json(episode / 'episode.json')
    if meta['route'].replace('-', '_') != route or episode.parent.name.replace('-', '_') != route:
        raise ValueError(f'{episode}: 路线目录或 episode.json 与 {route} 不一致。')
    if meta.get('units') != 'raw_encoder_ticks' or meta.get('tick_degrees') != TICK_DEGREES:
        raise ValueError(f'{episode}: 编码器单位与 Replay 不一致。')
    if 'joint_frames' in meta:
        with (episode / 'joints.jsonl').open('rb') as stream:
            frames = sum(bool(line.strip()) for line in stream)
        if frames != meta['joint_frames']:
            raise ValueError(f'{episode}: 关节帧数与录制元数据不一致。')
    return ReplayTrajectory(episode, meta['follower_calibration'] if calibration is None else calibration, rate, speed,
                            observed_body=observed_body)


def route_episodes(root, pair, route):
    group = root / LAYER_DIRS[pair]
    candidates = sorted(
        path for name in (route, route.replace('_', '-')) for path in (group / name).glob('*/episode.json')
    )
    complete = [path.parent for path in candidates if read_json(path).get('status') == 'complete']
    if len({path.name for path in complete}) != len(complete):
        raise ValueError(f'{group}: {route} 有重复采集版本。')
    return sorted(complete, key=lambda path: path.name)


def motion_plan(task, quantity_rule):
    return plan_task(task['inventory'], task['targets'], task.get('positions'),
                     edge_costs=task.get('edge_costs'), quantity_rule=quantity_rule)


def report_payload(report, path, field):
    if not isinstance(report, dict) or not isinstance(report.get(field), dict):
        raise ValueError(f'{path}: 缺少带现场编号的 {field}，请传入 observe 报告；普通九宫格结果不能直接用于 Replay。')
    if not report[field] or report.get('complete') is not True or report.get('stable') is not True:
        raise ValueError(f'{path}: 视觉识别未完成或未稳定。照片报告需先人工核对，再作为任务 JSON 提交。')
    return report[field]


def read_task(args):
    if args.input:
        if args.inventory or args.source_report:
            raise ValueError('--input 已包含库存与目标，不需要另一份库存。')
        return read_json(args.input)
    if not (args.inventory or args.source_report):
        raise ValueError('目标视觉报告需要 --source-report 或 --inventory 提供 S1–S4 库存。')
    target_report = read_json(args.target_report)
    if target_report.get('quantity_rule', args.quantity_rule) != args.quantity_rule:
        raise ValueError('目标识别和 Replay 规划的数量规则不同。')
    targets = report_payload(target_report, args.target_report, 'targets')
    inventory = (report_payload(read_json(args.source_report), args.source_report, 'inventory')
                 if args.source_report else read_json(args.inventory))
    return {'inventory': inventory, 'targets': targets}


def bind_replay(root, task, quantity_rule):
    plan = motion_plan(task, quantity_rule)
    moves = []
    for move in plan['moves']:
        pair, route = tuple(move['layer_pair']), move['route']
        episodes = route_episodes(root, pair, route)
        if not episodes:
            raise ValueError(f'缺少已确认的 Replay 数据：{LAYER_DIRS[pair]}/{route}。')
        episode = episodes[-1]
        trajectory = read_trajectory(episode, route)
        moves.append({
            **move, 'episode': episode.relative_to(root).as_posix(), 'sha256': fingerprint(episode),
            'duration_seconds': round(trajectory.duration, 3),
            'start_goal_raw': trajectory.rows[0]['goal_raw'],
        })
    return {'mode': 'replay_plan', 'hardware_executed': False, 'quantity_rule': quantity_rule,
            'task': task, 'required_counts': plan['required_counts'], 'moves': moves}


def audit(root):
    counts = defaultdict(int)
    for pair, group in LAYER_DIRS.items():
        for source in SLOTS:
            for target in TARGETS:
                route = f'{source}_{target}'
                episodes = route_episodes(root, pair, route)
                if not episodes:
                    raise ValueError(f'缺少已确认的 Replay 数据：{group}/{route}。')
                for episode in episodes:
                    read_trajectory(episode, route)
                    counts[group] += 1
    return dict(counts)


def run_step(root, bound, step, port=None, calibration=None):
    if bound.get('mode') != 'replay_plan':
        raise ValueError('请传入 replay-plan 生成的计划。')
    expected = motion_plan(bound['task'], bound['quantity_rule'])['moves']
    moves = bound['moves']
    if len(moves) != len(expected) or not 1 <= step <= len(moves):
        raise ValueError('计划搬运数量或步骤编号无效。')
    move, expected_move = moves[step - 1], expected[step - 1]
    if any(move.get(key) != value for key, value in expected_move.items()):
        raise ValueError('步骤的颜色、路线或层位与上层决策不一致。')
    pair, route = tuple(move['layer_pair']), move['route']
    episode = root / move['episode']
    if episode not in route_episodes(root, pair, route) or not episode.resolve().is_relative_to(root.resolve()):
        raise ValueError('选中的示范不属于该路线与层位。')
    if fingerprint(episode) != move['sha256']:
        raise ValueError('示范与计划绑定时的文件校验值不同，请重新生成计划。')
    if bool(port) != bool(calibration):
        raise ValueError('实机回放须同时提供 --follower-port 和 --follower-calibration。')
    active_cal = None
    if calibration:
        from teleoperate_koch import load_calibration
        active_cal = load_calibration(calibration)
    trajectory = read_trajectory(episode, route, active_cal)
    print(f'第 {step}/{len(moves)} 步 {move["color"]} | {move["source"]} 第 {pair[0]} 层 '
          f'至 {move["target"]} 第 {pair[1]} 层\n示范目录 {episode}\n'
          f'起点编码器 {trajectory.rows[0]["goal_raw"]}', flush=True)
    if not port:
        print('离线核对通过。')
        return
    from serial.tools.list_ports import comports
    if port.casefold() not in {item.device.casefold() for item in comports()}:
        raise ValueError(f'未枚举到从臂串口 {port}。')
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError('实机回放需要交互终端，以便确认起点和按键中止。')
    from collect_koch import replay
    print('确认当前 S/T 方块与这一步一致。完成后检查抓放和层位，再调用下一步。', flush=True)
    replay(SimpleNamespace(follower_port=port, follower_calibration=calibration, speed=SPEED, replay_rate=RATE), episode)


def configure_cli(sub):
    for name in ('replay-audit', 'replay-plan', 'replay-run'):
        parser = sub.add_parser(name)
        parser.add_argument('--data', type=Path, default=DATA_ROOT, help='包含四个 koch_st 系列目录的数据集根目录。')
        if name == 'replay-plan':
            task = parser.add_mutually_exclusive_group(required=True)
            task.add_argument('--input', type=Path, help='人工核对后的完整 inventory/targets JSON。')
            task.add_argument('--target-report', type=Path, help='observe targets 生成的完整稳定报告。')
            stock = parser.add_mutually_exclusive_group()
            stock.add_argument('--inventory', type=Path, help='现场手填 S1–S4，方块颜色从下往上。')
            stock.add_argument('--source-report', type=Path, help='observe source 生成的完整稳定报告。')
            parser.add_argument('--quantity-rule', required=True, choices=('parity', 'direct'))
            parser.add_argument('--output', type=Path, required=True)
        elif name == 'replay-run':
            parser.add_argument('--plan', type=Path, required=True)
            parser.add_argument('--step', type=int, required=True, help='计划中的步号，从 1 开始。')
            parser.add_argument('--follower-port')
            parser.add_argument('--follower-calibration', type=Path)


def run_cli(args):
    root = args.data.resolve()
    if args.command == 'replay-audit':
        counts = audit(root)
        print('\n'.join(f'{name}  16 条路线，{count} 条有效示范' for name, count in counts.items()))
    elif args.command == 'replay-plan':
        bound = bind_replay(root, read_task(args), args.quantity_rule)
        save_json(args.output, bound)
        print(f'已绑定 {len(bound["moves"])} 步 Replay。计划文件 {args.output.resolve()}')
    else:
        run_step(root, read_json(args.plan), args.step, args.follower_port, args.follower_calibration)
