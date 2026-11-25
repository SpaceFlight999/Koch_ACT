"""Run one selected legacy episode and record joint tracking feedback."""
from datetime import datetime
from time import monotonic

from koch_robot.dataset import REPLAY_DEFAULT_RATE, REPLAY_DEFAULT_SPEED
from koch_robot.teleop import NAMES, TICK_DEGREES, Teleoperator, connect_bus, load_calibration, make_bus

from .planner import SLOTS, TARGETS
from .replay_auto import ReplayChain, configure_legacy_alignment, eligible_episodes, gap_degrees
from .replay_bridge import LAYER_DIRS, data_and_exclusions, read_trajectory
from .reverse import ROOT, read_json, save_json


def select_episode(site_path, source, target, pair):
    if source not in SLOTS or target not in TARGETS or pair not in LAYER_DIRS:
        raise ValueError('请选择有效的 S/T 位置和层位。')
    site = read_json(site_path)
    data, excluded = data_and_exclusions(site_path, site)
    route = f'{source}_{target}'
    episodes = eligible_episodes(data, pair, route, excluded)
    if not episodes:
        raise ValueError(f'{source} 第 {pair[0]} 层到 {target} 第 {pair[1]} 层没有可用录制。')
    return episodes[-1]


def run_single(site_path, source, target, pair, episode, cancelled, publish):
    if select_episode(site_path, source, target, pair) != episode:
        raise ValueError('所选录制已更新，请在面板上重新选择。')
    site = read_json(site_path)
    calibration = load_calibration(site_path.parent / site['follower_calibration'])
    route = f'{source}_{target}'
    trajectory = configure_legacy_alignment(read_trajectory(
        episode, route, calibration, rate=REPLAY_DEFAULT_RATE, speed=REPLAY_DEFAULT_SPEED), trim=True)
    path = ROOT / 'outputs/koch_replay_auto' / datetime.now().strftime('%Y%m%d_%H%M%S_%f') / 'single-replay.json'
    report = {'episode': str(episode), 'route': route, 'layer_pair': pair, 'outcome': 'incomplete',
              'rate': REPLAY_DEFAULT_RATE, 'speed_degrees_per_second': REPLAY_DEFAULT_SPEED, 'samples': []}
    peak = dict.fromkeys(NAMES, 0.)
    last_update = 0.
    previous_goal = None
    chain = ReplayChain([trajectory], publish, cancelled)

    def feedback(frame):
        nonlocal previous_goal, last_update
        if not frame['sent']:
            return
        actual, goal = frame['follower_raw'], frame['goal_raw']
        compared_goal = previous_goal or actual
        errors = {joint: abs(compared_goal[joint] - actual[joint]) * TICK_DEGREES for joint in NAMES}
        previous_goal = goal
        for joint, error in errors.items():
            peak[joint] = max(peak[joint], error)
        stamp = monotonic()
        if stamp - last_update >= .2:
            last_update = stamp
            sample = {'time_monotonic': stamp, 'feedback_raw': actual, 'goal_raw': goal,
                      'compared_goal_raw': compared_goal,
                      'error_degrees': {joint: round(error, 2) for joint, error in errors.items()}}
            report['samples'].append(sample)
            publish('joint_feedback', errors=sample['error_degrees'], peak=max(peak.values()))
        chain.accept_frame(frame)

    def targets(elapsed, actual, previous):
        if 'start_gap_degrees' not in report:
            report['start_gap_degrees'] = round(gap_degrees(actual, trajectory.rows[0]['goal_raw']), 2)
            publish('start_gap', degrees=report['start_gap_degrees'])
        return chain(elapsed, actual, previous)

    publish('plan', moves=[{'source': source, 'target': target, 'source_layer': pair[0],
                            'target_layer': pair[1], 'color': '现场', 'route': route}])
    publish('connecting')
    if cancelled():
        raise KeyboardInterrupt
    follower = connect_bus('follower', site['follower_port'], factory=make_bus)
    try:
        Teleoperator(None, follower, None, calibration, REPLAY_DEFAULT_SPEED).run(
            target_source=targets, stop=chain.stop, on_frame=feedback)
        if chain.completed != 1:
            raise RuntimeError('单条 Replay 未播到终点。')
        report['outcome'] = 'trajectory_completed'
    except BaseException as error:
        report['error'] = str(error) or type(error).__name__
        raise
    finally:
        try:
            follower.disconnect(disable_torque=False)
        finally:
            report['peak_error_degrees'] = {joint: round(error, 2) for joint, error in peak.items()}
            save_json(path, report)
            publish('single_log', path=str(path))
    publish('single_complete', path=str(path))
