"""Reviewable task protocol and deterministic offline execution verification."""
import copy
import math
import time

from .planner import SCAN_ORDER, plan_task, validate_inventory, validate_targets


def commands(plan):
    steps = [{'kind': 'observe_source', 'camera_role': 'global', 'requires_stable': True}]
    for target in SCAN_ORDER:
        steps.extend([{'kind': 'scan_move', 'target': target, 'gripper': 'hold',
                       'trajectory': f'scan_to_{target}', 'requires_calibrated_path': True},
                      {'kind': 'observe_target', 'target': target, 'camera_role': 'wrist', 'requires_stable': True}])
    for index, move in enumerate(plan['moves']):
        steps.extend([{'kind': 'move_to_source', 'move': index, **move,
                       'requires_verified_empty_gripper_transit': True},
                      {'kind': 'check_joint_start', 'move': index, **move},
                      {'kind': 'model_pick_place', 'move': index, **move,
                       'skills': ['gripper:open', 'approach_align', 'gripper:close', 'lift',
                                  'transfer', 'lower', 'release'], 'camera_role': 'global'},
                      {'kind': 'verify_pick_place', 'move': index, **move,
                       'requires_source_and_target_observation': True}])
    return steps


def commit_observed_move(inventory, placed, move, observed_inventory, observed_placed):
    """Advance only after exact post-move inventory evidence, not a timeout.

    Callers must obtain fresh stable independent observations. This pure function
    checks their contents and leaves prior state untouched on failure.
    """
    expected_s, expected_t = copy.deepcopy(inventory), copy.deepcopy(placed)
    source, target = move['source'], move['target']
    stack = expected_s[source]
    if not stack or len(stack) != move['source_layer'] or stack[-1] != move['color']:
        raise ValueError('夹取位置不是当前可取的最上层。')
    if len(expected_t[target]) + 1 != move['target_layer']:
        raise ValueError('放置层缺少下层支撑或已有物体。')
    expected_t[target].append(stack.pop())
    if observed_inventory != expected_s or observed_placed != expected_t:
        raise ValueError('实测库存变化不符合计划；暂停并重新识别，不能自动记为成功。')
    return expected_s, expected_t


def commit_from_reports(inventory, placed, move, source_report, target_report, completed_after, now=None, max_age=.6):
    """Accept only fresh, stable independent S/T stack observations after motion.

    No scheduler calls this on real hardware yet. Keeping this boundary separate
    prevents an eventual adapter from treating motion duration as task success.
    """
    now = time.monotonic() if now is None else now
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (completed_after, now, max_age)) or max_age <= 0:
        raise ValueError('观察时间及新鲜度上限无效。')
    for report, region in ((source_report, 'S'), (target_report, 'T')):
        stamp = report.get('captured_monotonic')
        start = report.get('stable_since_monotonic')
        required, matching = report.get('required_frames'), report.get('matching_frames')
        if (report.get('complete') is not True or report.get('stable') is not True or
                report.get('empty_reference_used') is not True or
                report.get('view_pose_verified') is not True or
                report.get('scope') != f'{region}_fixed_camera_visible_faces' or
                type(required) is not int or required < 2 or type(matching) is not int or matching < required or
                not all(type(v) in (int, float) and math.isfinite(v) for v in (start, stamp)) or
                not completed_after < start < stamp <= now or now - start > max_age):
            raise ValueError(f'{region} 区需要动作之后的新鲜稳定堆栈观测、已标定空位证据和固定视角检查。')
    return commit_observed_move(inventory, placed, move, source_report['inventory'], target_report['placed'])


def simulate(inventory, targets, positions=None, edge_costs=None):
    validate_inventory(inventory)
    validate_targets(targets)
    plan = plan_task(inventory, targets, positions=positions, edge_costs=edge_costs)
    current, placed = copy.deepcopy(inventory), {t: [] for t in targets}
    events = []
    for move in plan['moves']:
        observed_s, observed_t = copy.deepcopy(current), copy.deepcopy(placed)
        observed_t[move['target']].append(observed_s[move['source']].pop())
        current, placed = commit_observed_move(current, placed, move, observed_s, observed_t)
        events.append({'move': move, 'inventory': copy.deepcopy(current), 'placed': copy.deepcopy(placed),
                       'verification': 'synthetic_simulation_only'})
    return {'mode': 'offline_simulation', 'hardware_executed': False, 'plan': plan,
            'commands': commands(plan), 'events': events, 'remaining_inventory': current, 'placed': placed,
            'live_blockers': ['八块独立侧面及目标编号的现场标定', '安全高度扫描路径及腕部视野验证',
                              '空爪 T→S 连续过渡示范及验收', '抓取/放置后的实时视觉闭环验收']}
