"""Exact small-state planner: source top first, destination bottom first."""
from collections import Counter
from functools import lru_cache
import math
import re

SLOTS = tuple(f'S{i}' for i in range(1, 5))
TARGETS = tuple(f'T{i}' for i in range(1, 5))
SCAN_ORDER = ('T1', 'T2', 'T4', 'T3')


def validate_inventory(inventory):
    if not isinstance(inventory, dict) or set(inventory) != set(SLOTS):
        raise ValueError('库存必须包含 S1–S4，数组顺序为第一层、第二层。')
    result = []
    for slot in SLOTS:
        stack = inventory[slot]
        if not isinstance(stack, list) or len(stack) > 2:
            raise ValueError(f'{slot} 必须是最多两层的颜色数组。')
        if any(not isinstance(c, str) or not c.strip() or c in ('unknown', '?') for c in stack):
            raise ValueError(f'{slot} 存在未知颜色，不能规划。')
        result.append(tuple(stack))
    return tuple(result)


def validate_targets(targets, quantity_rule='parity'):
    if not isinstance(targets, dict) or set(targets) != set(TARGETS):
        raise ValueError('任务必须包含 T1–T4。')
    if quantity_rule not in ('parity', 'direct'):
        raise ValueError('数量规则必须为 parity 或 direct。')
    colors, numbers, demand = [], [], []
    for slot in TARGETS:
        item = targets[slot]
        if not isinstance(item, dict):
            raise ValueError(f'{slot} 必须是包含 color 和 number 的对象。')
        color, number = item.get('color'), item.get('number')
        if not isinstance(color, str) or not color.strip() or color in ('unknown', '?'):
            raise ValueError(f'{slot} 颜色未知。')
        if isinstance(number, bool) or not re.fullmatch(r'[0-9]+', str(number)):
            raise ValueError(f'{slot} 数字必须是非负整数。')
        number = int(number)
        colors.append(color)
        numbers.append(number)
        if quantity_rule == 'direct' and number > 2:
            raise ValueError(f'{slot} 要求放 {number} 块，当前 Replay 库只覆盖目标第一层和第二层。')
        demand.append((1 if number % 2 else 2) if quantity_rule == 'parity' else number)
    if quantity_rule == 'parity' and (len(set(colors)) != 4 or len(set(numbers)) != 4):
        raise ValueError('四张目标彩纸须颜色各异、数字各异；不猜测重复识别结果。')
    return tuple(colors), tuple(demand)


def plan_task(inventory, targets, positions=None, start='T3', edge_costs=None, *, quantity_rule='parity'):
    """Minimize sum of supplied directed costs, or 2D distance as an explicit proxy.

    Only requested deliveries are legal. An unwanted blocking block requires a
    separately defined buffer; inventing a buffer position is not safe.
    """
    stacks = validate_inventory(inventory)
    colors, demand = validate_targets(targets, quantity_rule)
    available = Counter(c for stack in stacks for c in stack)
    requested = Counter()
    for color, count in zip(colors, demand):
        requested[color] += count
    missing = dict(requested - available)
    if missing:
        raise ValueError(f'颜色库存不足：{missing}')
    nodes = set(SLOTS + TARGETS)
    if start not in nodes:
        raise ValueError('规划起点须为 S1–S4 或 T1–T4。')
    if positions is None:
        positions = {**{s: [i % 2, i // 2] for i, s in enumerate(SLOTS)},
                     **{t: [3 + i % 2, i // 2] for i, t in enumerate(TARGETS)}}
        metric = 'layout_distance_proxy_not_physical_optimum'
    else:
        metric = 'calibrated_xy_distance_not_collision_checked'
    if not isinstance(positions, dict) or set(positions) != nodes or any(
            not isinstance(p, (list, tuple)) or len(p) != 2 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in p) for p in positions.values()):
        raise ValueError('位置须完整包含八个 S/T 点，每点两个有限坐标。')
    if edge_costs is not None:
        if not isinstance(edge_costs, dict):
            raise ValueError('路径成本必须是有向边到数值的对象。')
        metric = 'provided_directed_edge_cost'
        required = {(a, b) for a in nodes for b in nodes if a[0] != b[0]}
        if any(f'{a}->{b}' not in edge_costs for a, b in required):
            raise ValueError('路径成本须包含全部 S→T 和 T→S 有向边。')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
               for v in edge_costs.values()):
            raise ValueError('路径成本须是有限非负数。')

    def cost(a, b):
        if a == b:
            return 0.
        if edge_costs is not None:
            if f'{a}->{b}' not in edge_costs:
                raise ValueError(f'缺少路径成本 {a}->{b}')
            return float(edge_costs[f'{a}->{b}'])
        return math.dist(positions[a], positions[b])

    @lru_cache(None)
    def solve(heights, remaining, current):
        if not any(remaining):
            return 0., ()
        best = (math.inf, ())
        for si, height in enumerate(heights):
            if height == 0:
                continue
            color = stacks[si][height - 1]
            for ti, count in enumerate(remaining):
                if not count or colors[ti] != color:
                    continue
                hs, rs = list(heights), list(remaining)
                hs[si] -= 1
                rs[ti] -= 1
                rest, actions = solve(tuple(hs), tuple(rs), TARGETS[ti])
                total = cost(current, SLOTS[si]) + cost(SLOTS[si], TARGETS[ti]) + rest
                if total < best[0]:
                    action = (si, height, ti, demand[ti] - count + 1, color)
                    best = (total, (action,) + actions)
        return best

    total, actions = solve(tuple(map(len, stacks)), demand, start)
    if not math.isfinite(total):
        raise ValueError('库存数量足够，但所需下层被无法投递的上层挡住；需要明确的临时缓冲区及其搬运示范。')
    moves = [{'source': SLOTS[s], 'source_layer': sl, 'target': TARGETS[t],
              'target_layer': tl, 'color': c, 'route': f'{SLOTS[s]}_{TARGETS[t]}',
              'layer_pair': [sl, tl]} for s, sl, t, tl, c in actions]
    return {'moves': moves, 'cost': total, 'cost_metric': metric,
            'optimal_for_supplied_costs': True, 'collision_checked': False,
            'start': start, 'scan_order': list(SCAN_ORDER),
            'required_counts': dict(zip(TARGETS, demand)), 'search_states': solve.cache_info().currsize}
