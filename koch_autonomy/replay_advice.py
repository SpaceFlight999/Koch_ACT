"""Suggest physical rearrangements that have a complete recorded Replay chain."""
from copy import copy
from functools import lru_cache
from itertools import combinations, product

from teleoperate_koch import load_calibration

from .planner import SLOTS, validate_inventory, validate_targets
from .replay_auto import load_scan, prepare_chain
from .replay_bridge import read_trajectory
from .replay_policy import data_and_exclusions
from .reverse import read_json


COLOR_LABELS = {'red': '红', 'green': '绿', 'yellow': '黄', 'pink': '粉', 'blue': '蓝'}


def color_label(color):
    return COLOR_LABELS.get(color, color)


def layout_swaps(inventory):
    positions = [(slot, i) for slot in SLOTS for i in range(len(inventory[slot]))]
    for (a, i), (b, j) in combinations(positions, 2):
        ca, cb = inventory[a][i], inventory[b][j]
        if ca == cb:
            continue
        candidate = {slot: stack.copy() for slot, stack in inventory.items()}
        candidate[a][i], candidate[b][j] = cb, ca
        if a == b:
            yield (f'把 {a} 的上、下层方块互换。',), candidate
            continue
        lift = [f'{slot} 上层{color_label(inventory[slot][-1])}块' for slot, layer in ((a, i), (b, j))
                if layer < len(inventory[slot]) - 1]
        instruction = f'交换 {a} 第 {i + 1} 层{color_label(ca)}块和 {b} 第 {j + 1} 层{color_label(cb)}块。'
        steps = (f'先临时拿起 {"和".join(lift)}。', instruction,
                 '把临时拿起的上层方块放回原位。') if lift else (instruction,)
        yield steps, candidate


def spare_replacements(inventory, colors, count):
    occupied = [slot for slot in SLOTS if inventory[slot]]
    for slots in combinations(occupied, count):
        options = ([color for color in colors if color != inventory[slot][-1]] for slot in slots)
        for replacements in product(*options):
            candidate = {slot: stack.copy() for slot, stack in inventory.items()}
            steps = []
            for slot, color in zip(slots, replacements):
                layer, prior = len(inventory[slot]), inventory[slot][-1]
                steps.append(f'把 {slot} 第 {layer} 层{color_label(prior)}块换成备用{color_label(color)}块。')
                candidate[slot][-1] = color
            yield tuple(steps), candidate


def suggest_layouts(site_path, task, cancelled=lambda: False):
    site = read_json(site_path)
    base = site_path.resolve().parent
    data, excluded = data_and_exclusions(site_path, site)
    calibration = load_calibration(base / site['follower_calibration'])
    scan = load_scan(base / site['scan_episode'], calibration) if site.get('scan_episode') else None
    home = scan[1].rows[-1]['goal_raw'] if scan else None
    legacy = site.get('legacy_replay') is True
    inventory = task['inventory']
    validate_inventory(inventory)
    colors, demand = validate_targets(task['targets'], site['quantity_rule'])
    required_colors = tuple(dict.fromkeys(color for color, count in zip(colors, demand) if count))

    @lru_cache(None)
    def load(path, route, options):
        return read_trajectory(path, route, calibration, **dict(options))

    def reuse(path, route, _, **options):
        return copy(load(path, route, tuple(sorted(options.items()))))

    def check(placing, fallback=False):
        if cancelled():
            raise KeyboardInterrupt
        try:
            plan, _ = prepare_chain(data, {**task, 'inventory': placing}, site['quantity_rule'], calibration,
                                    home=home, trim=True, legacy_replay=legacy, observed_body=not legacy,
                                    latest_only=True, allow_older_fallback=fallback, excluded_paths=excluded,
                                    load_trajectory=reuse)
        except ValueError:
            return None
        return plan

    def option(title, instructions, placing, plan):
        return {'title': title, 'instructions': instructions, 'inventory': placing,
                'moves': plan['moves'], 'older_episode_steps': plan.get('older_episode_steps', [])}

    def cheapest(title, placements):
        selected = None
        for instructions, placing in placements:
            plan = check(placing)
            if plan is not None and (selected is None or plan['cost'] < selected[0]):
                selected = plan['cost'], option(title, instructions, placing, plan)
        return selected[1] if selected else None

    suggestions = []
    current = check(inventory, fallback=True)
    if current:
        suggestions.append(option('沿用当前摆法', ('保持 S 区方块原位。',), inventory, current))
    swap = cheapest('交换现有方块', layout_swaps(inventory))
    if swap:
        suggestions.append(swap)
    for count in (1, 2):
        spare = cheapest('更换备用方块', spare_replacements(inventory, required_colors, count))
        if spare:
            suggestions.append(spare)
            break
    return suggestions
