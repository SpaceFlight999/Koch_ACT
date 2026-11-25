"""从已保留的完整示范中另提取提起、搬运、下放、松爪；不改写原数据。"""
from koch_robot.paths import project_path, read_project_json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from koch_act_pipeline.prepare_skills import ROOT, NAMES, jsonl, save_json, contiguous_runs, IMAGE_SIZE

STAGES = ('lift', 'transfer', 'lower', 'release')


def find_stages(times, goals, actual, fractions, actual_fractions, reach_end):
    contact = None
    for k in np.flatnonzero((times >= reach_end) & (fractions < .65) & (actual_fractions - fractions > .08)):
        last = int(np.searchsorted(times, times[k] + .25))
        if (last > k + 2 and last <= len(times) and times[-1] >= times[k] + .25
                and np.all(actual_fractions[k:last] - fractions[k:last] > .08)
                and np.ptp(actual_fractions[k:last]) < .06):
            contact = int(k)
            break
    if contact is None:
        raise ValueError('无法确认首次稳定夹持，需人工标注阶段')
    opening = np.flatnonzero((fractions[:-1] < .5) & (fractions[1:] >= .5)
                             & (times[1:] > times[contact] + 1)) + 1
    if not len(opening):
        raise ValueError('找不到搬运后的松爪阶段')
    open_mid = int(opening[0])
    starts = np.flatnonzero((fractions[:open_mid] < .3) & (fractions[1:open_mid + 1] >= .3)) + 1
    starts = starts[starts > contact]
    if not len(starts):
        raise ValueError('找不到松爪起点')
    release_start = int(starts[-1])
    pan_delta = goals[release_start, 0] - goals[contact, 0]
    if abs(pan_delta) < 200:
        raise ValueError('底座搬运角度不足以分阶段，需人工标注')
    progress = (goals[:, 0] - goals[contact, 0]) / pan_delta
    middle = np.arange(contact, release_start + 1)
    early = middle[progress[middle] >= .1]
    late = middle[progress[middle] < .9]
    if not len(early) or not len(late):
        raise ValueError('无法识别搬运主运动')
    lift_end, lower_start = int(early[0]), int(late[-1] + 1)
    closing = np.flatnonzero((fractions[:-1] >= .6) & (fractions[1:] < .6)
                             & (times[1:] > times[open_mid] + .3)) + 1
    release_limit = int(closing[0]) if len(closing) else len(times) - 1
    peak = float(np.max(fractions[open_mid:release_limit + 1]))
    ends = np.flatnonzero(fractions[open_mid:release_limit + 1] >= peak - .02) + open_mid
    release_end = min(release_limit, int(np.searchsorted(times, times[int(ends[0])] + .2)))
    indices = (contact, lift_end, lower_start, release_start, release_end)
    if any(b <= a or times[b] - times[a] < .2 for a, b in zip(indices, indices[1:])):
        raise ValueError('阶段时长不足或顺序异常，需人工标注')
    return indices


def prepare(base_path):
    base = read_project_json(project_path(base_path))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / 'outputs/koch_skills' / stamp
    cache = ROOT / 'datasets/koch_skills' / stamp
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    assets = {c['source']: c['images'] for s in base['skills'].values() for c in s['clips']}
    clips = {name: [] for name in STAGES}
    annotations, sheets = [], {}
    for segment in base['segments']:
        source, route = Path(segment['source']), segment['route']
        all_rows = jsonl(source / 'joints.jsonl')
        rows = [r for r in all_rows if r.get('sent') and r.get('command_time') is not None]
        times = np.array([r['command_time'] for r in rows])
        goals = np.array([[r['goal_raw'][n] for n in NAMES] for r in rows], dtype=np.float32)
        actual = np.array([[r['follower_raw'][n] for n in NAMES] for r in rows], dtype=np.float32)
        cal = base['follower_calibration']
        grip = cal['motor_names'].index('gripper')
        closed, opened = cal['start_pos'][grip], cal['end_pos'][grip]
        fraction = np.clip((goals[:, 5] - closed) / (opened - closed), 0, 1)
        actual_fraction = (actual[:, 5] - closed) / (opened - closed)
        boundaries = find_stages(times, goals, actual, fraction, actual_fraction, segment['reach_end'])
        camera_times = np.array([r['t'] for r in jsonl(source / 'camera_timestamps.jsonl')])
        feedback_times = np.array([r['follower_time'] for r in all_rows])
        feedback = np.array([[r['follower_raw'][n] for n in NAMES] for r in all_rows], dtype=np.float32)
        images = np.load(assets[str(source)], mmap_mode='r')
        pictures = []
        src_id, target_id = int(route[1]) - 1, int(route[-1]) - 1
        validation = src_id == target_id
        for number, name in enumerate(STAGES):
            first, last = float(times[boundaries[number]]), float(times[boundaries[number + 1]])
            grid = np.arange(first, last + 1e-6, 1 / base['fps'])
            ai = np.searchsorted(times, grid, side='right') - 1
            si = np.searchsorted(feedback_times, grid, side='right') - 1
            vi = np.searchsorted(camera_times, grid, side='right') - 1
            valid = ((ai >= 0) & (si >= 0) & (vi >= 0) & (grid <= camera_times[-1])
                     & (grid - times[ai] <= .2) & (grid - feedback_times[si] <= .2)
                     & (grid - camera_times[vi] <= .25))
            for part, run in enumerate(contiguous_runs(np.flatnonzero(valid))):
                if len(run) < 6:
                    continue
                states = feedback[si[run]]
                actions = goals[ai[run]]
                if name == 'release':
                    actions = actions[:, 5:6]
                else:
                    states, actions = states[:, :5], actions[:, :5]
                    code = np.eye(4, dtype=np.float32)[src_id if name == 'lift' else target_id]
                    states = np.concatenate((states, np.tile(code, (len(run), 1))), axis=1)
                path = cache / f'{route}_{source.name}_{name}_{part}.npz'
                np.savez(path, state=states, action=actions, image_index=vi[run], timestamp=grid[run])
                clips[name].append({'skill': name, 'source': str(source), 'route': route,
                                    'episode_id': f'{route}_{source.name}', 'intent': name,
                                    'split': 'validation' if validation else 'train', 'frames': len(run),
                                    'arrays': str(path.resolve()), 'images': assets[str(source)],
                                    'start': float(grid[run[0]]), 'end': float(grid[run[-1]])})
            image_index = max(0, min(len(images) - 1, np.searchsorted(camera_times, last, side='right') - 1))
            pictures.append(np.array(images[image_index]))
        images._mmap.close()
        annotation = {'source': str(source), 'route': route,
                      'times': dict(zip(('grasp_confirmed', 'lift_end', 'transfer_end', 'release_start', 'release_end'),
                                        [float(times[i]) for i in boundaries])),
                      'method': 'stable_gripper_contact_then_base_pan_10pct_90pct_and_release',
                      'manually_verified_in_hardware': False}
        annotations.append(annotation)
        sheets.setdefault(route.split('_')[0], []).append((route, pictures))
        print(route, {name: round(times[b] - times[a], 2) for name, a, b in zip(STAGES, boundaries, boundaries[1:])}, flush=True)
    manifest = dict(base)
    manifest.update(output=str(output), cache=str(cache), previous_manifest=str(project_path(base_path).resolve()),
                    stage_annotations=annotations,
                    validation_note='到达技能留整条T4示范；搬运四阶段留对角S1T1/S2T2/S3T3/S4T4，所有源/目标均有训练数据；不是实物成功率')
    manifest['skills'] = dict(base['skills'])
    manifest['model_dirs'] = dict(base.get('model_dirs', {}))
    for name in base['skills']:
        manifest['model_dirs'].setdefault(name, str(Path(base['output']) / 'models' / name))
    for name in STAGES:
        if not all(any(c['split'] == split for c in clips[name]) for split in ('train', 'validation')):
            raise ValueError(f'{name} 的训练/验证数据不足')
        train_codes = {c['route'][1 if name == 'lift' else -1] for c in clips[name] if c['split'] == 'train'}
        if name != 'release' and train_codes != set('1234'):
            raise ValueError(f'{name} 的训练集没有覆盖全部源/目标条件')
        condition = 'source_S' if name == 'lift' else 'target_T'
        manifest['skills'][name] = {'clips': clips[name],
            'state_names': list(NAMES) if name == 'release' else list(NAMES[:5]) + [f'{condition}{i}' for i in range(1, 5)],
            'action_names': ['gripper'] if name == 'release' else list(NAMES[:5])}
    save_json(output / 'manifest.json', manifest)
    for group, rows in sheets.items():
        sheet = Image.new('RGB', (4 * IMAGE_SIZE[0], len(rows) * (IMAGE_SIZE[1] + 28)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (route, pictures) in enumerate(rows):
            for j, image in enumerate(pictures):
                x, y = j * IMAGE_SIZE[0], i * (IMAGE_SIZE[1] + 28)
                sheet.paste(Image.fromarray(image), (x, y + 28))
                draw.text((x + 8, y + 6), f'{route}: {STAGES[j]} end', fill='black')
        sheet.save(output / f'stages_{group}.jpg')
    save_json(ROOT / 'outputs/koch_skills/latest_preparation.json', {'manifest': str(output / 'manifest.json')})
    print('新阶段准备完成：', output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    args = parser.parse_args()
    if args.base is None:
        args.base = Path(read_project_json(ROOT / 'outputs/koch_skills/latest_preparation.json')['manifest'])
    prepare(args.base)


if __name__ == '__main__':
    main()
