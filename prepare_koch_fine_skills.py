"""离线细分取物/返程，选取真实一致休息姿态，只生成新训练副本。"""
from koch_project_paths import project_path, read_project_json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_koch_skills import ROOT, NAMES, IMAGE_SIZE, jsonl, save_json, contiguous_runs, sha256

NEW_SKILLS = ('rest_lift', 'approach_move', 'approach_align', 'return_lift', 'return_move', 'rest_align')
TICK_DEGREES = 360 / 4096


def pose_error(a, b):
    return np.max(np.abs((np.asarray(a) - np.asarray(b) + 2048) % 4096 - 2048), axis=-1) * TICK_DEGREES


def consensus_pose(poses, tolerance=5):
    """选择最大一致簇的实际示范姿态，不平均生成未见过的姿态。"""
    poses = np.asarray(poses)
    distances = pose_error(poses[:, None], poses[None, :])
    counts = np.sum(distances <= tolerance, axis=1)
    index = min(range(len(poses)), key=lambda i: (-counts[i], np.median(distances[i])))
    return int(index), poses[index].copy(), distances[index]


def rotation_phases(times, goals, first, last, minimum=.25):
    """从示范主搬运转角划分粗阶段，不把结果解释为已测量的末端高度。"""
    first_index = int(np.searchsorted(times, first))
    last_index = min(len(times) - 1, int(np.searchsorted(times, last)))
    delta = goals[last_index, 0] - goals[first_index, 0]
    if abs(delta) < 80:
        raise ValueError('可分辨的主转角不足')
    progress = (goals[:, 0] - goals[first_index, 0]) / delta
    candidates = np.arange(first_index, last_index + 1)
    early = candidates[(progress[candidates] >= .1) & (times[candidates] >= first + minimum)]
    if not len(early):
        raise ValueError('缺少提起结束候选')
    lift_end = int(early[0])
    late = candidates[(progress[candidates] >= .9) & (times[candidates] >= times[lift_end] + minimum)]
    align_start = None
    for k in late:
        stop = min(last_index + 1, int(np.searchsorted(times, times[k] + .2)))
        if stop > k and np.all(progress[k:stop] >= .85) and np.all(np.diff(times[k:stop]) <= .2):
            align_start = int(k)
            break
    if align_start is None or last - times[align_start] < minimum:
        raise ValueError('缺少稳定到达/姿态调整阶段')
    return first_index, lift_end, align_start, last_index


def first_settled_pose(times, goals, actual, target, first, tolerance=5, hold=.25):
    good = (pose_error(goals[:, :5], target) <= tolerance) & (pose_error(actual[:, :5], target) <= tolerance + 2)
    for i in np.flatnonzero(good & (times >= first)):
        j = int(np.searchsorted(times, times[i] + hold))
        if (j < len(times) and j > i + 2 and np.all(good[i:j + 1])
                and np.all(np.diff(times[i:j + 1]) <= .2)):
            return int(i), j
    return None


def prune_redundant(times, goals, actual, tolerance=.75, min_seconds=.4, max_seconds=4, max_excursion=8):
    """只裁剪近似同一完整身体姿态的小回环和静止停顿，不删大幅绕行。"""
    if len(times) < 3:
        return np.arange(len(times)), []
    keep, removed, i = [0], [], 0
    while i < len(times) - 1:
        candidates = np.flatnonzero((times >= times[i] + min_seconds) & (times <= times[i] + max_seconds)
                                    & (pose_error(goals, goals[i]) <= tolerance)
                                    & (pose_error(actual, actual[i]) <= 2))
        selected = None
        for j in reversed(candidates):
            if (np.max(pose_error(goals[i:j + 1], goals[i])) <= max_excursion
                    and np.max(pose_error(actual[i:j + 1], actual[i])) <= max_excursion):
                selected = int(j)
                break
        if selected is None:
            i += 1
        else:
            excursion = float(np.max(pose_error(goals[i:selected + 1], goals[i])))
            removed.append({'kind': 'stationary_pause' if excursion <= tolerance else 'small_pose_loop',
                            'source_start': float(times[i]), 'source_end': float(times[selected]),
                            'removed_seconds': float(times[selected] - times[i]),
                            'seam_degrees': float(pose_error(goals[selected], goals[i])),
                            'excursion_degrees': excursion})
            i = selected
        keep.append(i)
    return np.asarray(keep), removed


def load_episode(segment):
    rows = jsonl(Path(segment['source']) / 'joints.jsonl')
    sent = [r for r in rows if r.get('sent') and r.get('command_time') is not None]
    return {'rows': rows, 'times': np.array([r['command_time'] for r in sent]),
            'goals': np.array([[r['goal_raw'][n] for n in NAMES] for r in sent], dtype=np.float32),
            'actual': np.array([[r['follower_raw'][n] for n in NAMES] for r in sent], dtype=np.float32)}


def add_clips(name, first, last, episode, segment, image_path, cache, fps, condition=None, clean=False):
    times, goals = episode['times'], episode['goals']
    rows = episode['rows']
    feedback_times = np.array([r['follower_time'] for r in rows])
    feedback = np.array([[r['follower_raw'][n] for n in NAMES] for r in rows], dtype=np.float32)
    cameras = np.array([r['t'] for r in jsonl(Path(segment['source']) / 'camera_timestamps.jsonl')])
    grid = np.arange(first, last + 1e-6, 1 / fps)
    ai, si, vi = (np.searchsorted(t, grid, side='right') - 1 for t in (times, feedback_times, cameras))
    valid = ((ai >= 0) & (si >= 0) & (vi >= 0) & (grid <= cameras[-1]) & (grid - times[ai] <= .2)
             & (grid - feedback_times[si] <= .2) & (grid - cameras[vi] <= .25))
    result, cuts = [], []
    for part, indices in enumerate(contiguous_runs(np.flatnonzero(valid))):
        if len(indices) < 6:
            continue
        q, actual = goals[ai[indices], :5], feedback[si[indices], :5]
        keep, removed = prune_redundant(grid[indices], q, actual) if clean else (np.arange(len(indices)), [])
        cuts.extend(removed)
        indices, q, actual = indices[keep], q[keep], actual[keep]
        if len(indices) < 6:
            continue
        state = actual
        if condition is not None:
            code = np.eye(4, dtype=np.float32)[condition]
            state = np.concatenate((actual, np.tile(code, (len(indices), 1))), axis=1)
        path = cache / f'{segment["route"]}_{Path(segment["source"]).name}_{name}_{part}.npz'
        np.savez(path, state=state, action=q, image_index=vi[indices],
                 timestamp=grid[indices[0]] + np.arange(len(indices)) / fps, source_timestamp=grid[indices])
        result.append({'skill': name, 'source': segment['source'], 'route': segment['route'],
                       'episode_id': f'{segment["route"]}_{Path(segment["source"]).name}', 'intent': name,
                       'split': 'validation' if segment['route'][1] == segment['route'][-1] else 'train',
                       'arrays': str(path.resolve()), 'images': image_path, 'frames': len(indices),
                       'start': float(grid[indices[0]]), 'end': float(grid[indices[-1]]),
                       'cleaned': clean, 'training_duration_s': float((len(indices) - 1) / fps)})
    return result, cuts


def prepare(base_path):
    base = read_project_json(project_path(base_path))
    for item in base['source_hashes']:
        if any(sha256(Path(item['path']) / n) != digest for n, digest in item['hashes'].items()):
            raise ValueError('原始成功示范已改变，停止细分')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output, cache = ROOT / 'outputs/koch_skills' / stamp, ROOT / 'datasets/koch_skills' / stamp
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    assets = {c['source']: c['images'] for s in base['skills'].values() for c in s['clips']}
    episodes = [load_episode(s) for s in base['segments']]
    end_poses, tail_indices = [], []
    for episode in episodes:
        t, q = episode['times'], episode['goals']
        tail = np.flatnonzero(t >= t[-1] - .5)
        median = np.median(q[tail, :5], axis=0)
        index = int(tail[np.argmin(pose_error(q[tail, :5], median))])
        tail_indices.append(index)
        end_poses.append(q[index, :5])
    representative, rest, distances = consensus_pose(end_poses)
    rest_info = {'joint_names': list(NAMES[:5]), 'goal_raw': rest.astype(int).tolist(), 'tolerance_degrees': 5,
                 'source': base['segments'][representative]['source'],
                 'source_timestamp': float(episodes[representative]['times'][tail_indices[representative]]),
                 'method': 'largest_5_degree_cluster_then_observed_medoid',
                 'final_pose_distances_degrees': distances.tolist(), 'height_centimeters': None}
    save_json(output / 'rest_pose.json', rest_info)
    clips = {name: [] for name in NEW_SKILLS}
    annotations, omitted, photos = [], [], {}
    transport = {x['source']: x for x in base['stage_annotations']}
    cal = base['follower_calibration']
    grip = cal['motor_names'].index('gripper')
    closed, opened = cal['start_pos'][grip], cal['end_pos'][grip]
    for segment, episode in zip(base['segments'], episodes):
        t, q, actual = episode['times'], episode['goals'], episode['actual']
        route, source = segment['route'], segment['source']
        annotation = {'source': source, 'route': route, 'inbound': {}, 'return': {}, 'removed': []}
        pictures = []
        images = np.load(assets[source], mmap_mode='r')
        cameras = np.array([x['t'] for x in jsonl(Path(source) / 'camera_timestamps.jsonl')])
        def picture(when):
            i = max(0, min(len(images) - 1, int(np.searchsorted(cameras, when, side='right') - 1)))
            return np.array(images[i])
        try:
            incoming = rotation_phases(t, q, float(t[0]), segment['reach_end'])
            fractions = np.clip((q[:, 5] - closed) / (opened - closed), 0, 1)
            first_open = np.flatnonzero((fractions[:-1] < .3) & (fractions[1:] >= .3)
                                        & (t[1:] < segment['reach_end'])) + 1
            annotation['opening_overlaps_body_transfer'] = bool(len(first_open) and t[first_open[0]] < t[incoming[2]])
            for name, a, b in zip(NEW_SKILLS[:3], incoming, incoming[1:]):
                extracted, _ = add_clips(name, float(t[a]), float(t[b]), episode, segment, assets[source],
                                         cache, base['fps'], condition=int(route[1]) - 1)
                clips[name].extend(extracted)
                annotation['inbound'][name] = [float(t[a]), float(t[b])]
                pictures.append((name, picture(t[b])))
        except ValueError as exc:
            omitted.append({'source': source, 'stage': 'inbound', 'reason': str(exc)})
        data = transport[source]['times']
        release_start = data['release_start']
        actual_fraction = (actual[:, 5] - closed) / (opened - closed)
        fractions = np.clip((q[:, 5] - closed) / (opened - closed), 0, 1)
        hold = actual_fraction[(t >= release_start - .3) & (t < release_start)]
        threshold = min(.95, float(np.median(hold)) + .08)
        clear = np.flatnonzero((t >= release_start) & (actual_fraction >= threshold) & (fractions >= threshold))
        if len(clear):
            first = int(clear[0])
            settled = first_settled_pose(t, q, actual, rest, float(t[first]))
            # 不一致的末尾仍能提供返程前两段，最终姿态只用已进入统一休息区的数据。
            end = settled[1] if settled else int(np.argmin(pose_error(q[first:, :5], rest)) + first)
            try:
                returning = rotation_phases(t, q, float(t[first]), float(t[end]))
                for name, a, b in zip(NEW_SKILLS[3:5], returning[:2], returning[1:3]):
                    extracted, removed = add_clips(name, float(t[a]), float(t[b]), episode, segment, assets[source],
                                                   cache, base['fps'], condition=int(route[-1]) - 1, clean=True)
                    clips[name].extend(extracted)
                    annotation['removed'].extend(removed)
                    annotation['return'][name] = [float(t[a]), float(t[b])]
                    pictures.append((name, picture(t[b])))
                if settled:
                    a, b = returning[2], settled[1]
                    extracted, removed = add_clips('rest_align', float(t[a]), float(t[b]), episode, segment,
                                                   assets[source], cache, base['fps'], clean=True)
                    clips['rest_align'].extend(extracted)
                    annotation['removed'].extend(removed)
                    annotation['return']['rest_align'] = [float(t[a]), float(t[b])]
                    annotation['trailing_seconds_omitted'] = float(t[-1] - t[b])
                    pictures.append(('rest_align', picture(t[b])))
                else:
                    omitted.append({'source': source, 'stage': 'rest_align',
                                    'reason': '没有稳定进入统一休息姿态 5° 范围；不学习差异大的末尾调整'})
            except ValueError as exc:
                omitted.append({'source': source, 'stage': 'return', 'reason': str(exc)})
        else:
            omitted.append({'source': source, 'stage': 'return', 'reason': '无法确认松爪后物体已释放'})
        images._mmap.close()
        photos.setdefault(route.split('_')[0], []).append((route, pictures))
        annotations.append(annotation)
        print(route, 'inbound', list(annotation['inbound']), 'return', list(annotation['return']),
              'cuts', len(annotation['removed']), flush=True)
    manifest = dict(base)
    manifest.update(output=str(output), cache=str(cache), previous_manifest=str(project_path(base_path).resolve()),
                    fine_annotations=annotations, fine_omissions=omitted, rest_pose=rest_info)
    manifest['validation_note'] = base['validation_note'] + '；取物/返程细分同样留整条对角路径作验证，统一休息姿态只用稳定进入目标范围的示范'
    manifest['skills'] = dict(base['skills'])
    # 再次细分时也只继承旧九项，新增六项必须写入本次目录，避免错误跳过训练。
    manifest['model_dirs'] = {name: base.get('model_dirs', {}).get(name, str(Path(base['output']) / 'models' / name))
                              for name in base['skills'] if name not in NEW_SKILLS}
    for name in NEW_SKILLS:
        if not all(any(c['split'] == split for c in clips[name]) for split in ('train', 'validation')):
            raise ValueError(f'{name} 的训练/验证片段不足，候选标注已保存于 {output}')
        prefix = 'source_S' if name in NEW_SKILLS[:3] else 'from_T'
        if name != 'rest_align':
            code_index = 1 if name in NEW_SKILLS[:3] else -1
            if {c['route'][code_index] for c in clips[name] if c['split'] == 'train'} != set('1234'):
                raise ValueError(f'{name} 的条件没有覆盖四个区域')
        manifest['skills'][name] = {'clips': clips[name], 'state_names': list(NAMES[:5]) +
            ([] if name == 'rest_align' else [f'{prefix}{i}' for i in range(1, 5)]), 'action_names': list(NAMES[:5])}
    manifest['requested_workflow'] = [
        {'skill': 'rest_lift'}, {'skill': 'approach_move'}, {'skill': 'gripper', 'intent': 'open'},
        {'skill': 'approach_align'}, {'skill': 'gripper', 'intent': 'close'}, {'skill': 'lift'},
        {'skill': 'transfer'}, {'skill': 'lower'}, {'skill': 'release'}, {'skill': 'return_lift'},
        {'skill': 'return_move'}, {'skill': 'rest_align'}]
    manifest['workflow_validated_in_hardware'] = False
    manifest['workflow_note'] = '原示范开爪与身体运动有重叠；此处训练独立组件，不伪造严格新顺序的连续专家演示。没有末端高度标定。'
    save_json(output / 'fine_annotation_report.json', {'rest': rest_info, 'annotations': annotations, 'omitted': omitted})
    save_json(output / 'manifest.json', manifest)
    for group, rows in photos.items():
        sheet = Image.new('RGB', (6 * IMAGE_SIZE[0], len(rows) * (IMAGE_SIZE[1] + 28)), 'white')
        draw = ImageDraw.Draw(sheet)
        for row, (route, pictures) in enumerate(rows):
            for col, (name, image) in enumerate(pictures):
                x, y = col * IMAGE_SIZE[0], row * (IMAGE_SIZE[1] + 28)
                sheet.paste(Image.fromarray(image), (x, y + 28))
                draw.text((x + 6, y + 6), f'{route}: {name}', fill='black')
        sheet.save(output / f'fine_{group}.jpg')
    save_json(ROOT / 'outputs/koch_skills/latest_preparation.json', {'manifest': str(output / 'manifest.json')})
    print('细分完成：', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    args = parser.parse_args()
    if args.base is None:
        args.base = Path(read_project_json(ROOT / 'outputs/koch_skills/latest_preparation.json')['manifest'])
    prepare(args.base)
