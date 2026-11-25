"""离线审计、隔离无效示范并提取到达 S1–S4 / 夹爪技能；不访问机器人。"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from koch_robot.paths import ROOT, portable_references

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
ROUTES = tuple(f'S{s}_T{t}' for s in range(1, 5) for t in range(1, 5))
FILES = ('episode.json', 'joints.jsonl', 'camera_timestamps.jsonl', 'camera.avi')
IMAGE_SIZE = (384, 216)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(portable_references(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Windows 的只读句柄也可能暂时禁止替换目标，例如训练看板轮询时。
    for attempt in range(8):
        try:
            temp.replace(path)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(.025 * 2**attempt, .25))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ordered(values, title):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f'{title} 缺失、非有限或没有严格递增')
    return values


def audit_episode(path, route, decode=True):
    path = Path(path)
    meta = json.loads((path / 'episode.json').read_text(encoding='utf-8'))
    if meta.get('status') != 'complete':
        raise ValueError(f'状态为 {meta.get("status", "missing")}，未确认成功')
    if meta.get('route') != route or meta.get('joint_names') != list(NAMES):
        raise ValueError('路径或关节名称不一致')
    for name in FILES:
        if not (path / name).is_file():
            raise ValueError(f'缺少 {name}')
    rows, cameras = jsonl(path / 'joints.jsonl'), jsonl(path / 'camera_timestamps.jsonl')
    sent = [r for r in rows if r.get('sent') is True and r.get('command_time') is not None]
    ct = ordered([r['command_time'] for r in sent], '运动时间戳')
    ordered([r['follower_time'] for r in rows], '反馈时间戳')
    vt = ordered([r['t'] for r in cameras], '相机时间戳')
    if len(sent) < 20 or ct[-1] - ct[0] < 1:
        raise ValueError('动作不足一秒或有效帧不足 20')
    if [r['frame'] for r in cameras] != list(range(len(cameras))):
        raise ValueError('视频帧索引不连续')
    for row in rows:
        for key in ('follower_raw', 'goal_raw'):
            if set(row[key]) != set(NAMES) or any(type(v) is not int or abs(v) > 8191 for v in row[key].values()):
                raise ValueError('关节值缺失或超出原始读取范围')
    if meta.get('joint_frames') != len(rows) or meta.get('camera_frames') != len(cameras):
        raise ValueError('元数据帧数不一致')
    if vt[-1] < ct[0] or vt[0] > ct[-1]:
        raise ValueError('相机和动作时间没有交集')
    cap = cv2.VideoCapture(str(path / 'camera.avi'))
    try:
        if not cap.isOpened():
            raise ValueError('视频无法打开')
        count = 0
        if decode:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if [image.shape[1], image.shape[0]] != meta['camera']['size']:
                    raise ValueError('视频尺寸不一致')
                count += 1
        else:
            count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count != len(cameras):
            raise ValueError(f'视频实际可读 {count} 帧，时间戳为 {len(cameras)} 帧')
    finally:
        cap.release()
    return {'path': str(path.resolve()), 'route': route, 'status': 'valid',
            'duration': float(ct[-1] - ct[0]), 'joint_frames': len(rows), 'video_frames': len(cameras),
            'hashes': {name: sha256(path / name) for name in FILES}}


def audit_all(source):
    valid, invalid = [], []
    for route in ROUTES:
        for path in sorted((Path(source) / route).iterdir()):
            if not path.is_dir():
                continue
            try:
                item = audit_episode(path, route)
                valid.append(item)
                print(f'通过 {route}/{path.name} | {item["video_frames"]} 视频帧', flush=True)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                invalid.append({'path': str(path.resolve()), 'route': route, 'reason': str(exc)})
    missing = [r for r in ROUTES if not any(x['route'] == r for x in valid)]
    if missing:
        raise ValueError(f'以下路径没有完整示范，不移动任何数据：{missing}')
    return valid, invalid


def quarantine_invalid(source, archive, invalid):
    """只移动明确列出的示范目录；发生失败时反向恢复，不递归删除。"""
    source, archive = Path(source).resolve(), Path(archive).resolve()
    if not source.is_relative_to(ROOT) or not archive.is_relative_to(ROOT) or archive.is_relative_to(source):
        raise ValueError('数据移动路径必须位于项目内，隔离目录须在原数据目录外')
    moved = []
    try:
        for item in invalid:
            old = Path(item['path']).resolve()
            new = (archive / item['route'] / old.name).resolve()
            if (not old.is_relative_to(source) or not new.is_relative_to(archive)
                    or old.parent != source / item['route'] or new.exists()):
                raise ValueError('隔离路径校验失败')
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            moved.append((old, new))
            item['archived_to'] = str(new)
    except BaseException:
        for old, new in reversed(moved):
            new.rename(old)
        raise
    return len(moved)


def grab_boundary(times, fractions):
    """先确认后续完整闭合，再回溯最近 60% 开度下降点，排除早期试夹。"""
    armed = False
    for i, value in enumerate(fractions):
        if value > .6:
            armed = True
        if armed and value < .15 and times[-1] > times[i] + .3:
            stop = np.searchsorted(times, times[i] + .3)
            if np.all(fractions[i:stop] < .25):
                crossings = np.flatnonzero((fractions[:i] >= .6) & (fractions[1:i + 1] < .6)) + 1
                if len(crossings):
                    return int(crossings[-1]), i
    raise ValueError('不能可靠识别首次抓取边界，需要人工标注')


def contiguous_runs(indices):
    if len(indices) == 0:
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)


def prepare_episode(item, cache, fps=20):
    path = Path(item['path'])
    meta = json.loads((path / 'episode.json').read_text(encoding='utf-8'))
    rows = jsonl(path / 'joints.jsonl')
    sent = [r for r in rows if r.get('sent') and r.get('command_time') is not None]
    times = np.array([r['command_time'] for r in sent])
    feedback_times = np.array([r['follower_time'] for r in rows])
    cameras = jsonl(path / 'camera_timestamps.jsonl')
    camera_times = np.array([r['t'] for r in cameras])
    goals = np.array([[r['goal_raw'][n] for n in NAMES] for r in sent], dtype=np.float32)
    feedback = np.array([[r['follower_raw'][n] for n in NAMES] for r in rows], dtype=np.float32)
    cal = meta['follower_calibration']
    j = cal['motor_names'].index('gripper')
    closed, opened = cal['start_pos'][j], cal['end_pos'][j]
    fractions = np.clip((goals[:, 5] - closed) / (opened - closed), 0, 1)
    close_start, close_end = grab_boundary(times, fractions)
    episode_id = f'{item["route"]}_{path.name}'
    directory = cache / episode_id
    directory.mkdir(parents=True, exist_ok=False)
    image_path = directory / 'images.npy'
    images = np.lib.format.open_memmap(image_path, mode='w+', dtype=np.uint8,
                                      shape=(len(cameras), IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
    cap = cv2.VideoCapture(str(path / 'camera.avi'))
    try:
        for index in range(len(cameras)):
            ok, image = cap.read()
            if not ok:
                raise ValueError('审计后视频读取失败')
            images[index] = cv2.cvtColor(cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
        images.flush()
    begin = max(times[0], feedback_times[0], camera_times[0])
    end = min(times[-1], camera_times[-1])
    grid = np.arange(begin, end + 1e-6, 1 / fps)
    ai = np.searchsorted(times, grid, side='right') - 1
    si = np.searchsorted(feedback_times, grid, side='right') - 1
    vi = np.searchsorted(camera_times, grid, side='right') - 1
    valid = ((grid - times[ai] <= .2) & (grid - feedback_times[si] <= .2)
             & (grid - camera_times[vi] <= .25))
    candidates = [('reach_' + item['route'].split('_')[0], begin, times[close_start], None)]
    # 开/合指令条件化；同一模型中的两种动作有显式指令，不混为随机开合。
    crossings = np.flatnonzero((fractions[:-1] < .5) != (fractions[1:] < .5)) + 1
    previous = -math.inf
    for k in crossings:
        t = times[k]
        if t - previous < .6:
            continue
        first, last = max(begin, t - .4), min(end, t + .8)
        window = fractions[(times >= first) & (times <= last)]
        if len(window) < 2 or np.ptp(window) < .2:
            continue
        intent = 'close' if fractions[k] < fractions[k - 1] else 'open'
        candidates.append(('gripper', first, last, intent))
        previous = t
    clips = []
    for number, (skill, first, last, intent) in enumerate(candidates):
        indices = np.flatnonzero(valid & (grid >= first) & (grid <= last))
        for run_index, run in enumerate(contiguous_runs(indices)):
            if len(run) < 6:
                continue
            states, actions = feedback[si[run]], goals[ai[run]]
            if skill.startswith('reach_'):
                states, actions = states[:, :5], actions[:, :5]
            else:
                flag = np.tile([float(intent == 'open'), float(intent == 'close')], (len(run), 1))
                states = np.concatenate((states, flag), axis=1).astype(np.float32)
                actions = actions[:, 5:6]
            clip_path = directory / f'{skill}_{number}_{run_index}.npz'
            np.savez(clip_path, state=states, action=actions, image_index=vi[run], timestamp=grid[run])
            clips.append({'skill': skill, 'source': str(path), 'route': item['route'], 'episode_id': episode_id,
                          'intent': intent, 'split': 'validation' if item['route'].endswith('T4') else 'train',
                          'arrays': str(clip_path.resolve()), 'images': str(image_path.resolve()),
                          'frames': len(run), 'start': float(grid[run[0]]), 'end': float(grid[run[-1]])})
    endpoint_index = max(0, int(np.searchsorted(camera_times, times[close_start], side='right') - 1))
    thumbnail = np.array(images[endpoint_index])
    del images
    return clips, {'source': str(path), 'route': item['route'], 'reach_end': float(times[close_start]),
                   'reach_seconds': float(times[close_start] - times[0]),
                   'confirmed_closed': float(times[close_end]), 'method': 'first_confirmed_grasp_before_60pct_close'}, thumbnail


def run(args):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / 'outputs/koch_skills' / stamp if args.output is None else args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = ROOT / 'datasets/koch_skills' / stamp
    valid, invalid = audit_all(args.source)
    calibrations = [json.loads((Path(x['path']) / 'episode.json').read_text(encoding='utf-8'))['follower_calibration'] for x in valid]
    if any(cal != calibrations[0] for cal in calibrations[1:]):
        raise ValueError('成功示范的从臂校准不一致，先停止清理和训练以核实坐标')
    archive = ROOT / 'datasets/koch_st_invalid_archive' / stamp
    audit = {'source': str(args.source.resolve()), 'valid': valid, 'invalid': invalid,
             'archive': str(archive), 'policy': '保留全部有效完整示范，只隔离未完成或无效记录'}
    save_json(output / 'audit.json', audit)
    if args.quarantine:
        quarantine_invalid(args.source, archive, invalid)
        save_json(output / 'audit.json', audit)
        save_json(archive / 'restore_manifest.json', {'moved': invalid})
    cache.mkdir(parents=True, exist_ok=False)
    clips, segments, thumbnails = [], [], []
    for item in valid:
        extracted, segment, image = prepare_episode(item, cache, args.fps)
        clips.extend(extracted)
        segments.append(segment)
        thumbnails.append((segment, image))
        print(f'提取 {item["route"]} 到达片段 {segment["reach_seconds"]:.2f}s；技能片段 {len(extracted)}', flush=True)
    skills = {}
    for name in ('reach_S1', 'reach_S2', 'reach_S3', 'reach_S4', 'gripper'):
        selected = [c for c in clips if c['skill'] == name]
        if not any(c['split'] == 'train' for c in selected) or not any(c['split'] == 'validation' for c in selected):
            raise ValueError(f'{name} 没有完整的训练/验证片段')
        skills[name] = {'clips': selected, 'state_names': list(NAMES[:5]) if name.startswith('reach_')
                        else list(NAMES) + ['intent_open', 'intent_close'],
                        'action_names': list(NAMES[:5]) if name.startswith('reach_') else ['gripper']}
    manifest = {'schema_version': 1, 'output': str(output.resolve()), 'cache': str(cache.resolve()),
                'fps': args.fps, 'image_size': list(IMAGE_SIZE), 'units': 'raw_encoder_ticks',
                'image_preprocessing': 'RGB / 255 then ImageNet mean/std; resize to 384x216',
                'follower_calibration': calibrations[0], 'skills': skills,
                'validation_note': '整条 T4 示范留作验证；每个到达技能只有一条验证示范，不能当作实物成功率',
                'segments': segments, 'source_hashes': valid}
    save_json(output / 'manifest.json', manifest)
    sheet = Image.new('RGB', (IMAGE_SIZE[0] * 4, (IMAGE_SIZE[1] + 30) * math.ceil(len(thumbnails) / 4)), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (segment, image) in enumerate(thumbnails):
        x, y = index % 4 * IMAGE_SIZE[0], index // 4 * (IMAGE_SIZE[1] + 30)
        sheet.paste(Image.fromarray(image), (x, y + 30))
        draw.text((x + 8, y + 8), f'{segment["route"]} reach end {segment["reach_seconds"]:.2f}s', fill='black')
    sheet.save(output / 'reach_endpoints.jpg')
    # 整理后复核原始完整示范的字节完全一致。
    for item in valid:
        if any(sha256(Path(item['path']) / name) != digest for name, digest in item['hashes'].items()):
            raise ValueError('完整示范在处理期间被修改，停止训练')
    save_json(ROOT / 'outputs/koch_skills/latest_preparation.json', {'manifest': str(output / 'manifest.json')})
    print(f'准备完成：保留 {len(valid)} 条完整示范，隔离 {len(invalid) if args.quarantine else 0} 条；{output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'datasets/koch_st')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--quarantine', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error('fps 须为 1–60')
    run(args)


if __name__ == '__main__':
    main()
