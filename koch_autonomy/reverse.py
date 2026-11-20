"""Audit demonstrations and fit an offline reverse-transfer reference only.

Reversing a loaded transport video does not produce empty-gripper observations
or demonstrated inverse dynamics. These artifacts are intentionally incompatible
with ModelPack/Actor and must never be promoted as a live ACT policy.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from koch_project_paths import ROOT, project_path, read_project_json

BODY = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
TICK_DEGREES = 360 / 4096


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_route_layers(route, pair):
    if not isinstance(route, str) or not re.fullmatch(r'S[1-4]_T[1-4]', route):
        raise ValueError('路径必须为 S1_T1 到 S4_T4。')
    if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(type(v) is not int or v not in (1, 2) for v in pair):
        raise ValueError('层级必须为 [1|2, 1|2]。')


def reverse_samples(times, actual, first, last, fps=20):
    times, actual = np.asarray(times, dtype=float), np.asarray(actual, dtype=float)
    if (times.ndim != 1 or actual.shape != (len(times), 5) or len(times) < 2 or
            not np.isfinite(times).all() or not np.isfinite(actual).all() or
            np.any(np.diff(times) <= 0) or not 0 < last - first or fps <= 0 or
            first < times[0] or last > times[-1]):
        raise ValueError('逆向候选的时间、关节或阶段边界无效。')
    begin = max(0, np.searchsorted(times, first, side='right') - 1)
    end = min(len(times), np.searchsorted(times, last, side='left') + 1)
    if np.max(np.diff(times[begin:end])) > .2:
        raise ValueError('搬运阶段反馈中断超过 0.2 秒，不跨缺口逆向。')
    if np.max(np.abs(np.diff(actual[begin:end], axis=0))) > 1024:
        raise ValueError('关节反馈出现大跳变，不直接镜像或跨圈插值。')
    grid = np.linspace(first, last, max(6, int(np.ceil((last - first) * fps)) + 1))
    q = np.stack([np.interp(grid, times, actual[:, i]) for i in range(5)], axis=1)[::-1].copy()
    return np.linspace(0., last - first, len(grid)), q, grid[::-1].copy()


def prepare(output, batch=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    batch = project_path(batch or read_project_json(ROOT / 'outputs/koch_layers/latest_batch.json')['batch'])
    records, rejected, model_packs, camera_records, scan_candidates = [], [], [], [], []
    calibration = None
    for path in sorted(batch.glob('*/manifest.json')):
        manifest = read_project_json(path)
        pair = manifest['layer_pair']
        for annotation in manifest.get('stage_annotations', []):
            validate_route_layers(annotation.get('route'), pair)
        if calibration is not None and calibration != manifest['follower_calibration']:
            raise ValueError('四组示范的从臂校准不一致。')
        calibration = manifest['follower_calibration']
        index_path = path.parent / 'trained_models/index.json'
        index = read_json(index_path) if index_path.exists() else {'models': {}}
        model_packs.append({'manifest': str(path.resolve()), 'layer_pair': pair,
                            'exported_models': len(index['models']),
                            'hardware_verified': manifest.get('workflow_validated_in_hardware', False)})
        for annotation in manifest.get('stage_annotations', []):
            source = Path(annotation['source'])
            joint_file = source / 'joints.jsonl'
            row_info = {'source': str(source), 'route': annotation['route'], 'layer_pair': pair}
            try:
                rows = [json.loads(line) for line in joint_file.read_text(encoding='utf-8').splitlines() if line.strip()]
                times = np.array([r['follower_time'] for r in rows])
                actual = np.array([[r['follower_raw'][n] for n in BODY] for r in rows])
                bounds = annotation['times']
                # Only the annotated middle transport; grasp/release are never reversed.
                t, q, original_t = reverse_samples(times, actual, bounds['lift_end'], bounds['transfer_end'])
                name = f'S{pair[0]}T{pair[1]}_{annotation["route"]}_{source.name}.npz'
                np.savez(output / name, timestamp=t.astype(np.float32), body_raw=q.astype(np.float32),
                         original_timestamp=original_t, progress=(t / t[-1]).astype(np.float32))
                row_info.update(arrays=str(output / name), frames=len(t), duration_s=float(t[-1]),
                                split='validation' if annotation['route'][1] == annotation['route'][-1] else 'train',
                                body_min_raw=q.min(0).tolist(), body_max_raw=q.max(0).tolist(),
                                max_speed_degrees_s=float(np.max(np.abs(np.diff(q, axis=0) / np.diff(t)[:, None])) * TICK_DEGREES),
                                source_sha256=hashlib.sha256(joint_file.read_bytes()).hexdigest())
                records.append(row_info)
                scan_candidates.append({**{k: row_info[k] for k in ('source', 'route', 'layer_pair')},
                    'target': annotation['route'][-2:], 'body_raw': dict(zip(BODY, q[0].round().astype(int).tolist())),
                    'source_time': float(original_t[0]), 'stage': 'transfer_end_before_lower',
                    'height_mm': None, 'gripper_command': 'keep_unchanged',
                    'scan_view_verified': False, 'safe_to_execute': False})
                episode = read_json(source / 'episode.json')
                camera_records.append({'source': str(source), 'camera': episode.get('camera')})
            except (ValueError, KeyError, OSError) as exc:
                rejected.append({**row_info, 'reason': str(exc)})
    if not records:
        raise ValueError('没有完整的可用搬运片段；请检查报告源数据。')
    result = {'format': 'koch_reverse_reference_v1', 'deployable': False, 'body_names': list(BODY),
              'units': 'raw_encoder_ticks', 'fps': 20, 'follower_calibration': calibration,
              'records': records, 'rejected': rejected, 'model_packs': model_packs,
              'camera_records': camera_records, 'scan_candidates': scan_candidates,
              'limits': ['仅逆向五个身体关节的实际轨迹；未反转夹爪及带物图像。',
                         '不是空爪 T→S 示教，不覆盖 T 放置位起步或 S 抓取对齐。',
                         '关节包络不是末端高度或碰撞保证；扫描候选未验证视野。']}
    save_json(output / 'manifest.json', result)
    return result


def features(record, progress):
    validate_route_layers(record.get('route'), record.get('layer_pair'))
    pair = tuple(record['layer_pair'])
    route = record['route']
    code = np.zeros(12, np.float32)
    code[int(route[1]) - 1] = 1
    code[4 + int(route[-1]) - 1] = 1
    code[8 + ((1, 1), (1, 2), (2, 1), (2, 2)).index(pair)] = 1
    return np.column_stack([np.tile(code, (len(progress), 1)), progress]).astype(np.float32)


def train_reference(manifest_path, epochs=300):
    """Fit joint positions vs route/layers/phase; no motor or camera connection."""
    import torch
    if type(epochs) is not int or epochs < 1:
        raise ValueError('训练轮数须为正整数。')
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get('format') != 'koch_reverse_reference_v1' or manifest.get('deployable') is not False:
        raise ValueError('仅接受已审计、禁止部署的逆向参考数据。')
    splits = {}
    sources = {}
    for split in ('train', 'validation'):
        xs, ys, weights = [], [], []
        sources[split] = set()
        for record in manifest['records']:
            if record['split'] != split:
                continue
            sources[split].add(record['source'])
            with np.load(record['arrays']) as arrays:
                xs.append(features(record, arrays['progress']))
                ys.append(arrays['body_raw'])
                weights.extend([1 / len(arrays['body_raw'])] * len(arrays['body_raw']))
        if not xs:
            raise ValueError('训练和整条示范留出验证集都必须非空。')
        splits[split] = (np.concatenate(xs), np.concatenate(ys), np.asarray(weights, np.float32))
    if sources['train'] & sources['validation']:
        raise ValueError('同一示范不能跨训练和验证集。')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    mean = splits['train'][1].mean(0)
    scale = np.maximum(splits['train'][1].std(0), 1)
    model = torch.nn.Sequential(torch.nn.Linear(13, 96), torch.nn.Tanh(),
                                torch.nn.Linear(96, 96), torch.nn.Tanh(), torch.nn.Linear(96, 5))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    x, y, weights = splits['train']
    x, y, weights = torch.from_numpy(x), torch.from_numpy((y - mean) / scale), torch.from_numpy(weights)
    vx, vy, vw = splits['validation']
    vx = torch.from_numpy(vx)
    history, best, best_state = [], float('inf'), None
    for epoch in range(1, epochs + 1):
        model.train()
        loss = (((model(x) - y) ** 2).mean(1) * weights).sum() / weights.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.inference_mode():
                error = np.abs(model(vx).numpy() * scale + mean - vy) * TICK_DEGREES
            mae = float(np.average(error.mean(1), weights=vw))
            history.append({'epoch': epoch, 'loss': float(loss.detach()), 'validation_mae_degrees': mae})
            if mae < best:
                best, best_state = mae, {key: value.clone() for key, value in model.state_dict().items()}
            if epoch % 50 == 0 or epoch == epochs:
                print(f'reverse reference epoch={epoch}/{epochs}, held-out MAE={mae:.3f} deg', flush=True)
    destination = manifest_path.parent / 'reverse_reference.pt'
    torch.save({'format': 'koch_reverse_reference_v1', 'deployable': False, 'state_dict': best_state,
                'mean': mean.tolist(), 'scale': scale.tolist(), 'input_dim': 13,
                'architecture': [13, 96, 96, 5], 'body_names': list(BODY)}, destination)
    report = {'model': str(destination.resolve()), 'epochs': epochs, 'best_validation_mae_degrees': best,
              'train_sources': len(sources['train']), 'validation_sources': len(sources['validation']),
              'history': history, 'deployable': False,
              'meaning': '对倒序示范关节路径的留出拟合误差，不是逆向机器人执行成功率；验证集也用于选模。'}
    save_json(manifest_path.parent / 'training_report.json', report)
    return report
