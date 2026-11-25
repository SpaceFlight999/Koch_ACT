"""离线训练主动起步和去掉开头等待的四组合并模型。"""
from koch_robot.paths import project_path
import argparse
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import koch_act_pipeline.train as train
from koch_act_pipeline.evaluate import evaluate_skill
from koch_act_pipeline.capture_policy import check_capture_dates
from koch_act_pipeline.prepare_skills import NAMES, ROOT, save_json, sha256
from koch_act_pipeline.train_layers import GROUPS, prune_checkpoints, smooth_forward
from koch_act_pipeline.train_merged import LAYER_FEATURES, layer_state, load_policy, read

OUTPUT_ROOT = ROOT / 'outputs/koch_active'
LIFT = 'active_takeoff'
FULL = 'merged_active_full'
TITLES = {LIFT: '主动起步提起', FULL: '完整搬运（合并·无起步等待）'}
STEPS = {LIFT: 1200, FULL: 2000}
TAKEOFF_FRAMES = 24
SMOOTH = {'velocity_matching_weight': .25, 'acceleration_matching_weight': .05}


def launch_index(state, action, clock, first_time):
    """0.4 秒内实测抬升和同轴指令一起上升，再回到首次上升指令。"""
    h = 8
    if len(clock) <= h:
        return None
    rise = state[h:, 1:3] - state[:-h, 1:3]
    target = action[h:, 1:3] - action[:-h, 1:3]
    axes = (rise >= [25, 60]) & (target >= [16, 30])
    hits = np.flatnonzero(axes.any(axis=1) & (clock[:-h] - first_time <= 10))
    if not len(hits):
        return None
    i = int(hits[0])
    ax = int(np.argmax(np.where(axes[i], rise[i] / [25, 60], -1)))
    column = 1 + ax
    j = np.flatnonzero(action[i + 1:i + h + 1, column] - action[i, column] >= [7, 12][ax])[0]
    return i + 1 + int(j)


def prepare(source, output, old):
    manifests = [read(source / group / 'manifest.json') for group in GROUPS]
    cal = manifests[0]['follower_calibration']
    if any(m['follower_calibration'] != cal for m in manifests):
        raise ValueError('四组从臂校准不一致。')
    if old['follower_calibration'] != cal or old['layer_conditions'] != LAYER_FEATURES:
        raise ValueError('旧合并模型与新训练的层级或校准不一致。')
    old_validation = {c['source'] for c in old['skills']['merged_full']['clips'] if c['split'] == 'validation'}
    names = manifests[0]['skills']['full_task']['state_names']
    cache = output / 'arrays'
    cache.mkdir(exist_ok=True)
    clips = {LIFT: [], FULL: []}
    cuts = []
    for group, m in enumerate(manifests):
        full = m['skills']['full_task']
        if full['state_names'] != names or full['action_names'] != list(NAMES):
            raise ValueError('四组模型的关节或格子定义不一致。')
        by_source = defaultdict(list)
        for clip in full['clips']:
            by_source[clip['source']].append(clip)
        if set(by_source) != {item['path'] for item in m['source_hashes']}:
            raise ValueError('完整轨迹未覆盖全部有效示范。')
        route_counts = Counter(items[0]['route'] for items in by_source.values())
        # 继承从未进旧模型梯度训练的整条验证示范，并保证每个格子组合仍有训练样本。
        held = set()
        for original, items in sorted(by_source.items()):
            route = items[0]['route']
            if original in old_validation and route_counts[route] > 1:
                held.add(original)
                route_counts[route] -= 1
        if not held:
            raise ValueError(f'{GROUPS[group]} 没有独立的验证示范。')
        for episode, (original, items) in enumerate(sorted(by_source.items())):
            items.sort(key=lambda c: c['start'])
            first_time = items[0]['start']
            active = False
            split = 'validation' if original in held else 'train'
            for part, clip in enumerate(items):
                with np.load(clip['arrays']) as raw:
                    state, action, images, clock = (raw[k] for k in ('state', 'action', 'image_index', 'timestamp'))
                start = 0 if active else launch_index(state, action, clock, first_time)
                if start is None:
                    continue
                if not active and len(clock) - start < 12:
                    raise ValueError(f'起步跨越记录缺口，无法可靠切分：{original}')
                prefix = f'{group}_{episode:03}_{part:03}'

                def save_clip(name, key, end, axes):
                    dest = cache / f'{prefix}_{key}.npz'
                    t = clock[start:end]
                    np.savez(dest, state=layer_state(state[start:end], group), action=action[start:end, :axes],
                             image_index=images[start:end], timestamp=t)
                    clips[name].append(dict(clip, skill=name, arrays=str(dest), layer_group=group,
                                            split=split, frames=len(t), start=float(t[0]), end=float(t[-1])))

                save_clip(FULL, 'full', len(clock), 6)
                if not active:
                    end = min(len(clock), start + TAKEOFF_FRAMES)
                    # 独立起步只训五个身体关节，夹爪保持当前位置。
                    save_clip(LIFT, 'lift', end, 5)
                    if split == 'train':
                        # 在完整模型中增加起步的采样机会，同时保留后续动作上下文。
                        save_clip(FULL, 'focus', min(len(clock), end + 15), 6)
                        clips[FULL].extend([dict(clips[FULL][-1]) for _ in range(2)])
                    cuts.append(dict(source=original, route=clip['route'], layer_group=group, split=split,
                                     first=float(first_time), motion_start=float(clock[start]),
                                     removed_seconds=round(float(clock[start] - first_time), 3)))
                active = True
            if not active:
                raise ValueError(f'没有检测到可信的主动起步：{original}')
    wanted = {(group, f'S{s}_T{t}') for group in range(4) for s in range(1, 5) for t in range(1, 5)}
    for name, rows in clips.items():
        check_capture_dates(rows)
        if {(c['layer_group'], c['route']) for c in rows if c['split'] == 'train'} != wanted:
            raise ValueError(f'{name} 训练路线缺失。')
        if {c['source'] for c in rows if c['split'] == 'train'} & {c['source'] for c in rows if c['split'] == 'validation'}:
            raise ValueError('训练和验证共享了原始示范。')
    save_json(output / 'trim_report.json', {'episodes': len(cuts), 'details': cuts})
    return dict(output=str(output), fps=20, image_size=[384, 216], units='raw_encoder_ticks',
        source_batch=str(source), follower_calibration=cal, layer_conditions=LAYER_FEATURES,
        validation_note='以整条示范隔离，沿用旧合并模型留出样本中的子集；参与选模，不是实机验证。',
        smooth_training=SMOOTH,
        skills={name: dict(clips=rows, state_names=names + LAYER_FEATURES,
                          action_names=list(NAMES[:5] if name == LIFT else NAMES)) for name, rows in clips.items()})


def run(source, output, old_path):
    completed = []
    def status(state, **fields):
        save_json(output / 'queue_state.json', dict(status=state, pid=os.getpid(),
            updated_at=datetime.now().astimezone().isoformat(), source_batch=str(source),
            completed=list(completed), steps=STEPS, **fields))
    manifest_path = output / 'manifest.json'
    old = read(old_path)
    status('preparing')
    if not manifest_path.exists():
        save_json(manifest_path, prepare(source, output, old))
    manifest = read(manifest_path)
    if manifest['source_batch'] != str(source):
        raise ValueError('续训批次与原训练来源不一致。')
    save_json(OUTPUT_ROOT / 'latest_batch.json', dict(output=str(output), source_batch=str(source)))
    if not torch.cuda.is_available():
        raise ValueError('本次训练需要 CUDA GPU。')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    os.environ['HF_HUB_OFFLINE'] = '1'
    device = torch.device('cuda')
    warm = old_path.parent / 'trained_models/merged_full'
    old_index = read(warm.parent / 'index.json')['models']['merged_full']
    for file in ('config.json', 'model.safetensors', 'normalization.json', 'skill_schema.json'):
        if sha256(warm / file) != old_index['sha256'][file]:
            raise ValueError(f'旧合并模型校验失败：{file}')
    old_schema = read(warm / 'skill_schema.json')
    if (old_schema['state_names'] != manifest['skills'][FULL]['state_names']
            or old_schema['action_names'] != list(NAMES)):
        raise ValueError('旧合并模型的输入输出与新模型不匹配。')
    factory, checkpoint_save = train.build_policy, train.save_checkpoint

    def build(*args):
        policy = load_policy(warm, device) if name == FULL else factory(*args)
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = .01
        return policy

    def save(policy, optimizer, scheduler, directory, step, stats, schema, val_mae):
        schema.update(layer_conditions=LAYER_FEATURES, smooth_training=SMOOTH, initial_wait_removed=True)
        result = checkpoint_save(policy, optimizer, scheduler, directory, step, stats, schema, val_mae)
        prune_checkpoints(directory, result, output)
        return result

    train.build_policy, train.save_checkpoint = build, save
    try:
        for name, steps in STEPS.items():
            status('training', skill=name)
            if shutil.disk_usage(output).free < 2 * 1024**3:
                raise ValueError('可用磁盘不足 2 GB，停止训练。')
            index_path = output / 'trained_models/index.json'
            index = read(index_path) if index_path.exists() else dict(models={}, hardware_verified=False)
            target = output / 'trained_models' / name
            existing = index['models'].get(name)
            if not existing or not all((target/f).is_file() and sha256(target/f) == h
                                       for f, h in existing['sha256'].items()):
                args = SimpleNamespace(steps=steps, resume=True, chunk_size=16, batch_size=8,
                    eval_every=200, checkpoint_every=400, policy_forward=smooth_forward,
                    normalization=read(warm / 'normalization.json') if name == FULL else None)
                train.train_skill(name, manifest['skills'][name], manifest, args, device)
                train.update_model_index(manifest)
                status('evaluating', skill=name)
                evaluation = evaluate_skill(name, manifest, device)
                save_json(output / f'evaluation_{name}.json', evaluation)
                target.mkdir(parents=True, exist_ok=True)
                hashes = {}
                for file in Path(evaluation['model']).iterdir():
                    if file.is_file():
                        shutil.copy2(file, target / file.name)
                        hashes[file.name] = sha256(target / file.name)
                        if hashes[file.name] != sha256(file):
                            raise ValueError(f'导出校验失败：{file.name}')
                index['models'][name] = dict(directory=name, title=TITLES[name], sha256=hashes,
                    checkpoint_step=evaluation['checkpoint_step'],
                    first_action_mae_degrees=evaluation['first_action_mae_degrees'])
                save_json(index_path, index)
            completed.append(name)
        status('complete')
    except BaseException as exc:
        status('error', reason=str(exc) or type(exc).__name__)
        raise
    finally:
        train.build_policy, train.save_checkpoint = factory, checkpoint_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = project_path(args.output).resolve()
    if output.parent != OUTPUT_ROOT.resolve():
        parser.error('输出必须是 outputs/koch_active 下独立的目录。')
    source = Path(read(ROOT / 'outputs/koch_layers/latest_batch.json')['batch']).resolve()
    old_path = Path(read(ROOT / 'outputs/koch_merged/latest_batch.json')['output']).resolve() / 'manifest.json'
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'training.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        run(source, output, old_path)
    except BaseException as exc:
        state_path = output / 'queue_state.json'
        state = read(state_path) if state_path.exists() else {}
        if state.get('status') != 'error':
            save_json(state_path, dict(state, status='error', reason=str(exc) or type(exc).__name__))
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
