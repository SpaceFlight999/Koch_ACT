"""离线训练 Koch 分步 ACT 模型；无相机/串口控制接口。"""
from koch_robot.paths import project_path, read_project_json
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from koch_act_pipeline.prepare_skills import ROOT, save_json, sha256
from koch_act_pipeline.capture_policy import check_capture_dates

SKILLS = ('reach_S1', 'reach_S2', 'reach_S3', 'reach_S4', 'gripper', 'lift', 'transfer', 'lower', 'release',
          'rest_lift', 'approach_move', 'approach_align', 'return_lift', 'return_move', 'rest_align')
IMAGE_KEY = 'observation.images.front'
IMAGE_MEAN = (.485, .456, .406)
IMAGE_STD = (.229, .224, .225)


class SkillDataset(Dataset):
    def __init__(self, clips, chunk_size=16, stats=None):
        self.clips, self.chunk_size, self.images = [], chunk_size, {}
        for clip in clips:
            with np.load(project_path(clip['arrays'])) as arrays:
                self.clips.append(dict(clip, state=arrays['state'], action=arrays['action'],
                                       image_index=arrays['image_index']))
        if not self.clips:
            raise ValueError('没有技能片段')
        self.ends = np.cumsum([len(c['state']) for c in self.clips])
        if stats is None:
            state = np.concatenate([c['state'] for c in self.clips])
            action = np.concatenate([c['action'] for c in self.clips])
            minimum = np.ones(state.shape[1], dtype=np.float32)
            if state.shape[1] == 8:
                minimum[-2:] = .05
            elif state.shape[1] == 9:
                minimum[-4:] = .05
            stats = {'state_mean': state.mean(0).tolist(),
                     'state_std': np.maximum(state.std(0), minimum).tolist(),
                     'action_mean': action.mean(0).tolist(),
                     'action_std': np.maximum(action.std(0), 1).tolist()}
        self.stats = stats
        self.sm, self.ss = (np.asarray(stats[k], dtype=np.float32) for k in ('state_mean', 'state_std'))
        self.am, self.astd = (np.asarray(stats[k], dtype=np.float32) for k in ('action_mean', 'action_std'))
        # 同一示范的通信缺口可能产生多段，按原始示范而非片段数量均衡采样。
        totals = {}
        for clip in self.clips:
            source = clip.get('source', clip['arrays'])
            totals[source] = totals.get(source, 0) + len(clip['state'])
        self.weights = np.concatenate([np.full(len(c['state']), 1 / totals[c.get('source', c['arrays'])])
                                       for c in self.clips])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        clip_index = int(np.searchsorted(self.ends, index, side='right'))
        clip = self.clips[clip_index]
        local = index - (int(self.ends[clip_index - 1]) if clip_index else 0)
        path = project_path(clip['images'])
        if path not in self.images:
            self.images[path] = np.load(path, mmap_mode='r')
        image = np.array(self.images[path][clip['image_index'][local]], copy=True)
        future = np.arange(local, local + self.chunk_size)
        pad = future >= len(clip['action'])
        future = np.minimum(future, len(clip['action']) - 1)
        return {IMAGE_KEY: torch.from_numpy(image).permute(2, 0, 1),
                'observation.state': torch.from_numpy((clip['state'][local] - self.sm) / self.ss),
                'action': torch.from_numpy((clip['action'][future] - self.am) / self.astd),
                'action_is_pad': torch.from_numpy(pad)}

    def close(self):
        for image in self.images.values():
            image._mmap.close()
        self.images.clear()


def build_policy(state_dim, action_dim, device, chunk=16):
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    config = ACTConfig(
        input_features={'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
                        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 216, 384))},
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        chunk_size=chunk, n_action_steps=min(8, chunk), dim_model=256, dim_feedforward=1024,
        device=str(device), push_to_hub=False, optimizer_lr=1e-4, optimizer_lr_backbone=1e-5)
    return ACTPolicy(config).to(device)


def device_batch(batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    image = batch[IMAGE_KEY].float() / 255
    mean = torch.tensor(IMAGE_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(IMAGE_STD, device=device)[None, :, None, None]
    batch[IMAGE_KEY] = (image - mean) / std
    return batch


@torch.no_grad()
def evaluate(policy, loader, device, action_std, amp):
    policy.eval()
    error, count = 0.0, 0
    scale = torch.as_tensor(action_std, device=device)
    for batch in loader:
        batch = device_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            predicted = policy.predict_action_chunk(batch)
        valid = (~batch['action_is_pad']).unsqueeze(-1)
        absolute = (predicted.float() - batch['action']).abs() * scale
        error += float((absolute * valid).sum())
        count += int(valid.sum()) * predicted.shape[-1]
    policy.train()
    return error / max(1, count) * 360 / 4096


def save_checkpoint(policy, optimizer, scheduler, directory, step, stats, schema, val_mae):
    checkpoint = directory / 'checkpoints' / f'step_{step:06d}'
    if (checkpoint / 'checkpoint.json').exists():
        raise ValueError('拒绝覆盖已完成的检查点')
    checkpoint.mkdir(parents=True, exist_ok=True)
    model = checkpoint / 'pretrained_model'
    policy.save_pretrained(model)
    save_json(model / 'normalization.json', dict(stats, image_mean=IMAGE_MEAN, image_std=IMAGE_STD))
    save_json(model / 'skill_schema.json', schema)
    torch.save({'step': step, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'torch_rng': torch.get_rng_state()}, checkpoint / 'training_state.pt')
    save_json(checkpoint / 'checkpoint.json', {'step': step, 'validation_mae_degrees': val_mae,
                                              'model': str(model.resolve())})
    save_json(directory / 'latest_checkpoint.json', {'checkpoint': str(checkpoint.resolve())})
    return checkpoint


def model_directory(manifest, name):
    output = project_path(manifest['output'])
    return project_path(manifest.get('model_dirs', {}).get(name, output / 'models' / name))


def update_model_index(manifest):
    index = {}
    for name in manifest['skills']:
        best = model_directory(manifest, name) / 'best_checkpoint.json'
        if best.exists():
            value = read_project_json(best)
            index[name] = dict(value, model=str(Path(value['checkpoint']) / 'pretrained_model'))
    save_json(project_path(manifest['output']) / 'model_index.json', index)


def train_skill(name, skill, manifest, args, device):
    check_capture_dates(skill['clips'])
    directory = model_directory(manifest, name)
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / 'progress.json'
    if progress_path.exists() and not args.resume:
        raise ValueError(f'{name} 已有训练结果；请使用 --resume 继续或另建训练目录')
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'complete' and previous.get('step', 0) >= args.steps:
            print(f'{name} 已完成，保留现有模型。', flush=True)
            return previous
    train_clips = [c for c in skill['clips'] if c['split'] == 'train']
    val_clips = [c for c in skill['clips'] if c['split'] == 'validation']
    if {c['source'] for c in train_clips} & {c['source'] for c in val_clips}:
        raise ValueError('同一原始示范不能同时进入训练和验证')
    training = SkillDataset(train_clips, args.chunk_size, getattr(args, 'normalization', None))
    validation = SkillDataset(val_clips, args.chunk_size, training.stats)
    save_json(directory / 'normalization.json', training.stats)
    sampler = WeightedRandomSampler(training.weights, num_samples=len(training), replacement=True)
    train_loader = DataLoader(training, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == 'cuda')
    schema = {'skill': name, 'fps': manifest['fps'], 'units': manifest['units'],
              'state_names': skill['state_names'], 'action_names': skill['action_names'],
              'image_key': IMAGE_KEY, 'image_size': manifest['image_size'],
              'follower_calibration': manifest['follower_calibration'],
              'preprocessing': 'Apply normalization.json to raw state/action and RGB image; see koch_act_pipeline/train.py',
              'validation': manifest['validation_note'], 'hardware_verified': False}
    if name in ('rest_lift', 'approach_move', 'approach_align', 'return_lift', 'return_move', 'rest_align'):
        schema.update(rest_pose=manifest['rest_pose'], workflow_note=manifest['workflow_note'])
    policy = build_policy(len(skill['state_names']), len(skill['action_names']), device, args.chunk_size)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-4)
    def lr_factor(step):
        return min(1, (step + 1) / 50) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    step, best, started = 0, math.inf, time.monotonic()
    if args.resume and (directory / 'latest_checkpoint.json').exists():
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        checkpoint = Path(read_project_json(directory / 'latest_checkpoint.json')['checkpoint'])
        del policy
        config = ACTConfig.from_pretrained(checkpoint / 'pretrained_model', local_files_only=True)
        config.device = str(device)
        policy = ACTPolicy.from_pretrained(checkpoint / 'pretrained_model', config=config, local_files_only=True).to(device)
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        state = torch.load(checkpoint / 'training_state.pt', map_location=device, weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng'].cpu())
        step = state['step']
        if (directory / 'best_checkpoint.json').exists():
            best = json.loads((directory / 'best_checkpoint.json').read_text())['validation_mae_degrees']
    if getattr(args, 'policy_forward', None) is not None:
        policy.forward = args.policy_forward.__get__(policy, type(policy))
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    progress = {'skill': name, 'status': 'running', 'step': step, 'steps': args.steps,
                'train_frames': len(training), 'validation_frames': len(validation),
                'device': str(device), 'parameters': sum(p.numel() for p in policy.parameters())}
    save_json(progress_path, progress)
    print(f'开始 {name}: train={len(training)} val={len(validation)}，输出 {len(skill["action_names"])} 关节', flush=True)
    iterator, latest_val = iter(train_loader), None
    initial_step = step
    try:
        with (directory / 'metrics.csv').open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'l1', 'kl', 'validation_mae_degrees', 'elapsed_s'])
            if stream.tell() == 0:
                writer.writeheader()
            while step < args.steps:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                batch = device_batch(batch, device)
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    loss, losses = policy(batch)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('训练损失出现 NaN/Inf，停止，保留已有检查点')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                step += 1
                elapsed = time.monotonic() - started
                if step == 1 or step % args.eval_every == 0 or step == args.steps:
                    latest_val = evaluate(policy, val_loader, device, training.astd, amp)
                    print(f'{name} step={step}/{args.steps} loss={float(loss.detach()):.5f} val_mae={latest_val:.3f}deg', flush=True)
                writer.writerow({'step': step, 'loss': float(loss.detach()), 'l1': losses['l1_loss'],
                                 'kl': losses.get('kld_loss', 0), 'validation_mae_degrees': latest_val if step % args.eval_every == 0 or step == 1 else '',
                                 'elapsed_s': round(elapsed, 3)})
                improved = latest_val is not None and latest_val < best and (step == 1 or step % args.eval_every == 0 or step == args.steps)
                if step in (1, 50) or step % args.checkpoint_every == 0 or step == args.steps or improved:
                    checkpoint = save_checkpoint(policy, optimizer, scheduler, directory, step, training.stats, schema, latest_val)
                    if improved:
                        best = latest_val
                        save_json(directory / 'best_checkpoint.json', {'checkpoint': str(checkpoint), 'validation_mae_degrees': best, 'step': step})
                if step % 25 == 0 or step == 1 or step == args.steps:
                    progress.update(step=step, last_loss=float(loss.detach()), best_validation_mae_degrees=best,
                                    elapsed_s=round(elapsed, 2), steps_per_second=round((step - initial_step) / max(elapsed, .001), 2))
                    save_json(progress_path, progress)
                    stream.flush()
                    if step % 100 == 0:
                        print(f'{name} {step}/{args.steps} | {progress["steps_per_second"]} steps/s', flush=True)
        progress.update(status='complete', step=step)
        save_json(progress_path, progress)
        return progress
    except BaseException as exc:
        progress.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                        step=step, reason=str(exc) or type(exc).__name__)
        save_json(progress_path, progress)
        raise
    finally:
        training.close()
        validation.close()
        del policy, optimizer, scheduler
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def resolve_manifest(value):
    if value:
        return project_path(value).resolve()
    pointer = read_project_json(ROOT / 'outputs/koch_skills/latest_preparation.json')
    return Path(pointer['manifest'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument('--skills', nargs='+', choices=SKILLS, default=list(SKILLS))
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    manifest_path = resolve_manifest(args.manifest)
    manifest = read_project_json(manifest_path)
    output = Path(manifest['output'])
    if args.status:
        for name in manifest['skills']:
            progress = model_directory(manifest, name) / 'progress.json'
            if progress.exists():
                value = json.loads(progress.read_text(encoding='utf-8'))
                print(f'{name:10s} {value["status"]:12s} {value["step"]}/{value["steps"]} '
                      f'验证 MAE={value.get("best_validation_mae_degrees", "--")}°')
            else:
                print(f'{name:10s} 等待训练')
        print(f'结果目录：{output}')
        return
    if min(args.steps, args.batch_size, args.chunk_size, args.eval_every, args.checkpoint_every) < 1:
        parser.error('步数、批量、chunk 和保存/验证频率必须为正数')
    lock = output / 'training.lock'
    if lock.exists():
        raise ValueError(f'此目录已有训练锁 {lock}；请先确认已有训练结束，避免重复启动')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    selected = [name for name in args.skills if name in manifest['skills']]
    queue = {'status': 'running', 'pid': os.getpid(), 'manifest': str(manifest_path),
             'manifest_sha256': sha256(manifest_path), 'skills': selected, 'completed': [],
             'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None}
    try:
        save_json(output / 'train_config.json', vars(args) | {'manifest': str(manifest_path), 'device': str(device)})
        save_json(output / 'queue_state.json', queue)
        for name in selected:
            queue['current_skill'] = name
            save_json(output / 'queue_state.json', queue)
            train_skill(name, manifest['skills'][name], manifest, args, device)
            queue['completed'].append(name)
            update_model_index(manifest)
        queue['status'] = 'complete'
    except BaseException as exc:
        queue.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error', reason=str(exc))
        raise
    finally:
        save_json(output / 'queue_state.json', queue)
        lock.unlink()


if __name__ == '__main__':
    main()
