"""只读加载分步模型做离线验证，不连接机械臂或相机。"""
from koch_project_paths import read_project_json
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare_koch_skills import save_json
from train_koch_skills import (SKILLS, IMAGE_KEY, SkillDataset, device_batch, resolve_manifest, model_directory)


def evaluate_skill(name, manifest, device):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    directory = model_directory(manifest, name)
    best = read_project_json(directory / 'best_checkpoint.json')
    model_path = Path(best['checkpoint']) / 'pretrained_model'
    config = ACTConfig.from_pretrained(model_path, local_files_only=True)
    config.device = str(device)
    policy = ACTPolicy.from_pretrained(model_path, config=config, local_files_only=True).to(device).eval()
    stats = json.loads((model_path / 'normalization.json').read_text(encoding='utf-8'))
    clips = [c for c in manifest['skills'][name]['clips'] if c['split'] == 'validation']
    dataset = SkillDataset(clips, config.chunk_size, stats)
    loader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)
    scale = torch.as_tensor(dataset.astd, device=device)
    all_error, all_count, first_error, first_count = 0., 0, 0., 0
    try:
        with torch.no_grad():
            for batch in loader:
                batch = device_batch(batch, device)
                predicted = policy.predict_action_chunk(batch)
                if predicted.shape != batch['action'].shape or not bool(torch.isfinite(predicted).all()):
                    raise ValueError('模型输出维度错误或出现非有限值')
                error = (predicted - batch['action']).abs() * scale
                valid = (~batch['action_is_pad']).unsqueeze(-1)
                all_error += float((error * valid).sum())
                all_count += int(valid.sum()) * predicted.shape[-1]
                first_error += float(error[:, 0].sum())
                first_count += len(error) * predicted.shape[-1]
        return {'skill': name, 'checkpoint_step': best['step'], 'model': str(model_path),
                'validation_sources': sorted(set(c['source'] for c in clips)), 'frames': len(dataset),
                'future_chunk_mae_degrees': all_error / all_count * 360 / 4096,
                'first_action_mae_degrees': first_error / first_count * 360 / 4096,
                'hardware_verified': False, 'split': 'validation_used_for_model_selection'}
    finally:
        dataset.close()
        del policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--skills', nargs='+', choices=SKILLS, default=list(SKILLS))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    manifest = read_project_json(resolve_manifest(args.manifest))
    torch.set_num_threads(4)
    results = []
    for name in args.skills:
        if name not in manifest['skills']:
            continue
        result = evaluate_skill(name, manifest, torch.device(args.device))
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        save_json(Path(manifest['output']) / f'evaluation_{name}.json', result)
    save_json(Path(manifest['output']) / 'offline_evaluation.json', {'results': results,
              'note': '离线教师观测下的误差；不是闭环控制成功率，验证集参与了最佳模型选择'})


if __name__ == '__main__':
    main()
