"""Bind an offline plan to existing ACT contracts without loading or running them."""
from pathlib import Path

from .reverse import ROOT, read_json, validate_route_layers


def bind_plan(plan, verify_hashes=True):
    from koch_skill_runtime import ModelPack, Stage, latest_pack_paths
    from teleoperate_koch import load_calibration
    from prepare_koch_skills import sha256
    manifests, _ = latest_pack_paths()
    calibration = load_calibration(ROOT / 'calibration/koch_legacy/follower/my_follower.json')
    selected = (Stage('gripper', '开爪', 'open'), Stage('approach_align', '对齐源位置'),
                Stage('gripper', '夹取', 'close'), Stage('lift', '提起'),
                Stage('transfer', '转移'), Stage('lower', '下放'), Stage('release', '松爪'))
    packs, files_checked, moves = {}, set(), []
    for move in plan['moves']:
        pair, route = tuple(move['layer_pair']), move['route']
        validate_route_layers(route, pair)
        if route != f'{move["source"]}_{move["target"]}' or pair != (move['source_layer'], move['target_layer']):
            raise ValueError('规划中的路径、格子和层级不一致。')
        if pair not in packs:
            packs[pair] = ModelPack(manifests[pair])
        pack = packs[pair]
        if tuple(pack.manifest.get('layer_pair', ())) != pair or pack.manifest['follower_calibration'] != calibration:
            raise ValueError('动作包层级或从臂校准与当前任务不一致。')
        stages = []
        for stage in selected:
            pack.check_skill(stage.skill)
            info = pack.index[stage.skill]
            directory = (pack.root / info['directory']).resolve()
            if not directory.is_relative_to(pack.root.resolve()):
                raise ValueError('动作模型目录超出模型包。')
            required = {'config.json', 'model.safetensors', 'skill_schema.json', 'normalization.json'}
            if not required.issubset(info['sha256']):
                raise ValueError('动作模型校验清单不完整。')
            for filename in required:
                key = str(directory / filename)
                if key not in files_checked:
                    if not Path(key).is_file():
                        raise ValueError(f'缺少动作模型文件：{key}')
                    if verify_hashes and sha256(key) != info['sha256'][filename]:
                        raise ValueError(f'动作模型校验失败：{key}')
                    files_checked.add(key)
            schema = read_json(directory / 'skill_schema.json')
            if schema.get('follower_calibration') != calibration or schema.get('units') != 'raw_encoder_ticks' or schema.get('fps') != 20:
                raise ValueError('动作模型输入校准、单位或频率不匹配。')
            clips = pack.clips(stage, route)
            stages.append({'skill': stage.skill, 'intent': stage.intent, 'directory': str(directory),
                           'camera_role': 'global', 'image_size': schema['image_size'],
                           'state_names': schema['state_names'], 'action_names': schema['action_names'],
                           'reference_sources': sorted({c['source'] for c in clips}),
                           'inference_limit_s': pack.duration(stage, route)})
        starts = pack.start_poses(Stage('approach_align', '对齐源位置'), route)
        moves.append({**move, 'manifest': str(pack.path), 'stages': stages,
                      'source_alignment_start_candidates': starts})
    return {'mode': 'offline_model_binding', 'model_files_checked': len(files_checked),
            'sha256_verified': verify_hashes, 'hardware_executed': False, 'deployable': False,
            'moves': moves, 'note': '模型文件/校准/条件已匹配；未验证扫描、空爪过渡、模型接缝或实机成功。'}
