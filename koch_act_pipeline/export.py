"""将已离线验证的最佳技能模型复制到独立目录，保留原检查点及 Replay。"""
from koch_robot.paths import project_path, read_project_json
import argparse
import shutil
from pathlib import Path

from koch_act_pipeline.prepare_skills import ROOT, save_json, sha256

TITLES = {'reach_S1': '到达 S1', 'reach_S2': '到达 S2', 'reach_S3': '到达 S3', 'reach_S4': '到达 S4',
          'gripper': '夹爪开合', 'lift': '向上提起', 'transfer': '搬运到指定 T',
          'lower': '向下放置', 'release': '放置后松爪', 'rest_lift': '休息位提起',
          'approach_move': '移到指定 S 上方', 'approach_align': '向下对齐 S',
          'return_lift': 'T 松爪后提起', 'return_move': '移向休息位', 'rest_align': '调整到统一休息姿态'}


def export_models(manifest_path):
    manifest = read_project_json(project_path(manifest_path))
    output = Path(manifest['output']).resolve()
    if not output.is_relative_to((ROOT / 'outputs/koch_skills').resolve()):
        raise ValueError('仅向项目技能结果目录导出')
    index = read_project_json(output / 'model_index.json')
    if set(index) != set(manifest['skills']):
        raise ValueError('还有未完成的模型，不能导出完整技能包')
    exported, evaluations = {}, []
    for name, info in index.items():
        source = Path(info['model']).resolve()
        if not source.is_relative_to((ROOT / 'outputs/koch_skills').resolve()):
            raise ValueError('来源模型路径超出训练结果目录')
        report = read_project_json(output / f'evaluation_{name}.json')
        if Path(report['model']).resolve() != source:
            raise ValueError(f'{name} 的离线验证不是当前最佳模型')
        evaluations.append(report)
        target = output / 'trained_models' / name
        target.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for file in source.iterdir():
            if not file.is_file():
                continue
            destination = target / file.name
            expected = sha256(file)
            if destination.exists() and sha256(destination) != expected:
                raise ValueError('拒绝覆盖不同内容的已有模型')
            if not destination.exists():
                shutil.copy2(file, destination)
            if sha256(destination) != expected:
                raise ValueError('导出文件校验失败')
            hashes[file.name] = expected
        if not {'config.json', 'normalization.json', 'skill_schema.json'}.issubset(hashes):
            raise ValueError('模型配置或标准化参数缺失')
        if not any(n.endswith('.safetensors') for n in hashes):
            raise ValueError('模型权重缺失')
        exported[name] = {'title': TITLES[name], 'directory': name, 'checkpoint_step': info['step'],
                           'first_action_mae_degrees': report['first_action_mae_degrees'],
                           'future_chunk_mae_degrees': report['future_chunk_mae_degrees'], 'sha256': hashes}
    save_json(output / 'trained_models/index.json', {'models': exported, 'hardware_verified': False,
              'rest_pose': manifest.get('rest_pose'), 'requested_workflow': manifest.get('requested_workflow'),
              'workflow_note': manifest.get('workflow_note'),
              'note': '每个子目录含模型、配置、标准化参数和输入输出定义；尚未串联或实机验证'})
    save_json(output / 'offline_evaluation.json', {'results': evaluations,
              'note': '离线教师观测下的验证误差，验证集参与选模，不是实机成功率'})
    rows = ['# Koch 独立技能训练结果', '',
            f'已保存 {len(exported)} 个 ACT 模型。模型均已重新加载并验证输出维度与数值有限性；'
            '下表为验证集选择的最佳检查点，并非训练总步数。', '',
            '| 技能 | 最佳检查点 | 验证首动作 MAE（度） | 验证未来动作 MAE（度） |',
            '|---|---:|---:|---:|']
    for name, value in exported.items():
        rows.append(f'| {value["title"]} | {value["checkpoint_step"]} | '
                    f'{value["first_action_mae_degrees"]:.3f} | {value["future_chunk_mae_degrees"]:.3f} |')
    rows.extend(['', '统一模型包：`trained_models/`。每个模型的输入/输出关节和区域指令见其 `skill_schema.json`，'
                 '推理必须应用随包的 `normalization.json`。', '',
                 '到达/提起/搬运/下放只输出五个身体关节；夹爪/松爪只输出夹爪。'
                 '提起带源区域条件，搬运和下放带目标 T 区条件。', '',
                 '17 条完整原始 Replay、68 个原始数据文件和原回放程序均保留，20 条无效记录在隔离目录中。'
                 '见 `preservation_report.json` 与第一批的 `audit.json`。', '',
                 '当前每条路径示范很少，模型为小样本基线；上述验证集参与了最佳检查点选择。'
                 '未接机械臂，未测试闭环执行、阶段衔接或实际搬运成功率。'])
    if manifest.get('rest_pose'):
        rows.extend(['', '统一休息位取自最大 5° 一致簇中的真实示范姿态，原始关节编码为：`' +
                     str(manifest['rest_pose']['goal_raw']) + '`。', '',
                     '新增取物三段带 S 区指令；返程提起和移动带出发 T 区指令，最终调整统一到上述休息姿态。', '',
                     '组合目标：休息位提起 → 移到 S 上方 → 开爪 → 下对齐 → 夹取 → 提起 → '
                     '移到 T → 下放 → 松爪 → 提起 → 返回 → 调整休息姿态。', '',
                     manifest['workflow_note'], '',
                     '仅从训练副本删除近似同姿态的小回环、停顿及稳定休息后的多余调整，'
                     '详见 fine_annotation_report.json；不修改原始 Replay。'])
    if 'return_move' in exported:
        value = exported['return_move']
        rows.extend(['', f'本轮返程移动的验证首动作 MAE 为 {value["first_action_mae_degrees"]:.3f}°，'
                     f'未来动作 MAE 为 {value["future_chunk_mae_degrees"]:.3f}°。'
                     '这个阶段仍需补充不同 T 返回统一休息位的示范，并继续验证；不能认为已可靠完成自动返程。'])
    (output / 'TRAINING_RESULT.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(f'导出完成：{len(exported)} 个模型 → {output / "trained_models"}', flush=True)
    return output / 'trained_models'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = Path(read_project_json(ROOT / 'outputs/koch_skills/latest_preparation.json')['manifest'])
    export_models(args.manifest)
