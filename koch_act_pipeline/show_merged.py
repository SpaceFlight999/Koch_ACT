"""查看独立合并组的进度。"""
from koch_robot.paths import ROOT, read_project_json
import argparse
import json
import subprocess
import sys
from pathlib import Path




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    pointer=read_project_json(ROOT/'outputs/koch_merged/latest_batch.json')
    output=Path(pointer['output'])
    if args.resume:
        return subprocess.call([sys.executable,'-B','-X','utf8','-m','koch_act_pipeline.train_merged',
            '--batch',pointer['source_batch'],'--output',str(output)],cwd=ROOT)
    status=json.loads((output/'queue_state.json').read_text())
    names={'preparing_full':'准备完整轨迹','waiting_for_skills':'等待分项模型导出',
           'teacher_labels':'生成分项辅助标签','pretraining':'分项辅助预训练',
           'finetuning':'完整轨迹联合微调','evaluating':'离线验证','complete':'已完成','error':'已停止'}
    print('合并组：',names.get(status['status'],status['status']))
    if 'source_completed' in status:print(f'原四组已导出 {status["source_completed"]}/{status["source_total"]}')
    if 'skill' in status:print('当前分项：',status['group'],status['skill'])
    if 'reason' in status:print('原因：',status['reason'])
    for phase,path in [('预训练',output/'phase_pretrain'),('完整微调',output)]:
        f=path/'models/merged_full/progress.json'
        if f.exists():
            p=json.loads(f.read_text());print(phase,f'{p["step"]}/{p["steps"]} 步',p['status'])
    print('保存目录：',output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
