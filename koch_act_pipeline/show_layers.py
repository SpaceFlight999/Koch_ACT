"""只读查看四组训练队列；--resume 继续同一批次未完成的模型。"""
from koch_robot.paths import ROOT, read_project_json
import argparse
import json
import subprocess
import sys
from pathlib import Path

TITLES={'koch_st':'第一层S→第一层T','koch_st_layer2_to_layer1':'第二层S→第一层T',
        'koch_st_layer1_to_layer2':'第一层S→第二层T','koch_st_layer2_to_layer2':'第二层S→第二层T'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    batch=Path(read_project_json(ROOT/'outputs/koch_layers/latest_batch.json')['batch'])
    if args.resume:
        return subprocess.call([sys.executable,'-B','-X','utf8','-m','koch_act_pipeline.train_layers','--batch',str(batch),'--steps','2000'],cwd=ROOT)
    state=json.loads((batch/'queue_state.json').read_text())
    print('队列状态：',state['status'],'；已验证导出：',len(state.get('completed',[])),'/64')
    print('当前：',TITLES.get(state.get('group'),state.get('group','--')),state.get('skill','准备数据'))
    if state.get('reason'):
        print('停止原因：',state['reason'])
    for group,title in TITLES.items():
        print('\n'+title)
        m=batch/group/'manifest.json'
        if not m.exists():
            print('  正在等待数据准备');continue
        for skill in json.loads(m.read_text())['skills']:
            progress=batch/group/'models'/skill/'progress.json'
            if progress.exists():
                p=json.loads(progress.read_text())
                print(f'  {skill:16s} {p["status"]:10s} {p["step"]}/{p["steps"]} 验证MAE={p.get("best_validation_mae_degrees","--")}°')
            else:
                print(f'  {skill:16s} 等待训练')
    print('\n结果目录：',batch)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
