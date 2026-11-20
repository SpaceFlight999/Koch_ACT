"""独立合并组：等待 60 个分项模型，辅助预训练后用完整轨迹微调。"""
from koch_project_paths import project_path, read_project_json
import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train_koch_skills as train
from evaluate_koch_skills import evaluate_skill
from koch_capture_policy import check_capture_dates
from prepare_koch_skills import NAMES, ROOT, jsonl, save_json, sha256
from train_koch_layers import GROUPS, prune_checkpoints, smooth_forward, verified_export


MERGED_ROOT = ROOT / 'outputs/koch_merged'
MODEL = 'merged_full'
PHASE_STEPS, FULL_STEPS = 1200, 2400
LAYER_FEATURES = [f'layer_S{s}_T{t}' for s,t in ((1,1),(2,1),(1,2),(2,2))]
TEACHER_WEIGHT = .25
TEACHER_LIMIT_TICKS = 8 * 4096 / 360


def read(path):
    return read_project_json(path)


def layer_state(state, group_number):
    if state.ndim != 2 or state.shape[1] != 14:
        raise ValueError('完整示范输入必须是六关节＋S/T 八位指令')
    flags = np.eye(4, dtype=np.float32)[group_number]
    return np.c_[state, np.tile(flags, (len(state), 1))].astype(np.float32)


def teacher_train_sources(manifest):
    return {c['source'] for name in train.SKILLS for c in manifest['skills'][name]['clips'] if c['split']=='train'}


def prepare_full(manifests, output):
    """保留原有完整轨迹和间隔切分；新增数组只增加层级条件。"""
    cache = output/'full_arrays'
    cache.mkdir(exist_ok=True)
    full_clips, validation_sources = [], set()
    for number,m in enumerate(manifests):
        teacher_seen = teacher_train_sources(m)
        inputs = m['skills']['full_task']
        if inputs['action_names'] != list(NAMES) or len(inputs['state_names']) != 14:
            raise ValueError('完整轨迹的关节或指令定义不一致')
        held_out = {c['source'] for c in inputs['clips'] if c['split']=='validation'} - teacher_seen
        if not held_out:
            raise ValueError(f'{m["source_group"]} 没有可共同留出的验证示范')
        validation_sources.update(held_out)
        for i,c in enumerate(inputs['clips']):
            destination = cache/f'{number}_{i:04d}.npz'
            with np.load(c['arrays']) as raw:
                np.savez(destination, state=layer_state(raw['state'],number), action=raw['action'],
                         image_index=raw['image_index'], timestamp=raw['timestamp'])
            # 未共同留出的原验证示范也不加入训练，保留与旧完整模型的验证隔离。
            if c['split']=='validation' and c['source'] not in held_out:
                continue
            full_clips.append(dict(c, arrays=str(destination), layer_group=number))
    check_capture_dates(full_clips)
    training_sources={c['source'] for c in full_clips if c['split']=='train'}
    if training_sources & validation_sources:
        raise ValueError('合并训练和验证来源重叠')
    names=manifests[0]['skills']['full_task']['state_names'] + LAYER_FEATURES
    return dict(clips=full_clips,state_names=names,action_names=list(NAMES))


def raw_episode(source):
    rows=jsonl(Path(source)/'joints.jsonl')
    sent=[r for r in rows if r.get('sent') and r.get('command_time') is not None]
    return (np.array([r['follower_time'] for r in rows]),
            np.array([[r['follower_raw'][n] for n in NAMES] for r in rows],dtype=np.float32),
            np.array([r['command_time'] for r in sent]),
            np.array([[r['goal_raw'][n] for n in NAMES] for r in sent],dtype=np.float32))


def aligned_demo(source_values, time_grid, route, layer):
    st,actual,at,goals=source_values
    si=np.searchsorted(st,time_grid,side='right')-1
    ai=np.searchsorted(at,time_grid,side='right')-1
    if np.any(si<0) or np.any(ai<0):
        raise ValueError('分项片段时间早于原始示范')
    code=np.r_[np.eye(4)[int(route[1])-1],np.eye(4)[int(route[-1])-1]].astype(np.float32)
    state=layer_state(np.c_[actual[si],np.tile(code,(len(si),1))],layer)
    return state,goals[ai].copy()


def blended_labels(demo, indexes, anchors, predicted):
    """只在教师负责的轴上添加有界修正；所有帧保留原始时序。"""
    if predicted.shape!=(len(anchors),len(indexes)) or not np.isfinite(predicted).all():
        raise ValueError('分项模型辅助输出无效')
    result=demo.copy()
    t=np.arange(len(demo))
    correction=np.clip(predicted-demo[anchors][:,indexes],-TEACHER_LIMIT_TICKS,TEACHER_LIMIT_TICKS)
    for i,axis in enumerate(indexes):
        result[:,axis] += TEACHER_WEIGHT*np.interp(t,anchors,correction[:,i])
    return result


def load_policy(directory,device):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    config=ACTConfig.from_pretrained(directory,local_files_only=True)
    config.device=str(device)
    config.pretrained_backbone_weights=None
    return ACTPolicy.from_pretrained(directory,config=config,local_files_only=True).to(device).eval()


@torch.inference_mode()
def teacher_outputs(policy, stats, state, images, frames, anchors, device):
    result=[]
    sm,ss,am,astd=(np.asarray(stats[k],dtype=np.float32) for k in ('state_mean','state_std','action_mean','action_std'))
    for start in range(0,len(anchors),32):
        a=anchors[start:start+32]
        batch=train.device_batch({train.IMAGE_KEY:torch.from_numpy(np.array(images[frames[a]],copy=True)).permute(0,3,1,2),
            'observation.state':torch.from_numpy((state[a]-sm)/ss)},device)
        prediction=policy.predict_action_chunk(batch)[:,0].float().cpu().numpy()*astd+am
        result.append(prediction)
    return np.concatenate(result)


def prepare_teachers(manifests, full, output, device, status):
    cache=output/'teacher_arrays';cache.mkdir(exist_ok=True)
    validation=[c for c in full['clips'] if c['split']=='validation']
    held_out={c['source'] for c in validation}
    all_clips=[]
    episodes={}
    for number,m in enumerate(manifests):
        for name in train.SKILLS:
            status('teacher_labels',group=m['source_group'],skill=name,clips=len(all_clips))
            if not verified_export(m,name,2000):
                raise ValueError(f'{m["source_group"]}/{name} 未完成或导出校验失败')
            folder=Path(m['output'])/'trained_models'/name
            stats=read(folder/'normalization.json')
            teacher=load_policy(folder,device)
            teacher_skill=m['skills'][name]
            indexes=[list(NAMES).index(n) for n in teacher_skill['action_names']]
            clips=[c for c in teacher_skill['clips'] if c['split']=='train' and c['source'] not in held_out]
            check_capture_dates(clips)
            for i,c in enumerate(clips):
                with np.load(c['arrays']) as x:
                    state=x['state']; frames=x['image_index']
                    clock=x['source_timestamp'] if 'source_timestamp' in x.files else x['timestamp']
                    if c['source'] not in episodes:
                        episodes[c['source']]=raw_episode(c['source'])
                    merged, action=aligned_demo(episodes[c['source']],clock,c['route'],number)
                    # 先用该阶段原标签，再添加教师有界修正；未负责的轴仍用原示范。
                    action[:,indexes]=x['action']
                    anchors=np.unique(np.r_[np.arange(0,len(state),3),len(state)-1]).astype(int)
                    images=np.load(c['images'],mmap_mode='r')
                    try:
                        pred=teacher_outputs(teacher,stats,state,images,frames,anchors,device)
                    finally:
                        images._mmap.close()
                    target=blended_labels(action,indexes,anchors,pred)
                    destination=cache/f'{number}_{name}_{i:04d}.npz'
                    np.savez(destination,state=merged,action=target,image_index=frames,timestamp=x['timestamp'])
                all_clips.append(dict(c,arrays=str(destination),teacher=name,layer_group=number))
            del teacher
    if not all_clips:
        raise ValueError('没有可用的分项教师片段')
    return dict(full,clips=all_clips+validation)


def manifest_for(manifests,output,skill,phase):
    return dict(output=str(output), model_dirs={}, fps=20, image_size=[384,216], units='raw_encoder_ticks',
        follower_calibration=manifests[0]['follower_calibration'], skills={MODEL:skill}, phase=phase,
        validation_note='整条示范共同留出；不进入分项梯度训练或合并训练，参与选模，不是独立实机测试',
        capture_date='2026-11-04', layer_conditions=LAYER_FEATURES,
        smoothing={'velocity_matching_weight':.25,'acceleration_matching_weight':.05,'temporal_ensemble_coeff':.01})


def train_phase(manifest, steps, device, normalization, batch, warm_start=None):
    factory,save=train.build_policy,train.save_checkpoint
    def build(*args,**kwargs):
        policy=load_policy(warm_start,device) if warm_start else factory(*args,**kwargs)
        policy.config.n_action_steps=1
        policy.config.temporal_ensemble_coeff=.01
        return policy
    def checkpoint(policy,optimizer,scheduler,directory,step,stats,schema,val_mae):
        schema.update(layer_conditions=LAYER_FEATURES,training_phase=manifest['phase'],smooth_training=manifest['smoothing'])
        result=save(policy,optimizer,scheduler,directory,step,stats,schema,val_mae)
        prune_checkpoints(directory,result,batch)
        return result
    train.build_policy,train.save_checkpoint=build,checkpoint
    try:
        train.train_skill(MODEL,manifest['skills'][MODEL],manifest,
            SimpleNamespace(steps=steps,resume=True,chunk_size=16,batch_size=8,eval_every=200,
                            checkpoint_every=500,normalization=normalization,policy_forward=smooth_forward),device)
        train.update_model_index(manifest)
        return Path(read(Path(manifest['output'])/'model_index.json')[MODEL]['model'])
    finally:
        train.build_policy,train.save_checkpoint=factory,save


def run(source,output):
    def status(phase,**fields):
        save_json(output/'queue_state.json',dict(status=phase,pid=os.getpid(),source_batch=str(source),
            updated_at=datetime.now().astimezone().isoformat(),**fields))
    manifests=[read(source/g/'manifest.json') for g in GROUPS]
    if any(m['follower_calibration']!=manifests[0]['follower_calibration'] for m in manifests):
        raise ValueError('四组从臂校准不一致，不能合并')
    full_path=output/'manifest.json'
    if not full_path.exists():
        status('preparing_full')
        full=prepare_full(manifests,output)
        save_json(full_path,manifest_for(manifests,output,full,'full_finetune'))
    final=read(full_path)
    full=final['skills'][MODEL]
    while True:
        q=read(source/'queue_state.json')
        completed=len(q.get('completed',[]))
        if q.get('status')=='complete' and completed==64:
            break
        status('waiting_for_skills', source_status=q.get('status'),source_completed=completed,source_total=64)
        time.sleep(15)
    if not torch.cuda.is_available():
        raise ValueError('合并训练需要 CUDA GPU')
    os.environ['HF_HUB_OFFLINE']='1'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    device=torch.device('cuda')
    pre_dir=output/'phase_pretrain'; pre_dir.mkdir(exist_ok=True)
    pre_path=pre_dir/'manifest.json'
    if not pre_path.exists():
        phases=prepare_teachers(manifests,full,output,device,status)
        save_json(pre_path,manifest_for(manifests,pre_dir,phases,'skill_teacher_pretrain'))
    data=train.SkillDataset([c for c in full['clips'] if c['split']=='train'])
    normalization=data.stats
    data.close()
    status('pretraining',steps=PHASE_STEPS)
    initial=train_phase(read(pre_path),PHASE_STEPS,device,normalization,output)
    status('finetuning',steps=FULL_STEPS)
    model=train_phase(final,FULL_STEPS,device,normalization,output,initial)
    status('evaluating')
    result=evaluate_skill(MODEL,final,device)
    save_json(output/'offline_evaluation.json',result)
    target=output/'trained_models'/MODEL;target.mkdir(parents=True,exist_ok=True)
    for f in model.iterdir():
        if f.is_file() and not (target/f.name).exists():
            shutil.copy2(f,target/f.name)
        if f.is_file() and sha256(f)!=sha256(target/f.name):
            raise ValueError('合并模型导出哈希不一致')
    hashes={f.name:sha256(f) for f in target.iterdir() if f.is_file()}
    save_json(output/'trained_models/index.json',{'models':{MODEL:{'directory':MODEL,'title':'四组联合完整搬运',
        'checkpoint_step':result['checkpoint_step'],'sha256':hashes}},'hardware_verified':False})
    status('complete',model=str(target),first_action_mae_degrees=result['first_action_mae_degrees'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True,help='原四组训练批次')
    parser.add_argument('--output',type=Path,required=True,help='独立合并组的结果目录')
    args=parser.parse_args()
    source,output=project_path(args.batch).resolve(),project_path(args.output).resolve()
    if not source.is_relative_to(ROOT/'outputs/koch_layers') or not output.is_relative_to(MERGED_ROOT) or output==MERGED_ROOT:
        parser.error('批次或结果目录超出指定范围')
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'training.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        run(source,output)
    except BaseException as exc:
        save_json(output/'queue_state.json',dict(status='error',reason=str(exc),source_batch=str(source)))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
