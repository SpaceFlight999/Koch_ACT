"""四种 S/T 层级独立训练：完整搬运 + 原有 15 项技能，仅使用今天的训练副本。"""
from koch_robot.paths import project_path, read_project_json
import argparse
import json
import os
import shutil
import types
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import koch_act_pipeline.prepare_skills as prep
import koch_act_pipeline.prepare_transport as transport
import koch_act_pipeline.prepare_fine_skills as fine
import koch_act_pipeline.train as train
from koch_act_pipeline.evaluate import evaluate_skill

ROOT = prep.ROOT
GROUPS = ('koch_st', 'koch_st_layer2_to_layer1', 'koch_st_layer1_to_layer2', 'koch_st_layer2_to_layer2')
TODAY = '2026-11-04'


def temporal_loss(predicted, target, padding):
    pair = (~padding[:, 1:] & ~padding[:, :-1]).unsqueeze(-1)
    delta = (predicted[:, 1:] - predicted[:, :-1]) - (target[:, 1:] - target[:, :-1])
    velocity = (delta.abs() * pair).sum() / (pair.sum() * predicted.shape[-1]).clamp_min(1)
    triplet = pair[:, 1:] & pair[:, :-1]
    acceleration = ((delta[:, 1:] - delta[:, :-1]).abs() * triplet).sum() / (triplet.sum() * predicted.shape[-1]).clamp_min(1)
    return velocity, acceleration


def smooth_forward(self, batch):
    batch = dict(batch)
    if self.config.image_features:
        batch['observation.images'] = [batch[k] for k in self.config.image_features]
    predicted, (mu, logvar) = self.model(batch)
    mask = (~batch['action_is_pad']).unsqueeze(-1)
    l1 = ((predicted - batch['action']).abs() * mask).sum() / (mask.sum() * predicted.shape[-1]).clamp_min(1)
    kl = (-.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean() if logvar is not None else l1 * 0
    velocity, acceleration = temporal_loss(predicted, batch['action'], batch['action_is_pad'])
    loss = l1 + self.config.kl_weight * kl + .25 * velocity + .05 * acceleration
    return loss, {'l1_loss': l1.item(), 'kld_loss': kl.item(), 'velocity_loss': velocity.item(), 'acceleration_loss': acceleration.item()}


def decode_images(source, destination):
    meta = json.loads((source / 'episode.json').read_text(encoding='utf-8'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(destination, mode='w+', dtype=np.uint8,
        shape=(meta['camera_frames'], 216, 384, 3))
    capture = cv2.VideoCapture(str(source / 'camera.avi'))
    try:
        for i in range(len(images)):
            ok, frame = capture.read()
            if not ok:
                raise ValueError('视频解码失败：' + str(source))
            images[i] = cv2.cvtColor(cv2.resize(frame, (384,216), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
        images.flush()
        images._mmap.close()


def full_clips(item, image_path, cache):
    source = Path(item['path'])
    rows = prep.jsonl(source / 'joints.jsonl')
    sent = [r for r in rows if r.get('sent') and r.get('command_time') is not None]
    at = np.array([r['command_time'] for r in sent])
    st = np.array([r['follower_time'] for r in rows])
    vt = np.array([r['t'] for r in prep.jsonl(source / 'camera_timestamps.jsonl')])
    goals = np.array([[r['goal_raw'][n] for n in prep.NAMES] for r in sent], dtype=np.float32)
    actual = np.array([[r['follower_raw'][n] for n in prep.NAMES] for r in rows], dtype=np.float32)
    grid = np.arange(max(at[0],st[0],vt[0]), min(at[-1],vt[-1]), .05)
    ai, si, vi = [np.searchsorted(t,grid,side='right')-1 for t in (at,st,vt)]
    valid = (grid-at[ai]<=.2)&(grid-st[si]<=.2)&(grid-vt[vi]<=.25)
    s,t = int(item['route'][1])-1,int(item['route'][-1])-1
    code = np.r_[np.eye(4)[s],np.eye(4)[t]].astype(np.float32)
    result = []
    for i,run in enumerate(prep.contiguous_runs(np.flatnonzero(valid))):
        if len(run)<6:
            continue
        destination = cache / f'{item["route"]}_{source.name}_full_{i}.npz'
        np.savez(destination,state=np.c_[actual[si[run]],np.tile(code,(len(run),1))],
                 action=goals[ai[run]],image_index=vi[run],timestamp=grid[run])
        result.append(dict(skill='full_task',source=str(source),route=item['route'],
            split='validation' if s==t else 'train',arrays=str(destination),images=str(image_path),
            frames=len(run),start=float(grid[run[0]]),end=float(grid[run[-1]])))
    return result


def prepare_group(batch, group, audited):
    out = batch / group
    out.mkdir(exist_ok=True)
    valid, reach_valid, transport_valid, omissions = [], [], set(), []
    for item in audited:
        source = Path(item['path'])
        for name,digest in item['hashes'].items():
            if prep.sha256(source/name)!=digest:
                raise ValueError('原始示范已改变：'+str(source))
        valid.append(item)
        episode = fine.load_episode({'source':str(source)})
        times,goals,actual = episode['times'],episode['goals'],episode['actual']
        cal = json.loads((source/'episode.json').read_text(encoding='utf-8'))['follower_calibration']
        i = cal['motor_names'].index('gripper')
        lo,hi = cal['start_pos'][i],cal['end_pos'][i]
        fraction=np.clip((goals[:,5]-lo)/(hi-lo),0,1)
        try:
            reach,_ = prep.grab_boundary(times,fraction)
            reach_valid.append(item)
            transport.find_stages(times,goals,actual,fraction,(actual[:,5]-lo)/(hi-lo),times[reach])
            transport_valid.add(str(source))
        except ValueError as exc:
            omissions.append(dict(source=str(source),reason=str(exc)))
    # 复用已验证的数据提取，不移动原始数据，也不改旧模型窗口的 latest 指针。
    pointer = ROOT/'outputs/koch_skills/latest_preparation.json'
    real_save = prep.save_json
    def local_save(path,value):
        if Path(path).resolve()!=pointer.resolve():
            real_save(path,value)
    old_audit = prep.audit_all
    original_saves = [m.save_json for m in (prep,transport,fine)]
    try:
        prep.audit_all = lambda _: (reach_valid, [])
        for module in (prep,transport,fine):
            module.save_json = local_save
        reached = prep.run(SimpleNamespace(source=ROOT/'datasets'/group,output=out/'reach_preparation',fps=20,quarantine=False))
        base = read_project_json(reached/'manifest.json')
        base['segments']=[s for s in base['segments'] if s['source'] in transport_valid]
        real_save(out/'transport_input.json',base)
        transported = transport.prepare(out/'transport_input.json')
        prepared = fine.prepare(transported/'manifest.json')
    finally:
        prep.audit_all = old_audit
        for module,save in zip((prep,transport,fine),original_saves):
            module.save_json=save
    manifest=read_project_json(prepared/'manifest.json')
    assets={c['source']:Path(c['images']) for skill in manifest['skills'].values() for c in skill['clips']}
    cache=Path(base['cache'])
    complete=[]
    for item in valid:
        source=Path(item['path'])
        images=assets.get(str(source),cache/f'{item["route"]}_{source.name}'/'images.npy')
        if not images.exists():
            decode_images(source,images)
        complete.extend(full_clips(item,images,cache))
    manifest.update(output=str(out),model_dirs={},source_hashes=valid,source_group=group,
        layer_pair={'koch_st':[1,1],'koch_st_layer2_to_layer1':[2,1],
                    'koch_st_layer1_to_layer2':[1,2],'koch_st_layer2_to_layer2':[2,2]}[group],
        transport_omissions=omissions,capture_date=TODAY,
        smooth_training={'velocity_matching_weight':.25,'acceleration_matching_weight':.05,
                         'temporal_ensemble_coeff':.01,'raw_trajectory_smoothing':False})
    manifest['skills']={'full_task':dict(clips=complete,state_names=list(prep.NAMES)+
        [f'source_S{i}' for i in range(1,5)]+[f'target_T{i}' for i in range(1,5)],action_names=list(prep.NAMES)),**manifest['skills']}
    for name,skill in manifest['skills'].items():
        train_sources={c['source'] for c in skill['clips'] if c['split']=='train'}
        val_sources={c['source'] for c in skill['clips'] if c['split']=='validation'}
        if not train_sources or not val_sources or train_sources&val_sources:
            raise ValueError(f'{group}/{name} 数据划分无效')
    real_save(out/'manifest.json',manifest)
    real_save(out/'preparation_summary.json',{n:dict(train_frames=sum(c['frames'] for c in s['clips'] if c['split']=='train'),
        validation_frames=sum(c['frames'] for c in s['clips'] if c['split']=='validation')) for n,s in manifest['skills'].items()})
    return manifest


def prune_checkpoints(directory,current,batch):
    directory,current,batch=Path(directory).resolve(),Path(current).resolve(),Path(batch).resolve()
    if not directory.is_relative_to(batch) or not current.is_relative_to(directory/'checkpoints'):
        raise ValueError('检查点清理路径越界')
    best_file=directory/'best_checkpoint.json'
    best=Path(read_project_json(best_file)['checkpoint']).resolve() if best_file.exists() else current
    for checkpoint in (directory/'checkpoints').iterdir():
        target=checkpoint.resolve()
        if not target.is_relative_to(batch) or checkpoint.is_symlink() or not checkpoint.name.startswith('step_'):
            raise ValueError('非预期检查点目录')
        if target not in (best,current):
            shutil.rmtree(target)
        elif target!=current and (target/'training_state.pt').exists():
            (target/'training_state.pt').unlink()


def verified_export(manifest, name, steps):
    """恢复队列时跳过已有完整且通过哈希校验的导出。"""
    out = Path(manifest['output'])
    directory = train.model_directory(manifest, name)
    try:
        progress = json.loads((directory/'progress.json').read_text())
        final = json.loads((directory/'finalized.json').read_text())
        best = read_project_json(directory/'best_checkpoint.json')
        evaluation = read_project_json(out/f'evaluation_{name}.json')
        record = json.loads((out/'trained_models/index.json').read_text())['models'][name]
    except (OSError, ValueError, KeyError):
        return False
    if (progress.get('status') != 'complete' or progress.get('step', 0) < steps
            or final.get('steps', 0) < steps or not final.get('best_weights_preserved')
            or record.get('directory') != name or record.get('checkpoint_step') != best.get('step')
            or evaluation.get('checkpoint_step') != best.get('step')):
        return False
    hashes = record.get('sha256', {})
    required = {'config.json', 'model.safetensors', 'skill_schema.json', 'normalization.json'}
    target = out/'trained_models'/name
    if not required.issubset(hashes):
        return False
    return all((target/f).is_file() and prep.sha256(target/f) == hashes[f] for f in required)


def train_batch(batch,steps):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():
        raise ValueError('本批次需要可用 CUDA GPU')
    os.environ['HF_HUB_OFFLINE']='1'
    factory,checkpoint_save=train.build_policy,train.save_checkpoint
    def build(*args,**kwargs):
        policy=factory(*args,**kwargs)
        policy.config.temporal_ensemble_coeff=.01
        policy.config.n_action_steps=1
        policy.forward=types.MethodType(smooth_forward,policy)
        return policy
    def save(policy,optimizer,scheduler,directory,step,stats,schema,val_mae):
        schema.update(layer_pair=current_manifest['layer_pair'],source_group=current_manifest['source_group'],
                      smooth_training=current_manifest['smooth_training'])
        result=checkpoint_save(policy,optimizer,scheduler,directory,step,stats,schema,val_mae)
        prune_checkpoints(directory,result,batch)
        return result
    train.build_policy,train.save_checkpoint=build,save
    manifests=[read_project_json(batch/g/'manifest.json') for g in GROUPS]
    jobs=[(m,n) for n in ('full_task',*train.SKILLS) for m in manifests]
    state=dict(status='training',pid=os.getpid(),total_models=len(jobs),steps_per_model=steps,completed=[])
    args=SimpleNamespace(steps=steps,resume=True,chunk_size=16,batch_size=8,eval_every=200,checkpoint_every=500,
                         policy_forward=smooth_forward)
    try:
        for current_manifest,name in jobs:
            state.update(group=current_manifest['source_group'],skill=name)
            prep.save_json(batch/'queue_state.json',state)
            if verified_export(current_manifest, name, steps):
                train.check_capture_dates(current_manifest['skills'][name]['clips'])
                state['completed'].append(dict(group=current_manifest['source_group'],skill=name))
                prep.save_json(batch/'queue_state.json',state)
                continue
            if shutil.disk_usage(batch).free<2*1024**3:
                raise ValueError('可用磁盘不足 2GB，保留检查点后停止')
            train.train_skill(name,current_manifest['skills'][name],current_manifest,args,torch.device('cuda'))
            train.update_model_index(current_manifest)
            # 每个已完成模型立即验证并导出最佳模型；使用普通复制兼容受限 Windows 文件系统。
            evaluation=evaluate_skill(name,current_manifest,torch.device('cuda'))
            out=Path(current_manifest['output'])
            prep.save_json(out/f'evaluation_{name}.json',evaluation)
            source=Path(evaluation['model'])
            target=out/'trained_models'/name
            target.mkdir(parents=True,exist_ok=True)
            for file in source.iterdir():
                if file.is_file() and not (target/file.name).exists():
                    shutil.copy2(file,target/file.name)
                if file.is_file() and prep.sha256(file)!=prep.sha256(target/file.name):
                    raise ValueError('导出模型哈希不一致：'+str(target/file.name))
            record=dict(directory=name,title=name,checkpoint_step=evaluation['checkpoint_step'],
                first_action_mae_degrees=evaluation['first_action_mae_degrees'],
                sha256={f.name:prep.sha256(f) for f in target.iterdir() if f.is_file()})
            index_path=out/'trained_models/index.json'
            index=json.loads(index_path.read_text()) if index_path.exists() else dict(models={},hardware_verified=False)
            index['models'][name]=record
            prep.save_json(index_path,index)
            # 完成训练的模型保留最佳/最新权重；优化器只用于未完成模型断点续训。
            directory=train.model_directory(current_manifest,name).resolve()
            if not directory.is_relative_to(batch.resolve()):
                raise ValueError('优化器状态清理路径越界')
            for optimizer_state in directory.glob('checkpoints/step_*/training_state.pt'):
                if not optimizer_state.resolve().is_relative_to(batch.resolve()) or optimizer_state.is_symlink():
                    raise ValueError('优化器状态路径异常')
                optimizer_state.unlink()
            prep.save_json(directory/'finalized.json',dict(steps=steps,best_weights_preserved=True,
                optimizer_state_kept=False,note='同目标步数恢复队列会跳过；增加训练步数请另建批次'))
            state['completed'].append(dict(group=current_manifest['source_group'],skill=name))
            prep.save_json(batch/'queue_state.json',state)
        state['status']='complete'
    except BaseException as exc:
        state.update(status='error',reason=str(exc))
        raise
    finally:
        prep.save_json(batch/'queue_state.json',state)
        train.build_policy,train.save_checkpoint=factory,checkpoint_save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000)
    args=parser.parse_args()
    batch=project_path(args.batch).resolve()
    if not batch.is_relative_to(ROOT/'outputs/koch_layers'):
        parser.error('批次须在 outputs/koch_layers 下')
    batch.mkdir(parents=True,exist_ok=True)
    lock=batch/'training.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        cleanup=read_project_json(ROOT/'outputs/dataset_cleanup/20261104_204708_554952/manifest.json')
        for group in GROUPS:
            if not (batch/group/'manifest.json').exists():
                prep.save_json(batch/'queue_state.json',dict(status='preparing',group=group,pid=os.getpid(),total_models=64))
                audited=[i['audit'] for i in cleanup['items'] if i['group']==group and i['action']=='keep']
                prepare_group(batch,group,audited)
        train_batch(batch,args.steps)
    except BaseException as exc:
        path=batch/'queue_state.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(status='error',reason=str(exc))
        prep.save_json(path,state)
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
