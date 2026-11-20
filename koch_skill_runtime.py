"""独立 ACT 模型运行时：模型线程不接触串口，现有控制器负责电机与退出。"""
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from prepare_koch_skills import ROOT, sha256
from koch_capture_policy import check_capture_dates
from koch_project_paths import project_path, read_project_json
from teleoperate_koch import NAMES, TICK_DEGREES, TeleopError, checked_angles, gripper_closing_contact


@dataclass(frozen=True)
class Stage:
    skill: str
    title: str
    intent: str | None = None


STAGES = (
    Stage('rest_lift', '休息位提起'), Stage('approach_move', '移到 S 上方'),
    Stage('gripper', '打开夹爪', 'open'), Stage('approach_align', '向下对齐 S'),
    Stage('gripper', '闭合夹取', 'close'), Stage('lift', '夹持后提起'),
    Stage('transfer', '移动到 T'), Stage('lower', '向下放置'), Stage('release', '放置后松爪'),
    Stage('return_lift', '松爪后提起'), Stage('return_move', '返回休息位'),
    Stage('rest_align', '调整休息姿态'))
LAYER_GROUPS = {(1, 1): 'koch_st', (2, 1): 'koch_st_layer2_to_layer1',
                (1, 2): 'koch_st_layer1_to_layer2', (2, 2): 'koch_st_layer2_to_layer2'}
FULL_TASK = Stage('full_task', '完整搬运（本层模型）')
MERGED_TASK = Stage('merged_full', '完整搬运（四组合并模型）')
ACTIVE_LIFT = Stage('active_takeoff', '主动起步提起')
ACTIVE_FULL = Stage('merged_active_full', '完整搬运（合并·无起步等待）')
ACTIVE_STAGES = (ACTIVE_LIFT, ACTIVE_FULL)
FULL_STAGES = (FULL_TASK, MERGED_TASK, ACTIVE_FULL)
LAYER_STAGES = (MERGED_TASK, *ACTIVE_STAGES)
ALL_STAGES = STAGES + (FULL_TASK,) + tuple(Stage(f'reach_S{i}', f'到达 S{i}') for i in range(1, 5)) + (MERGED_TASK, *ACTIVE_STAGES)
ACTION_NAMES = (tuple(NAMES[:5]), ('gripper',), tuple(NAMES))


def latest_pack_paths():
    def pointer(folder, key):
        path = ROOT / 'outputs' / folder / 'latest_batch.json'
        return project_path(read_project_json(path)[key])
    layers = pointer('koch_layers', 'batch')
    return ({pair: layers / group / 'manifest.json' for pair, group in LAYER_GROUPS.items()},
            pointer('koch_merged', 'output') / 'manifest.json'
            if (ROOT / 'outputs/koch_merged/latest_batch.json').is_file() else None)


def latest_active_path():
    folder = ROOT / 'outputs/koch_active'
    pointer = folder / 'latest_batch.json'
    if not pointer.is_file():
        return None
    output = project_path(read_project_json(pointer)['output']).resolve()
    if output.parent != folder.resolve():
        raise TeleopError('主动起步训练包路径无效。')
    return output / 'manifest.json' if (output / 'trained_models/index.json').is_file() else None


def make_state(names, raw, route, intent, means, layer_pair=None):
    if not re.fullmatch(r'S[1-4]_T[1-4]', route):
        raise TeleopError('路径必须为 S1_T1 到 S4_T4。')
    values = []
    for i, name in enumerate(names):
        if name in NAMES:
            value = raw[name]
            if name != 'gripper':
                value += 4096 * round((means[i] - value) / 4096)
        elif name in ('intent_open', 'intent_close'):
            if intent not in ('open', 'close'):
                raise TeleopError('夹爪模型需要明确的开/合指令。')
            value = int(name == 'intent_' + intent)
        elif re.fullmatch(r'source_S[1-4]', name):
            value = int(name[-1] == route[1])
        elif re.fullmatch(r'(target_T|from_T)[1-4]', name):
            value = int(name[-1] == route[-1])
        elif re.fullmatch(r'layer_S[12]_T[12]', name):
            if layer_pair not in LAYER_GROUPS:
                raise TeleopError('合并模型需要明确的 S 层和 T 层。')
            value = int((int(name[7]), int(name[10])) == layer_pair)
        else:
            raise TeleopError(f'未知模型输入：{name}')
        values.append(value)
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise TeleopError('模型输入包含非有限值。')
    return result


def action_goals(values, names, previous, calibration):
    values = np.asarray(values)
    if values.shape != (len(names),) or not np.isfinite(values).all() or np.any(np.abs(values) > 8191):
        raise TeleopError('模型动作数值或维度无效，停止。')
    if tuple(names) not in ACTION_NAMES:
        raise TeleopError('模型输出关节定义不匹配。')
    goals = dict(previous)
    for name, value in zip(names, values):
        if name == 'gripper':
            i = calibration['motor_names'].index(name)
            low, high = sorted((calibration['start_pos'][i], calibration['end_pos'][i]))
            value = np.clip(value, low, high)
        else:
            value += 4096 * round((previous[name] - value) / 4096)
        goals[name] = round(float(value))
    checked_angles(goals, calibration)
    return goals


class Actor:
    def __init__(self, directory, device, calibration, layer_pair=None):
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        self.torch = torch
        self.schema = json.loads((directory / 'skill_schema.json').read_text(encoding='utf-8'))
        if self.schema['follower_calibration'] != calibration:
            raise TeleopError('模型与当前从臂校准不一致，不能执行。')
        if self.schema['units'] != 'raw_encoder_ticks' or self.schema['fps'] != 20:
            raise TeleopError('本程序要求原始 tick、20 Hz 训练的模型。')
        self.names = tuple(self.schema['action_names'])
        if self.names not in ACTION_NAMES:
            raise TeleopError('模型输出关节定义不匹配。')
        self.state_names = self.schema['state_names']
        self.layer_pair = layer_pair
        stats = json.loads((directory / 'normalization.json').read_text(encoding='utf-8'))
        self.sm, self.ss, self.am, self.astd = [np.asarray(stats[k], dtype=np.float32) for k in
            ('state_mean', 'state_std', 'action_mean', 'action_std')]
        for a, length in ((self.sm, len(self.state_names)), (self.ss, len(self.state_names)),
                          (self.am, len(self.names)), (self.astd, len(self.names))):
            if a.shape != (length,) or not np.isfinite(a).all():
                raise TeleopError('模型标准化参数无效。')
        if np.any(self.ss <= 0) or np.any(self.astd <= 0):
            raise TeleopError('模型标准差必须为正。')
        self.device = torch.device(device)
        self.image_mean = torch.tensor(stats['image_mean'], device=self.device)[None, :, None, None]
        self.image_std = torch.tensor(stats['image_std'], device=self.device)[None, :, None, None]
        config = ACTConfig.from_pretrained(directory, local_files_only=True)
        config.device = str(device)
        # 完整权重已在包内，避免初始化骨干时另行下载 ImageNet 权重。
        config.pretrained_backbone_weights = None
        self.policy = ACTPolicy.from_pretrained(directory, config=config, local_files_only=True).to(device).eval()

    def predict(self, rgb, raw, route, intent=None):
        torch = self.torch
        if rgb.shape != (216, 384, 3) or rgb.dtype != np.uint8:
            raise TeleopError('模型图像必须为 384×216 的 RGB uint8。')
        state = make_state(self.state_names, raw, route, intent, self.sm, self.layer_pair)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].to(self.device).float() / 255
        batch = {self.schema['image_key']: (image - self.image_mean) / self.image_std,
                 'observation.state': torch.from_numpy((state - self.sm) / self.ss)[None].to(self.device)}
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)
        if tuple(chunk.shape) != (1, self.policy.config.chunk_size, len(self.names)) or not torch.isfinite(chunk).all():
            raise TeleopError('模型预测维度异常或出现 NaN/Inf。')
        return chunk[0, 0].float().cpu().numpy() * self.astd + self.am


class ModelPack:
    def __init__(self, manifest_path=None):
        if manifest_path is None:
            pointer = read_project_json(ROOT / 'outputs/koch_layers/latest_batch.json')
            manifest_path = project_path(pointer['batch']) / 'koch_st/manifest.json'
        self.path = project_path(manifest_path).resolve()
        self.manifest = read_project_json(self.path)
        self.root = self.path.parent / 'trained_models'
        index_path = self.root / 'index.json'
        if not index_path.is_file():
            raise TeleopError(f'训练包缺少统一模型索引：{index_path}')
        self.index = json.loads(index_path.read_text(encoding='utf-8'))['models']
        self._capture_dates = {}

    def check_skill(self, name):
        if name not in self.index or name not in self.manifest['skills']:
            raise TeleopError(f'{name} 尚未完成并导出。')
        try:
            check_capture_dates(self.manifest['skills'][name]['clips'], self._capture_dates)
        except ValueError as exc:
            raise TeleopError(f'{name}：{exc}') from exc

    def load(self, stages, device, calibration, layer_pair=None):
        if any(stage in LAYER_STAGES for stage in stages) and layer_pair not in LAYER_GROUPS:
            raise TeleopError('请选择合并模型的储存层和放置层。')
        for name in dict.fromkeys(stage.skill for stage in stages):
            self.check_skill(name)
        import torch
        torch.set_num_threads(4)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise TeleopError('CUDA 不可用，可选择 --device cpu。')
        actors = {}
        for stage in stages:
            if stage.skill in actors:
                continue
            info = self.index[stage.skill]
            directory = (self.root / info['directory']).resolve()
            if not directory.is_relative_to(self.root.resolve()):
                raise TeleopError('模型路径超出模型包。')
            required = {'config.json', 'model.safetensors', 'skill_schema.json', 'normalization.json'}
            if not required.issubset(info['sha256']):
                raise TeleopError('模型包文件校验清单不完整。')
            for filename in required:
                if sha256(directory / filename) != info['sha256'][filename]:
                    raise TeleopError(f'{stage.skill}/{filename} 校验失败。')
            print(f'加载模型：{stage.skill} ({device})', flush=True)
            actors[stage.skill] = Actor(directory, device, calibration, layer_pair)
        return actors

    def clips(self, stage, route, layer_pair=None):
        clips = [c for c in self.manifest['skills'][stage.skill]['clips']
                 if stage.intent is None or c.get('intent') == stage.intent]
        if stage in LAYER_STAGES:
            if layer_pair not in LAYER_GROUPS:
                raise TeleopError('请选择合并模型的储存层和放置层。')
            group = tuple(LAYER_GROUPS).index(layer_pair)
            clips = [c for c in clips if c['layer_group'] == group]
        matched = [c for c in clips if c['route'] == route]
        if stage in (*FULL_STAGES, ACTIVE_LIFT) and not matched:
            raise TeleopError(f'{route} 在所选层级中没有可用的{stage.title}示范。')
        if matched:
            return matched
        names = self.manifest['skills'][stage.skill]['state_names']
        if 'source_S1' in names:
            clips = [c for c in clips if c['route'][:2] == route[:2]]
        if 'target_T1' in names or 'from_T1' in names:
            clips = [c for c in clips if c['route'][-2:] == route[-2:]]
        if not clips:
            raise TeleopError(f'{stage.title} 缺少所选格子的可用示范。')
        return clips

    def duration(self, stage, route, layer_pair=None):
        clips = self.clips(stage, route, layer_pair)
        if stage in FULL_STAGES:
            bounds = {}
            for c in clips:
                start, end = bounds.get(c['source'], (c['start'], c['end']))
                bounds[c['source']] = min(start, c['start']), max(end, c['end'])
            return float(np.clip(np.median([end - start for start, end in bounds.values()]), .5, 120))
        # 只估计单段推理时限，不将示范时长当作动作成功判据。
        durations = [c.get('training_duration_s', c['end'] - c['start']) for c in clips]
        return float(np.clip(np.median(durations), .5, 30))

    def start_poses(self, stage, route, layer_pair=None):
        clips = self.clips(stage, route, layer_pair)
        # 每条示范只取最早的片段，避免误把通信中断后的片段当成动作起点。
        first = {}
        for clip in sorted(clips, key=lambda c: c['start']):
            first.setdefault(clip['source'], clip)
        names = self.manifest['skills'][stage.skill]['state_names']
        poses = []
        for clip in first.values():
            with np.load(clip['arrays']) as data:
                poses.append({n: round(float(data['state'][0, names.index(n)])) for n in NAMES[:5]})
        return poses


class InferenceWorker:
    def __init__(self, actors, route, clock=time.monotonic):
        self.actors, self.route, self.clock = actors, route, clock
        self.condition = threading.Condition()
        self.pending = self.result = self.error = None
        self.ended = False
        self.counter = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name='koch-model-inference')

    def start(self):
        self.thread.start()

    def submit(self, epoch, stage, raw, rgb, camera_time):
        with self.condition:
            self.counter += 1
            self.pending = (self.counter, epoch, stage, dict(raw), rgb, camera_time, self.clock())
            self.condition.notify()

    def snapshot(self):
        with self.condition:
            if self.error is not None:
                raise TeleopError(f'模型推理失败：{self.error}')
            return self.result

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.ended or self.pending is not None)
                    if self.ended:
                        return
                    job, self.pending = self.pending, None
                serial, epoch, stage, raw, rgb, camera_time, observed = job
                started = self.clock()
                if started - min(camera_time, observed) > .3:
                    continue
                value = self.actors[stage.skill].predict(rgb, raw, self.route, stage.intent)
                result = dict(serial=serial, epoch=epoch, values=value, observed=observed,
                              camera_time=camera_time, inference_seconds=self.clock() - started)
                with self.condition:
                    self.result = result
                    self.condition.wait_for(lambda: self.ended, timeout=max(0, .05 - (self.clock() - started)))
        except BaseException as exc:
            with self.condition:
                self.error = str(exc) or type(exc).__name__

    def close(self):
        with self.condition:
            self.ended = True
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise TeleopError('推理线程未退出；机械臂控制已经结束，请退出程序。')


class SkillSource:
    def __init__(self, actors, stages, durations, route, calibration, camera, emit=print, *,
                 clock=time.monotonic, start_tolerance=10):
        if not stages or len(stages) != len(durations) or any(not math.isfinite(d) or not .1 <= d <= 120 for d in durations):
            raise TeleopError('每段推理时限须在 0.1–120 秒内。')
        self.actors, self.stages, self.durations = actors, stages, durations
        self.route, self.cal, self.camera, self.emit, self.clock = route, calibration, camera, emit, clock
        self.start_tolerance = start_tolerance
        self.worker = InferenceWorker(actors, route, clock)
        self.stage_index = 0
        self.waiting = self.settled = self.advance = self.initialized = False
        self.last_inference = None
        self.primed = None
        self.phase_start = self.last_submit = 0.
        self.serial = -1
        self.pending_since = None

    @property
    def stage(self):
        return self.stages[self.stage_index]

    @property
    def is_last(self):
        return self.stage_index == len(self.stages) - 1

    def observation(self):
        self.camera.check()
        with self.camera.lock:
            stamp, frame = self.camera.latest
        if self.clock() - stamp > .3:
            raise TeleopError('相机画面超过 0.3 秒未更新，停止模型控制。')
        return stamp, cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (384, 216), interpolation=cv2.INTER_AREA)

    def prime(self, actual):
        _, rgb = self.observation()
        actor = self.actors[self.stage.skill]
        self.primed = actor.predict(rgb, actual, self.route, self.stage.intent)
        self.check_start(actual)

    def prime_live(self, actual):
        """回到起点后用实时图像预热，不阻塞串口控制循环。"""
        now = self.clock()
        if self.pending_since is None:
            self.pending_since = now
            self.last_submit = -math.inf
        result = self.worker.snapshot()
        if result is not None and result['epoch'] == -1 and now - min(result['camera_time'], result['observed']) <= .3:
            self.primed = result['values']
            self.check_start(actual)
            self.pending_since = None
            return True
        if now - self.pending_since >= 3:
            raise TeleopError('归位后未取得及时的模型预测，停止。')
        if now - self.last_submit >= .1:
            stamp, rgb = self.observation()
            self.worker.submit(-1, self.stage, actual, rgb, stamp)
            self.last_submit = now
        return False

    def check_start(self, actual):
        if self.primed is None:
            raise TeleopError('模型尚未预热。')
        actor = self.actors[self.stage.skill]
        target = action_goals(self.primed, actor.names, actual, self.cal)
        far = [n for n in actor.names if n != 'gripper'
               and abs(target[n] - actual[n]) * TICK_DEGREES > self.start_tolerance]
        if far:
            distances = '、'.join(
                f'{n} {abs(target[n] - actual[n]) * TICK_DEGREES:.1f}°' for n in far)
            self.emit(f'模型首目标与当前位置差距较大（{distances}），'
                      '将从当前姿态按控制速度平滑接近。')

    def start(self):
        self.worker.start()

    def close(self):
        self.worker.close()

    def request_next(self):
        if self.waiting and self.settled and not self.is_last:
            self.advance = True
            return True
        return False

    def __call__(self, elapsed, actual, previous):
        now = self.clock()
        if not self.initialized:
            self.check_start(actual)
            self.initialized = True
            self.target = self.origin = dict(actual)
            self.blend_start = elapsed
            self.last_submit = -math.inf
            return dict(actual)
        stamp, rgb = self.observation()
        result = self.worker.snapshot()
        if self.advance:
            self.stage_index += 1
            self.advance = self.waiting = self.settled = False
            self.phase_start, self.blend_start = elapsed, elapsed
            self.target = self.origin = dict(previous)
            self.pending_since = now
            self.last_submit = -math.inf
            self.last_inference = None
            self.emit(f'进入第 {self.stage_index + 1} 段：{self.stage.title}')
        if not self.waiting and elapsed - self.phase_start >= self.durations[self.stage_index]:
            # 即使模型无有效输出，也不能把空等时限记成段完成。
            if self.last_inference is None:
                raise TeleopError('本段没有及时获得有效预测，停止。')
            self.waiting = True
            self.emit(f'{self.stage.title} 推理时限结束，保持末目标；确认动作后按 N，托稳后按 F 结束。')
        actor = self.actors[self.stage.skill]
        if not self.waiting:
            if now - self.last_submit >= .04:
                self.worker.submit(self.stage_index, self.stage, actual, rgb, stamp)
                self.last_submit = now
            fresh = (result is not None and result['epoch'] == self.stage_index
                     and now - min(result['observed'], result['camera_time']) <= .3)
            if fresh:
                self.pending_since = None
                if result['serial'] != self.serial:
                    self.target = action_goals(result['values'], actor.names, previous, self.cal)
                    self.origin, self.blend_start = dict(previous), elapsed
                    self.serial = result['serial']
                    self.last_inference = {k: v for k, v in result.items() if k != 'values'}
            else:
                if self.pending_since is None:
                    self.pending_since = now
                if now - self.pending_since >= 1:
                    raise TeleopError('超过 1 秒没有新鲜模型预测，停止。')
                # 通信恢复或推理延迟时保持实际已下发的目标，不继续追赶旧预测。
                self.target = self.origin = dict(previous)
                self.blend_start = elapsed
                return dict(previous)
        weight = min(1., max(0., (elapsed - self.blend_start) / .05))
        goals = dict(previous)
        for name in actor.names:
            goals[name] = round(self.origin[name] + weight * (self.target[name] - self.origin[name]))
        checked_angles(goals, self.cal)
        self.settled = self.waiting and all(abs(goals[n] - actual[n]) * TICK_DEGREES <= 3
            or (n == 'gripper' and gripper_closing_contact(actual[n], previous[n], goals[n], self.cal)) for n in NAMES)
        return goals
