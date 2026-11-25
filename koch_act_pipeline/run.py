"""独立模型控制窗口；打开窗口不连接机械臂，点击执行后才启用从臂。"""
import argparse
import math
import queue
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from koch_robot.collect import disconnect
from koch_robot.dataset import Camera, EpisodeRecorder
from koch_act_pipeline.runtime import (ACTIVE_STAGES, ALL_STAGES, FULL_TASK, LAYER_GROUPS, MERGED_TASK, ModelPack,
                                SkillSource, STAGES, action_goals, latest_active_path, latest_pack_paths)
from koch_robot.teleop import (ROOT, NAMES, TICK_DEGREES, Teleoperator, TeleopError,
                              checked_angles, connect_bus, make_bus, load_calibration, limit_speed)

RUNS = ROOT / 'outputs/koch_model_control/runs'
CALIBRATION = ROOT / 'calibration/koch_legacy/follower/my_follower.json'
HOME_WARNING_DEGREES = 15.0
HOME_SPEED = 10.0
HOME_PROGRESS_SECONDS = 3.0
MODEL_START_SETTLE_DEGREES = 3.0
MODEL_START_WARNING_DEGREES = 10.0
SHOULDER = 'shoulder_lift'
HOME_ACTION = '单独回休息位'


def stages_for(index, route):
    if index == 0:
        return list(STAGES)
    stage = ALL_STAGES[index - 1]
    if stage.skill.startswith('reach_') and stage.skill[-2:] != route[:2]:
        raise TeleopError(f'请将储存区选为 {stage.skill[-2:]}，与所选到达模型一致。')
    return [stage]


def rest_poses(manifest):
    pose = manifest['rest_pose']
    names, positions = pose['joint_names'], pose['goal_raw']
    if tuple(names) != NAMES[:5] or len(positions) != len(names) or any(type(x) is not int for x in positions):
        raise TeleopError('训练包的休息位定义无效。')
    return [dict(zip(names, positions))]


class StartAlignment:
    """从当前姿态低速收敛到示范起点；夹爪保持原状并持续读取反馈。"""
    def __init__(self, poses, calibration, source, label, speed, emit):
        self.poses, self.calibration, self.source, self.label = poses, calibration, source, label
        self.speed, self.run_speed, self.emit = min(HOME_SPEED, speed), speed, emit
        self.target = self.origin = self.stable_since = self.handoff = None
        self.last_command = None
        self.remainder = {}
        self.initial_error = self.progress_error = self.progress_since = None
        self.settled = False

    def __call__(self, elapsed, actual, previous):
        if self.handoff is not None:
            since = elapsed - self.handoff
            goals = self.source(since, actual, previous)
            if since < 1:
                speed = self.speed + since * (self.run_speed - self.speed)
                goals = limit_speed(goals, previous, speed, elapsed - self.last_command, self.remainder)
            self.last_command = elapsed
            return goals
        if self.target is None:
            candidates = [dict(actual, **{n: v + 4096 * round((actual[n] - v) / 4096)
                                         for n, v in pose.items()}) for pose in self.poses]
            def distance(pose):
                return max(abs(pose[n] - actual[n]) for n in NAMES[:5])
            self.target = min(candidates, key=lambda pose: (distance(pose),
                sum(abs(pose[n] - actual[n]) for n in NAMES[:5])))
            checked_angles(self.target, self.calibration)
            self.initial_error = distance(self.target) * TICK_DEGREES
            self.progress_error, self.progress_since = self.initial_error, elapsed
            self.origin = dict(actual)
            self.last_command = elapsed
            if self.initial_error > HOME_WARNING_DEGREES:
                self.emit(f'检测到{self.label}偏差 {self.initial_error:.1f}°，'
                          f'将以不超过 {self.speed:g}°/s 自动修正，夹爪保持当前位置。')
            else:
                self.emit(f'正在低速对齐{self.label}（偏差 {self.initial_error:.1f}°），夹爪保持当前位置。')
            return dict(actual)
        error = max(abs(self.target[n] - actual[n]) for n in NAMES[:5]) * TICK_DEGREES
        if error < self.progress_error - .2:
            self.progress_error, self.progress_since = error, elapsed
        goals = dict(previous)
        for n in NAMES[:5]:
            delta = self.target[n] - self.origin[n]
            step = min(abs(delta), self.speed * elapsed / TICK_DEGREES)
            desired = self.origin[n] + round(math.copysign(step, delta))
            # 不累积远离反馈的命令；阻滞突然解除时不会追赶一串旧目标。
            goals[n] = min(actual[n] + 34, max(actual[n] - 34, desired))
        goals = limit_speed(goals, previous, self.speed, elapsed - self.last_command, self.remainder)
        self.last_command = elapsed
        ramp_done = all(goals[n] == self.target[n] for n in NAMES[:5])
        # 示范起点和分步到位用相同的反馈容差；单独归位仍保持 2°。
        tolerance = MODEL_START_SETTLE_DEGREES if self.source is not None else 2.0
        if error <= tolerance and ramp_done:
            if self.stable_since is None:
                self.stable_since = elapsed
            if elapsed - self.stable_since >= .25:
                if not self.settled:
                    self.settled = True
                    self.emit(f'已到达{self.label}。')
                if self.source is not None and self.source.prime_live(actual):
                    self.handoff = self.last_command = elapsed
                    self.emit(f'开始执行：{self.source.stage.title}。')
                    self.source(0, actual, goals)
                    return goals
        else:
            self.stable_since = None
            self.settled = False
        if not self.settled and elapsed - self.progress_since > HOME_PROGRESS_SECONDS:
            joint = max(NAMES[:5], key=lambda n: abs(self.target[n] - actual[n]))
            raise TeleopError(f'未能对齐{self.label}：ID {NAMES.index(joint) + 1} {joint} 连续 '
                              f'{HOME_PROGRESS_SECONDS:g} 秒没有明显接近，仍相差 {error:.1f}°，停止。')
        if not self.settled and elapsed > self.initial_error / self.speed + 5:
            joint = max(NAMES[:5], key=lambda n: abs(self.target[n] - actual[n]))
            raise TeleopError(f'未能对齐{self.label}：ID {NAMES.index(joint) + 1} {joint} '
                              f'仍相差 {error:.1f}°，停止。')
        return goals


class ShoulderWatch:
    """模型调用专用：目标持续偏离、肩关节又没有靠近目标时停止。"""
    def __init__(self):
        self.since, self.baseline, self.position = None, 0, 0

    def check(self, frame):
        actual = frame['follower_raw'][SHOULDER]
        error = (frame['goal_raw'][SHOULDER] - actual) * TICK_DEGREES
        progress = (actual - self.position) * TICK_DEGREES * math.copysign(1, error)
        now = frame['follower_time']
        if abs(error) < 6:
            self.since = None
        elif self.since is None or error * self.baseline < 0 or abs(error) < abs(self.baseline) - 1 or progress > 1:
            self.since, self.baseline, self.position = now, error, actual
        elif now - self.since >= 2:
            raise TeleopError(f'从臂 ID 2 {SHOULDER} 持续未接近目标，误差 {abs(error):.1f}°，停止以免持续施力。')


def fault_detail(bus, reason):
    match = re.search(r'ID ([1-6])\b', reason)
    if not match:
        return reason
    motor_id = int(match[1])
    try:
        data = bus.read_fault_snapshot(NAMES[motor_id - 1])
        detail = (f'故障读取：ID {motor_id} 状态码 0x{data["fault"]:02X}，'
                  f'电压 {data["voltage"]:.1f}V，温度 {data["temperature"]}°C，扭矩 {data["torque"]}。')
    except Exception as exc:
        detail = f'故障详情未能读取：{exc}'
    return reason + '\n' + detail


def run_session(config, messages, stop, finish, next_stage):
    camera = source = recorder = None
    buses, errors = [], []
    status, reason = 'model_stopped', ''
    def emit(message):
        messages.put(('log', str(message)))
    try:
        if config.preview:
            camera = Camera(config.camera)
            messages.put(('camera', camera))
            emit('相机预览中，不连接机械臂。')
            while not stop.wait(.05) and not finish.is_set():
                camera.check()
            return
        home_only = getattr(config, 'home_only', False)
        layers = getattr(config, 'layers', None)
        pack = ModelPack(config.manifest)
        calibration = load_calibration(CALIBRATION)
        if pack.manifest['follower_calibration'] != calibration:
            raise TeleopError('模型与当前从臂校准不一致，不能执行。')
        stages = [] if home_only else stages_for(config.stage, config.route)
        example_pack = pack
        if stages == [MERGED_TASK]:
            # 合并训练索引未包含每一条路线的示范，归位和时长用同层同格的完整示范。
            example_pack = ModelPack(config.rest_manifest)
            if (example_pack.manifest['follower_calibration'] != calibration
                    or tuple(example_pack.manifest.get('layer_pair', ())) != layers):
                raise TeleopError('合并模型的启动示范与所选层级或当前从臂校准不一致。')
        poses, label = None, '动作起点'
        if home_only:
            poses, label = rest_poses(pack.manifest), '休息位'
        elif stages[0].skill not in {'gripper', 'release'}:
            example_stage = FULL_TASK if stages[0] == MERGED_TASK else stages[0]
            poses = example_pack.start_poses(example_stage, config.route, layers)
            label = '示范起点'
        if not home_only:
            emit('正在加载模型，期间可随时停止。')
            actors = pack.load(stages, config.device, calibration, layers)
            durations = [config.duration or example_pack.duration(
                FULL_TASK if s == MERGED_TASK else s, config.route, layers) for s in stages]
        if stop.is_set() or finish.is_set():
            return
        if not home_only:
            camera = Camera(config.camera)
            messages.put(('camera', camera))
            if stop.is_set() or finish.is_set():
                return
            recorder = EpisodeRecorder(RUNS, config.route, camera, None, calibration)
            recorder.meta.update(mode='trained_model', models=str(pack.root), layer_pair=layers,
                stage_names=[s.title for s in stages], stage_durations=durations,
                task_success_verified=False, transition='manual_confirmation', speed=config.speed)
            messages.put(('path', str(recorder.path)))
            camera.attach(recorder)
        bus = connect_bus('follower', config.port, factory=make_bus)
        buses.append(bus)
        control = Teleoperator(None, bus, None, calibration, min(config.speed, HOME_SPEED) if home_only else config.speed)
        if not home_only:
            source = SkillSource(actors, stages, durations, config.route, calibration, camera, emit=emit,
                                 start_tolerance=MODEL_START_WARNING_DEGREES)
            if poses is None:
                _, actual = control.preflight()
                source.prime(actual)
            if stop.is_set() or finish.is_set():
                return
            source.start()
            if poses is None:
                emit(f'开始执行：{source.stage.title}。')
        alignment = StartAlignment(poses, calibration, source, label, config.speed, emit) if poses is not None else None
        target = alignment if alignment is not None else source
        watch = ShoulderWatch()
        def halted():
            if stop.is_set() or finish.is_set():
                return True
            if recorder is not None:
                recorder.check()
                camera.check()
            if alignment is not None and alignment.handoff is None:
                messages.put(('state', (1, '对齐' + label, alignment.settled, alignment.settled, True)))
                next_stage.clear()
                return False
            if next_stage.is_set():
                next_stage.clear()
                if not source.request_next():
                    emit('当前尚未到段末并稳定，或已是最后一段。')
            messages.put(('state', (source.stage_index + 1, source.stage.title,
                source.waiting, source.settled, source.is_last)))
            return False
        def record_frame(frame):
            if source is None:
                return
            aligning = alignment is not None and alignment.handoff is None
            if aligning and 'startup_target_raw' not in recorder.meta:
                recorder.meta.update(startup_reference=label, startup_target_raw=dict(alignment.target))
            frame.update(model_stage='startup_alignment' if aligning else source.stage.skill, model_intent=source.stage.intent,
                         model_inference=source.last_inference)
            recorder.joint_frame(frame)
            if not aligning:
                watch.check(frame)
        control.run(target_source=target, stop=halted, on_frame=record_frame, emit=emit)
        if source is not None and finish.is_set() and source.waiting and source.settled and source.is_last:
            status = 'model_sequence_ended'
        emit('控制已结束，从臂扭矩已释放。')
    except BaseException as exc:
        status, reason = 'error', str(exc) or type(exc).__name__
        if buses:
            reason = fault_detail(buses[-1], reason)
        emit('停止：' + reason)
    finally:
        disconnect(buses)
        messages.put(('camera', None))
        if camera is not None:
            camera.attach(None)
        if recorder is not None:
            try:
                recorder.finish(status, reason)
            except Exception as exc:
                errors.append('记录保存：' + str(exc))
        for resource in (source, camera):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(str(exc))
        for error in errors:
            emit('退出清理：' + error)
        messages.put(('done', None))


def probe(manifest, stage_index, route, device, layers=(1, 1)):
    stages = stages_for(stage_index, route)
    is_merged = stages == [MERGED_TASK]
    is_active = stages[0] in ACTIVE_STAGES
    if manifest is None or is_merged:
        paths, merged = latest_pack_paths()
    if manifest is None:
        manifest = latest_active_path() if is_active else merged if is_merged else paths[layers]
        if manifest is None:
            raise TeleopError('所选模型尚未导出。')
    pack = ModelPack(manifest)
    calibration = pack.manifest['follower_calibration']
    actors = pack.load(stages, device, calibration, layers)
    examples = ModelPack(paths[layers]) if is_merged else pack
    if examples.manifest['follower_calibration'] != calibration:
        raise TeleopError('离线示范与模型校准不一致。')
    for stage in stages:
        example_stage = FULL_TASK if is_merged else stage
        clip = examples.clips(example_stage, route, layers)[0]
        with np.load(clip['arrays']) as data:
            i = len(data['state']) // 2
            raw = dict(zip(NAMES, calibration['start_pos']))
            for name, value in zip(examples.manifest['skills'][example_stage.skill]['state_names'], data['state'][i]):
                if name in NAMES:
                    raw[name] = round(float(value))
            images = np.load(clip['images'], mmap_mode='r')
            try:
                rgb = np.array(images[int(data['image_index'][i])])
            finally:
                images._mmap.close()
            actor = actors[stage.skill]
            values = actor.predict(rgb, raw, route, stage.intent)
            goals = action_goals(values, actor.names, raw, calibration)
            print(f'{stage.title}: ' + ' '.join(f'{n}={goals[n]}' for n in actor.names), flush=True)
    print('离线模型调用通过；没有打开相机或串口。')


class ModelWindow:
    def __init__(self, root, manifest=None):
        import tkinter as tk
        from tkinter import ttk
        self.root = root
        self.messages, self.stop, self.finish, self.next_stage = queue.Queue(), threading.Event(), threading.Event(), threading.Event()
        self.worker = self.camera = None
        self.closing = False
        self.last_stamp = None
        self.packs, self.merged, self.active = {}, None, None
        errors = []
        def load(path):
            try:
                return ModelPack(path)
            except (OSError, ValueError, KeyError, TeleopError) as exc:
                errors.append(str(exc))
                return None
        try:
            if manifest is None:
                paths, merged_path = latest_pack_paths()
                for pair, path in paths.items():
                    if pack := load(path):
                        self.packs[pair] = pack
                if merged_path:
                    self.merged = load(merged_path)
                if active_path := latest_active_path():
                    self.active = load(active_path)
            else:
                if pack := load(manifest):
                    if MERGED_TASK.skill in pack.index or any(s.skill in pack.index for s in ACTIVE_STAGES):
                        if MERGED_TASK.skill in pack.index:
                            self.merged = pack
                        else:
                            self.active = pack
                        paths, _ = latest_pack_paths()
                        for pair, path in paths.items():
                            if local := load(path):
                                self.packs[pair] = local
                    else:
                        self.packs[tuple(pack.manifest.get('layer_pair', (1, 1)))] = pack
        except (OSError, ValueError, KeyError, TeleopError) as exc:
            errors.append(str(exc))
        self.pair_labels = {f'第 {s} 层 S → 第 {t} 层 T': (s, t) for s, t in self.packs}
        self.available_actions = {}
        root.title('Koch · 训练成果演示')
        root.geometry(f'1120x{min(880, root.winfo_screenheight() - 90)}')
        root.minsize(940, 680)
        root.configure(bg='#edf1f3')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 11))
        style.configure('TButton', padding=(10, 5))
        style.configure('Start.TButton', background='#087f78', foreground='white')
        style.configure('Stop.TButton', background='#b9343a', foreground='white')
        top = ttk.Frame(root, padding=18)
        top.pack(fill='x')
        ttk.Label(top, text='训练成果演示', font=('Microsoft YaHei UI', 22, 'bold')).pack(anchor='w')
        ttk.Label(top, text='选择层级、格子和动作。').pack(anchor='w', pady=(4, 0))
        body = ttk.Frame(root, padding=(18, 0, 18, 10))
        body.pack(fill='both', expand=True)
        left = ttk.Frame(body, width=270, padding=12)
        left.pack(side='left', fill='y')
        right = ttk.Frame(body, padding=(14, 0, 0, 0))
        right.pack(side='left', fill='both', expand=True)
        self.variables = {}
        self.inputs = []
        pairs = list(self.pair_labels) or ['无可用模型']
        model_summary = f'已发现 {len(self.packs)} 个分层训练包'
        if self.merged is not None:
            model_summary += ' + 1 个四组合并模型'
        if self.active is not None:
            model_summary += f'\n无等待新模型已导出 {len(self.active.index)}/2'
        fields = [('layers', '储存层 → 放置层', pairs[0], pairs),
            ('s', '储存区 S', 'S1', ['S1','S2','S3','S4']), ('t', '放置区 T', 'T1', ['T1','T2','T3','T4']),
            ('stage', '执行动作', '请选择动作', ['请选择动作']),
            ('camera', '相机编号', '2', None), ('port', '从臂串口', 'COM5', None),
            ('speed', '最大速度（°/秒）', '90', None), ('duration', '单段推理秒数（留空用示范时长）', '', None),
            ('device', '推理设备', 'auto', ['auto', 'cuda', 'cpu'])]
        ttk.Label(left, text=model_summary, foreground='#28706b').pack(anchor='w', pady=(0, 8))
        for name, label, value, options in fields:
            ttk.Label(left, text=label).pack(anchor='w', pady=(4, 1))
            var = tk.StringVar(root, value)
            self.variables[name] = var
            widget = ttk.Combobox(left, textvariable=var, values=options, state='readonly', width=27) if options else ttk.Entry(left, textvariable=var, width=29)
            widget.pack(fill='x')
            self.inputs.append((widget, 'readonly' if options else 'normal'))
            if name == 'stage':
                self.action_widget = widget
            if name == 'layers':
                widget.bind('<<ComboboxSelected>>', self.update_actions)
        self.start_button = ttk.Button(left, text='开始所选动作', style='Start.TButton', command=self.start)
        self.start_button.pack(fill='x', pady=(12, 0))
        self.status = tk.StringVar(root, '待机 · 请选择路径和动作')
        ttk.Label(right, textvariable=self.status, font=('Microsoft YaHei UI', 14, 'bold'), wraplength=650).pack(anchor='w', pady=(8, 10))
        controls = ttk.Frame(right)
        controls.pack(fill='x', pady=(0, 8))
        self.preview_button = ttk.Button(controls, text='仅预览相机', command=lambda: self.start(True))
        self.preview_button.pack(side='left', padx=(0, 6))
        self.next_button = ttk.Button(controls, text='下一段（N）', command=self.next_stage.set, state='disabled')
        self.next_button.pack(side='left', padx=(0, 6))
        ttk.Button(controls, text='结束并释放（F）', command=self.finish.set).pack(side='left', padx=(0, 6))
        self.stop_button = ttk.Button(controls, text='停止（Esc）', style='Stop.TButton', command=self.stop.set)
        self.stop_button.pack(side='left')
        self.image = tk.Label(right, text='相机尚未打开\n点击“仅预览相机”或“开始执行模型”', bg='#18252d', fg='#dbe4e9', font=('Microsoft YaHei UI', 14))
        self.image.pack(fill='both', expand=True)
        self.path = tk.StringVar(root, '运行视频和关节记录保存到 outputs/koch_model_control/runs')
        ttk.Label(right, textvariable=self.path, wraplength=650).pack(anchor='w', pady=8)
        ttk.Label(right, text='完整搬运和分步演示会先从当前姿态自动收敛到示范起点。\n'
                              '分步动作到时保持；确认后进入下一段。停止会释放扭矩。', wraplength=650).pack(anchor='w')
        self.log = tk.Text(root, height=7, bg='#ffffff', fg='#25343e', font=('Microsoft YaHei UI', 10), state='disabled')
        self.log.pack(fill='x', padx=18, pady=(0, 16))
        self.update_actions()
        if errors:
            message = '部分训练包尚不可用：' + errors[0]
            self.messages.put(('log', message))
        root.bind('<Escape>', lambda e: self.stop.set())
        root.bind('<KeyPress-f>', lambda e: self.finish.set())
        root.bind('<KeyPress-n>', lambda e: self.next_stage.set())
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.after_id = root.after(80, self.poll)

    def update_actions(self, event=None):
        pair = self.pair_labels.get(self.variables['layers'].get())
        pack = self.packs.get(pair)
        actions = {}
        if pack is not None:
            available = set()
            for stage in ALL_STAGES:
                current = self.active if stage in ACTIVE_STAGES else self.merged if stage == MERGED_TASK else pack
                if current is None:
                    continue
                try:
                    current.check_skill(stage.skill)
                    available.add(stage.skill)
                except TeleopError:
                    continue
            if all(s.skill in available for s in STAGES):
                actions['整套分步演示'] = 0
            for i, stage in enumerate(ALL_STAGES, 1):
                if stage.skill in available:
                    actions[stage.title] = i
            if pack.manifest.get('rest_pose'):
                actions[HOME_ACTION] = -1
        self.available_actions = actions
        self.action_widget.configure(values=list(actions))
        if self.variables['stage'].get() not in actions:
            self.variables['stage'].set('请选择动作' if actions else '无可用动作')
        self.start_button.configure(state='normal' if actions else 'disabled')

    def start(self, preview=False):
        from tkinter import messagebox
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            v = {n: value.get() for n, value in self.variables.items()}
            if not preview and v['stage'] not in self.available_actions:
                raise ValueError('请先选择已训练并导出的动作。')
            pair = self.pair_labels.get(v['layers'])
            local = self.packs.get(pair)
            stage = 0 if preview else self.available_actions[v['stage']]
            if stage > 0 and ALL_STAGES[stage - 1].skill.startswith('reach_'):
                v['s'] = ALL_STAGES[stage - 1].skill[-2:]
                self.variables['s'].set(v['s'])
            merged = stage > 0 and ALL_STAGES[stage - 1] == MERGED_TASK
            active = stage > 0 and ALL_STAGES[stage - 1] in ACTIVE_STAGES
            pack = self.active if active else self.merged if merged else local
            home_only = stage == -1
            config = SimpleNamespace(preview=preview, manifest=pack.path if pack else None,
                rest_manifest=local.path if merged else None, layers=pair, home_only=home_only,
                route=v['s']+'_'+v['t'], stage=stage, camera=0 if home_only else int(v['camera']),
                port=v['port'].strip(), speed=90 if preview else float(v['speed']),
                duration=float(v['duration']) if not (home_only or preview) and v['duration'].strip() else None,
                device=v['device'])
            if config.camera < 0:
                raise ValueError('相机编号须为非负整数。')
            if not preview and (not config.port or not math.isfinite(config.speed) or not 2 <= config.speed <= 120):
                raise ValueError('串口不能为空，速度须为 2–120°/秒。')
            if config.duration is not None and (not math.isfinite(config.duration) or not .1 <= config.duration <= 120):
                raise ValueError('单段秒数须为 0.1–120，或留空。')
        except ValueError as exc:
            messagebox.showerror('参数有误', str(exc), parent=self.root)
            return
        for event in (self.stop, self.finish, self.next_stage):
            event.clear()
        self.last_stamp = None
        for widget, _ in self.inputs:
            widget.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.preview_button.configure(state='disabled')
        self.status.set('正在打开相机…' if preview else '正在准备从臂…' if config.home_only else '正在准备模型与从臂…')
        self.worker = threading.Thread(target=run_session, args=(config, self.messages, self.stop, self.finish, self.next_stage), daemon=True)
        self.worker.start()

    def poll(self):
        from PIL import Image, ImageTk
        # 控制线程会频繁发布状态；每次只保留最后状态，避免阻塞界面停止按钮。
        latest_state = None
        for _ in range(800):
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            if kind == 'camera':
                self.camera = value
            elif kind == 'state':
                latest_state = value
            elif kind == 'path':
                self.path.set(value)
            elif kind == 'log':
                self.log.configure(state='normal')
                self.log.insert('end', value + '\n')
                self.log.see('end')
                self.log.configure(state='disabled')
            elif kind == 'done':
                self.status.set('已停止 · 从臂执行线程结束，查看下方日志')
                for widget, state in self.inputs:
                    widget.configure(state=state)
                self.start_button.configure(state='normal' if self.available_actions else 'disabled')
                self.preview_button.configure(state='normal')
                self.next_button.configure(state='disabled')
                latest_state = None
        if latest_state:
            index, title, waiting, settled, last = latest_state
            hold_status = '已到位，保持末目标' if settled else '未到位，保持末目标'
            self.status.set(f'第 {index} 段 · {title} · ' +
                (hold_status if waiting else '正在低速修正' if title.startswith('对齐') else '模型执行中'))
            self.next_button.configure(state='normal' if waiting and settled and not last else 'disabled')
        if self.camera is not None:
            with self.camera.lock:
                stamp, frame = self.camera.latest
            if stamp != self.last_stamp:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.thumbnail((max(320, self.image.winfo_width()), max(180, self.image.winfo_height())))
                photo = ImageTk.PhotoImage(image, master=self.root)
                self.image.configure(image=photo, text='')
                self.image.photo = photo
                self.last_stamp = stamp
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.after_id = self.root.after(80, self.poll)

    def close(self):
        self.closing = True
        self.stop.set()
        self.status.set('正在停止并等待释放，请稍候…')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--probe', action='store_true', help='使用录制数据离线检查模型，不访问硬件')
    parser.add_argument('--stage', type=int, choices=range(len(ALL_STAGES) + 1), default=0)
    parser.add_argument('--layers', nargs=2, type=int, choices=(1, 2), metavar=('S层', 'T层'), default=(1, 1),
                        help='离线检查使用的层级组合')
    parser.add_argument('--route', default='S1_T1')
    parser.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    args = parser.parse_args(argv)
    if args.probe:
        from koch_robot.collect import route_name
        probe(args.manifest, args.stage, route_name(args.route), args.device, tuple(args.layers))
        return 0
    import tkinter as tk
    root = tk.Tk()
    ModelWindow(root, args.manifest)
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
