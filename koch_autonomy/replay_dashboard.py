"""Run Replay with the original GridApp camera and recognition widgets."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
from threading import Event, Thread
from time import monotonic
from tkinter import StringVar, Tk, Toplevel, messagebox, ttk
from types import SimpleNamespace

from camera_vision_app import CameraStream
from grid_reader.app import GridApp
from grid_reader.cameras import resolve_named_camera
from grid_reader.worker import replace_latest
from teleoperate_koch import NAMES

from .replay_advice import color_label, suggest_layouts
from .replay_auto import run_auto
from .replay_policy import route_is_disabled, toggle_route
from .replay_single import run_single, select_episode
from .reverse import read_json


JOINT_LABELS = ('底座', '肩', '肘', '腕俯仰', '腕旋转', '夹爪')


class ReplayDashboard(GridApp):
    def __init__(self, root, site, observe):
        self.site, self.observe = site.resolve(), observe
        self.events, self.views = SimpleQueue(), Queue(maxsize=1)
        self.cancel = Event()
        self.job = self.monitor_stream = None
        self.close_requested = False
        self.moves = []
        self.active_step = None
        self.advice_window = None
        self.last_view, self.camera_error = None, None
        super().__init__(root, SimpleNamespace(mode='combined'), task_camera=True)
        root.title('Koch · 扫描与 Replay 控制台')
        root.bind('<Escape>', self.request_stop)
        self.canvas.create_text(24, 26, anchor='nw', fill='white', text='启动识别后在这里显示相机画面。')
        self.status.set('相机待启动')

        section = ttk.Frame(self.status_area.master)
        section.pack(fill='x', pady=(8, 0), before=self.status_area)
        actions = ttk.Frame(section)
        actions.pack(fill='x')
        use_global = bool(read_json(self.site).get('global_targets'))
        self.preview_button = ttk.Button(actions, text='仅识别预览',
                                         command=lambda: self.start(True, global_targets_only=use_global))
        self.preview_button.pack(side='left')
        self.run_button = ttk.Button(actions, text='开始自动搬运', style='Go.TButton',
                                     command=lambda: self.start(False, global_targets_only=use_global))
        self.run_button.pack(side='left', padx=8)
        self.advice_button = ttk.Button(actions, text='摆法建议',
                                        command=lambda: self.start(True, global_targets_only=use_global, advice=True))
        self.advice_button.pack(side='left', padx=(0, 8))
        self.launch_buttons = [self.preview_button, self.run_button, self.advice_button]
        if use_global:
            wrist_button = ttk.Button(actions, text='腕扫并搬运', command=lambda: self.start(False))
            wrist_button.pack(side='left', padx=(0, 8))
            self.launch_buttons.append(wrist_button)
        self.scan_button = ttk.Button(actions, text='只扫描 T 区', command=lambda: self.start(False, scan_only=True))
        self.scan_button.pack(side='left', padx=(0, 8))
        self.launch_buttons.append(self.scan_button)
        self.stop_button = ttk.Button(actions, text='停止任务（Esc）', command=self.request_stop, state='disabled')
        self.stop_button.pack(side='left')

        single = ttk.Labelframe(section, text='单条 Replay 与路线管理', padding=(8, 3))
        single.pack(fill='x', pady=(8, 0))
        self.source = StringVar(value='S3')
        self.source_layer = StringVar(value='2')
        self.target = StringVar(value='T4')
        self.target_layer = StringVar(value='1')
        self.selectors = []
        for label, var, choices in (
            ('取料', self.source, ('S1', 'S2', 'S3', 'S4')), ('层', self.source_layer, ('1', '2')),
            ('放置', self.target, ('T1', 'T2', 'T3', 'T4')), ('层', self.target_layer, ('1', '2')),
        ):
            ttk.Label(single, text=label).pack(side='left', padx=(2, 4))
            selector = ttk.Combobox(single, textvariable=var, values=choices, width=4, state='readonly')
            selector.pack(side='left', padx=(0, 8))
            selector.bind('<<ComboboxSelected>>', self.update_single_choice)
            self.selectors.append(selector)
        self.single_button = ttk.Button(single, text='播放选中动作', command=lambda: self.start(False, single=True))
        self.single_button.pack(side='left', padx=(3, 8))
        self.route_toggle_button = ttk.Button(single, text='停用路线', command=self.toggle_selected_route)
        self.route_toggle_button.pack(side='left', padx=(0, 8))
        self.episode_label = StringVar()
        ttk.Label(single, textvariable=self.episode_label, foreground='#53655F').pack(side='left')
        self.joint_feedback = StringVar()
        self.replay_record = StringVar()
        ttk.Label(section, textvariable=self.joint_feedback, font=('Microsoft YaHei UI', 10)).pack(
            fill='x', pady=(5, 0))
        ttk.Label(section, textvariable=self.replay_record, foreground='#53655F').pack(fill='x')
        self.update_single_choice()

        headline = ttk.Frame(section)
        headline.pack(fill='x', pady=(8, 4))
        self.current = StringVar(value='当前路线：—')
        self.count = StringVar(value='已执行 0 / 0 条')
        ttk.Label(headline, textvariable=self.current, font=('Microsoft YaHei UI', 16, 'bold')).pack(side='left')
        ttk.Label(headline, textvariable=self.count, font=('Microsoft YaHei UI', 12, 'bold')).pack(side='right')
        self.progress = ttk.Progressbar(section, mode='determinate', maximum=1)
        self.progress.pack(fill='x', pady=(0, 6))

        route_list = ttk.Frame(section)
        route_list.pack(fill='x')
        self.routes = ttk.Treeview(route_list, columns=('step', 'source', 'target', 'color', 'state'),
                                   show='headings', height=3, selectmode='browse')
        self.routes.bind('<<TreeviewSelect>>', self.select_plan_route)
        for name, title, width in (('step', '顺序', 70), ('source', '取料', 165), ('target', '放置', 165),
                                   ('color', '颜色', 145), ('state', '路线状态', 330)):
            self.routes.heading(name, text=title)
            self.routes.column(name, width=width, anchor='center', stretch=name == 'state')
        self.routes.tag_configure('active', background='#CDE9DA', foreground='#155F4A')
        self.routes.tag_configure('done', foreground='#53655F')
        scroll = ttk.Scrollbar(route_list, orient='vertical', command=self.routes.yview)
        self.routes.configure(yscrollcommand=scroll.set)
        self.routes.pack(fill='x', expand=True, side='left')
        scroll.pack(fill='y', side='right')
        self.task_status = StringVar(value='待启动 · 仅识别预览不会驱动机械臂。')
        self.task_label = ttk.Label(section, textvariable=self.task_status, anchor='w', justify='left', wraplength=1060)
        self.task_label.pack(fill='x', pady=(5, 0))

    def single_selection(self):
        return self.source.get(), self.target.get(), (int(self.source_layer.get()), int(self.target_layer.get()))

    def update_single_choice(self, event=None):
        if self.job is not None:
            return
        self.single_button.configure(state='disabled')
        self.route_toggle_button.configure(state='disabled')
        selection = self.single_selection()
        try:
            disabled = route_is_disabled(self.site, *selection)
            self.route_toggle_button.configure(text='恢复路线' if disabled else '停用路线', state='normal')
            if disabled:
                self.episode_label.set('该取放层位已停用')
                return
            episode = select_episode(self.site, *selection)
        except (OSError, ValueError, KeyError) as exc:
            self.episode_label.set(str(exc))
        else:
            self.episode_label.set('最新录制 ' + episode.name)
            self.single_button.configure(state='normal')

    def select_plan_route(self, _=None):
        selected = self.routes.selection()
        if self.job is not None or not selected:
            return
        move = self.moves[int(selected[0]) - 1]
        self.source.set(move['source'])
        self.source_layer.set(str(move['source_layer']))
        self.target.set(move['target'])
        self.target_layer.set(str(move['target_layer']))
        self.update_single_choice()

    def toggle_selected_route(self):
        if self.job is not None:
            return
        source, target, pair = self.single_selection()
        try:
            disabled = toggle_route(self.site, source, target, pair)
        except (OSError, ValueError, KeyError) as exc:
            self.set_status(str(exc), '#9C342B')
            return
        self.update_single_choice()
        for index, move in enumerate(self.moves, 1):
            if (move['source'], move['target'], (move['source_layer'], move['target_layer'])) == (source, target, pair):
                self.routes.set(str(index), 'state', '已停用 · 请重新预览' if disabled else '已恢复 · 请重新预览')
        if self.advice_window and self.advice_window.winfo_exists():
            self.advice_window.destroy()
        action = '已停用' if disabled else '已恢复'
        self.set_status(f'{action} {source} 第 {pair[0]} 层 → {target} 第 {pair[1]} 层。下次规划会使用新设置。')

    def start(self, preview, *, scan_only=False, single=False, global_targets_only=False, advice=False):
        if self.job is not None:
            return
        selected = None
        if single:
            selection = self.single_selection()
            try:
                episode = select_episode(self.site, *selection)
            except (OSError, ValueError, KeyError) as exc:
                self.set_status(str(exc), '#9C342B')
                return
            selected = (*selection, episode)
            source, target, (sl, tl) = selection
            title = '启动单条 Replay'
            message = (f'{source} 第 {sl} 层 → {target} 第 {tl} 层\n录制 {episode.name}\n\n'
                       '确认源与目标层位已摆好、从臂在录制休息位附近、活动范围已清空。')
        elif scan_only:
            title, message = '启动 T 区扫描', '确认四张任务卡无遮挡、从臂在扫描休息位附近、活动范围已清空。'
        elif global_targets_only:
            title = '全局读卡并搬运'
            message = '全局相机读取四张卡后，直接执行搬运。确认从臂在录制休息位、T 区只有任务卡、活动范围已清空。'
        else:
            title, message = '启动自动搬运', '确认从臂已在录制休息位，T 区只有任务卡，活动范围已清空。'
        if not preview and not messagebox.askokcancel(title, message, parent=self.root):
            return
        self.cancel.clear()
        self.last_view, self.camera_error = None, None
        self.canvas.delete('all')
        self.routes.delete(*self.routes.get_children())
        self.show_report({'cells': []})
        self.moves = []
        self.active_step = None
        self.progress.configure(maximum=1, value=0)
        self.current.set('当前路线：—')
        self.count.set('已执行 0 / 0 条')
        self.joint_feedback.set('等待关节回读…' if single else '')
        self.replay_record.set('')
        self.status.set('正在准备相机…')
        self.set_status('正在读取现场配置…')
        for control in (*self.launch_buttons, self.single_button, self.route_toggle_button, *self.selectors):
            control.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.job = Thread(target=self.run_task, args=(preview, scan_only, selected, global_targets_only, advice),
                          name='koch-replay-dashboard')
        self.job.start()

    def publish(self, kind, **payload):
        self.events.put((kind, payload))

    @contextmanager
    def motion_camera(self, source, site_dir):
        index = resolve_named_camera(source['camera_name'])[0] if source.get('camera_name') else source['camera']
        config = read_json(site_dir / source['config'])
        stream = CameraStream(index, config['frame_size'])
        self.publish('monitor_started', stream=stream)
        try:
            yield
        finally:
            stream.stop()
            stream.thread.join(timeout=3)
            self.publish('monitor_stopped')
        if stream.thread.is_alive():
            raise RuntimeError('运行画面相机尚未释放，无法复查 S 区剩余库存。')

    def run_task(self, preview, scan_only, single, global_targets_only, advice=False):
        observe = partial(self.observe, cancelled=self.cancel.is_set,
                          on_view=lambda kind, frame: replace_latest(self.views, (kind, frame)))
        try:
            if single is not None:
                site = read_json(self.site)
                with self.motion_camera(site['source'], self.site.parent):
                    run_single(self.site, *single, self.cancel.is_set, self.publish)
            else:
                task = run_auto(self.site, preview, observe, on_event=self.publish, scan_only=scan_only,
                                global_targets_only=global_targets_only, cancelled=self.cancel.is_set,
                                monitor=self.motion_camera, observe_only=advice)
                if advice and task is not None:
                    self.publish('preparing_advice')
                    results = suggest_layouts(self.site, task, self.cancel.is_set)
                    self.publish('layout_advice', suggestions=results)
        except KeyboardInterrupt:
            self.publish('stopped')
        except Exception as exc:
            self.publish('error', message=str(exc) or type(exc).__name__)

    def set_status(self, message, color='#17382D'):
        self.task_status.set(message)
        self.task_label.configure(foreground=color)

    def update_event(self, kind, payload):
        labels = {
            'observe_source': '正在识别 S 区库存…', 'observe_targets': '正在识别 T 区任务卡…',
            'preparing_plan': '任务卡已识别，正在核对路线并加载 Replay…',
            'preparing_advice': '正在核验原摆法、互换和备块替换的 Replay 路线…',
            'connecting': '正在连接从臂并检查起点…', 'checking_empty': '路线已全部执行，正在核对 S 区剩余库存…',
            'scan_approach': '从休息位前往 T 区扫描位…', 'scan_return': '扫描结束，返回休息位…',
            'empty': 'S 区原本为空，未执行动作。', 'complete': '任务完成 · S 区剩余库存与计划一致。',
            'preview': '预览完成 · 路线未执行，未连接机械臂。', 'stopped': '任务已停止，不会自动继续执行。',
        }
        if kind in labels:
            self.set_status(labels[kind])
            if kind in ('complete', 'empty', 'preview'):
                self.current.set('当前路线：—')
            if kind == 'stopped':
                self.end_active_route('未执行完 · 已中止')
        elif kind == 'error':
            self.end_active_route('未执行完')
            self.set_status(f'任务未完成 · {payload["message"]}', '#9C342B')
        elif kind == 'target_report':
            self.show_report(payload['report'])
        elif kind == 'layout_advice':
            self.show_layout_advice(payload['suggestions'])
        elif kind == 'observe_global_targets':
            self.set_status('腕部扫描已返回，正在用全局相机读卡…' if payload['wrist_error'] else
                            '正在用全局相机读卡…')
        elif kind == 'scan_complete':
            self.set_status('扫描完成 · 已记录四张卡并回到休息位。')
            self.replay_record.set('记录目录  ' + Path(payload['directory']).name)
        elif kind == 'start_gap':
            self.joint_feedback.set(f'起点最大偏差 {payload["degrees"]:.2f}°，正在对位。')
        elif kind == 'joint_feedback':
            joints = '   '.join(f'{label} {payload["errors"][joint]:.2f}°'
                                 for label, joint in zip(JOINT_LABELS, NAMES))
            self.joint_feedback.set(f'{joints}     本次峰值 {payload["peak"]:.2f}°')
        elif kind == 'single_log':
            path = Path(payload['path'])
            self.replay_record.set(f'诊断记录  {path.parent.name} / {path.name}')
        elif kind == 'single_complete':
            self.set_status('单条 Replay 已播完 · 请核对落点。')
        elif kind == 'plan':
            self.moves = payload['moves']
            if payload.get('older_episode_steps'):
                steps = '、'.join(map(str, payload['older_episode_steps']))
                self.replay_record.set(f'第 {steps} 步使用可衔接的较早录制。')
            self.progress.configure(maximum=max(1, len(self.moves)))
            self.count.set(f'已执行 0 / {len(self.moves)} 条')
            for step, move in enumerate(self.moves, 1):
                self.routes.insert('', 'end', iid=str(step), values=(step, f'{move["source"]} 第 {move["source_layer"]} 层',
                                   f'{move["target"]} 第 {move["target_layer"]} 层', move['color'], '未执行'))
        elif kind == 'move_aligning':
            step = payload['step']
            self.active_step = str(step)
            self.routes.set(self.active_step, 'state', '正在对准起点')
            self.routes.see(self.active_step)
            self.set_status(f'正在对准第 {step} 条路线的录制起点。')
        elif kind in ('move_started', 'move_finished'):
            step = payload['step']
            row = str(step)
            active = kind == 'move_started'
            self.routes.set(row, 'state', '正在执行' if active else '已执行')
            self.routes.item(row, tags=('active' if active else 'done',))
            self.routes.see(row)
            if active:
                self.active_step = row
                move = self.moves[step - 1]
                self.current.set(f'当前路线：{move["source"]} 第 {move["source_layer"]} 层 → '
                                 f'{move["target"]} 第 {move["target_layer"]} 层')
                self.set_status(f'正在执行第 {step} 条路线。')
            else:
                self.active_step = None
                self.current.set('当前路线：—')
                self.progress.configure(value=step)
                self.count.set(f'已执行 {step} / {len(self.moves)} 条')
        elif kind == 'monitor_started':
            self.monitor_stream = payload['stream']
            self.camera_error = None
        elif kind == 'monitor_stopped':
            self.monitor_stream, self.camera_error = None, None

    def show_layout_advice(self, suggestions):
        if not suggestions:
            self.set_status('原摆法、互换两块和换一至两块备料都未找到完整 Replay。', '#9C342B')
            return
        if self.advice_window and self.advice_window.winfo_exists():
            self.advice_window.destroy()
        window = self.advice_window = Toplevel(self.root)
        window.title('按当前库存和任务卡生成的摆法建议')
        window.geometry('760x440')
        window.minsize(640, 380)
        window.bind('<Escape>', lambda _: window.destroy())
        panel = ttk.Frame(window, padding=14)
        panel.pack(fill='both', expand=True)
        tabs = ttk.Notebook(panel)
        tabs.pack(fill='both', expand=True)
        for choice in suggestions:
            page = ttk.Frame(tabs, padding=12)
            tabs.add(page, text=choice['title'])
            ttk.Label(page, text='\n'.join(choice['instructions']), justify='left', wraplength=570).pack(anchor='w')
            positions = '    '.join(f'{slot} {"／".join(map(color_label, stack)) or "空"}'
                                     for slot, stack in choice['inventory'].items())
            ttk.Label(page, text=f'摆好后（下层／上层）  {positions}').pack(anchor='w', pady=(8, 6))
            table = ttk.Treeview(page, columns=('step', 'color', 'source', 'target'), show='headings',
                                 height=max(4, len(choice['moves'])), selectmode='none')
            for column, title, width in (('step', '顺序', 60), ('color', '方块', 75),
                                         ('source', '取料', 165), ('target', '放置', 165)):
                table.heading(column, text=title)
                table.column(column, width=width, anchor='center', stretch=column in ('source', 'target'))
            for index, move in enumerate(choice['moves'], 1):
                table.insert('', 'end', values=(index, color_label(move['color']),
                    f'{move["source"]} 第 {move["source_layer"]} 层', f'{move["target"]} 第 {move["target_layer"]} 层'))
            table.pack(fill='x')
            if choice['older_episode_steps']:
                steps = '、'.join(map(str, choice['older_episode_steps']))
                ttk.Label(page, text=f'第 {steps} 步使用可衔接的较早录制。').pack(anchor='w', pady=(6, 0))
        ttk.Button(panel, text='关闭', command=window.destroy).pack(anchor='e', pady=(8, 0))
        self.set_status(f'已核验 {len(suggestions)} 种摆法。整理好方块后可用「仅识别预览」复查。')

    def end_active_route(self, status):
        if self.active_step is not None:
            self.routes.set(self.active_step, 'state', status)
            self.routes.item(self.active_step, tags=())
            self.active_step = None
        self.current.set('当前路线：—')

    def tick(self):
        while not self.events.empty():
            self.update_event(*self.events.get_nowait())
        if self.monitor_stream is not None:
            try:
                _, frame = self.monitor_stream.frames.get_nowait()
                self.show_view('运行画面 · S 区', frame)
            except Empty:
                pass
            try:
                self.camera_error = self.monitor_stream.messages.get_nowait()
            except Empty:
                pass
        else:
            try:
                kind, frame = self.views.get_nowait()
                self.show_view('任务卡 · T 区' if kind == 'targets' else '库存 · S 区', frame)
            except Empty:
                pass
        if self.camera_error:
            self.status.set(f'运行相机不可用 · {self.camera_error}')
        elif self.last_view is not None and monotonic() - self.last_view[1] > 1:
            self.status.set(f'{self.last_view[0]} · 保留最后一帧，当前没有新画面')
        if self.job is not None and not self.job.is_alive() and self.events.empty():
            self.job = None
            for button in self.launch_buttons:
                button.configure(state='normal')
            for selector in self.selectors:
                selector.configure(state='readonly')
            self.update_single_choice()
            self.stop_button.configure(state='disabled')
            if self.close_requested:
                super().close()
                return
        self.root.after(50, self.tick)

    def show_view(self, label, frame):
        self.render(frame)
        self.last_view = label, monotonic()
        self.status.set(f'{label} · {frame.shape[1]}×{frame.shape[0]}')

    def request_stop(self, event=None):
        if self.job is not None and self.job.is_alive():
            self.cancel.set()
            self.stop_button.configure(state='disabled')
            self.set_status('正在停止，等待本次执行结束…')

    def close(self):
        if self.job is not None:
            self.close_requested = True
            self.request_stop()
        else:
            super().close()


def open_dashboard(site, observe):
    root = Tk()
    ReplayDashboard(root, site, observe)
    root.mainloop()
