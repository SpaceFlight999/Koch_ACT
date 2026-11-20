"""Run Replay with the original GridApp camera and recognition widgets."""
from contextlib import contextmanager
from functools import partial
from queue import Empty, Queue, SimpleQueue
from threading import Event, Thread
from time import monotonic
from tkinter import StringVar, Tk, messagebox, ttk
from types import SimpleNamespace

from camera_vision_app import CameraStream
from grid_reader.app import GridApp
from grid_reader.cameras import resolve_named_camera
from grid_reader.worker import replace_latest

from .replay_auto import run_auto
from .reverse import read_json


class ReplayDashboard(GridApp):
    def __init__(self, root, site, observe):
        self.site, self.observe = site.resolve(), observe
        self.events, self.views = SimpleQueue(), Queue(maxsize=1)
        self.cancel = Event()
        self.job = self.monitor_stream = None
        self.close_requested = False
        self.moves = []
        self.active_step = None
        self.last_view, self.camera_error = None, None
        super().__init__(root, SimpleNamespace(mode='combined'), task_camera=True)
        root.title('Koch 自动搬运 · 实时路线')
        root.bind('<Escape>', self.request_stop)
        self.canvas.create_text(24, 26, anchor='nw', fill='white', text='启动识别后在这里显示相机画面。')
        self.status.set('相机待启动')

        section = ttk.Frame(self.status_area.master)
        section.pack(fill='x', pady=(8, 0), before=self.status_area)
        actions = ttk.Frame(section)
        actions.pack(fill='x')
        self.preview_button = ttk.Button(actions, text='仅识别预览', command=lambda: self.start(True))
        self.preview_button.pack(side='left')
        self.run_button = ttk.Button(actions, text='开始自动搬运', style='Go.TButton', command=lambda: self.start(False))
        self.run_button.pack(side='left', padx=8)
        self.stop_button = ttk.Button(actions, text='停止任务（Esc）', command=self.request_stop, state='disabled')
        self.stop_button.pack(side='left')
        ttk.Label(actions, text='已指定现场配置', foreground='#53655F').pack(side='right')

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
                                   show='headings', height=3, selectmode='none')
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

    def start(self, preview):
        if self.job is not None:
            return
        if not preview and not messagebox.askokcancel(
            '启动自动搬运', '确认机械臂已在录制公共起始位，T 区为空，人员已离开活动范围。\n'
            '识别和路线校验通过后会自动运动，不会逐条确认。', parent=self.root,
        ):
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
        self.status.set('正在准备相机…')
        self.set_status('正在读取现场配置…')
        self.preview_button.configure(state='disabled')
        self.run_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.job = Thread(target=self.run_task, args=(preview,), name='koch-replay-dashboard')
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
            raise RuntimeError('运行画面相机尚未释放，无法复查 S 区是否清空。')

    def run_task(self, preview):
        observe = partial(self.observe, cancelled=self.cancel.is_set,
                          on_view=lambda kind, frame: replace_latest(self.views, (kind, frame)))
        try:
            run_auto(self.site, preview, observe, on_event=self.publish,
                     cancelled=self.cancel.is_set, monitor=self.motion_camera)
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
            'connecting': '路线已就绪，正在连接从臂并检查起点…', 'checking_empty': '路线已全部执行，正在确认 S 区是否清空…',
            'empty': '任务完成 · S 区原本为空，没有执行动作。', 'complete': '任务完成 · 已确认 S 区清空。',
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
        elif kind == 'plan':
            self.moves = payload['moves']
            self.progress.configure(maximum=max(1, len(self.moves)))
            self.count.set(f'已执行 0 / {len(self.moves)} 条')
            for step, move in enumerate(self.moves, 1):
                self.routes.insert('', 'end', iid=str(step), values=(step, f'{move["source"]} 第 {move["source_layer"]} 层',
                                   f'{move["target"]} 第 {move["target_layer"]} 层', move['color'], '未执行'))
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
            self.preview_button.configure(state='normal')
            self.run_button.configure(state='normal')
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
