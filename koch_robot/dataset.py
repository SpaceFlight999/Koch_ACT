"""Koch S→T 采集文件、异步相机记录与独立从臂回放轨迹。"""
import bisect
import json
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from koch_robot.teleop import (COLLECTION_SPEED, MAX_SPEED, MIN_SPEED, NAMES, TICK_DEGREES, TeleopError,
                              checked_angles, gripper_closing_contact)

ROUTES = tuple(f'S{s}_T{t}' for s in range(1, 5) for t in range(1, 5))
REPLAY_START_ALIGNMENT_DEGREES = 10
REPLAY_DEFAULT_RATE = 1.0
REPLAY_DEFAULT_SPEED = COLLECTION_SPEED


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def completed_episodes(root, route):
    episodes = []
    for path in sorted((Path(root) / route).glob('*/episode.json')):
        try:
            if json.loads(path.read_text(encoding='utf-8')).get('status') == 'complete':
                episodes.append(path.parent)
        except (ValueError, OSError):
            continue
    return episodes


class PreviewWindow:
    """与项目相机工具一致，用 Tk 显示，兼容已安装的 headless OpenCV。"""
    def __init__(self, title='Koch 相机预览'):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title(title)
        self.image = tk.Label(self.root)
        self.image.pack()
        self.help = tk.Label(self.root, text='Enter：确认画面　F：结束记录　Esc/空格：中断',
                             font=('Microsoft YaHei UI', 12), pady=8)
        self.help.pack()
        self.key, self.closed, self.last_stamp = -1, False, None
        self.root.protocol('WM_DELETE_WINDOW', self._close)
        self.root.bind('<KeyPress>', self._key)

    def _close(self):
        self.closed = True

    def _key(self, event):
        if self.key in (27, 32, 3):
            return
        special = {'Escape': 27, 'Return': 13, 'space': 32}
        self.key = special.get(event.keysym, ord(event.char.lower()) if len(event.char) == 1 else -1)

    def show(self, frame=None, stamp=None, label=None):
        from PIL import Image, ImageTk
        if self.closed:
            return 27
        if frame is not None and stamp != self.last_stamp:
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), master=self.root)
            self.image.configure(image=photo)
            self.image.photo = photo
            self.last_stamp = stamp
        if label:
            self.root.title(label)
        self.root.update()
        key, self.key = self.key, -1
        return 27 if self.closed else key

    def close(self):
        self.root.destroy()


class Camera:
    """相机读取独立于串口控制；不在控制线程里等待图像。"""
    def __init__(self, index, width=1280, height=720, fps=20):
        if index < 0:
            raise ValueError('相机编号必须非负。')
        self.index, self.fps = index, fps
        self.latest, self.error, self.sink = None, None, None
        self.lock = threading.Lock()
        self.ended = threading.Event()
        self.window = None
        self.last_preview = 0.0
        self.capture = None
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            capture = cv2.VideoCapture(index, backend)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            if ok:
                self.capture = capture
                self.latest = (time.monotonic(), frame)
                break
            capture.release()
        if self.capture is None:
            raise TeleopError(f'相机 {index} 无画面；可换编号，检查 USB 和占用相机的软件。')
        self.size = (frame.shape[1], frame.shape[0])
        self.thread = threading.Thread(target=self._read, daemon=True, name='koch-camera')
        self.thread.start()

    def _read(self):
        try:
            while not self.ended.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    raise TeleopError('相机断开或读取失败。')
                stamp = time.monotonic()
                if (frame.shape[1], frame.shape[0]) != self.size:
                    raise TeleopError('相机分辨率在录制中改变。')
                with self.lock:
                    self.latest = (stamp, frame)
                    # 同一把锁保证 detach 返回后，不再向已经关闭的记录器写入。
                    if self.sink is not None:
                        self.sink.camera_frame(stamp, frame)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.capture.release()

    def attach(self, recorder):
        with self.lock:
            self.sink = recorder

    def check(self):
        if self.error:
            raise TeleopError(self.error)
        if time.monotonic() - self.latest[0] > 1:
            raise TeleopError('相机超过 1 秒没有新画面，本条采集将保留为异常记录。')

    def show(self, label='Camera preview'):
        self.check()
        if self.window is None:
            self.window = PreviewWindow()
        now = time.monotonic()
        # 视频仍记录每个相机帧；仅将耗时的界面缩放/绘制限制到 10 Hz。
        if now - self.last_preview >= .1:
            with self.lock:
                stamp, frame = self.latest
            self.last_preview = now
            return self.window.show(frame, stamp, label)
        return self.window.show(label=label)

    def close(self):
        self.ended.set()
        self.thread.join(timeout=3)
        if self.window is not None:
            self.window.close()
        if self.thread.is_alive():
            raise TeleopError('相机驱动未结束读取，请退出本程序后重试。')


class EpisodeRecorder:
    """有界队列写视频和 JSONL；磁盘阻塞或队列溢出明确报错，禁止静默漏帧。"""
    def __init__(self, root, route, camera, leader_cal, follower_cal):
        if route not in ROUTES:
            raise ValueError('路径必须为 S1–S4 到 T1–T4。')
        self.path = Path(root) / route / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path.mkdir(parents=True, exist_ok=False)
        self.origin = time.monotonic()
        self.queue = queue.Queue(maxsize=128)
        self.closing = threading.Event()
        self.error = None
        self.last_camera = None
        self.camera_count = self.joint_count = 0
        self.meta = {
            'schema_version': 1, 'status': 'recording', 'route': route,
            'created_at': datetime.now().astimezone().isoformat(),
            'joint_names': list(NAMES), 'units': 'raw_encoder_ticks',
            'tick_degrees': TICK_DEGREES, 'monotonic_origin': self.origin,
            'leader_calibration': leader_cal, 'follower_calibration': follower_cal,
            'camera': {'index': camera.index, 'size': list(camera.size), 'video_fps': camera.fps},
            'timestamp_note': '共用 monotonic 软件时钟；图像为读取完成时间，不是硬件曝光同步。',
        }
        self.video = cv2.VideoWriter(str(self.path / 'camera.avi'),
                                     cv2.VideoWriter_fourcc(*'MJPG'), camera.fps, camera.size)
        if not self.video.isOpened():
            self.video.release()
            raise TeleopError('无法创建 MJPG 视频，请检查保存目录和磁盘空间。')
        try:
            write_json(self.path / 'episode.json', self.meta)
        except BaseException:
            self.video.release()
            raise
        self.thread = threading.Thread(target=self._write, daemon=True, name='koch-dataset-writer')
        self.thread.start()

    def _put(self, kind, value):
        self.check()
        if self.closing.is_set():
            raise TeleopError('记录器已关闭。')
        try:
            self.queue.put_nowait((kind, value))
        except queue.Full as exc:
            self.error = '记录队列已满，磁盘写入跟不上，停止本次采集。'
            raise TeleopError(self.error) from exc

    def camera_frame(self, stamp, frame):
        index = self.camera_count
        self._put('camera', (stamp - self.origin, frame, index))
        self.last_camera = (index, stamp - self.origin)
        self.camera_count += 1

    def joint_frame(self, frame):
        row = dict(frame)
        for key in ('leader_time', 'follower_time', 'command_time'):
            if row[key] is not None:
                row[key] -= self.origin
        row['camera_reference'] = self.last_camera
        self._put('joint', row)
        self.joint_count += 1

    def _write(self):
        try:
            with (self.path / 'joints.jsonl').open('w', encoding='utf-8') as joints, \
                    (self.path / 'camera_timestamps.jsonl').open('w', encoding='utf-8') as stamps:
                written = 0
                while not self.closing.is_set() or not self.queue.empty():
                    try:
                        kind, value = self.queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if kind == 'camera':
                        stamp, frame, index = value
                        self.video.write(frame)
                        stamps.write(json.dumps({'frame': index, 't': stamp}) + '\n')
                    else:
                        joints.write(json.dumps(value, allow_nan=False) + '\n')
                    written += 1
                    if written % 20 == 0:
                        joints.flush()
                        stamps.flush()
        except Exception as exc:
            self.error = f'数据写入失败：{exc}'
        finally:
            self.video.release()

    def check(self):
        if self.error:
            raise TeleopError(self.error)

    def finish(self, status, reason=''):
        self.closing.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.error = '磁盘线程没有按时结束，记录不完整。'
        if not self.error and status == 'review':
            capture = cv2.VideoCapture(str(self.path / 'camera.avi'))
            try:
                ok, _ = capture.read()
                if (not ok or round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != self.camera_count
                        or self.camera_count == 0 or self.joint_count < 2):
                    self.error = '视频/轨迹不完整，本条不能计入完成进度。'
            finally:
                capture.release()
        self.meta.update(status='error' if self.error else status, reason=self.error or reason,
                         camera_frames=self.camera_count, joint_frames=self.joint_count)
        write_json(self.path / 'episode.json', self.meta)
        self.check()

    def accept(self, success):
        if self.meta['status'] != 'review':
            raise TeleopError('本条未正常结束，不能标记为完成。')
        self.meta['status'] = 'complete' if success else 'rejected'
        write_json(self.path / 'episode.json', self.meta)


class ReplayTrajectory:
    """用实际发送成功的从臂目标回放，夹持时保留闭合目标；不需要主臂。"""
    def __init__(self, episode, follower_cal, rate=REPLAY_DEFAULT_RATE, max_speed=REPLAY_DEFAULT_SPEED,
                 *, observed_body=False):
        self.path = Path(episode)
        self.meta = json.loads((self.path / 'episode.json').read_text(encoding='utf-8'))
        if (self.meta.get('schema_version') != 1 or self.meta.get('status') != 'complete'
                or self.meta.get('joint_names') != list(NAMES)):
            raise TeleopError('只能回放本程序已确认完成的完整记录。')
        if self.meta.get('follower_calibration') != follower_cal:
            raise TeleopError('当前从臂校准与采集时不一致，禁止直接回放。')
        rows = [r for r in read_jsonl(self.path / 'joints.jsonl') if r.get('sent') is True]
        if observed_body:
            rows = [{**row, 'goal_raw': {**row['follower_raw'], 'gripper': row['goal_raw']['gripper']}}
                    for row in rows]
        self._initialize(rows, follower_cal, rate, max_speed)

    @classmethod
    def from_rows(cls, path, label, rows, follower_cal, rate, max_speed):
        trajectory = cls.__new__(cls)
        trajectory.path, trajectory.meta = Path(path), {'route': label}
        trajectory._initialize(rows, follower_cal, rate, max_speed)
        return trajectory

    def _initialize(self, rows, follower_cal, rate, max_speed):
        if not math.isfinite(rate) or not .1 <= rate <= 1:
            raise TeleopError('回放倍率须为 0.1–1，默认 1。')
        if not math.isfinite(max_speed) or not MIN_SPEED <= max_speed <= MAX_SPEED:
            raise TeleopError(f'速度须为 {MIN_SPEED:g}–{MAX_SPEED:g}°/s。')
        self.cal, self.requested_rate = follower_cal, rate
        self.rows = rows
        if len(self.rows) < 2:
            raise TeleopError('有效动作帧不足 2 帧，无法回放。')
        absolute = [r['command_time'] for r in self.rows]
        if any(type(t) not in (float, int) or not math.isfinite(t) for t in absolute):
            raise TeleopError('动作时间戳无效。')
        if any(b <= a for a, b in zip(absolute, absolute[1:])):
            raise TeleopError('动作时间戳必须严格递增。')
        for row in self.rows:
            checked_angles(row['goal_raw'], follower_cal)
            checked_angles(row['follower_raw'], follower_cal)
        self.recorded_duration = absolute[-1] - absolute[0]
        motor_speed = max(1, math.floor(max_speed / (0.229 * 6))) * (0.229 * 6)
        safe_speed = .9 * min(max_speed, motor_speed)
        self.safe_speed = safe_speed
        # 只延长超速的相邻段，避免一个抖动时间戳把整条记录压成慢动作。
        rows, self.times = [self.rows[0]], [0.0]
        for a, b, ta, tb in zip(self.rows, self.rows[1:], absolute, absolute[1:]):
            recorded_gap = tb - ta
            travel = max(abs(b['goal_raw'][n] - a['goal_raw'][n]) for n in NAMES) * TICK_DEGREES / safe_speed
            if recorded_gap > .5 and travel > 0:
                # 失联区间保持旧目标，到区间末尾后再限速接入新目标。
                self.times.append(self.times[-1] + recorded_gap / rate)
                rows.append(a)
                duration = travel
            else:
                duration = max(recorded_gap / rate, travel)
            self.times.append(self.times[-1] + duration)
            rows.append(b)
        self.rows = rows
        self.duration = self.times[-1]
        self.rate = self.recorded_duration / self.duration
        self.shift = None
        self.at_end = self.settled = False
        self.align_start = False
        self.start_tolerance = 2.5
        self.alignment_speed = None
        self.alignment_transition = None
        self.alignment_hold_seconds = 0.
        self.aligned_since = None
        self.use_recorded_endpoint_feedback = False
        self.allow_endpoint_hold = False
        self.endpoint_tolerance = 3.
        self.follow_feedback = False
        self.minimum_feedback_pace = 0.
        self.feedback_time = self.last_elapsed = 0.
        self.trimmed_seconds = 0.
        self.motion_origin = None

    def trim_wait(self, *, after_settle=False):
        first = self.rows[0]['goal_raw']
        def delta(goal_a, goal_b):
            return max(abs(goal_a[n] - goal_b[n]) for n in NAMES) * TICK_DEGREES

        def departure_from(start):
            goal = self.rows[start]['goal_raw']
            return next((i for i in range(start + 1, len(self.rows))
                         if delta(goal, self.rows[i]['goal_raw']) > .75), None)

        if after_settle:
            # Some demos move a few degrees to settle at home before the operator waits.
            # Preserve this movement and remove only an early near-static section.
            for start, row in enumerate(self.rows):
                if self.times[start] > 1.4 or delta(first, row['goal_raw']) > REPLAY_START_ALIGNMENT_DEGREES:
                    break
                departure = departure_from(start)
                if departure is None or self.times[departure] - self.times[start] < 1.2:
                    continue
                resume = bisect.bisect_right(self.times, self.times[departure] - .15) - 1
                transition = max(.15, delta(row['goal_raw'], self.rows[resume]['goal_raw']) / self.safe_speed)
                cut = self.times[resume] - self.times[start] - transition
                if cut <= 0:
                    continue
                stationary_recorded = self.rows[resume]['command_time'] - row['command_time']
                self.rows = self.rows[:start+1] + self.rows[resume:]
                self.times = self.times[:start+1] + [value - cut for value in self.times[resume:]]
                self.duration, self.trimmed_seconds = self.times[-1], cut
                self.rate = (self.recorded_duration - stationary_recorded + transition * self.requested_rate) / self.duration
                return
            return

        departure = departure_from(0)
        if departure is None:
            return
        # Preserve the last 150 ms before the first detected departure.
        index = max(0, bisect.bisect_right(self.times, self.times[departure] - .15) - 1)
        cut = self.times[index]
        self.trimmed_seconds = cut
        self.rows = self.rows[index:]
        self.times = [time - cut for time in self.times[index:]]
        self.duration = self.times[-1]
        recorded = self.rows[-1]['command_time'] - self.rows[0]['command_time']
        self.rate = recorded / self.duration

    def __call__(self, elapsed, actual, previous):
        if self.shift is None:
            first = self.rows[0]['goal_raw']
            self.shift = {n: 4096 * round((actual[n] - first[n]) / 4096) for n in NAMES}
            difference = [f'{n}={abs(first[n] + self.shift[n] - actual[n]) * TICK_DEGREES:.1f}°'
                          for n in NAMES if abs(first[n] + self.shift[n] - actual[n]) * TICK_DEGREES
                          > REPLAY_START_ALIGNMENT_DEGREES]
            if difference:
                self.shift = None
                raise TeleopError(f'回放起点相差超过 {REPLAY_START_ALIGNMENT_DEGREES}°，'
                                  '请在扭矩关闭时手动对齐：' + '、'.join(difference))
        if self.align_start:
            if self.motion_origin is None:
                first = {n: self.rows[0]['goal_raw'][n] + self.shift[n] for n in NAMES}
                if self.alignment_speed is not None:
                    if self.alignment_transition is None:
                        if elapsed == 0:
                            return previous
                        travel = max(abs(first[n] - previous[n]) for n in NAMES) * TICK_DEGREES
                        # Smoothstep peaks at 1.5 times its average speed.
                        self.alignment_transition = dict(previous), elapsed, 1.5 * travel / self.alignment_speed
                    origin, started, duration = self.alignment_transition
                    if elapsed - started < duration:
                        phase = (elapsed - started) / duration
                        weight = phase * phase * (3 - 2 * phase)
                        goals = {n: round(origin[n] + weight * (first[n] - origin[n])) for n in NAMES}
                        checked_angles(goals, self.cal)
                        return goals
                errors = {n: first[n] - actual[n] for n in NAMES}
                if max(map(abs, errors.values())) * TICK_DEGREES > self.start_tolerance:
                    self.aligned_since = None
                    if elapsed > 4:
                        raise TeleopError('4 秒内未对准录制起点，停止 Replay。')
                    # A small, gradual offset can overcome static joint error at rest.
                    # Keep the recorded gripper target throughout alignment.
                    bias = min(22, int(max(0, elapsed - .6) * 12))
                    goals = {n: first[n] + (int(math.copysign(bias, errors[n]))
                             if n != 'gripper' and abs(errors[n]) * TICK_DEGREES > self.start_tolerance else 0)
                             for n in NAMES}
                    checked_angles(goals, self.cal)
                    return goals
                # The preflight call uses elapsed=0 and cannot start the motion clock.
                if elapsed > 0:
                    if self.aligned_since is None:
                        self.aligned_since = elapsed
                    if elapsed - self.aligned_since >= self.alignment_hold_seconds:
                        self.motion_origin = elapsed
            elapsed = max(0, elapsed - (self.motion_origin if self.motion_origin is not None else elapsed))
        if self.follow_feedback:
            lag = max(abs(previous[n] - actual[n]) for n in NAMES[:-1]) * TICK_DEGREES
            pace = max(self.minimum_feedback_pace, (8. / max(8., lag)) ** 3)
            self.feedback_time += max(0., elapsed - self.last_elapsed) * pace
            self.last_elapsed = elapsed
            elapsed = self.feedback_time
        position = min(self.times[-1], max(0, elapsed))
        right = min(len(self.times) - 1, bisect.bisect_right(self.times, position))
        left = max(0, right - 1)
        gap = self.times[right] - self.times[left]
        weight = ((position - self.times[left]) / gap if gap else 1)
        a, b = self.rows[left]['goal_raw'], self.rows[right]['goal_raw']
        goals = {n: round(a[n] + weight * (b[n] - a[n])) + self.shift[n] for n in NAMES}
        checked_angles(goals, self.cal)
        self.at_end = elapsed >= self.duration
        endpoint = self.rows[-1]['follower_raw'] if self.use_recorded_endpoint_feedback else None
        self.settled = self.at_end and all(
            abs((endpoint[n] + self.shift[n] if endpoint is not None else goals[n]) - actual[n])
            * TICK_DEGREES <= self.endpoint_tolerance
            or (n == 'gripper' and gripper_closing_contact(actual[n], previous[n], goals[n], self.cal))
            for n in NAMES)
        if self.at_end and not self.settled and not self.allow_endpoint_hold and elapsed > self.duration + 5:
            raise TeleopError('回放末端 5 秒内未到位，停止；请检查负载和轨迹。')
        return goals
