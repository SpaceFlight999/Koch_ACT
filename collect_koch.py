"""Koch 16 条 S→T 示范采集与从臂 replay；菜单操作，不修改校准。"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from koch_dataset import (Camera, EpisodeRecorder, PreviewWindow, ReplayTrajectory, ROUTES,
                          REPLAY_START_ALIGNMENT_DEGREES,
                          REPLAY_DEFAULT_RATE, REPLAY_DEFAULT_SPEED,
                          completed_episodes, read_jsonl)
from teleoperate_koch import (COLLECTION_SPEED, DEFAULT_SPEED, NAMES, ROOT, SPECS, TICK_DEGREES,
                              TeleopError, Teleoperator, connect_bus, load_calibration, make_bus)

DEFAULT_OUTPUT = ROOT / 'datasets/koch_st'


def route_name(value):
    value = value.strip().upper().replace('→', '_').replace('->', '_').replace(' ', '')
    if value.isdigit() and 1 <= int(value) <= 16:
        return ROUTES[int(value) - 1]
    if value in ROUTES:
        return value
    raise TeleopError('请输入 1–16 或 S1_T1 这样的路径名称。')


def show_progress(root):
    count = sum(bool(completed_episodes(root, route)) for route in ROUTES)
    print(f'\n首轮进度 {count}/16 （✓ 表示至少一条已确认成功的记录）')
    for start in range(0, 16, 4):
        print(' | '.join(f'{i + 1:2d}. {ROUTES[i].replace("_", "→")} '
                         f'{"✓" if completed_episodes(root, ROUTES[i]) else "待采集"}'
                         for i in range(start, start + 4)))
    print(f'保存目录：{root}')


class SessionKeys:
    def __init__(self, camera=None, recorder=None, trajectory=None):
        self.camera, self.recorder, self.trajectory = camera, recorder, trajectory
        self.finished = self.aborted = self.announced = False

    def __call__(self):
        import msvcrt
        if self.recorder is not None:
            self.recorder.check()
        key = -1
        if self.camera is not None:
            label = (f'REC {self.recorder.meta["route"]} | F: finish | ESC: abort'
                     if self.recorder else 'Preview | ENTER: ready | ESC: cancel')
            key = self.camera.show(label)
        if key not in (27, 32, 3) and msvcrt.kbhit():
            key = ord(msvcrt.getwch().lower())
        if self.trajectory is not None and self.trajectory.settled and not self.announced:
            print('\n已到回放末帧，正在保持。请托稳从臂，按 F 结束并释放扭矩。', flush=True)
            self.announced = True
        if key in (27, 32, 3, ord('q')):
            self.aborted = True
        if key == ord('f'):
            self.finished = self.trajectory is None or self.trajectory.settled
            self.aborted = not self.finished
        return self.finished or self.aborted


def preview(camera):
    import msvcrt
    print('相机预览已打开。调整画面使 S 区、T 区和夹爪都清楚可见。')
    print('确认相机正确后，在预览窗口或终端按 Enter；Esc 取消。')
    while True:
        key = camera.show()
        if msvcrt.kbhit():
            key = ord(msvcrt.getwch())
        if key in (13, 10):
            return True
        if key in (27, 3, ord('q')):
            return False
        time.sleep(.01)


def disconnect(buses):
    for bus in reversed(buses):
        try:
            bus.disconnect(disable_torque=False)
        except Exception as exc:
            print(f'释放串口失败 {bus.port}：{exc}', file=sys.stderr)


def calibrations(args):
    return load_calibration(args.leader_calibration), load_calibration(args.follower_calibration)


def record(args, route):
    camera, recorder, buses = None, None, []
    result, reason = 'aborted', '用户取消或示范未正常结束。'
    try:
        # 相机、目录与视频编码器均就绪后，才允许启用机械臂。
        camera = Camera(args.camera)
        if not preview(camera):
            return
        leader_cal, follower_cal = calibrations(args)
        for arm, port in (('leader', args.leader_port), ('follower', args.follower_port)):
            buses.append(connect_bus(arm, port, factory=make_bus))
            print(f'{SPECS[arm]["label"]}：{buses[-1].port}')
        smooth = not args.standard_control
        speed = args.speed if args.speed is not None else (COLLECTION_SPEED if smooth else DEFAULT_SPEED)
        control = Teleoperator(*buses, leader_cal, follower_cal, speed, smooth_collection=smooth)
        control.preflight()
        print(f'\n采集 {route.replace("_", "→")}：请先把木块放在对应 S 格。')
        print(f'{"流畅" if smooth else "标准"}采集：目标 {1 / control.period:g} Hz，'
              f'最大 {speed:g}°/s；实际频率取决于串口响应。')
        print('两臂放到相近起始姿态。开始后用主臂抓取、搬运到对应 T 格、松开并退回休息位。')
        print('F：正常结束；Esc/空格：放弃。结束会释放从臂扭矩，请先放下物体并托稳从臂。')
        input('准备好后按 Enter 开始录制并跟随：')
        camera.check()
        recorder = EpisodeRecorder(args.output, route, camera, leader_cal, follower_cal)
        recorder.meta['control'] = {'mode': 'smooth' if smooth else 'standard',
                                    'target_hz': 1 / control.period, 'speed_degrees_per_second': speed}
        print(f'记录目录：{recorder.path}', flush=True)
        camera.attach(recorder)
        deadline = time.monotonic() + 2
        while recorder.camera_count == 0:
            camera.show('Preparing recording...')
            if time.monotonic() > deadline:
                raise TeleopError('没有录到相机首帧。')
            time.sleep(.01)
        keys = SessionKeys(camera, recorder)
        control.run(stop=keys, on_frame=recorder.joint_frame)
        if keys.finished and recorder.joint_count >= 2 and recorder.camera_count:
            result, reason = 'review', ''
    except BaseException as exc:
        result, reason = 'error', str(exc) or type(exc).__name__
        raise
    finally:
        # 无论写盘/相机是否失败，控制器已先执行扭矩释放，再做磁盘清理。
        original = sys.exception()
        cleanup_errors = []
        disconnect(buses)
        if camera is not None:
            camera.attach(None)
        if recorder is not None:
            recorder.meta['control']['discarded_position_samples'] = control.discarded_samples
        try:
            if recorder is not None:
                recorder.finish(result, reason)
        except Exception as exc:
            cleanup_errors.append(str(exc))
        try:
            if camera is not None:
                camera.close()
        except Exception as exc:
            cleanup_errors.append(str(exc))
        if cleanup_errors:
            detail = '；'.join(cleanup_errors)
            if original is not None:
                raise TeleopError(f'原始停止原因：{original}\n记录清理：{detail}') from original
            raise TeleopError(detail)
    if recorder is not None:
        if result == 'review':
            success = input('这条示范是否成功把木块从指定 S 格放到 T 格？输入 y 确认保存为完成，其余保留为重录：').strip().lower() == 'y'
            recorder.accept(success)
            print('已记入完成进度。' if success else '已保留为失败示范，不计入 16 条进度。')
        else:
            print('本条未计入进度，已有文件保留。')
        print(f'保存位置：{recorder.path}')


def resolve_episode(args, value):
    candidate = Path(value)
    if candidate.is_dir():
        return candidate
    route = route_name(value)
    available = completed_episodes(args.output, route)
    if not available:
        raise TeleopError(f'{route} 还没有已确认成功的记录。')
    if len(available) == 1:
        return available[0]
    print(f'{route} 有 {len(available)} 条记录（均保留）：')
    for index, path in enumerate(available, 1):
        print(f'{index}: {path.name}')
    selected = input('选择版本，Enter 使用最新：').strip()
    if not selected:
        return available[-1]
    if not selected.isdigit() or not 1 <= int(selected) <= len(available):
        raise TeleopError('版本编号无效。')
    return available[int(selected) - 1]


def replay(args, episode):
    follower_cal = load_calibration(args.follower_calibration)
    speed = args.speed if args.speed is not None else REPLAY_DEFAULT_SPEED
    source = ReplayTrajectory(episode, follower_cal, args.replay_rate, speed)
    first = source.rows[0]['goal_raw']
    print(f'\n从臂回放 {source.meta["route"]} | 请求 {source.requested_rate:g} 倍速，'
          f'平均 {source.rate:.3f} 倍 | 预计 {source.duration:.1f}s')
    if source.rate + 1e-6 < args.replay_rate:
        print('仅对超速的小段延长时间，其余按请求速度回放；所有目标点保留。')
    print('不需要连接主臂；不自动识别木块位置，S/T 方格、相机和木块须保持采集时的布置。')
    print('起点原始编码器值：' + ' '.join(f'{n}={first[n]}' for n in NAMES))
    print('Esc/空格随时停止并释放；到末帧会保持，托稳后按 F 结束。')
    buses, log_stream = [], None
    def report(message):
        nonlocal log_stream
        print(message, flush=True)
        if log_stream is not None:
            try:
                log_stream.write(f'{datetime.now().astimezone().isoformat()} {message}\n')
                log_stream.flush()
            except OSError as exc:
                print(f'回放日志写入失败：{exc}', file=sys.stderr)
                try:
                    log_stream.close()
                except OSError:
                    pass
                log_stream = None
    try:
        log_dir = ROOT / 'outputs/koch_replay'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.log')
        log_stream = log_path.open('x', encoding='utf-8')
        print(f'本次回放诊断日志：{log_path}')
        report(f'轨迹：{Path(episode).resolve()}；请求倍率={source.requested_rate:g}；'
               f'平均倍率={source.rate:g}；预计时长={source.duration:.3f}s；最大速度={speed:g}°/s')
        bus = connect_bus('follower', args.follower_port, factory=make_bus)
        buses.append(bus)
        control = Teleoperator(None, bus, None, follower_cal, speed)
        report(f'回放目标更新频率：{1 / control.period:g} Hz；实际频率将在运行中显示。')
        _, actual = control.preflight()
        print('请在扭矩关闭时手动对齐，下面显示各关节到起点的最短角度差：')
        print(' '.join(f'{n}={((first[n] - actual[n] + 2048) % 4096 - 2048) * TICK_DEGREES:+.1f}°'
                       for n in NAMES))
        input(f'放好木块、托稳对齐，确认活动空间后按 Enter 开始回放'
              f'（起点须在 {REPLAY_START_ALIGNMENT_DEGREES}° 内）：')
        # run 内再次预检并验证起点，用户确认后的位置不能直接沿用。
        keys = SessionKeys(trajectory=source)
        control.run(stop=keys, target_source=source, emit=report)
        report('回放已完成并释放从臂。' if keys.finished else '回放已中断并释放从臂。')
    except BaseException as exc:
        report(f'回放停止：{exc or type(exc).__name__}')
        raise
    finally:
        disconnect(buses)
        if log_stream is not None:
            try:
                log_stream.close()
            except OSError as exc:
                print(f'关闭回放日志失败：{exc}', file=sys.stderr)


def video_preview(episode):
    stamps = read_jsonl(Path(episode) / 'camera_timestamps.jsonl')
    capture = cv2.VideoCapture(str(Path(episode) / 'camera.avi'))
    if not capture.isOpened() or not stamps:
        capture.release()
        raise TeleopError('录制视频无法读取。')
    print('仅播放记录视频，不连接机械臂。Esc/空格退出。')
    window = None
    try:
        window = PreviewWindow('Koch 已录制视频')
        started = time.monotonic()
        for row in stamps:
            ok, frame = capture.read()
            if not ok:
                raise TeleopError('视频帧数少于时间戳记录，文件可能损坏。')
            remaining = started + row['t'] - stamps[0]['t'] - time.monotonic()
            while remaining > 0:
                if window.show() in (27, 32):
                    return
                time.sleep(min(.02, remaining))
                remaining = started + row['t'] - stamps[0]['t'] - time.monotonic()
            if window.show(frame, row['frame']) in (27, 32):
                return
    finally:
        capture.release()
        if window is not None:
            window.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--camera', type=int, default=2, help='相机编号，当前 USB 相机为 2，可在菜单中切换')
    parser.add_argument('--leader-port')
    parser.add_argument('--follower-port')
    parser.add_argument('--leader-calibration', type=Path, default=ROOT / 'calibration/koch_legacy/leader/my_leader.json')
    parser.add_argument('--follower-calibration', type=Path, default=ROOT / 'calibration/koch_legacy/follower/my_follower.json')
    parser.add_argument('--speed', type=float, help='最大速度 2–120°/s；流畅采集/回放默认 120，标准采集默认 90')
    parser.add_argument('--standard-control', action='store_true', help='采集恢复为原来的 20 Hz、90°/s 控制方式')
    parser.add_argument('--replay-rate', type=float, default=REPLAY_DEFAULT_RATE,
                        help='请求回放倍率 0.1–1，默认 1（录制速度）；仅超速小段会进一步降速')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--record', help='直接选择采集路径，例如 S1_T1 或 1')
    modes.add_argument('--replay', help='回放路径编号、名称或记录目录')
    modes.add_argument('--video', help='只看记录的视频，不连接机械臂')
    modes.add_argument('--list', action='store_true', help='只显示 16 条采集进度')
    modes.add_argument('--camera-check', action='store_true', help='仅预览相机，不连接机械臂')
    args = parser.parse_args(argv)
    try:
        if args.list:
            show_progress(args.output)
        elif args.camera_check:
            camera = Camera(args.camera)
            try:
                preview(camera)
            finally:
                camera.close()
        elif args.record:
            record(args, route_name(args.record))
        elif args.replay:
            replay(args, resolve_episode(args, args.replay))
        elif args.video:
            video_preview(resolve_episode(args, args.video))
        else:
            while True:
                show_progress(args.output)
                print(f'当前相机编号：{args.camera}；C：换相机/预览；P+编号：从臂回放；V+编号：看视频；Q：退出')
                command = input('输入 1–16 选择采集路径（已有记录会另存，不覆盖）：').strip().lower()
                if command == 'q':
                    break
                try:
                    if command == 'c':
                        value = input('相机编号（通常 0 或 1）：').strip()
                        args.camera = int(value)
                        camera = Camera(args.camera)
                        try:
                            preview(camera)
                        finally:
                            camera.close()
                    elif command.startswith(('p', 'v')):
                        episode = resolve_episode(args, command[1:])
                        (replay if command[0] == 'p' else lambda a, e: video_preview(e))(args, episode)
                    else:
                        record(args, route_name(command))
                except (KeyboardInterrupt, EOFError):
                    # 中断后退出菜单；不在状态可能未知时自动重启运动。
                    raise
                except (TeleopError, ValueError, OSError) as exc:
                    print(f'本次未完成：{exc}', file=sys.stderr)
                    if not isinstance(exc, ValueError):
                        print('请确认机械臂已托稳，解决原因后再选择路径。')
    except (KeyboardInterrupt, EOFError):
        print('\n已取消；中途记录不计入完成进度。')
        return 130
    except Exception as exc:
        print(f'停止：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
