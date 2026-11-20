"""实时调用已训练的分步 ACT 模型，用相机和从臂反馈自主执行 S→T 搬运。"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from koch_dataset import Camera, ROUTES
from koch_skill_runtime import ModelPack, STAGES, Stage, SkillSource, action_goals
from teleoperate_koch import (
    NAMES,
    ROOT,
    TICK_DEGREES,
    TeleopError,
    Teleoperator,
    connect_bus,
    load_calibration,
    make_bus,
)

ALL_SKILLS = (
    'reach_S1', 'reach_S2', 'reach_S3', 'reach_S4', 'gripper', 'lift', 'transfer',
    'lower', 'release', 'rest_lift', 'approach_move', 'approach_align',
    'return_lift', 'return_move', 'rest_align',
)
TITLES = {
    'reach_S1': '到达 S1', 'reach_S2': '到达 S2', 'reach_S3': '到达 S3',
    'reach_S4': '到达 S4', 'gripper': '夹爪开合', 'lift': '夹持后提起',
    'transfer': '移动到 T', 'lower': '向下放置', 'release': '放置后松爪',
    'rest_lift': '休息位提起', 'approach_move': '移到 S 上方',
    'approach_align': '向下对齐 S', 'return_lift': '松爪后提起',
    'return_move': '返回休息位', 'rest_align': '调整休息姿态',
}
DEFAULT_LOG_ROOT = ROOT / 'outputs/koch_model_control'


def parse_route(value):
    route = value.strip().upper().replace('→', '_').replace('->', '_').replace(' ', '')
    if route.isdigit() and 1 <= int(route) <= 16:
        return ROUTES[int(route) - 1]
    if route not in ROUTES:
        raise TeleopError('路径须为 1–16 或 S1_T1 到 S4_T4。')
    return route


def choose_route(value=None):
    if value:
        return parse_route(value)
    print('请选择 S→T 路径：')
    for start in range(0, 16, 4):
        print(' | '.join(f'{index + 1:2d}. {ROUTES[index].replace("_", "→")}'
                         for index in range(start, start + 4)))
    return parse_route(input('输入 1–16 或路径名称：'))


def selected_stages(skill, intent):
    if skill == 'workflow':
        return list(STAGES)
    if skill not in ALL_SKILLS:
        raise TeleopError(f'未知技能：{skill}')
    if skill == 'gripper':
        if intent not in ('open', 'close'):
            raise TeleopError('单独调用 gripper 时必须指定 --intent open 或 --intent close。')
        return [Stage('gripper', TITLES['gripper'], intent)]
    if intent is not None:
        raise TeleopError('--intent 只用于 gripper。')
    return [Stage(skill, TITLES[skill])]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


class RunLog:
    def __init__(self, root, info):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path = Path(root) / stamp
        self.path.mkdir(parents=True, exist_ok=False)
        self.events = (self.path / 'events.log').open('x', encoding='utf-8')
        self.frames = (self.path / 'joints.jsonl').open('x', encoding='utf-8')
        self.frame_count = 0
        self.info = dict(info, created_at=datetime.now().astimezone().isoformat(),
                         status='starting')
        write_json(self.path / 'run.json', self.info)

    def __call__(self, message):
        line = f'{datetime.now().astimezone().isoformat()} {message}'
        print(message, flush=True)
        self.events.write(line + '\n')
        self.events.flush()

    def joint_frame(self, row):
        value = dict(row, wall_time=time.time())
        self.frames.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        self.frame_count += 1
        if self.frame_count % 60 == 0:
            self.frames.flush()

    def finish(self, status, reason=''):
        self.info.update(status=status, reason=reason, finished_at=datetime.now().astimezone().isoformat(),
                         control_frames=self.frame_count)
        write_json(self.path / 'run.json', self.info)

    def close(self):
        self.events.close()
        self.frames.close()


class StopController:
    def __init__(self, camera, status, completed):
        self.camera = camera
        self.status = status
        self.completed = completed
        self.aborted = False

    def __call__(self):
        import msvcrt
        key = self.camera.show(self.status())
        if msvcrt.kbhit():
            key = ord(msvcrt.getwch().lower())
        if key in (27, 32, 3, ord('q')):
            self.aborted = True
        return self.aborted or self.completed()


class AutoTarget:
    """在每段达到示范时限且关节稳定后自动切换下一段。"""

    def __init__(self, source, emit=print, *, clock=time.monotonic, settle_timeout=12.0):
        if not math.isfinite(settle_timeout) or settle_timeout <= 0:
            raise ValueError('settle_timeout 必须为正数。')
        self.source = source
        self.emit = emit
        self.clock = clock
        self.settle_timeout = settle_timeout
        self.waiting_since = None
        self.completed = False

    def start(self):
        self.source.start()

    def close(self):
        self.source.close()

    def __call__(self, elapsed, actual, previous):
        if self.completed:
            return dict(previous)
        goals = self.source(elapsed, actual, previous)
        if not self.source.waiting:
            self.waiting_since = None
            return goals
        if self.waiting_since is None:
            self.waiting_since = self.clock()
        if self.source.settled:
            completed = self.source.stage.title
            self.waiting_since = None
            if self.source.is_last:
                self.completed = True
                self.emit(f'全部动作完成：{completed} 已稳定。')
            else:
                self.emit(f'{completed} 已稳定，准备进入下一段。')
                self.source.request_next()
        elif self.clock() - self.waiting_since > self.settle_timeout:
            raise TeleopError(
                f'{self.source.stage.title} 的推理时限结束后，'
                f'{self.settle_timeout:g} 秒内仍未到达末目标，停止。')
        return goals


def preview_camera(camera):
    import msvcrt
    print('相机预览：确认 S 区、T 区、方块和夹爪都完整可见。Enter 确认，Esc 取消。')
    while True:
        key = camera.show('调用训练模型 | 相机预览')
        if msvcrt.kbhit():
            key = ord(msvcrt.getwch())
        if key in (13, 10):
            return True
        if key in (27, 3, ord('q')):
            return False
        time.sleep(.01)


def disconnect(buses):
    errors = []
    for bus in reversed(buses):
        try:
            bus.disconnect(disable_torque=False)
        except Exception as exc:
            errors.append(f'释放串口失败 {bus.port}：{exc}')
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', help='S1_T1 到 S4_T4，也可输入编号 1–16')
    parser.add_argument('--skill', choices=('workflow',) + ALL_SKILLS, default='workflow',
                        help='workflow 执行完整 12 段流程；也可只调用一个已训练技能')
    parser.add_argument('--intent', choices=('open', 'close'),
                        help='单独调用 gripper 时的开/合指令')
    parser.add_argument('--model-manifest', type=Path,
                        help='训练包 manifest.json；省略时读取 latest_preparation.json')
    parser.add_argument('--follower-port', help='从臂串口；默认自动识别')
    parser.add_argument('--follower-calibration', type=Path,
                        default=ROOT / 'calibration/koch_legacy/follower/my_follower.json')
    parser.add_argument('--camera', type=int, default=2, help='相机编号，默认 2')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--speed', type=float, default=90.0, help='从臂最大速度，单位 °/s')
    parser.add_argument('--settle-timeout', type=float, default=12.0,
                        help='每段末目标稳定等待上限，单位秒')
    parser.add_argument('--check', action='store_true',
                        help='只读检查相机、串口、模型和首段预测，不启用扭矩')
    parser.add_argument('--camera-check', action='store_true', help='只预览相机')
    parser.add_argument('--yes', action='store_true', help='跳过开始前的人工确认')
    parser.add_argument('--log-root', type=Path, default=DEFAULT_LOG_ROOT)
    args = parser.parse_args(argv)

    camera = None
    bus = None
    source = None
    log = None
    status = 'error'
    reason = ''
    try:
        camera = Camera(args.camera)
        if args.camera_check:
            preview_camera(camera)
            status, reason = 'complete', 'camera_check'
            return 0

        if not preview_camera(camera):
            status, reason = 'cancelled', 'camera_preview_cancelled'
            return 0
        route = choose_route(args.route)
        stages = selected_stages(args.skill, args.intent)
        follower_cal = load_calibration(args.follower_calibration)
        pack = ModelPack(args.model_manifest)
        missing = [stage.skill for stage in stages if stage.skill not in pack.index]
        if missing:
            raise TeleopError('训练包缺少模型：' + '、'.join(sorted(set(missing))))
        durations = [pack.duration(stage, route) for stage in stages]

        log = RunLog(args.log_root, {
            'route': route, 'skill': args.skill, 'intent': args.intent,
            'camera_index': args.camera, 'device': args.device,
            'model_manifest': str(pack.path), 'stages': [
                {'skill': stage.skill, 'title': stage.title, 'intent': stage.intent,
                 'duration_s': duration}
                for stage, duration in zip(stages, durations)
            ],
        })
        event = log
        event(f'模型包：{pack.path}')
        event(f'路径 {route.replace("_", "→")}；{len(stages)} 段；推理目标按训练数据的 20 Hz 轨迹时长控制。')

        actors = pack.load(stages, args.device, follower_cal)
        source = SkillSource(actors, stages, durations, route, follower_cal, camera, emit=event)
        target = AutoTarget(source, emit=event, settle_timeout=args.settle_timeout)

        bus = connect_bus('follower', args.follower_port, factory=make_bus)
        event(f'从臂：{bus.port}')
        control = Teleoperator(None, bus, None, follower_cal, args.speed)
        _, actual = control.preflight()
        source.prime(actual)
        first = action_goals(source.primed, actors[source.stage.skill].names, actual, follower_cal)
        event('首段目标相对当前位置：' + ' '.join(
            f'{name}={((first[name] - actual[name]) * TICK_DEGREES):+.1f}°' for name in NAMES))
        if args.check:
            event('只读检查通过；未启用扭矩，也未发送运动目标。')
            status, reason = 'complete', 'check_only'
            return 0

        print(f'控制频率 60 Hz；模型动作时长 20 Hz；最大速度 {args.speed:g}°/s。')
        print('已自动校正编码器整圈；模型校准必须与当前从臂校准完全一致。')
        print('Esc/空格/Q：停止并释放从臂。执行期间请保持相机与 S/T 区布置不变。')
        if not args.yes:
            input('确认活动空间、方块位置和从臂起始姿态后按 Enter 开始：')

        def completed():
            return target.completed

        def state_label():
            if target.completed:
                return '调用训练模型 | 已完成'
            return f'调用训练模型 | {source.stage_index + 1}/{len(stages)} {source.stage.title}'

        stop = StopController(camera, state_label, completed)
        target.start()
        control.run(stop=stop, target_source=target, emit=event, on_frame=log.joint_frame)
        if target.completed:
            status, reason = 'complete', 'workflow_complete'
            event('自主流程完成，已释放从臂扭矩。')
            return 0
        if stop.aborted:
            status, reason = 'cancelled', 'operator_stopped'
            event('操作员停止，已释放从臂扭矩。')
            return 130
        raise TeleopError('控制结束，但没有确认完整流程完成。')
    except (KeyboardInterrupt, EOFError) as exc:
        status, reason = 'cancelled', str(exc) or type(exc).__name__
        print('\n已停止。', file=sys.stderr)
        return 130
    except BaseException as exc:
        status, reason = 'error', str(exc) or type(exc).__name__
        print(f'\n停止：{reason}', file=sys.stderr)
        return 1
    finally:
        cleanup = []
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                cleanup.append(f'关闭推理线程：{exc}')
        cleanup.extend(disconnect([bus] if bus is not None else []))
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:
                cleanup.append(f'关闭相机：{exc}')
        if log is not None:
            try:
                log.finish(status, reason if not cleanup else reason + '；' + '；'.join(cleanup))
            except Exception as exc:
                print(f'关闭诊断日志失败：{exc}', file=sys.stderr)
            finally:
                try:
                    log.close()
                except Exception as exc:
                    print(f'关闭诊断日志失败：{exc}', file=sys.stderr)
        for message in cleanup:
            print(message, file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
