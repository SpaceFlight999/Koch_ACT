"""Run with python -m koch_autonomy --help."""
import argparse
import json
from pathlib import Path
import time

from .planner import plan_task
from .reverse import ROOT, prepare, read_json, save_json, train_reference
from .workflow import commands, simulate
from . import replay_bridge
from koch_project_paths import resolve_config_reference


def load_target_reader(rotation=None, target=None, config_path=None, quantity_rule='parity'):
    config_path = Path(config_path).resolve() if config_path else ROOT / 'configs/grid_runtime.json'
    config = read_json(config_path)
    model = resolve_config_reference(config['model_path'], config_path)
    colors = resolve_config_reference(config['color_config_path'], config_path)
    verifier = (resolve_config_reference(config['verifier_model_path'], config_path)
                if config.get('verifier_model_path') else ROOT / 'outputs/digits/mnist_cnn_v1/best.pt')
    for label, path in (('纸面数字模型', model), ('纸面颜色标定', colors), ('独立数字复核模型', verifier)):
        if not path.is_file():
            raise ValueError(f'缺少{label}：{path}。请恢复原目录结构或在 --config 中指定实际文件路径。')
    from handwritten_digits.model import Predictor
    from .vision import TargetReader
    return TargetReader(Predictor(model), read_json(colors), rotation, target,
                        verifier=Predictor(verifier), quantity_rule=quantity_rule,
                        locate_four_corners=config.get('locate_four_corners', False))


def camera_observe(args, *, on_view=None, cancelled=None):
    import cv2
    from camera_vision import read_image, write_image
    from .vision import StableRead, load_stack_reader, mark_source, rectify_marked_plane
    alignment = None
    if getattr(args, 'alignment', None) is not None:
        from .camera_alignment import CameraAlignment, SOURCE_ADJUST_HINT
        alignment = CameraAlignment(args.alignment)
        if args.kind != 'source' or args.camera_name != alignment.profile['camera_name']:
            raise ValueError('S 区相机名须与全局相机对位配置一致。')
    plane = read_json(args.plane) if args.plane else None
    if plane and (args.kind != 'targets' or plane.get('scope') != 'target_paper_plane'):
        raise ValueError('平面校正仅用于 T 彩纸平面，不能当作三维方块或 ACT 相机视角补偿。')
    if plane and args.rotation not in (None, 0):
        raise ValueError('标记平面已定义正方向，不能再次旋转编号。')
    def read_frame(frame):
        if alignment is not None:
            aligned = alignment.measure(frame)
            registered = alignment.register_source(frame, aligned)
            if registered is None:
                marked = frame.copy()
                hint = (SOURCE_ADJUST_HINT if alignment.source_bounds else
                        '请打开「全局相机对位」调整支架。')
                return ({'complete': False, 'stable': False, 'alignment': aligned,
                         'status': f'全局相机暂不能定位 S 区。{hint}'}, marked)
            aligned['source_registration_applied'] = registered is not frame
            frame = registered
        if plane:
            frame, visible, _ = rectify_marked_plane(frame, plane)
        if args.roi:
            x, y, w, h = args.roi
            if min(x, y) < 0 or min(w, h) < 10 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                raise ValueError('T 区 ROI 超出画面或过小。')
            frame = frame[y:y+h, x:x+w]
            if plane and not (visible[y:y+h, x:x+w] == 255).all():
                raise ValueError('校正后 T 区有不可见部分，请调整取景。')
        elif plane and not (visible == 255).all():
            raise ValueError('校正画面有不可见边界，请用 --roi 限定完全可见的 T 区。')
        result = reader.read(frame, registered=True) if alignment is not None else reader.read(frame)
        report, marked = result if isinstance(result, tuple) else (result, frame)
        if args.kind == 'source':
            marked = mark_source(frame, report)
        report.update(input_roi=args.roi, marked_plane_registration=bool(plane))
        if alignment is not None:
            report['alignment'] = aligned
        return report, marked
    if args.kind == 'targets':
        reader = load_target_reader(0 if plane else args.rotation, args.target, args.config,
                                    getattr(args, 'quantity_rule', 'parity'))
    else:
        if args.roi:
            raise ValueError('堆栈使用八面标定中的区域，不能再用 --roi 改变像素坐标。')
        if not args.config:
            raise ValueError('堆栈识别需要 --config 指定当前视角的八面标定。')
        reader = load_stack_reader(Path(args.config).resolve(), args.kind)
    if args.image:
        frame = read_image(args.image)
        report, marked = read_frame(frame)
        report.update(input_image=str(args.image.resolve()), stable=False, hardware_executed=False)
        save_json(args.output, report)
        write_image(args.output.with_suffix('.png'), marked)
        print(report['status'])
        return
    if args.camera is None and not args.camera_name:
        raise ValueError('实时识别须明确指定 --camera 或 --camera-name，不自动猜设备。')
    if args.camera_name:
        from grid_reader.cameras import resolve_named_camera
        camera_index, camera_name = resolve_named_camera(args.camera_name)
    else:
        camera_index, camera_name = args.camera, None
    if args.seconds <= 0:
        raise ValueError('识别超时须大于零。')
    if cancelled is not None and cancelled():
        raise KeyboardInterrupt('用户中止视觉读取。')
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    try:
        if not capture.isOpened():
            raise ValueError(f'无法打开指定相机 {camera_index}。')
        size = reader.config['frame_size'] if args.kind != 'targets' else [1280, 720]
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        stable = StableRead()
        until = time.monotonic() + args.seconds
        report = {'complete': False, 'stable': False, 'status': '等待画面。'}
        while time.monotonic() < until:
            if cancelled is not None and cancelled():
                raise KeyboardInterrupt('用户中止视觉读取。')
            ok, frame = capture.read()
            stamp = time.monotonic()
            if not ok:
                raise ValueError('指定相机未返回图像。')
            report, marked = read_frame(frame)
            report = stable.update(report, stamp)
            report.update(camera_index=camera_index, camera_name=camera_name, hardware_executed=False)
            if on_view is not None:
                on_view(args.kind, marked)
            if args.show:
                cv2.imshow('Koch perception - ESC stops', marked)
                if cv2.waitKey(1) & 0xFF == 27:
                    report.update(stable=False, status='用户停止读取。')
                    break
            if report['stable']:
                break
        if 'marked' in locals():
            write_image(args.output.with_suffix('.png'), marked)
            if not report['stable']:
                raw_path = args.output.with_name(args.output.stem + '-raw.png')
                write_image(raw_path, frame)
                report['raw_frame'] = raw_path.name
        save_json(args.output, report)
        print(report['status'], 'stable=', report['stable'])
        if not report['stable']:
            raise ValueError(f'{report["status"]} 没有取得稳定读取；诊断已保存到 {args.output}')
        return report
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()


def calibrate_source(args):
    import cv2
    from camera_vision import read_image, fit_sample
    from .planner import SLOTS, TARGETS
    from .vision import SourceReader, TargetStackReader, appearance_difference, warp_quad, marker_centers
    frame = read_image(args.image)
    region = 'T' if args.command == 'calibrate-target-stack' else 'S'
    empty = read_image(args.empty_image) if args.empty_image else None
    if empty is not None and empty.shape != frame.shape:
        raise ValueError('有块/空场景图片必须来自同一固定相机、同一分辨率。')
    if empty is not None and (args.empty_max_difference is None or args.occupied_min_difference is None):
        raise ValueError('空位标定须明确提供 --empty-max-difference 和 --occupied-min-difference。')
    faces = {}
    confirmed_colors = {}
    colors = read_json(args.colors)
    # Each sampled face must have a user-confirmed color to bind the two cameras'
    # vocabularies. This avoids applying top-surface samples blindly to side faces.
    for slot in (SLOTS if region == 'S' else TARGETS):
        for layer in (1, 2):
            name = f'{slot}_L{layer}'
            print(f'{name}: 框选本块独立侧面内部（第一层在下），Enter确认；不要包含相邻块。', flush=True)
            x, y, w, h = map(int, cv2.selectROI(name, frame, showCrosshair=True))
            cv2.destroyWindow(name)
            if min(w, h) < 10:
                raise ValueError('采样取消或区域过小，未保存标定。')
            color = input(f'{name} 真实颜色键 {list(colors["colors"])}: ').strip()
            if color not in colors['colors']:
                raise ValueError('未知颜色键，未保存标定。')
            sample = fit_sample(frame[y:y+h, x:x+w])
            sample.update(source_image=str(args.image.resolve()), source_face=name)
            colors['colors'][color]['samples'].append(sample)
            faces[name] = [[x, y], [x+w-1, y], [x+w-1, y+h-1], [x, y+h-1]]
            confirmed_colors[name] = color
            if empty is not None:
                before, _ = warp_quad(empty, faces[name], 64)
                after, _ = warp_quad(frame, faces[name], 64)
                difference = appearance_difference(after, before)
                if difference < args.occupied_min_difference:
                    raise ValueError(f'{name} 有块与空场景差异仅 {difference:.2f}，不能可靠分离；未保存标定。')
    colors['calibration_required'] = False
    color_path = args.output.with_name(args.output.stem + '_colors.json').resolve()
    config = {'region': region, 'frame_size': list(frame.shape[1::-1]), 'view_validated': True, 'faces': faces,
              'color_config_path': str(color_path), 'reference_image': str(args.image.resolve()),
              'coordinate_scope': 'fixed_camera_visible_faces_not_robot_coordinates'}
    if args.view_marker_ids:
        config['view_markers'] = {'marker_ids': args.view_marker_ids,
                                  'reference_points_px': marker_centers(frame, args.view_marker_ids).tolist(),
                                  'max_shift_px': args.max_view_shift_px}
    if empty is not None:
        config.update(empty_reference_image=str(args.empty_image.resolve()),
                      empty_max_difference=args.empty_max_difference,
                      occupied_min_difference=args.occupied_min_difference)
    reader = (SourceReader if region == 'S' else TargetStackReader)(config, colors)
    report = reader.read(frame)
    if not report['complete']:
        raise ValueError(f'标定图自检未通过：{report["unknown"]}')
    if any(f['color'] != confirmed_colors[f['id']] for f in report['faces']):
        raise ValueError('采样后预测颜色与确认颜色不一致，未保存标定。')
    if empty is not None:
        empty_report = reader.read(empty)
        if not empty_report['complete'] or empty_report['observed_count'] != 0:
            raise ValueError('空场景自检未通过，未保存标定。')
    save_json(color_path, colors)
    save_json(args.output, config)
    print('八个侧面及颜色标定已保存：', args.output)


def demo():
    return {'inventory': {'S1': ['red', 'blue'], 'S2': ['yellow', 'green'],
                          'S3': ['red', 'blue'], 'S4': ['yellow', 'green']},
            'targets': {'T1': {'color': 'red', 'number': 1}, 'T2': {'color': 'blue', 'number': 2},
                        'T3': {'color': 'yellow', 'number': 3}, 'T4': {'color': 'green', 'number': 4}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='S/T 视觉识别、任务规划与从臂 Replay。')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'simulate'):
        p = sub.add_parser(name)
        p.add_argument('--input', type=Path, required=True, help='包含 inventory/targets 的 JSON，库存从下往上。')
        p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('demo')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/koch_autonomy/demo.json')
    p = sub.add_parser('observe')
    p.add_argument('kind', choices=('source', 'targets', 'target-stack'))
    p.add_argument('--config', type=Path)
    src = p.add_mutually_exclusive_group()
    src.add_argument('--image', type=Path)
    src.add_argument('--camera', type=int)
    src.add_argument('--camera-name')
    p.add_argument('--rotation', type=int, choices=(0, 90, 180, 270), help='仅在方向已标定时指定；默认自动判断。')
    p.add_argument('--quantity-rule', choices=('parity', 'direct'), default='parity', help='T 卡数字按奇偶或直接计数。')
    p.add_argument('--target', choices=('T1', 'T2', 'T3', 'T4'), help='已到达指定扫描位且画面只有一张纸时使用。')
    p.add_argument('--roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), help='仅框选 T 区；输出坐标相对裁剪图。')
    p.add_argument('--plane', type=Path, help='T 彩纸平面的四个 ArUco 标记配置；不用于 ACT。')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--show', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    for name in ('calibrate-source', 'calibrate-target-stack'):
        p = sub.add_parser(name)
        p.add_argument('--image', type=Path, required=True, help='八个层位均有独立可见方块的标定图。')
        p.add_argument('--empty-image', type=Path, required=name == 'calibrate-target-stack',
                       help='同一视角、所有层位为空的参考图；T区纸张位置颜色须与任务一致。')
        p.add_argument('--empty-max-difference', type=float)
        p.add_argument('--occupied-min-difference', type=float)
        p.add_argument('--view-marker-ids', type=int, nargs=4, help='固定背景上的四个 ArUco 标记，依序TL/TR/BR/BL。')
        p.add_argument('--max-view-shift-px', type=float, default=3., help='固定视角容差，默认3像素；不是位姿补偿。')
        p.add_argument('--colors', type=Path, default=ROOT / 'configs/grid_blocks_colors.json')
        p.add_argument('--output', type=Path, default=ROOT / ('configs/autonomy_source.json' if name == 'calibrate-source'
                                                            else 'configs/autonomy_target_stack.json'))
    p = sub.add_parser('prepare-reverse')
    p.add_argument('--output', type=Path, required=True, help='新的独立输出目录，不覆盖原始示范。')
    p.add_argument('--batch', type=Path)
    p = sub.add_parser('train-reference')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=300)
    p = sub.add_parser('bind-models')
    p.add_argument('--plan', type=Path, required=True, help='plan/simulate/demo 输出的 JSON。')
    p.add_argument('--output', type=Path, required=True)
    sub.add_parser('cameras')
    p = sub.add_parser('align-camera', help='显示全局相机参考点与实时偏移。')
    p.add_argument('--profile', type=Path)
    p.add_argument('--site', type=Path, help='与 Replay 面板共用 S 区配置并核对八个层位。')
    p.add_argument('--output', type=Path, help='对齐后可保存当前原图和偏移报告。')
    p = sub.add_parser('replay-auto', help='自动识别、连续执行、结束时核对 S 区剩余库存。')
    p.add_argument('--site', type=Path, required=True, help='同伴电脑的相机、从臂和路径配置。')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--preview', action='store_true', help='从当前取景识别并生成计划，不连接机械臂。')
    mode.add_argument('--scan-only', action='store_true', help='从休息位到 T 区扫描，返回休息位后结束。')
    p.add_argument('--reviewed-targets', type=Path, help='使用人工核对的 T1–T4 任务卡 JSON，跳过腕部扫描。')
    p.add_argument('--carry-rate', type=float, default=replay_bridge.RATE,
                   help='搬运示范的回放倍率，仍按关节限速重定时。')
    p = sub.add_parser('replay-ui', help='打开相机画面与自动 Replay 进度面板。')
    p.add_argument('--site', type=Path, required=True, help='自动搬运的现场配置。')
    replay_bridge.configure_cli(sub)
    args = parser.parse_args(argv)
    try:
        if args.command in ('plan', 'simulate', 'demo'):
            data = demo() if args.command == 'demo' else read_json(args.input)
            if args.command == 'plan':
                plan = plan_task(data['inventory'], data['targets'], data.get('positions'), edge_costs=data.get('edge_costs'))
                result = {'mode': 'offline_plan', 'hardware_executed': False, 'plan': plan, 'commands': commands(plan)}
            else:
                result = simulate(data['inventory'], data['targets'], data.get('positions'), data.get('edge_costs'))
            if args.command == 'demo':
                save_json(args.output.with_name('demo_input.json'), data)
            save_json(args.output, result)
            print(f'已保存离线结果：{args.output}；搬运 {len(result["plan"]["moves"])} 次；没有驱动机械臂。')
        elif args.command == 'observe':
            camera_observe(args)
        elif args.command in ('calibrate-source', 'calibrate-target-stack'):
            calibrate_source(args)
        elif args.command == 'prepare-reverse':
            result = prepare(args.output, args.batch)
            print('保留/排除片段：', len(result['records']), len(result['rejected']))
        elif args.command == 'train-reference':
            result = train_reference(args.manifest, args.epochs)
            print('逆向参考留出 MAE：', result['best_validation_mae_degrees'], '度；禁止部署。')
        elif args.command == 'bind-models':
            from .models import bind_plan
            result = bind_plan(read_json(args.plan)['plan'])
            save_json(args.output, result)
            print(f'离线绑定 {len(result["moves"])} 次搬运，校验 {result["model_files_checked"]} 个文件；未执行动作。')
        elif args.command == 'cameras':
            from grid_reader.cameras import video_devices
            print(json.dumps([{'index': i, 'name': name} for i, name in enumerate(video_devices())], ensure_ascii=False, indent=2))
        elif args.command == 'align-camera':
            from .camera_alignment import DEFAULT_PROFILE, open_alignment
            profile, source_config = args.profile, None
            if args.site:
                site_dir = args.site.resolve().parent
                source = read_json(args.site)['source']
                if profile is None and source.get('alignment'):
                    profile = site_dir / source['alignment']
                source_config = site_dir / source['config']
            if not open_alignment(profile or DEFAULT_PROFILE, args.output, source_config=source_config):
                parser.exit(1, '全局相机对位未确认。\n')
        elif args.command == 'replay-auto':
            from .replay_auto import run_auto
            if args.scan_only and args.reviewed_targets is not None:
                raise ValueError('--scan-only 与 --reviewed-targets 不能同时使用。')
            run_auto(args.site, args.preview, camera_observe, scan_only=args.scan_only,
                     reviewed_targets=args.reviewed_targets, carry_rate=args.carry_rate)
        elif args.command == 'replay-ui':
            from .replay_dashboard import open_dashboard
            open_dashboard(args.site, camera_observe)
        elif args.command.startswith('replay-'):
            replay_bridge.run_cli(args)
    except (KeyboardInterrupt, EOFError):
        parser.exit(130, '\n回放中止。\n')
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f'停止：{exc}\n')


if __name__ == '__main__':
    main()
