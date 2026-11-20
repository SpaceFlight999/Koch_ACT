"""Guided capture of real block placements followed by training and held-out evaluation."""

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import random
import time
from uuid import uuid4

import cv2
import numpy as np

from camera_vision import COLOR_NAMES, fit_sample, read_image, write_image
from grid_reader.cameras import resolve_named_camera
from .planner import SLOTS
from .reverse import ROOT, read_json, save_json
from .source_color_training import evaluate, fit_palette
from .source_capture_view import SourceCaptureView
from .vision import SourceReader, validate_quad, warp_quad


DEFAULT_SESSION = ROOT / 'outputs/koch_autonomy/source_color_collection'
DEFAULT_CONFIG = (ROOT / 'outputs/koch_autonomy/field_source_color_training_20261118_194128'
                  / 'source_trained_live.json')
LAYERS = tuple(f'{slot}_L{layer}' for slot in SLOTS for layer in (1, 2))
LAYOUT_ORDER = ('S4', 'S2', 'S3', 'S1')  # Front-left, front-right, rear-left, rear-right.
NARROW_FACES = ('S1_L1', 'S3_L1')
INVENTORY = ('red', 'red', 'pink', 'pink', 'green', 'green', 'blue', 'yellow')
OBSERVED_LAYOUTS = {
    ('pink', 'red', 'green', 'blue', 'green', 'yellow', 'red', 'pink'),
    ('green', 'blue', 'pink', 'red', 'yellow', 'green', 'pink', 'red'),
    ('red', 'blue', 'green', 'pink', 'green', 'red', 'pink', 'yellow'),
}
SWATCHES = {'yellow': (70, 218, 244), 'red': (85, 90, 236), 'blue': (247, 156, 66),
            'green': (110, 205, 97), 'pink': (196, 158, 243)}
WINDOW = 'S color collection'


def build_plan(seed, exclude=()):
    rng = random.Random(seed)
    seen = OBSERVED_LAYOUTS | {tuple(item) for item in exclude}
    plan = []
    for group in range(3):
        for _ in range(2000):
            pattern = rng.sample(INVENTORY, len(INVENTORY))
            rotations = [tuple(pattern[i:] + pattern[:i]) for i in range(len(pattern))]
            if len(set(rotations)) == len(pattern) and not seen.intersection(rotations):
                break
        else:
            raise ValueError('无法排出新的均衡摆法，请更换采集序号或减少已排除的摆法。')
        seen.update(rotations)
        rng.shuffle(rotations)
        for i, arrangement in enumerate(rotations):
            plan.append({'id': len(plan) + 1, 'split': 'train' if group < 2 else 'holdout',
                         'lighting': 'alternate' if group == 1 or (group == 2 and i % 2) else 'normal',
                         'labels': dict(zip(LAYERS, arrangement))})
    return plan


def describe(entry, order=LAYOUT_ORDER):
    kind = '训练' if entry['split'] == 'train' else '留出验收'
    lights = '另一种常用室内光线（方便时）' if entry['lighting'] == 'alternate' else '平时使用的光线'
    print(f"\n第 {entry['id']:02d}/24 套 · {kind} · {lights}", flush=True)
    for position, slot in zip(('前左', '前右', '后左', '后右'), order):
        lower, upper = (COLOR_NAMES[entry['labels'][f'{slot}_L{layer}']] for layer in (1, 2))
        print(f'  {position} {slot}   下层 {lower}   上层 {upper}', flush=True)


def open_session(directory, config_path, seed):
    plan_path = directory / 'plan.json'
    if plan_path.is_file():
        config = read_json(directory / 'capture_config.json')
        base = read_json(directory / 'base_colors.json')
        spec = read_json(plan_path)
    else:
        config = read_json(config_path)
        base = read_json(config['color_config_path'])
        spec = {'version': 1, 'seed': seed, 'inventory': dict(Counter(INVENTORY)),
                'source_slot_order': config.get('source_slot_order'),
                'base_config': str(config_path),
                'entries': build_plan(seed, base.get('training_arrangements', []))}
    if config.get('source_slot_order') != list(LAYOUT_ORDER):
        raise ValueError('本采集程序使用前左 S4、前右 S2、后左 S3、后右 S1，请检查配置。')
    if set(base['colors']) != set(INVENTORY):
        raise ValueError('标定须覆盖红、粉、绿、蓝、黄五种颜色。')
    reader = SourceReader(config, base)
    for quad in config['faces'].values():
        w, h = config['frame_size']
        validate_quad(quad, (h, w, 3))
    if not plan_path.is_file():
        save_json(directory / 'capture_config.json', config)
        save_json(directory / 'base_colors.json', base)
        save_json(plan_path, spec)
    return config, base, spec, reader


def capture_path(directory, entry):
    return directory / entry['split'] / f"layout_{entry['id']:02d}.json"


def load_captures(directory, entries, retake=()):
    records = []
    for entry in entries:
        if entry['id'] in retake:
            continue
        path = capture_path(directory, entry)
        if path.is_file():
            record = read_json(path)
            if any(record[key] != value for key, value in entry.items()):
                raise ValueError(f'第 {entry["id"]} 套记录与摆法表不一致，可用 --retake {entry["id"]} 补拍。')
            image = (directory / record['image']).resolve()
            if (image.parent != path.parent.resolve() or image.suffix != '.png' or
                    not image.stem.startswith(path.stem)):
                raise ValueError(f'第 {entry["id"]} 套照片路径与摆法不一致，可用 --retake {entry["id"]} 补拍。')
            try:
                read_image(image)
            except (ValueError, OSError):
                print(f'第 {entry["id"]} 套的照片缺失或损坏，加入待补拍列表。', flush=True)
                continue
            records.append(record)
    return records


def source_crop(config):
    quads = np.asarray(list(config['faces'].values()))
    start = np.maximum(0, np.floor(quads.min(axis=(0, 1))).astype(int) - 38)
    end = np.minimum(config['frame_size'], np.ceil(quads.max(axis=(0, 1))).astype(int) + 38)
    return start, (slice(start[1], end[1]), slice(start[0], end[0]))


def preview(frame, config, entry):
    """Small camera preview with a separate panel for the intended labels."""
    start, area = source_crop(config)
    crop = frame[area]
    factor = min(960 / crop.shape[1], 540 / crop.shape[0])
    fitted = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    view = np.full((540, 960, 3), (36, 39, 42), dtype=np.uint8)
    x, y = (960 - fitted.shape[1]) // 2, (540 - fitted.shape[0]) // 2
    view[y:y+fitted.shape[0], x:x+fitted.shape[1]] = fitted
    for quad in config['faces'].values():
        points = np.rint((np.asarray(quad) - start) * factor + [x, y]).astype(np.int32)
        cv2.polylines(view, [points], True, (250, 250, 250), 2, cv2.LINE_AA)
    panel = np.full((540, 338, 3), (23, 26, 29), dtype=np.uint8)

    def label(text, x, y, size=.54, color=(231, 233, 235)):
        cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    label(f"Placement {entry['id']:02d} / 24", 20, 38, .8)
    label('Holdout' if entry['split'] == 'holdout' else 'Training', 20, 66)
    for i, (position, slot) in enumerate(zip(('FRONT LEFT', 'FRONT RIGHT', 'REAR LEFT', 'REAR RIGHT'), LAYOUT_ORDER)):
        y = 120 + 76 * i
        label(f'{slot}   {position}', 20, y, .58)
        for layer, x in ((1, 20), (2, 172)):
            color = entry['labels'][f'{slot}_L{layer}']
            cv2.rectangle(panel, (x, y+15), (x+15, y+30), SWATCHES[color], cv2.FILLED)
            level = 'Low' if layer == 1 else 'Top'
            label(f'{level} {color}', x+23, y+29, .49)
    label('Space: capture and save', 20, 460)
    label('Esc: save progress and exit', 20, 507, .49)
    return cv2.hconcat([view, panel])


class CaptureWindow:
    """Display frames in Tk so headless OpenCV builds can still acquire and render them."""

    def __init__(self):
        import tkinter as tk
        from PIL import Image, ImageTk

        self.image, self.image_tk = Image, ImageTk
        self.root = tk.Tk()
        self.root.title(WINDOW)
        self.label = tk.Label(self.root, bg='#24272a')
        self.label.pack()
        self.status = tk.StringVar()
        tk.Label(self.root, textvariable=self.status, anchor='w', padx=12, pady=5).pack(fill='x')
        controls = tk.Frame(self.root, padx=10, pady=8)
        controls.pack(fill='x')
        self.buttons = {}
        for key, text in ((32, '拍照并保存，进入下一套'), (27, '暂停')):
            button = tk.Button(controls, text=text, padx=10, pady=3,
                               command=lambda code=key: setattr(self, 'key', code))
            button.pack(side='left', padx=4)
            self.buttons[key] = button
        self.maximum = (min(1298, self.root.winfo_screenwidth() - 56),
                        min(540, max(240, self.root.winfo_screenheight() - 175)))
        self.key = -1
        self.closed = False
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind('<KeyRelease>', self.press)
        self.root.focus_set()

    def press(self, event):
        key = {'space': 32, 'Escape': 27}.get(event.keysym)
        if key is not None:
            self.key = key

    def show(self, frame, actions, status):
        if self.closed:
            return 27
        rgb = self.image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rgb.thumbnail(self.maximum, self.image.Resampling.BILINEAR)
        self.photo = self.image_tk.PhotoImage(rgb, master=self.root)
        self.label.configure(image=self.photo)
        self.status.set(status)
        for key, button in self.buttons.items():
            button.configure(state='normal' if key in actions else 'disabled')
        self.root.update()
        key, self.key = self.key, -1
        return 27 if self.closed else key

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()


def inspect_capture(frame, config, reader, entry, aligner):
    evidence = {'status': 'fixed_reference'}
    faces = config['faces']
    if aligner is not None:
        try:
            faces, evidence = aligner.align(frame)
            evidence['status'] = 'aligned'
        except (ValueError, cv2.error) as exc:
            return None, {'status': 'needs_alignment', 'reason': str(exc)}
    if entry['split'] == 'train':
        unusable = []
        for face, quad in faces.items():
            patch, _ = warp_quad(frame, quad, 64)
            try:
                fit_sample(patch, min_saturation=reader.colors.get('sample_min_saturation', 45))
            except (ValueError, cv2.error):
                unusable.append(face)
        evidence['unusable_faces'] = unusable
    return faces, evidence


def capture_one(camera, config, reader, entry, show, aligner=None, previous_notice=''):
    describe(entry)
    print('按摆法表核对实物后，点击「拍照并保存，进入下一套」或按空格。', flush=True)
    previous = None
    stable_frames = 0
    last_frame = None
    _, area = source_crop(config)
    while True:
        ok, frame = camera.read()
        if not ok:
            if last_frame is None:
                width, height = config['frame_size']
                last_frame = np.zeros((height, width, 3), dtype=np.uint8)
            previous, stable_frames = None, 0
            if show(preview(last_frame, config, entry), {27}, '相机暂时没有新画面，正在重试。') == 27:
                return None
            time.sleep(.05)
            continue
        if list(frame.shape[1::-1]) != config['frame_size']:
            raise ValueError('相机返回分辨率与当前标定不符。')
        last_frame = frame
        view = cv2.resize(frame[area], (96, 96), interpolation=cv2.INTER_AREA)
        still = previous is not None and max(cv2.mean(cv2.absdiff(view, previous))[:3]) < 3.
        stable_frames = stable_frames + 1 if still else 1
        previous = view
        ready = stable_frames >= 3
        status = '摆好后点「拍照并保存，进入下一套」。' if ready else '等待画面稳定。'
        key = show(preview(frame, config, entry), {27, 32} if ready else {27}, previous_notice + status)
        if key == 27:
            return None
        if key == 32 and ready:
            captured = frame.copy()
            captured_at = datetime.now().astimezone().isoformat()
            if show(preview(captured, config, entry), set(), f'正在保存第 {entry["id"]} 套…') == 27:
                return None
            faces, evidence = inspect_capture(captured, config, reader, entry, aligner)
            return captured, captured_at, faces, evidence


def save_capture(directory, entry, captured, camera):
    frame, captured_at, faces, alignment = captured
    path = capture_path(directory, entry)
    image = path.with_name(f'{path.stem}_{uuid4().hex[:12]}.png')
    write_image(image, frame)
    save_json(path, {**entry, 'image': image.relative_to(directory).as_posix(),
                     'captured_at': captured_at, 'camera': camera, 'faces': faces,
                     'view_alignment': alignment})
    return image


def collect(directory, config, reader, remaining, camera_index=None):
    index, name = (resolve_named_camera(config['camera_device_name']) if camera_index is None
                   else (camera_index, f'DirectShow camera {camera_index}'))
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    window = None
    try:
        if not camera.isOpened():
            raise RuntimeError(f'无法打开 {name}。请先关闭占用相机的应用。')
        width, height = config['frame_size']
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(12):
            if not camera.grab():
                raise RuntimeError(f'{name} 未返回画面。')
        aligner = None
        view_path = directory / 'capture_views.json'
        if config.get('reference_image'):
            other_views = read_json(view_path) if view_path.is_file() else []
            aligner = SourceCaptureView(config, other_views)
            ok, frame = camera.read()
            if ok:
                try:
                    faces, _ = aligner.align(frame)
                    config = {**config, 'faces': faces}
                    reader = SourceReader(config, reader.colors)
                except ValueError as exc:
                    print(f'启动对齐：{exc} 拍照时会再尝试。', flush=True)
        window = CaptureWindow()
        notice = ''
        for entry in remaining:
            window.root.title(f'{WINDOW} — 第 {entry["id"]:02d}/24 套')
            captured = capture_one(camera, config, reader, entry, window.show, aligner, notice)
            if captured is None:
                print('进度已保存，再打开本程序会从下一套继续。', flush=True)
                return False
            while True:
                try:
                    image = save_capture(directory, entry, captured, name)
                    break
                except (OSError, ValueError, cv2.error) as exc:
                    message = f'照片保存失败，第 {entry["id"]} 套未跳过。请检查磁盘后点击重试。'
                    print(f'{message} {exc}', flush=True)
                    confirm = window.show(preview(captured[0], config, entry), {27, 32}, message)
                    while confirm not in (27, 32):
                        time.sleep(.04)
                        confirm = window.show(preview(captured[0], config, entry), {27, 32}, message)
                    if confirm == 27:
                        return False
            if captured[2] is not None:
                config = {**config, 'faces': captured[2]}
            needs_review = captured[2] is None or bool(captured[3].get('unusable_faces'))
            notice = f'第 {entry["id"]} 套已保存。'
            if needs_review:
                notice += '已标记待复核。'
            print(f'{notice} {image}', flush=True)
        return True
    finally:
        camera.release()
        if window is not None:
            window.close()


def train(directory, config, base, records):
    training = [record for record in records if record['split'] == 'train']
    holdout = [record for record in records if record['split'] == 'holdout']
    if (len(training), len(holdout)) != (16, 8):
        raise ValueError(f'需采齐 16 套训练和 8 套验收，目前为 {len(training)}/{len(holdout)} 套。')
    print('采集完成，开始训练并复查留出照片。', flush=True)
    candidate, coverage, skipped = fit_palette(config, base, training, directory)
    baseline = evaluate(config, base, holdout, directory)
    score = evaluate(config, candidate, holdout, directory)
    missing = [{'face': face, 'color': color} for face in LAYERS for color in base['colors'] if not coverage[face, color]]
    passed = (not missing and score['correct'] >= max(63, baseline['correct']) and score['wrong'] == 0 and
              not any(item['face'] in NARROW_FACES for item in score['differences']))
    metrics = {'training_layouts': len(training), 'holdout_layouts': len(holdout),
               'fitted_regions': sum(coverage.values()), 'skipped_training_regions': skipped,
               'missing_training_combinations': missing, 'baseline': baseline, 'candidate': score,
               'gate': {'minimum_correct': 63, 'maximum_wrong': 0, 'at_least_baseline_correct': True,
                        'require_narrow_faces': list(NARROW_FACES)},
               'passed': passed}
    save_json(directory / 'candidate_colors.json', candidate)
    save_json(directory / 'training_metrics.json', metrics)
    if passed:
        candidate['training_arrangements'] = base.get('training_arrangements', []) + [
            [record['labels'][key] for key in LAYERS] for record in records]
        accepted = directory / 'accepted' / uuid4().hex
        color_path = accepted / 'trained_colors.json'
        accepted_metrics_path = accepted / 'training_metrics.json'
        save_json(color_path, candidate)
        save_json(accepted_metrics_path, metrics)
        latest = max((item for item in records if item.get('faces', config['faces']) is not None),
                     key=lambda record: record.get('captured_at', ''))
        live_config = {**config, 'faces': latest.get('faces', config['faces']),
                       'reference_image': str(directory / latest['image'])}
        save_json(directory / 'source_trained.json', {**live_config, 'color_config_path': str(color_path),
                  'training_metrics_path': str(accepted_metrics_path)})
    print(f"验收 {score['correct']}/{score['total']} 正确，{score['wrong']} 错分，{score['unknown']} 未知。")
    if passed:
        print(f'通过本轮验收。识别配置：{directory / "source_trained.json"}')
    else:
        print(f'未达到本轮门槛。查看差异：{directory / "training_metrics.json"}')
        print('照片确有拍摄或摆放错误时用 --retake 套号补拍。有效样本的错分应补充新训练数据。')
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--session', type=Path, default=DEFAULT_SESSION)
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--camera', type=int, help='按设备编号开启相机，默认按名称查找')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--show-plan', action='store_true', help='只打印 24 套摆法')
    mode.add_argument('--train-only', action='store_true', help='只用已采齐的照片训练')
    mode.add_argument('--retake', type=int, nargs='+', metavar='套号', help='补拍指定套号，例如 --retake 3 5')
    args = parser.parse_args()
    directory = args.session.resolve()
    try:
        if args.camera is not None and args.camera < 0:
            raise ValueError('相机编号不能为负数。')
        if args.retake and (not (directory / 'plan.json').is_file() or any(not 1 <= n <= 24 for n in args.retake)):
            raise ValueError('补拍需要已有会话，且套号在 1 到 24 之间。')
        if args.show_plan:
            if (directory / 'plan.json').is_file():
                entries = read_json(directory / 'plan.json')['entries']
            else:
                base = read_json(read_json(args.config)['color_config_path'])
                entries = build_plan(args.seed, base.get('training_arrangements', []))
            for entry in entries:
                describe(entry)
            return 0
        if args.train_only and not (directory / 'plan.json').is_file():
            raise ValueError(f'采集会话不存在：{directory}')
        config, base, plan, reader = open_session(directory, args.config.resolve(), args.seed)
        entries = plan['entries']
        records = load_captures(directory, entries, args.retake or ())
        done = {record['id'] for record in records}
        remaining = [entry for entry in entries if entry['id'] not in done]
        print(f'采集目录：{directory}，已完成 {len(done)}/24 套。', flush=True)
        if remaining and not args.train_only:
            if not collect(directory, config, reader, remaining, args.camera):
                return 0
            records = load_captures(directory, entries)
        return 0 if train(directory, config, base, records) else 2
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(2, f'停止：{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
