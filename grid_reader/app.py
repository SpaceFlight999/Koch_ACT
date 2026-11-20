"""无人值守九宫格读取窗口；自动连接、自动定位、自动保存，通过检查才确认。"""

import argparse
import copy
import json
import queue
import time
import tkinter as tk
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, ttk

import cv2
import torch
from PIL import Image, ImageTk

from camera_vision import BASE, read_image, load_config
from camera_vision_app import CameraStream
from handwritten_digits.model import Predictor
from koch_project_paths import resolve_config_reference
from .core import GridReader, OUTPUT, Stability, save_report, annotate
from .cameras import resolve_named_camera, video_devices
from .worker import LatestWorker
from .collection import DATASET_ROOT, MATERIAL_NAMES, collect_frame, feedback_text


def write_state(report, path=OUTPUT / "current_state.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_at": datetime.now().isoformat(timespec="seconds"), **report},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_camera_settings(source, index, name, path=BASE / "configs/grid_runtime.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    settings.update(camera_source=source, camera_index=index, camera_device_name=name)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def configure_changed(widget, **options):
    changed = {key: value for key, value in options.items() if str(widget.cget(key)) != str(value)}
    if changed:
        widget.configure(**changed)


def live_overlay(frame, report, stamp, now, max_age=.8):
    if (not report or not report.get("cells") or report.get("source") != "live_camera" or
            not 0 <= now - stamp <= max_age or report.get("capture_size") != list(frame.shape[1::-1])):
        return frame
    return annotate(frame, report)


class GridApp:
    def __init__(self, root, reader, camera_index=1, image_path=None, preferred_camera=None, camera_size=(1920, 1080), camera_devices=None, material="cube", material_profiles=None, runtime_path=BASE / "configs/grid_runtime.json", task_camera=False):
        self.root, self.reader, self.camera_index = root, reader, camera_index
        self.runtime_path = runtime_path
        self.task_camera = task_camera
        if getattr(reader, "color_config", None):
            reader.color_config = {**reader.color_config, "fast_lookup": reader.color_config.get("fast_lookup", True)}
        self.preferred_camera = preferred_camera
        self.camera_name = None
        self.camera_size = list(camera_size)
        self.generation = 0
        self.worker = LatestWorker(reader)
        self.save_executor = ThreadPoolExecutor(max_workers=1)
        self.save_future = None
        self.pending_save = None
        self.capture_future = None
        self.capture_on_next_frame = False
        self.material = material
        self.material_profiles = material_profiles or {}
        self.result_raw = None
        self.result_stamp = 0
        self.last_display_stamp = 0
        self.last_status_stamp = 0
        self.displayed_has_cells = False
        self.display_report = None
        self.display_stamp = 0
        self.max_result_age = .8
        self.blocks = getattr(reader, "mode", "digits") == "blocks"
        self.combined = getattr(reader, "mode", "digits") == "combined"
        self.trial_saved = False
        self.stream = None
        self.raw = read_image(image_path) if image_path else None
        self.offline = bool(image_path)
        self.paused = False
        self.guard = Stability()
        self.report, self.marked, self.debug = None, None, []
        self.last_received, self.retry_at, self.last_state_write = 0, 0, 0
        self.connect_started = 0
        self.last_saved_signature = None
        self.last_saved_at = 0
        self.frame_captured_at = None
        self.closing = False
        root.title("九宫格 · 无数字方块颜色试验" if self.blocks else "九宫格方块 · 颜色与数字" if self.combined else "九宫格卡纸 · 颜色与数字")
        root.geometry("1360x950")
        root.minsize(1160, 870)
        root.configure(bg="#F3F5F4")
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#F3F5F4", foreground="#17382D")
        style.configure("TButton", padding=(12, 7))
        style.configure("Go.TButton", background="#155F4A", foreground="white")
        style.map("Go.TButton", background=[("active", "#104C3B")])
        style.configure("Treeview", rowheight=27, background="white", fieldbackground="white")
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        title = "Koch / 自动搬运" if task_camera else "九宫格 / 方块颜色试验" if self.blocks else "九宫格 / 颜色与数字"
        ttk.Label(outer, text=title, font=("Microsoft YaHei UI", 23, "bold")).pack(anchor="w")
        collection_bar = ttk.Frame(outer)
        collection_bar.pack(fill="x", pady=(4, 12))
        ttk.Label(collection_bar, text="材料").pack(side="left")
        self.material_choice = tk.StringVar(value=MATERIAL_NAMES[material])
        material_box = ttk.Combobox(collection_bar, textvariable=self.material_choice, values=tuple(MATERIAL_NAMES.values()), state="readonly", width=10)
        material_box.pack(side="left", padx=(6, 12))
        material_box.bind("<<ComboboxSelected>>", self.select_material)
        self.collection_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(collection_bar, text="数据采集模式", variable=self.collection_mode, command=self.update_collection_controls).pack(side="left")
        self.capture_button = ttk.Button(collection_bar, text="截图保存（F8）", command=self.capture_sample, state="disabled")
        self.capture_button.pack(side="left", padx=8)
        ttk.Button(collection_bar, text="复制采集反馈", command=self.copy_collection_feedback).pack(side="left")
        self.collection_notice = tk.StringVar(value="原图、裁剪和颜色特征按材料分开保存；标签待核对")
        ttk.Label(collection_bar, textvariable=self.collection_notice, foreground="#53655F", width=42).pack(side="left", padx=10)
        root.bind("<F8>", self.capture_sample)
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="相机来源").pack(side="left")
        self.camera_source = tk.StringVar(value="Camo" if preferred_camera and "camo" in preferred_camera.casefold() else "普通相机")
        source_box = ttk.Combobox(toolbar, textvariable=self.camera_source, values=("普通相机", "Camo"), state="readonly", width=9)
        source_box.pack(side="left", padx=(6, 10))
        source_box.bind("<<ComboboxSelected>>", self.update_camera_controls)
        self.camera_device_names = {i: name for i, name in enumerate(camera_devices or []) if "camo" not in name.casefold()}
        self.camera_choice = tk.StringVar(value=f"{camera_index} · {self.camera_device_names.get(camera_index, '普通相机')}")
        choices = [f"{i} · {self.camera_device_names.get(i, '普通相机')}" for i in range(max(6, len(camera_devices or [])))
                   if not camera_devices or i >= len(camera_devices) or "camo" not in camera_devices[i].casefold()]
        ttk.Label(toolbar, text="普通相机编号").pack(side="left")
        self.camera_box = ttk.Combobox(toolbar, textvariable=self.camera_choice, values=choices, state="readonly", width=25)
        self.camera_box.pack(side="left", padx=6)
        self.update_camera_controls()
        ttk.Button(toolbar, text="切换 / 连接", style="Go.TButton", command=self.apply_camera_selection).pack(side="left")
        self.pause_button = ttk.Button(toolbar, text="暂停画面", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=7)
        ttk.Button(toolbar, text="保存本次检查", command=self.save_manual).pack(side="left")
        ttk.Button(toolbar, text="打开测试图片（可选）", command=self.open_image).pack(side="right")
        self.status = tk.StringVar(value="正在准备实物颜色数据和相机…" if self.blocks else "正在准备本地模型和相机…")
        status_area = ttk.Frame(outer, height=40)
        self.status_area = status_area
        status_area.pack(fill="x", pady=9)
        status_area.pack_propagate(False)
        status_label = ttk.Label(status_area, textvariable=self.status, wraplength=1280, anchor="nw")
        status_label.pack(fill="both", expand=True)
        status_label.bind("<Configure>", lambda event: status_label.configure(wraplength=max(1, event.width)))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True)
        middle.columnconfigure(0, weight=1)
        middle.columnconfigure(1, weight=0)
        middle.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(middle, bg="#13241E", width=800, height=450, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        panel = ttk.Frame(middle, padding=(16, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="n")
        ttk.Label(panel, text=("任务卡识别结果 · 以现场 T 位编号" if task_camera else "b编号是检测列表；仅确认九格后显示g编号" if self.blocks or self.combined else
                              "格子编号按相机画面从上到下、从左到右"), wraplength=405).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.tiles = []
        tile_columns = 2 if task_camera else 3
        for i in range(4 if task_camera else 9):
            frame = tk.Frame(panel, width=128, height=100 if task_camera else 126, bg="#DFE5E2", highlightthickness=1, highlightbackground="#B7C4BD")
            frame.grid(row=1 + i // tile_columns, column=i % tile_columns, padx=3, pady=3)
            frame.pack_propagate(False)
            label = tk.Label(frame, text=f"T{i + 1}" if task_camera else f"g{i // 3 + 1}{i % 3 + 1}", bg="#DFE5E2", fg="#53655F", font=("Consolas", 10))
            label.pack(anchor="w", padx=7, pady=(5, 0))
            number = tk.Label(frame, text="—", bg="#DFE5E2", fg="#53655F", font=("Microsoft YaHei UI", 18 if self.blocks else 29, "bold"))
            number.pack()
            caption = tk.Label(frame, text="等待定位", bg="#DFE5E2", fg="#53655F", font=("Microsoft YaHei UI", 9))
            caption.pack()
            self.tiles.append((frame, label, number, caption))
        ttk.Label(panel, text=("识别任务卡后在此保留结果；\n执行期间左侧切换至 S 区实时画面。" if task_camera else "允许同色多个方块，不要求写数字。\n缺格和裁切会标记；位置仅为二维轮廓中心。" if self.blocks else
                              "允许同色重复，数字不清晰会显示问号。\n九格颜色与数字连续一致后自动保存。" if self.combined else
                              "待确认不会自动补答案。\n相近底色、重复数字和方向歧义会单独标记。"), wraplength=405, foreground="#53655F").grid(row=4, column=0, columnspan=3, sticky="w", pady=9)
        columns = ("id", "color", "rgb", "number", "score", "center", "state")
        table_frame = ttk.Frame(outer)
        table_frame.pack(fill="x", pady=(12, 5))
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=4 if task_camera else 7)
        for key, title, width in zip(columns, ("格子", "底色", "图像RGB", "数字", "数字分数", "中心 / px", "状态"), (65, 105, 120, 70, 85, 145, 520)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="center")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(side="left", fill="x", expand=True)
        if task_camera:
            collection_bar.pack_forget()
            toolbar.pack_forget()
            root.unbind("<F8>")
        else:
            ttk.Label(outer, text=("本地运行 · 已关闭数字检查 · 首次方块结果自动留档，完整九格连续一致后另存 · 当前状态写入 outputs/grid_reader/current_state.json" if self.blocks else
                                  "本地运行 · 自动保存稳定读取 · 数字模型分数不等于真实正确率 · 当前状态写入 outputs/grid_reader/current_state.json"), foreground="#53655F").pack(anchor="w")
        self.root.after(100, self.tick)
        if not self.offline and not task_camera:
            self.root.after(250, self.connect)

    def update_camera_controls(self, event=None):
        self.camera_box.configure(state="disabled" if self.camera_source.get() == "Camo" else "readonly")

    def update_collection_controls(self):
        self.capture_button.configure(state="normal" if self.collection_mode.get() else "disabled")

    def select_material(self, event=None):
        material = next(key for key, name in MATERIAL_NAMES.items() if name == self.material_choice.get())
        try:
            path = self.material_profiles[material]
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            config = load_config(path)
            reader = GridReader(self.reader.predictor, auto_orientation=self.reader.auto_orientation,
                                mode=self.reader.mode, color_config=config)
            runtime_path = self.runtime_path
            runtime = json.loads(runtime_path.read_text(encoding="utf8"))
            runtime.update(material=material, color_config_path=str(path))
            temporary = runtime_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf8")
            temporary.replace(runtime_path)
        except (KeyError, OSError, ValueError) as exc:
            self.material_choice.set(MATERIAL_NAMES[self.material])
            self.collection_notice.set(f"材料配置未切换：{exc}")
            return
        self.material = material
        self.reader = reader
        self.worker.reader = reader
        self.trial_saved = False
        self.raw = None
        self.no_frame(f"已切换为{MATERIAL_NAMES[material]}，等待新帧。")
        self.collection_notice.set("纸片颜色待采集实拍标定" if config.get("calibration_required") else f"当前使用{MATERIAL_NAMES[material]}颜色样本")

    def capture_sample(self, event=None):
        if not self.collection_mode.get():
            return
        if self.capture_future is not None:
            self.collection_notice.set("上一张正在保存，请稍候。")
            return "break"
        if self.raw is None or (not self.paused and not self.offline and time.monotonic()-self.last_received > self.max_result_age):
            self.collection_notice.set("没有新鲜的相机画面，请先连接相机。")
            return "break"
        metadata = {"material": self.material, "camera_name": self.camera_name, "camera_index": self.camera_index,
                    "source": "offline_image" if self.offline else "live_camera", "paused": self.paused,
                    "frame_captured_at": self.frame_captured_at, "capture_size": list(self.raw.shape[1::-1]),
                    "color_config_path": self.material_profiles.get(self.material), "generation": self.generation}
        self.capture_future = self.save_executor.submit(collect_frame, self.reader, self.raw.copy(), metadata)
        self.collection_notice.set("正在保存这张原始画面、裁剪和颜色特征…")
        return "break"

    def poll_capture(self):
        future = getattr(self, "capture_future", None)
        if future is None or not future.done():
            return
        try:
            record = future.result()
            self.collection_notice.set(f"已保存 {MATERIAL_NAMES[record['material']]} · {record['capture_id']} · 待核对")
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            self.collection_notice.set(f"截图未保存成功：{exc}")
        finally:
            self.capture_future = None

    def copy_collection_feedback(self):
        try:
            text = feedback_text()
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.collection_notice.set("采集反馈已复制，可粘贴到聊天中。")
        except (OSError, ValueError) as exc:
            self.collection_notice.set(f"尚无可复制的采集反馈：{exc}")

    def apply_camera_selection(self):
        index = int(self.camera_choice.get().split(" · ", 1)[0])
        camo = self.camera_source.get() == "Camo"
        try:
            save_camera_settings("camo" if camo else "normal", index, self.camera_device_names.get(index), self.runtime_path)
        except (OSError, ValueError) as exc:
            self.status.set(f"相机选择未能保存：{exc}")
            return
        self.camera_index = index
        self.camera_name = None
        self.preferred_camera = "Camo" if camo else self.camera_device_names.get(index)
        self.connect()

    def connect(self):
        self.offline, self.paused = False, False
        self.pause_button.configure(text="暂停画面")
        self.raw = None
        self.guard = Stability()
        self.last_received = 0
        self.last_saved_signature = None
        self.last_state_write = self.last_saved_at = 0
        self.trial_saved = False
        self.no_frame("正在自动连接相机，旧读取已清空。")
        if self.stream and self.stream.thread.is_alive():
            self.stream.stop()
            self.retry_at = time.monotonic() + 3
            self.status.set("正在释放旧连接，将自动重连。")
            return
        self.guard = Stability()
        self.raw = None
        self.last_received = 0
        self.stream = None
        if self.preferred_camera:
            try:
                self.camera_index, self.camera_name = resolve_named_camera(self.preferred_camera)
            except RuntimeError as exc:
                self.no_frame(str(exc))
                self.retry_at = time.monotonic() + 8
                return
        self.retry_at = 0
        self.stream = CameraStream(self.camera_index, self.camera_size)
        self.connect_started = time.monotonic()
        self.status.set(f"正在自动连接{self.camera_name or '相机'}（编号{self.camera_index}）…")

    def toggle_pause(self):
        self.paused = not self.paused
        self.generation += 1
        self.clear_result("画面已暂停。" if self.paused else "等待新帧识别。")
        self.pause_button.configure(text="恢复实时画面" if self.paused else "暂停画面")
        if not self.paused:
            self.guard = Stability()
            if self.report:
                self.report.update(stable=False, matching_frames=0)
        self.status.set("画面已暂停，显示冻结帧。" if self.paused else "已恢复实时检查，等待新帧确认。")

    def reset_display(self):
        self.show_report({"cells": []})

    def show_report(self, report):
        cells = report.get("cells", [])
        for i, (frame, label, number, caption) in enumerate(self.tiles):
            if i < len(cells):
                cell = cells[i]
                rgb = cell["color"]["rgb"]
                color = cell["color"]["hex"]
                text_color = "#14231B" if np_luminance(rgb) > 145 else "white"
                label_text = cell["id"]
                number_text = cell["color"]["name"] if self.blocks else cell["number"] if cell["number"] is not None else "?"
                caption_text = ("颜色区域" if self.blocks else cell["color"]["name"]) + (" ✓" if cell["confirmed"] else " · 待确认")
            else:
                color, text_color = "#DFE5E2", "#53655F"
                label_text = f"T{i + 1}" if self.task_camera else f"b{i + 1:02d}" if self.blocks or self.combined else f"g{i // 3 + 1}{i % 3 + 1}"
                number_text, caption_text = "—", "等待定位"
            configure_changed(frame, bg=color)
            configure_changed(label, bg=color, fg=text_color, text=label_text)
            configure_changed(number, bg=color, fg=text_color, text=number_text)
            configure_changed(caption, bg=color, fg=text_color, text=caption_text)
        rows = self.table.get_children()
        for i, cell in enumerate(cells):
            values = (cell["id"], cell["color"]["name"], str(cell["color"]["rgb"]),
                "不读取" if self.blocks else cell["number"] if cell["number"] is not None else "?",
                "—" if self.blocks else f"{cell['number_score']:.1%}",
                str(cell["center_px"]), " / ".join(cell["issues"]) or ("本帧通过" if cell["confirmed"] else "唯一性待确认"))
            if i < len(rows):
                if tuple(self.table.item(rows[i], "values")) != values:
                    self.table.item(rows[i], values=values)
            else:
                self.table.insert("", "end", values=values)
        if len(rows) > len(cells):
            self.table.delete(*rows[len(cells):])

    def render(self, frame):
        if not self.offline:
            frame = live_overlay(frame, self.display_report, self.display_stamp, time.monotonic(), self.max_result_age)
        w, h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(w / frame.shape[1], h / frame.shape[0])
        dw, dh = max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))
        preview = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(w / 2, h / 2, image=self.photo)

    def clear_result(self, message):
        self.report = {"grid_found": False, "complete": False, "stable": False, "cells": [], "status": message}
        self.guard.update(self.report)
        self.marked, self.debug = None, []
        self.result_raw, self.result_stamp = None, 0
        self.last_display_stamp = 0
        self.last_status_stamp = 0
        self.displayed_has_cells = False
        self.display_report = None
        self.display_stamp = 0
        self.show_report(self.report)
        self.status.set(message)

    def no_frame(self, message):
        self.generation = getattr(self, "generation", 0) + 1
        self.clear_result(message)
        self.canvas.delete("all")
        self.canvas.create_text(22, 30, anchor="nw", text=message, fill="white", width=680, font=("Microsoft YaHei UI", 14))

    def queue_save(self, raw, report, marked, debug, directory=None, trial=False, signature=None):
        if self.save_future is not None:
            return False
        args = (raw, copy.deepcopy(report), marked, debug)
        if directory is not None:
            args += (directory,)
        self.save_future = self.save_executor.submit(save_report, *args)
        self.pending_save = (trial, signature, self.generation)
        return True

    def poll_save(self, now):
        if self.save_future is None or not self.save_future.done():
            return
        trial, signature, generation = self.pending_save
        try:
            path = self.save_future.result()
            if generation == self.generation:
                if trial:
                    self.trial_saved = True
                if signature is not None:
                    self.last_saved_signature, self.last_saved_at = signature, now
                self.status.set(f"已保存识别对应帧：{path.name}")
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            self.status.set(f"保存失败：{exc}")
        finally:
            self.save_future, self.pending_save = None, None

    def accept_result(self, raw, stamp, value, elapsed):
        report, marked, debug = value
        now = time.monotonic()
        if not self.offline:
            self.guard.update(report)
        else:
            report.update(stable=False, matching_frames=0, validation_source="offline_image")
        self.report, self.marked, self.debug = report, marked, debug
        self.result_raw, self.result_stamp = raw, stamp
        captured_at = (datetime.now() - timedelta(seconds=max(0, now - stamp))).isoformat(timespec="milliseconds")
        report.update(frame_captured_at=captured_at,
                      source="offline_image" if self.offline else "live_camera",
                      camera_name=self.camera_name, camera_index=self.camera_index,
                      capture_size=list(raw.shape[1::-1]), requested_capture_size=self.camera_size,
                      inference_ms=round(elapsed * 1000, 1), frame_received_monotonic=stamp,
                      result_age_ms=round((now - stamp) * 1000, 1))
        report["material"] = getattr(self, "material", "cube")
        # Keep the camera and inference live; text panels refresh at most 5 Hz.
        empty_transition = not report.get("cells") and getattr(self, "displayed_has_cells", False)
        if self.offline or empty_transition or now - getattr(self, "last_display_stamp", 0) >= .2:
            self.show_report(report)
            self.last_display_stamp = now
            self.displayed_has_cells = bool(report.get("cells"))
            self.display_report = copy.deepcopy(report)
            self.display_stamp = stamp
        if self.offline:
            self.render(marked)
        self.update_status(now)
        if (self.blocks or self.combined) and not self.offline and not self.trial_saved and report.get("cells"):
            self.queue_save(raw, report, marked, debug,
                            OUTPUT / ("combined_trial/runtime" if self.combined else "blocks_trial/runtime"), trial=True)
        if report.get("stable"):
            signature = tuple((c["number"], c["color"]["name"], tuple(round(n / 8) for n in c["color"]["lab"]),
                               int(c["center_px"][0] // 20), int(c["center_px"][1] // 20)) for c in report["cells"])
            if signature != self.last_saved_signature and now - self.last_saved_at > 10:
                self.queue_save(raw, report, marked, debug, signature=signature)

    def update_status(self, now):
        if self.raw is None or now - getattr(self, "last_status_stamp", 0) < .2:
            return
        self.last_status_stamp = now
        h, w = self.raw.shape[:2]
        requested = (f"；请求 {self.camera_size[0]}×{self.camera_size[1]}，驱动仅提供 {w}×{h}；"
                     + ("可在 Camo Studio 调整输出画质" if "camo" in str(self.camera_name).casefold() else "按相机实际画幅识别")
                     if not self.offline and [w, h] != self.camera_size else "")
        state = self.report.get("status", "识别中…") if self.report else "识别中…"
        detail = f"；识别 {self.report['inference_ms']:.0f} ms，结果距今 {(now-self.result_stamp)*1000:.0f} ms" if self.result_stamp and self.report else ""
        self.status.set(("离线图片；" if self.offline else f"实时预览 · {self.camera_name or '相机'}（编号{self.camera_index}）；") + f"实际 {w}×{h}{requested}{detail}。{state}")

    def tick(self):
        if self.closing:
            return
        try:
            now = time.monotonic()
            fresh = self.offline and self.report is None
            if self.stream is None and not self.offline and self.retry_at and now >= self.retry_at:
                self.connect()
            if self.stream and not self.offline:
                try:
                    stamp, frame = self.stream.frames.get_nowait()
                    if not self.paused and not self.stream.stop_event.is_set():
                        self.raw, self.last_received, fresh = frame, stamp, True
                        self.frame_captured_at = datetime.now().isoformat(timespec="seconds")
                        if not frame[::32, ::32].any():
                            self.raw, fresh = None, False
                            hint = "请检查Camo的手机连接和视频源。" if "camo" in str(self.camera_name).casefold() else "请检查镜头遮挡、曝光或其他程序占用。"
                            self.no_frame(f"相机返回全黑图像（实际 {frame.shape[1]}×{frame.shape[0]}）。{hint}")
                except queue.Empty:
                    pass
                try:
                    message = self.stream.messages.get_nowait()
                    self.raw = None
                    self.last_received = 0
                    self.no_frame(message + " 将自动重试。")
                    self.retry_at = now + 8
                except queue.Empty:
                    pass
                if self.retry_at and now >= self.retry_at and not self.stream.thread.is_alive():
                    self.retry_at = 0
                    self.connect()
                    return  # connect可能清空stream；下一次tick继续，finally仍安排回调。
                if (not self.last_received and self.connect_started and now - self.connect_started > 8
                        and self.stream.thread.is_alive()):
                    self.no_frame("相机首帧尚未返回，等待驱动；当前没有有效九格结果。")
                if not self.paused and self.last_received and now - self.last_received > 3:
                    self.raw = None
                    self.last_received = 0
                    self.no_frame("相机超过3秒未更新，当前结果已清空。")
                    self.stream.stop()
                    self.retry_at = now + 5
            if fresh and self.raw is not None:
                if self.offline:
                    started = time.monotonic()
                    value = self.reader.read(self.raw)
                    self.accept_result(self.raw, started, value, time.monotonic() - started)
                else:
                    # The preview always shows the latest captured frame, never an old annotated frame.
                    self.render(self.raw)
                    self.worker.submit(self.generation, self.last_received, self.raw)
                    if getattr(self, "capture_on_next_frame", False):
                        self.capture_on_next_frame = False
                        self.capture_sample()
            try:
                generation, stamp, raw, value, error, elapsed = self.worker.results.get_nowait()
                if (generation == self.generation and not self.offline and not self.paused and self.raw is not None
                        and time.monotonic() - stamp <= self.max_result_age):
                    if error:
                        self.clear_result(f"识别失败：{error}")
                    else:
                        self.accept_result(raw, stamp, value, elapsed)
            except queue.Empty:
                pass
            if not self.offline and not self.paused and self.raw is not None:
                if self.result_stamp and time.monotonic() - self.result_stamp > self.max_result_age:
                    self.clear_result("结果已过期，等待最新识别。")
                self.update_status(time.monotonic())
            self.poll_save(now)
            self.poll_capture()
            if self.report and now - self.last_state_write >= 1:
                self.report["paused"] = self.paused
                write_state(self.report)
                self.last_state_write = now
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            self.guard = Stability()
            self.no_frame(f"当前检查未通过：{exc}")
        finally:
            self.root.after(15, self.tick)

    def save_manual(self):
        if self.result_raw is None or self.report is None:
            return
        try:
            if self.queue_save(self.result_raw, self.report, self.marked, self.debug):
                self.status.set("正在后台保存识别对应帧…")
        except OSError as exc:
            self.status.set(f"保存未成功：{exc}")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            try:
                image = read_image(path)
                if self.stream:
                    self.stream.stop()
                self.raw, self.report, self.offline, self.paused = image, None, True, False
                self.generation += 1
                self.result_raw, self.result_stamp = None, 0
                self.guard = Stability()
            except (ValueError, OSError) as exc:
                self.status.set(str(exc))

    def close(self):
        self.closing = True
        self.worker.stop()
        self.save_executor.shutdown(wait=False, cancel_futures=True)
        if self.stream:
            self.stream.stop()
        self.wait_close(time.monotonic())

    def wait_close(self, start):
        if self.stream and self.stream.thread.is_alive() and time.monotonic() - start < 2:
            self.root.after(50, lambda: self.wait_close(start))
        else:
            self.root.destroy()


def np_luminance(rgb):
    return .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]


def main():
    parser = argparse.ArgumentParser(description="自动识别九宫格卡纸的底色与手写数字")
    parser.add_argument("--camera", type=int, help="明确指定普通相机编号，覆盖保存的设备选择")
    parser.add_argument("--camera-source", choices=("normal", "camo"), help="普通相机或Camo；默认使用上次选择")
    parser.add_argument("--camera-name", help="按设备名称匹配，如Camo")
    parser.add_argument("--image")
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path, help="现场视觉运行配置 JSON")
    parser.add_argument("--collect", action="store_true", help="进入数据采集模式，按F8或点击截图保存")
    parser.add_argument("--capture-once", action="store_true", help="保存启动后的第一张有效相机帧")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--mode", choices=("combined", "blocks", "digits"), default="combined", help="combined实物颜色与数字；blocks仅颜色；digits通用卡纸模式")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or (args.camera is not None and args.camera < 0):
        parser.error("相机宽高必须为正数，设备编号不能为负数")
    runtime_path = args.config.resolve() if args.config else BASE / "configs/grid_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8-sig")) if args.config or runtime_path.exists() else {}
    for key in ("model_path", "color_config_path"):
        if runtime.get(key):
            runtime[key] = str(resolve_config_reference(runtime[key], runtime_path))
    runtime["material_profiles"] = {
        material: str(resolve_config_reference(path, runtime_path))
        for material, path in runtime.get("material_profiles", {}).items()
    }
    torch.set_num_threads(2)
    root = tk.Tk()
    root.withdraw()
    try:
        colors = load_config(runtime["color_config_path"]) if runtime.get("color_config_path") else None
        reader = GridReader(Predictor(args.model or runtime.get("model_path")) if args.mode != "blocks" else None,
                            mode=args.mode, color_config=colors)
        legacy_source = "camo" if "camo" in str(runtime.get("camera_name", "")).casefold() else "normal"
        source = args.camera_source or ("normal" if args.camera is not None else runtime.get("camera_source", legacy_source))
        preferred = args.camera_name or ("Camo" if source == "camo" else runtime.get("camera_device_name") if args.camera is None else None)
        try:
            devices = video_devices() if not args.image else []
        except RuntimeError:
            devices = []
        app = GridApp(root, reader, args.camera if args.camera is not None else runtime.get("camera_index", 0), args.image,
                      preferred_camera=preferred, camera_size=(args.width, args.height), camera_devices=devices,
                      material=runtime.get("material", "cube"), material_profiles=runtime.get("material_profiles", {}),
                      runtime_path=runtime_path)
        if args.collect or args.capture_once:
            app.collection_mode.set(True)
            app.update_collection_controls()
        app.capture_on_next_frame = args.capture_once
    except (OSError, ValueError, RuntimeError) as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "startup_error.txt").write_text(str(exc), encoding="utf-8")
        root.destroy()
        raise
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
