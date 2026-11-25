"""中文相机视觉实验台。OpenCV 处理图像，Tk 显示，不依赖 cv2.imshow。"""

import argparse
import copy
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from koch_vision.colors import (BASE, DEFAULT_CONFIG, annotate, detect, fit_sample, load_config,
                           make_plane, read_image, save_config, save_result, write_image)


def configure_global_capture(capture, size):
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    capture.set(cv2.CAP_PROP_FPS, 30)


class CameraStream:
    def __init__(self, index, size):
        self.index, self.size = index, size
        self.actual_size, self.actual_fps = None, None
        self.stop_event = threading.Event()
        self.frames = queue.Queue(maxsize=1)
        self.messages = queue.Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError("相机未打开。检查编号、是否被其他程序占用，以及 Windows 相机隐私权限。")
            configure_global_capture(cap, self.size)
            self.actual_fps = cap.get(cv2.CAP_PROP_FPS)
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("相机读取失败，已停止刷新。请断开后重新连接。")
                self.actual_size = [frame.shape[1], frame.shape[0]]
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait((time.monotonic(), frame))
        except Exception as exc:
            self.messages.put(str(exc))
        finally:
            if cap is not None:
                cap.release()

    def stop(self):
        self.stop_event.set()


class VisionApp:
    def __init__(self, root, config_path=DEFAULT_CONFIG, image_path=None):
        self.root, self.config_path = root, config_path
        self.config = load_config(config_path)
        self.stream = None
        self.frame = read_image(image_path) if image_path else None
        self.frozen = None
        self.mode = None
        self.points = []
        self.start = None
        self.display_transform = (1, 0, 0)
        self.closing = False
        self.last_received = 0
        self.items = []
        root.title("相机视觉实验台 · 五色方块")
        root.geometry("1180x820")
        root.minsize(1000, 730)
        root.configure(bg="#F3F5F4")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.cancel())
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#F3F5F4", foreground="#18392F")
        style.configure("TButton", padding=(10, 7))
        style.configure("Primary.TButton", background="#155F4A", foreground="white")
        style.map("Primary.TButton", background=[("active", "#114C3C")])
        style.configure("Treeview", rowheight=27, background="white", fieldbackground="white")

        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="相机 / 颜色与位置", font=("Microsoft YaHei UI", 21, "bold")).pack(anchor="w")
        ttk.Label(outer, text="实物颜色样本标定  →  二维区域检测  →  独立场景验收").pack(anchor="w", pady=(3, 12))
        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        ttk.Label(bar, text="相机编号").pack(side="left")
        self.camera = tk.IntVar(value=self.config.get("camera_index", 1))
        ttk.Combobox(bar, textvariable=self.camera, values=list(range(6)), width=3, state="readonly").pack(side="left", padx=6)
        self.connect_button = ttk.Button(bar, text="连接相机", style="Primary.TButton", command=self.connect)
        self.connect_button.pack(side="left", padx=4)
        ttk.Button(bar, text="断开", command=self.disconnect).pack(side="left", padx=4)
        ttk.Button(bar, text="打开图片", command=self.open_image).pack(side="left", padx=4)
        ttk.Button(bar, text="保存当前结果", command=self.save).pack(side="right")
        self.status = tk.StringVar(value="请选择连接相机，或打开图片。现有样本会自动加载。")
        ttk.Label(outer, textvariable=self.status, wraplength=1100).pack(anchor="w", pady=8)

        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(middle, bg="#13231F", highlightthickness=0, width=800, height=450)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        panel = ttk.Frame(middle, padding=(18, 0, 0, 0), width=260)
        panel.pack(side="right", fill="y")
        ttk.Label(panel, text="01  颜色样本", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        self.keys = list(self.config["colors"])
        self.color_box = ttk.Combobox(panel, values=[], state="readonly", width=23)
        self.color_box.pack(fill="x", pady=8)
        self.refresh_colors()
        ttk.Button(panel, text="采样此颜色（拖框）", command=lambda: self.begin("sample")).pack(fill="x")
        ttk.Button(panel, text="撤销本色最近一组样本", command=self.undo_sample).pack(fill="x", pady=5)
        ttk.Label(panel, text="只框方块纯色表面内部。\n每色建议补 3–5 个位置/明暗样本。", wraplength=240).pack(anchor="w", pady=(0, 14))
        ttk.Label(panel, text="02  工作区与坐标", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        ttk.Button(panel, text="框选工作区", command=lambda: self.begin("roi")).pack(fill="x", pady=(8, 5))
        ttk.Button(panel, text="四点平面标定（可选）", command=self.begin_plane).pack(fill="x")
        ttk.Button(panel, text="清除平面标定", command=self.clear_plane).pack(fill="x", pady=5)
        ttk.Label(panel, text="像素：左上角为原点，向右/下增加。\n毫米：仅标定平面参考，不是机器人坐标。", wraplength=240).pack(anchor="w", pady=(0, 14))
        ttk.Button(panel, text="取消操作 / 恢复画面（Esc）", command=self.cancel).pack(fill="x")
        ttk.Label(panel, text="轮廓角度仅供观察；倾斜机位下\n不代表方块顶面或夹爪角度。", wraplength=240).pack(anchor="w", pady=10)

        columns = ("id", "color", "x", "y", "angle", "mm", "state")
        self.table = ttk.Treeview(outer, columns=columns, show="headings", height=5)
        titles = ("序号（本帧）", "颜色", "X / px", "Y / px", "轮廓角 / °", "平面参考 / mm", "提示")
        for key, title, width in zip(columns, titles, (100, 110, 95, 95, 120, 190, 190)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="center")
        self.table.pack(fill="x", pady=(12, 4))
        ttk.Label(outer, text="当前只检测颜色区域，不判断卡纸/方块类别、占用、三维高度；数据保存在本机。", foreground="#53655F").pack(anchor="w")
        root.after(80, self.tick)

    def refresh_colors(self):
        index = max(0, self.color_box.current())
        self.color_box["values"] = [f"{self.config['colors'][key]['name']} · {len(self.config['colors'][key]['samples'])} 组" for key in self.keys]
        self.color_box.current(index)

    def connect(self):
        if self.stream and self.stream.thread.is_alive():
            self.status.set("请先断开，并等待相机释放后再连接。")
            return
        self.cancel()
        self.frame = None
        self.last_received = 0
        self.stream = CameraStream(self.camera.get(), self.config["frame_size"])
        self.status.set("正在连接相机，请稍候…")
        self.connect_button.configure(state="disabled")

    def commit_config(self, candidate):
        # 写盘成功后才让配置在内存中生效。
        save_config(candidate, self.config_path)
        self.config = candidate

    def accept_camera(self, index):
        if index != self.config.get("camera_index"):
            candidate = copy.deepcopy(self.config)
            candidate.update(camera_index=index, roi=None, plane=None)
            self.commit_config(candidate)

    def disconnect(self):
        if self.stream:
            self.stream.stop()
        self.cancel()
        self.status.set("已请求断开；当前保留最后一帧，可继续采样或保存。")

    def open_image(self):
        if self.stream and self.stream.thread.is_alive():
            self.status.set("请先断开相机并等待释放，再打开图片。")
            return
        path = filedialog.askopenfilename(filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("全部文件", "*.*")])
        if path:
            try:
                self.frame = read_image(path)
                self.cancel()
                self.status.set(f"离线图片：{path}")
            except ValueError as exc:
                messagebox.showerror("图片读取失败", str(exc))

    def begin(self, mode):
        if self.frame is None:
            self.status.set("还没有画面，请先连接相机或打开图片。")
            return False
        if list(self.frame.shape[1::-1]) != self.config["frame_size"]:
            self.status.set("画幅不符。请先按状态提示重设配置。")
            return False
        self.frozen = self.frame.copy()
        self.mode, self.points, self.start = mode, [], None
        self.status.set("画面已冻结：拖框选择纯色内部，松开鼠标保存样本。" if mode == "sample" else "画面已冻结：拖框选择完整工作区。Esc 取消。")
        return True

    def begin_plane(self):
        if not self.begin("plane"):
            return
        width = simpledialog.askfloat("平面实测尺寸", "输入矩形参考区宽度（毫米）：", minvalue=1, parent=self.root)
        height = simpledialog.askfloat("平面实测尺寸", "输入矩形参考区高度（毫米）：", minvalue=1, parent=self.root) if width else None
        if not width or not height:
            self.cancel()
            return
        self.plane_size = (width, height)
        self.status.set("依次单击同一参考平面的左上、右上、右下、左下角。Esc 取消。")

    def clear_plane(self):
        candidate = copy.deepcopy(self.config)
        candidate["plane"] = None
        try:
            self.commit_config(candidate)
            self.status.set("平面标定已清除，输出像素坐标。")
        except OSError as exc:
            self.status.set(f"配置未保存，原标定保留：{exc}")

    def undo_sample(self):
        key = self.keys[self.color_box.current()]
        samples = self.config["colors"][key]["samples"]
        if samples and messagebox.askyesno("撤销样本", "移除该颜色最近一组样本？原始采样图片仍会保留。"):
            candidate = copy.deepcopy(self.config)
            candidate["colors"][key]["samples"].pop()
            try:
                self.commit_config(candidate)
                self.refresh_colors()
            except OSError as exc:
                self.status.set(f"撤销未保存，样本保留：{exc}")

    def point(self, event):
        scale, ox, oy = self.display_transform
        x, y = (event.x - ox) / scale, (event.y - oy) / scale
        frame = self.frozen if self.frozen is not None else self.frame
        if frame is None or not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
            return None
        return [int(x), int(y)]

    def mouse_down(self, event):
        point = self.point(event)
        if not self.mode or point is None:
            return
        if self.mode == "plane":
            self.points.append(point)
            if len(self.points) == 4:
                try:
                    candidate = copy.deepcopy(self.config)
                    candidate["plane"] = make_plane(self.points, *self.plane_size)
                    self.commit_config(candidate)
                    self.cancel()
                    self.status.set("平面参考标定已保存；请用第五个已知点核验，方块高度仍会产生视差。")
                except (ValueError, OSError) as exc:
                    self.points = []
                    self.status.set(str(exc) + " 请重新点击四点。")
            return
        self.start = point
        self.canvas.delete("selection")

    def mouse_move(self, event):
        point = self.point(event)
        if self.start is None or point is None:
            return
        scale, ox, oy = self.display_transform
        self.canvas.delete("selection")
        self.canvas.create_rectangle(self.start[0] * scale + ox, self.start[1] * scale + oy,
                                     point[0] * scale + ox, point[1] * scale + oy,
                                     outline="#FDE68A", width=2, tags="selection")

    def mouse_up(self, event):
        point = self.point(event)
        if self.start is None or point is None or self.frozen is None:
            return
        x1, x2 = sorted((self.start[0], point[0]))
        y1, y2 = sorted((self.start[1], point[1]))
        try:
            candidate = copy.deepcopy(self.config)
            if min(x2 - x1, y2 - y1) < 6:
                raise ValueError("选区太小，请重新拖框。")
            if self.mode == "sample":
                key = self.keys[self.color_box.current()]
                patch = self.frozen[y1:y2, x1:x2]
                profile = fit_sample(patch)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                source = BASE / "outputs" / "camera_vision" / "samples" / f"{stamp}_{key}.png"
                write_image(source, patch)
                profile.update(source_image=str(source), source_roi=[x1, y1, x2, y2])
                candidate["colors"][key]["samples"].append(profile)
            elif self.mode == "roi":
                candidate["roi"] = [x1, y1, x2, y2]
            self.commit_config(candidate)
            self.refresh_colors()
            self.cancel()
            self.status.set("已保存配置。请换位置、换朝向测试，测试图不用于继续拟合。")
        except (ValueError, OSError) as exc:
            self.start = None
            self.status.set(str(exc))

    def cancel(self):
        self.frozen, self.mode, self.start, self.points = None, None, None, []
        self.canvas.delete("selection")
        self.status.set("已取消操作 / 恢复画面。")

    def save(self):
        frame = self.frozen if self.frozen is not None else self.frame
        if frame is None:
            self.status.set("还没有画面可以保存。")
            return
        try:
            path = save_result(frame, self.config, BASE / "outputs" / "camera_vision" / "results")
            self.status.set(f"原图、标注图、JSON 已保存：{path.name}")
        except (ValueError, OSError) as exc:
            messagebox.showerror("保存失败", str(exc))

    def tick(self):
        if self.closing:
            return
        try:
            self.update_frame()
        except (OSError, ValueError, cv2.error) as exc:
            if self.stream:
                self.stream.stop()
            self.cancel()
            self.frame = None
            self.status.set(f"处理暂停，旧配置保留；请检查后重新连接：{exc}")
        finally:
            self.root.after(100, self.tick)

    def update_frame(self):
        if self.stream:
            try:
                stamp, frame = self.stream.frames.get_nowait()
                self.accept_camera(self.stream.index)
                self.frame = frame
                self.last_received = stamp
                if (stamp and not self.mode and not self.stream.stop_event.is_set()
                        and self.status.get().startswith(("正在连接", "实时检测"))):
                    self.status.set("实时检测中 · 方块移动到工作区内 · 保存结果可留存验收证据")
            except queue.Empty:
                pass
            try:
                self.status.set(self.stream.messages.get_nowait())
                self.frame = None
            except queue.Empty:
                pass
            if not self.stream.thread.is_alive():
                self.connect_button.configure(state="normal")
            elif self.last_received and time.monotonic() - self.last_received > 3:
                self.status.set("超过 3 秒未收到新画面；显示可能已过期，请断开检查。")
        frame = self.frozen if self.frozen is not None else self.frame
        if frame is not None:
            if list(frame.shape[1::-1]) != self.config["frame_size"]:
                if self.stream:
                    self.stream.stop()
                self.status.set("实际画幅与配置不同，检测暂停。")
                actual = list(frame.shape[1::-1])
                if messagebox.askyesno("画幅变化", f"实际画幅 {actual}，是否采用新画幅并清除旧工作区与平面标定？"):
                    candidate = copy.deepcopy(self.config)
                    candidate.update(frame_size=actual, roi=None, plane=None)
                    self.commit_config(candidate)
                else:
                    self.frame = None
                return
            try:
                self.items = detect(frame, self.config)
                marked = annotate(frame, self.items, self.config)
                cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
                scale = min(cw / frame.shape[1], ch / frame.shape[0])
                dw, dh = max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))
                ox, oy = (cw - dw) / 2, (ch - dh) / 2
                self.display_transform = (scale, ox, oy)
                photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)).resize((dw, dh)))
                self.photo = photo
                self.canvas.delete("picture")
                self.canvas.create_image(ox, oy, anchor="nw", image=photo, tags="picture")
                self.canvas.tag_lower("picture")
                self.canvas.delete("point")
                for i, (x, y) in enumerate(self.points, 1):
                    self.canvas.create_text(x * scale + ox, y * scale + oy, text=str(i), fill="#FDE68A", font=("Arial", 18), tags="point")
                self.table.delete(*self.table.get_children())
                for item in self.items:
                    self.table.insert("", "end", values=(item["frame_object_number"], item["color_name"],
                        *item["center_px"], item["silhouette_angle_deg_mod90"] if not item["clipped"] else "—",
                        str(item["plane_reference_mm"]) if item["plane_reference_mm"] is not None else "未标定 / 范围外",
                        "触及边界，请重摆" if item["clipped"] else "二维颜色区域"))
            except (ValueError, cv2.error) as exc:
                self.status.set(f"检测暂停：{exc}")
        else:
            self.canvas.delete("all")
            self.canvas.create_text(30, 35, anchor="nw", text="等待相机画面", fill="white", font=("Microsoft YaHei UI", 18))
            self.table.delete(*self.table.get_children())

    def close(self):
        self.closing = True
        if self.stream:
            self.stream.stop()
        self.wait_close(time.monotonic())

    def wait_close(self, start):
        if self.stream and self.stream.thread.is_alive() and time.monotonic() - start < 2:
            self.root.after(50, lambda: self.wait_close(start))
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="相机颜色与位置标定窗口")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--image", help="离线打开图片，不访问相机")
    parser.add_argument("--connect", action="store_true", help="启动后连接配置中的相机")
    args = parser.parse_args()
    root = tk.Tk()
    app = VisionApp(root, args.config, args.image)
    if args.connect and not args.image:
        root.after(200, app.connect)
    root.mainloop()


if __name__ == "__main__":
    main()
