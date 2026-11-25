"""DirectShow设备枚举与明确选择；指定相机缺失时不误连其他设备。"""

import re
import shutil
import subprocess


def parse_video_devices(output):
    return re.findall(r'"([^"\r\n]+)"\s*\(video\)', output)


def video_devices():
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("未找到FFmpeg，无法按名称确认相机；可使用--camera指定编号。")
    try:
        result = subprocess.run([executable, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("相机设备枚举未完成，将自动重试。") from exc
    return parse_video_devices(result.stderr)


def resolve_named_camera(preferred):
    names = video_devices()
    matches = [(index, name) for index, name in enumerate(names) if preferred.casefold() in name.casefold()]
    if not matches:
        hint = "请保持Camo Studio和手机Camo Camera打开。" if "camo" in preferred.casefold() else "请检查USB连接和相机是否被其他程序占用。"
        raise RuntimeError(f"等待{preferred}摄像头：{hint}")
    if len(matches) != 1:
        raise RuntimeError(f"发现多个{preferred}设备，需使用--camera指定编号：{matches}")
    return matches[0]


if __name__ == "__main__":
    import json
    print(json.dumps([{"index": i, "name": name} for i, name in enumerate(video_devices())], ensure_ascii=False, indent=2))
