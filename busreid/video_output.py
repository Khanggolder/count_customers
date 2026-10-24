from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2


def ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def validate_video(path: Path, expected_frames: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < 128:
        raise RuntimeError(f"Video output rỗng hoặc không tồn tại: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Không mở được video output: {path}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = capture.read()
        if not ok or (frames > 0 and frames < expected_frames):
            raise RuntimeError(f"Video output bị lỗi hoặc thiếu frame: {path}")
    finally:
        capture.release()


def make_browser_video(intermediate: Path, destination: Path, expected_frames: int) -> None:
    executable = ffmpeg_executable()
    if executable is None:
        raise RuntimeError("FFmpeg chưa có. Cài imageio-ffmpeg hoặc thêm ffmpeg vào PATH để xuất H.264.")
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(intermediate),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = result.stderr.strip()[-500:]
        raise RuntimeError(f"Không chuyển mã H.264: {detail}")
    validate_video(destination, expected_frames)
    capture = cv2.VideoCapture(str(destination))
    try:
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).lower()
        if codec not in {"avc1", "h264"}:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"Video output không phải H.264 (codec={codec!r})")
    finally:
        capture.release()
    intermediate.unlink(missing_ok=True)
