from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VideoMetadata:
    source_type: str
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_s: float | None


def redact_source(source: str | int) -> str:
    if not isinstance(source, str) or not source.lower().startswith(("rtsp://", "rtsps://")):
        return str(source)
    parts = urlsplit(source)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    # RTSP query strings commonly contain access tokens; never persist them.
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


class VideoSource(ABC):
    @abstractmethod
    def open(self) -> VideoMetadata: ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None, float]: ...

    @abstractmethod
    def close(self) -> None: ...


class OpenCVVideoSource(VideoSource):
    def __init__(self, source: str | int, reconnect_attempts: int = 3, timeout_ms: int = 5000):
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.timeout_ms = timeout_ms
        self.capture: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None
        self.frame_index = 0
        self.generation = 0
        self._pts_origin_s: float | None = None
        self._pending: tuple[np.ndarray, float] | None = None

    @property
    def is_rtsp(self) -> bool:
        return isinstance(self.source, str) and self.source.lower().startswith(("rtsp://", "rtsps://"))

    def open(self) -> VideoMetadata:
        if isinstance(self.source, str) and not self.is_rtsp and not Path(self.source).is_file():
            raise FileNotFoundError(f"Không tìm thấy video: {self.source}")
        backend = cv2.CAP_FFMPEG if self.is_rtsp else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(self.source, backend)
        self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms)
        self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(f"Không mở được nguồn: {redact_source(self.source)}")
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            self.close()
            raise RuntimeError(f"Nguồn không có kích thước frame hợp lệ: {redact_source(self.source)}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 25.0
        count_raw = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = count_raw if count_raw > 0 and not self.is_rtsp else None
        duration = count / fps if count is not None and fps > 0 else None
        source_type = "rtsp" if self.is_rtsp else ("webcam" if isinstance(self.source, int) else "mp4")
        self.metadata = VideoMetadata(source_type, width, height, fps, count, duration)
        self.frame_index = 0
        self._pts_origin_s = None
        self._pending = None
        return self.metadata

    def _reconnect(self) -> bool:
        if not self.is_rtsp:
            return False
        self.close()
        for attempt in range(self.reconnect_attempts):
            try:
                time.sleep(min(0.5 * (attempt + 1), 2.0))
                self.open()
                self.generation += 1
                logger.info("Đã kết nối lại RTSP %s", redact_source(self.source))
                return True
            except RuntimeError:
                logger.warning("RTSP reconnect lần %d thất bại", attempt + 1)
        return False

    def seek(self, frame_index: int) -> None:
        """Position an MP4 at an absolute frame, falling back to exact sequential decode."""
        if self.capture is None or self.metadata is None:
            raise RuntimeError("Nguồn video chưa open")
        if self.metadata.source_type != "mp4":
            raise ValueError("Chỉ MP4 hỗ trợ chọn frame bắt đầu")
        if frame_index < 0 or (self.metadata.frame_count is not None and frame_index >= self.metadata.frame_count):
            raise ValueError("Frame bắt đầu vượt quá tổng số frame của video")
        positioned = self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        reported = self.capture.get(cv2.CAP_PROP_POS_FRAMES) if positioned else -1
        if positioned and abs(reported - frame_index) < 0.5:
            ok, frame = self.capture.read()
            after = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
            if ok and frame is not None and abs(after - (frame_index + 1)) < 0.5:
                position_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = position_ms / 1000 if position_ms > 0 else frame_index / self.metadata.fps
                self._pending = (frame, timestamp)
                self.frame_index = frame_index
                return
        # Some codecs/backends report an earlier keyframe or reject random access.
        # Reopen and decode/discard to guarantee the requested absolute frame.
        self.close()
        self.open()
        assert self.capture is not None
        for _ in range(frame_index):
            ok, _ = self.capture.read()
            if not ok:
                raise ValueError("Không thể tìm đến frame bắt đầu trong video")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise ValueError("Không thể đọc frame bắt đầu trong video")
        position_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = position_ms / 1000 if position_ms > 0 else frame_index / self.metadata.fps
        self._pending = (frame, timestamp)
        self.frame_index = frame_index

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self.capture is None or self.metadata is None:
            raise RuntimeError("Nguồn video chưa open")
        if self._pending is not None:
            frame, timestamp_s = self._pending
            self._pending = None
            self.frame_index += 1
            return True, frame, timestamp_s
        absolute_index = self.frame_index
        ok, frame = self.capture.read()
        if not ok and self._reconnect():
            assert self.capture is not None
            ok, frame = self.capture.read()
        timestamp_s = absolute_index / self.metadata.fps
        if ok:
            position_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)
            if position_ms > 0:
                position_s = position_ms / 1000.0
                if self.is_rtsp:
                    if self._pts_origin_s is None:
                        self._pts_origin_s = position_s
                    timestamp_s = max(0.0, position_s - self._pts_origin_s)
                else:
                    timestamp_s = position_s
            self.frame_index += 1
        return bool(ok), frame if ok else None, timestamp_s

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "OpenCVVideoSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def probe_video(path: str | Path) -> VideoMetadata:
    source = OpenCVVideoSource(str(path))
    try:
        return source.open()
    finally:
        source.close()
