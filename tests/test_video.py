from pathlib import Path

import cv2
import numpy as np

from busreid.video import OpenCVVideoSource, probe_video, redact_source


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for value in range(5):
        writer.write(np.full((48, 64, 3), value * 20, dtype=np.uint8))
    writer.release()


def test_video_source_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "fixture.mp4"
    make_video(path)
    metadata = probe_video(path)
    assert metadata.width == 64
    assert metadata.frame_count == 5
    source = OpenCVVideoSource(str(path))
    source.open()
    frames = 0
    while source.read()[0]:
        frames += 1
    source.close()
    assert frames == 5


def test_rtsp_credentials_are_redacted() -> None:
    result = redact_source("rtsp://admin:secret@10.0.0.2:554/live?token=topsecret")
    assert "admin" not in result and "secret" not in result and "token" not in result
    assert "10.0.0.2" in result


def test_rtsp_reconnect_advances_generation_and_source_clock(monkeypatch) -> None:
    from datetime import datetime, timezone

    from busreid.timebase import SourceClock

    class FakeCapture:
        def __init__(self, generation):
            self.generation = generation
            self.calls = 0

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH or prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 16
            if prop == cv2.CAP_PROP_FPS:
                return 10
            if prop == cv2.CAP_PROP_POS_MSEC:
                return self.calls * 100
            return 0

        def read(self):
            self.calls += 1
            if self.generation == 0 and self.calls == 2:
                return False, None
            return True, np.zeros((16, 16, 3), dtype=np.uint8)

        def release(self):
            pass

    generations = iter([0, 1])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCapture(next(generations)))
    monkeypatch.setattr("busreid.video.time.sleep", lambda _: None)
    source = OpenCVVideoSource("rtsp://user:password@example.invalid/live")
    source.open()
    clock = SourceClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    first = clock.map_pts(source.read()[2])
    assert source.read()[0]
    assert source.generation == 1
    clock.reconnect()
    second = clock.map_pts(0.1)
    assert second > first
    source.close()
