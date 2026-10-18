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
