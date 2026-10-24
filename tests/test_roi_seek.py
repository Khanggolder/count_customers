"""ROI Test MP4 seek tests use FakeDetector: software behavior, not AI accuracy."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.video_lab import start_job
from busreid.config import AppConfig, CameraConfig, DetectorConfig
from busreid.pipeline import CameraPipeline
from busreid.types import Detection, Direction
from busreid.video import OpenCVVideoSource


def make_fixture(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((100, 100, 3), index * 20, dtype=np.uint8))
    writer.release()


class FrameDetector:
    def __init__(self, camera: str, reverse: bool = False) -> None:
        self.camera = camera
        self.reverse = reverse

    def track(self, frame: np.ndarray) -> list[Detection]:
        index = round(float(frame.mean()) / 20)
        high = index < 6
        y = 80 if high == self.reverse else 25
        return [Detection(f"{self.camera}:track", (40, y - 10, 60, y), 0.9)]


@pytest.mark.parametrize("camera,direction", [("front", Direction.IN), ("rear", Direction.OUT)])
def test_roi_segment_uses_absolute_frame_and_timestamp(
    tmp_path: Path, camera: str, direction: Direction
) -> None:
    video = tmp_path / "fixture.mp4"
    make_fixture(video)
    config_camera = CameraConfig(
        camera, camera, direction, str(video),
        [(0, 0), (1, 0), (1, 0.45), (0, 0.45)],
        [(0, 0.55), (1, 0.55), (1, 1), (0, 1)],
        2, 4,
    )
    config = AppConfig("sqlite://", DetectorConfig(), {camera: config_camera})
    output = tmp_path / camera
    result = CameraPipeline(config_camera, config, detector=FrameDetector(camera)).run(
        str(video), output, max_frames=4, start_frame=4
    )
    assert result.frames == 4
    assert len(result.events) == 1
    assert result.events[0].direction == direction
    assert result.events[0].frame_index >= 6
    assert result.events[0].timestamp_s >= 0.6
    with (output / f"{camera}_events.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert int(rows[0]["frame_index"]) >= 6
    assert float(rows[0]["timestamp_s"]) >= 0.6
    summary = json.loads((output / f"{camera}_summary.json").read_text(encoding="utf-8"))
    assert summary["start_frame"] == 4
    assert summary["next_frame_index"] == 8
    capture = cv2.VideoCapture(str(output / f"{camera}_annotated.mp4"))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    finally:
        capture.release()


def test_no_fake_crossing_without_two_zone_transition(tmp_path: Path) -> None:
    video = tmp_path / "fixture.mp4"
    make_fixture(video)
    camera = CameraConfig(
        "front", "front", Direction.IN, str(video),
        [(0, 0), (1, 0), (1, 0.45), (0, 0.45)],
        [(0, 0.55), (1, 0.55), (1, 1), (0, 1)],
        2, 4,
    )
    config = AppConfig("sqlite://", DetectorConfig(), {"front": camera})
    result = CameraPipeline(camera, config, detector=FrameDetector("front", reverse=True)).run(
        str(video), tmp_path / "reverse", max_frames=4, start_frame=4
    )
    assert result.frames == 4
    assert result.events == []


def test_seek_rejects_out_of_range_and_non_mp4(tmp_path: Path) -> None:
    video = tmp_path / "fixture.mp4"
    make_fixture(video)
    source = OpenCVVideoSource(str(video))
    try:
        source.open()
        with pytest.raises(ValueError, match="vượt quá"):
            source.seek(10)
        with pytest.raises(ValueError, match="vượt quá"):
            source.seek(-1)
        source.seek(7)
        ok, frame, timestamp = source.read()
        assert ok and frame is not None and frame.mean() > 120
        assert timestamp >= 0.7
    finally:
        source.close()


def test_seek_falls_back_to_sequential_decode_when_backend_rejects_it(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.index = 0
            self.released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            return prop != cv2.CAP_PROP_POS_FRAMES

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 8,
                cv2.CAP_PROP_FRAME_HEIGHT: 8,
                cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 5,
                cv2.CAP_PROP_POS_MSEC: 0,
            }.get(prop, self.index)

        def read(self):
            value = self.index
            self.index += 1
            return True, np.full((8, 8, 3), value, dtype=np.uint8)

        def release(self):
            self.released = True

    instances = []

    def capture(*_args):
        instance = FakeCapture()
        instances.append(instance)
        return instance

    monkeypatch.setattr(cv2, "VideoCapture", capture)
    source = OpenCVVideoSource(__file__)
    source.open()
    source.seek(3)
    ok, frame, timestamp = source.read()
    assert ok and int(frame.mean()) == 3 and timestamp == 0.3
    source.close()
    assert len(instances) == 2 and all(item.released for item in instances)


def test_dashboard_job_passes_start_frame_only_to_roi_session(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def popen(command, **_kwargs):
        captured["command"] = command
        return object()

    monkeypatch.setattr("app.video_lab.subprocess.Popen", popen)
    root = Path(__file__).parents[1]
    output = tmp_path / "job"
    output.mkdir()
    start_job(root, output, {"front": "fixture.mp4"}, max_frames=4, start_frame=450, session_only=True)
    assert captured["command"][-4:] == ["--max-frames", "4", "--start-frame", "450"]
    snapshot = output / "cameras_snapshot.yaml"
    assert captured["command"][captured["command"].index("--config") + 1] == str(snapshot)
    assert snapshot.read_bytes() == (root / "configs" / "cameras.yaml").read_bytes()
