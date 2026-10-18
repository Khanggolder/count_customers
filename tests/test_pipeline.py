from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from busreid.config import AppConfig, CameraConfig, DetectorConfig
from busreid.pipeline import CameraPipeline
from busreid.types import Detection, Direction


class FakeDetector:
    def __init__(self) -> None:
        self.index = 0

    def track(self, frame: np.ndarray) -> list[Detection]:
        y = 25 if self.index < 3 else 80
        self.index += 1
        return [Detection("front:track00001", (40, y - 20, 60, y), 0.99)]


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    for _ in range(8):
        writer.write(np.full((100, 100, 3), 100, dtype=np.uint8))
    writer.release()


def test_pipeline_writes_outputs_from_deterministic_tracks(tmp_path: Path) -> None:
    source = tmp_path / "synthetic_fixture.mp4"
    make_video(source)
    camera = CameraConfig(
        "front",
        "front",
        Direction.IN,
        str(source),
        [(0, 0), (1, 0), (1, 0.45), (0, 0.45)],
        [(0, 0.55), (1, 0.55), (1, 1), (0, 1)],
        2,
        4,
    )
    config = AppConfig("sqlite://", DetectorConfig(), {"front": camera})
    result = CameraPipeline(camera, config, detector=FakeDetector()).run(str(source), tmp_path / "output")
    assert result.frames == 8
    assert [event.direction for event in result.events] == [Direction.IN]
    assert Path(result.annotated_path).is_file()
    assert "front:track00001" in Path(result.events_path).read_text(encoding="utf-8-sig")
