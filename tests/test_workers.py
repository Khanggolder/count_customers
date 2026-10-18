from __future__ import annotations

import time
from types import SimpleNamespace

from busreid.config import AppConfig, CameraConfig, DetectorConfig
from busreid.types import Direction
from busreid.workers import DualCameraRunner


def camera(name: str) -> CameraConfig:
    return CameraConfig(
        name, name, Direction.IN, None, [(0, 0), (1, 0), (1, 0.4)], [(0, 0.6), (1, 0.6), (1, 1)]
    )


def test_two_workers_run_concurrently(monkeypatch, tmp_path) -> None:
    class FakePipeline:
        def __init__(self, camera, config):
            self.camera = camera

        def run(self, source, output, max_frames=None):
            time.sleep(0.12)
            return SimpleNamespace(camera=self.camera.name, frames=1, events=[])

    monkeypatch.setattr("busreid.workers.CameraPipeline", FakePipeline)
    config = AppConfig("sqlite://", DetectorConfig(), {"front": camera("front"), "rear": camera("rear")})
    started = time.perf_counter()
    result = DualCameraRunner(config).run({"front": "a.mp4", "rear": "b.mp4"}, tmp_path)
    elapsed = time.perf_counter() - started
    assert set(result.results) == {"front", "rear"}
    assert elapsed < 0.22


def test_one_camera_failure_does_not_block_other(monkeypatch, tmp_path) -> None:
    class FakePipeline:
        def __init__(self, camera, config):
            self.camera = camera

        def run(self, source, output, max_frames=None):
            if self.camera.name == "front":
                raise RuntimeError("camera down")
            return SimpleNamespace(camera="rear", frames=1, events=[])

    monkeypatch.setattr("busreid.workers.CameraPipeline", FakePipeline)
    config = AppConfig("sqlite://", DetectorConfig(), {"front": camera("front"), "rear": camera("rear")})
    result = DualCameraRunner(config).run({"front": "a", "rear": "b"}, tmp_path)
    assert "front" in result.errors
    assert "rear" in result.results
