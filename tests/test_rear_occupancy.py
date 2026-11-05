"""Software integration with a fake head detector; not an AI accuracy measurement."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.video_lab import start_job
from busreid.config import AppConfig, CameraConfig, DetectorConfig, RearOccupancyConfig, load_config
from busreid.rear_occupancy import HeadDetection, RearOccupancyCounter, RearOccupancyPipeline
from busreid.types import Direction
from busreid.video_output import validate_video
from busreid.workers import DualCameraRunner
from scripts.run_count_session import main as session_main


def video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((100, 100, 3), 80, dtype=np.uint8))
    writer.release()


class FakeHeads:
    def __init__(self, counts: list[int], stop_file: Path | None = None) -> None:
        self.counts = iter(counts)
        self.stop_file = stop_file

    def detect(self, _frame):
        count = next(self.counts)
        if self.stop_file and count == 3:
            self.stop_file.touch()
        return [HeadDetection((10 + i * 10, 10, 18 + i * 10, 18), 0.9) for i in range(count)]


def test_headcount_is_per_frame_roi_filtered_and_has_no_events(tmp_path: Path) -> None:
    source = tmp_path / "rear.mp4"
    video(source)
    config = RearOccupancyConfig(
        roi=[(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
        ignore=[[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]],
        smoothing_window=3,
    )
    output = tmp_path / "run"
    result = RearOccupancyPipeline(config, FakeHeads([1, 2, 3, 2, 1])).run(str(source), output)
    assert result.status == "COMPLETED"
    assert result.events == ()
    assert result.frames == 5
    assert result.last_count == 0
    assert result.mean_count == 0.8
    assert (result.min_count, result.max_count) == (0, 2)
    with (output / "rear_occupancy.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["visible_head_count"]) for row in rows] == [0, 1, 2, 1, 0]
    assert [int(row["smoothed_count"]) for row in rows] == [0, 0, 1, 1, 1]
    assert [int(row["frame_index"]) for row in rows] == list(range(5))
    assert [float(row["timestamp_s"]) for row in rows] == pytest.approx([0, 0.1, 0.2, 0.3, 0.4])
    assert not (output / "rear_events.csv").exists()
    validate_video(output / "rear_annotated.mp4", 5)
    assert (output / "rear_preview.jpg").is_file()
    summary = json.loads((output / "rear_occupancy_summary.json").read_text())
    assert summary["mode"] == "OCCUPANCY_HEADCOUNT"
    assert summary["events"] == []
    assert json.loads((output / "rear_progress.json").read_text())["visible_head_count"] == 0


def test_stop_closes_source_and_empty_video_fails(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "rear.mp4"
    video(source)
    stop = tmp_path / "stop.request"
    from busreid.video import OpenCVVideoSource

    closed = []
    original = OpenCVVideoSource.close

    def spy(self):
        closed.append(True)
        original(self)

    monkeypatch.setattr(OpenCVVideoSource, "close", spy)
    result = RearOccupancyPipeline(RearOccupancyConfig(), FakeHeads([1, 2, 3, 4, 5], stop)).run(
        str(source), tmp_path / "stopped", stop_file=stop
    )
    assert result.status == "STOPPED" and result.frames == 3 and closed
    with pytest.raises(RuntimeError, match="no frames"):
        RearOccupancyPipeline(RearOccupancyConfig(), FakeHeads([])).run(
            str(source), tmp_path / "empty", stop_file=stop
        )
    progress = json.loads((tmp_path / "empty" / "rear_progress.json").read_text())
    assert progress["status"] == "FAILED" and progress["visible_head_count"] is None


def test_rear_mode_dispatch_and_front_isolation(tmp_path: Path, monkeypatch) -> None:
    calls = []

    class FakeDoorPipeline:
        def __init__(self, camera, _config):
            self.name = camera.name

        def run(self, *_args, **_kwargs):
            calls.append(self.name)
            return SimpleNamespace(status="COMPLETED", events=[], frames=1)

    class FakeHeadPipeline:
        def __init__(self, _config):
            calls.append("head_init")

        def run(self, *_args, **_kwargs):
            raise RuntimeError("missing head model")

    monkeypatch.setattr("busreid.workers.CameraPipeline", FakeDoorPipeline)
    monkeypatch.setattr("busreid.workers.RearOccupancyPipeline", FakeHeadPipeline)
    cameras = {
        name: CameraConfig(name, name, direction, None, [(0, 0), (1, 0), (0, 1)],
                           [(0, 0.5), (1, 0.5), (1, 1)])
        for name, direction in (("front", Direction.IN), ("rear", Direction.OUT))
    }
    runner = DualCameraRunner(AppConfig("sqlite://", DetectorConfig(), cameras))
    result = runner.run({"front": "front.mp4", "rear": "rear.mp4"}, tmp_path, rear_mode="occupancy_headcount")
    assert "front" in result.results and "rear" in result.errors
    assert calls.count("front") == 1 and "rear" not in calls and "head_init" in calls
    calls.clear()
    result = runner.run({"rear": "rear.mp4"}, tmp_path, rear_mode="door_out")
    assert "rear" in result.results and calls == ["rear"]


def test_config_validates_independent_roi_and_does_not_change_door_zones(tmp_path: Path) -> None:
    base = Path(__file__).parents[1] / "configs" / "cameras.yaml"
    config = load_config(base)
    assert config.rear_occupancy.roi == [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert config.cameras["front"].event_direction == Direction.IN
    assert config.cameras["rear"].event_direction == Direction.OUT
    text = base.read_text(encoding="utf-8").replace("smoothing_window: 1", "smoothing_window: 0")
    invalid = tmp_path / "bad.yaml"
    invalid.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="imgsz/window"):
        load_config(invalid)


def test_counter_rejects_outside_and_ignore_polygon() -> None:
    counter = RearOccupancyCounter(
        RearOccupancyConfig(roi=[(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
                            ignore=[[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]])
    )
    heads = [HeadDetection((10, 10, 18, 18), 0.9), HeadDetection((30, 10, 38, 18), 0.9),
             HeadDetection((70, 10, 78, 18), 0.9)]
    accepted, raw, smoothed = counter.count(heads, (100, 100, 3))
    assert (len(accepted), raw, smoothed) == (1, 1, 1)


def test_independent_cli_writes_occupancy_not_passenger_events(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "rear.mp4"
    video(source)
    monkeypatch.setattr("busreid.rear_occupancy.HeadDetector", lambda _config: FakeHeads([1, 2, 3, 2, 1]))
    output = tmp_path / "session"
    assert session_main([
        "--rear", str(source), "--rear-mode", "occupancy_headcount",
        "--output", str(output), "--stop-file", str(output / "stop.request"),
        "--max-frames", "5",
    ]) == 0
    summary = json.loads((output / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["rear_mode"] == "occupancy_headcount"
    assert summary["cameras"]["rear"]["count"] is None
    assert summary["cameras"]["rear"]["visible_head_count"] == 1
    assert not (output / "rear_events.csv").exists()
    assert not list(tmp_path.glob("*.db"))


def test_head_mode_is_only_passed_to_independent_rear_job(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "cameras.yaml").write_bytes(
        (Path(__file__).parents[1] / "configs" / "cameras.yaml").read_bytes()
    )
    commands = []

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr("app.video_lab.subprocess.Popen", fake_popen)
    start_job(tmp_path, tmp_path / "session", {"rear": "rear.mp4"},
              session_only=True, rear_mode="occupancy_headcount")
    assert commands[0][commands[0].index("--rear-mode") + 1] == "occupancy_headcount"
    assert "run_count_session.py" in " ".join(commands[0])
    with pytest.raises(ValueError, match="chỉ hỗ trợ REAR"):
        start_job(tmp_path, tmp_path / "bad", {"front": "front.mp4"},
                  session_only=True, rear_mode="occupancy_headcount")
    with pytest.raises(ValueError, match="chỉ hỗ trợ REAR"):
        start_job(tmp_path, tmp_path / "bad", {"rear": "rear.mp4"},
                  session_only=False, rear_mode="occupancy_headcount")


def test_dashboard_shows_headcount_separately_from_rear_out(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError())
    )
    (tmp_path / "rear_progress.json").write_text(json.dumps({
        "mode": "OCCUPANCY_HEADCOUNT", "status": "COMPLETED", "frames": 5,
        "visible_head_count": 3, "processing_fps": 2.0,
        "model": "11n-head.pt", "confidence": 0.35,
    }), encoding="utf-8")
    (tmp_path / "rear_occupancy.csv").write_text(
        "frame_index,timestamp_s,visible_head_count,smoothed_count\n4,0.4,3,3\n", encoding="utf-8"
    )
    app = AppTest.from_file("app/dashboard.py")
    app.session_state["ai_counting_job"] = {
        "process": SimpleNamespace(poll=lambda: 0, returncode=0),
        "output": tmp_path, "sources": {"rear": "fixture.mp4"},
        "rear_mode": "occupancy_headcount",
    }
    app.run(timeout=20)
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Số khách quan sát được"] == "3"
    assert metrics["Occupancy quan sát"] == "3"
    assert metrics["Tổng khách xuống"] == "N/A – Chế độ thử nghiệm"
    assert metrics["Tổng lượt ghi nhận"] == "N/A – thiếu REAR-OUT"


def test_front_only_ignores_unconnected_rear_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError())
    )
    monkeypatch.setattr("app.counting_tab.new_job_dir", lambda *_a: tmp_path)
    source = tmp_path / "front.mp4"
    video(source)
    launched = []

    def fake_start(*_args, **kwargs):
        launched.append(kwargs)
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr("app.counting_tab.start_job", fake_start)
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.selectbox if item.key == "counting_rear_mode").set_value(
        "Thử nghiệm đếm khách trong xe (Head Detection)"
    )
    next(item for item in app.selectbox if item.key == "counting_mode_front").set_value("Đường dẫn MP4")
    app.run(timeout=20)
    next(item for item in app.text_input if item.key == "counting_path_front").set_value(str(source))
    next(item for item in app.button if item.label == "Bắt đầu đếm").click()
    app.run(timeout=20)
    assert not app.exception
    assert launched[0]["rear_mode"] == "door_out"
    assert app.session_state["ai_counting_job"]["sources"] == {"front": str(source.resolve())}
