"""Software/UI tests with fake workers; these do not measure YOLO accuracy."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.video_lab import resolve_source, start_job
from busreid.config import AppConfig, CameraConfig, DetectorConfig
from busreid.pipeline import CameraPipeline, _replace_with_retry
from busreid.types import Detection, Direction
from busreid.video_output import make_browser_video, validate_video
from scripts.run_count_session import main as session_main


def fixture_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.full((100, 100, 3), 100, dtype=np.uint8))
    writer.release()


def cam(name: str) -> CameraConfig:
    return CameraConfig(
        name,
        name,
        Direction.IN if name == "front" else Direction.OUT,
        None,
        [(0, 0), (1, 0), (1, 0.45), (0, 0.45)],
        [(0, 0.55), (1, 0.55), (1, 1), (0, 1)],
        min_stable_frames=2,
    )


class Detector:
    def __init__(self, camera: str, *, reverse: bool = False, stop_file: Path | None = None):
        self.camera = camera
        self.reverse = reverse
        self.stop_file = stop_file
        self.index = 0

    def track(self, _frame):
        y = (80 if self.index < 3 else 25) if self.reverse else (25 if self.index < 3 else 80)
        self.index += 1
        if self.stop_file and self.index == 5:
            self.stop_file.touch()
        return [Detection(f"{self.camera}:1", (40, y - 20, 60, y), 0.9)]


@pytest.mark.parametrize("name,direction", [("front", "IN"), ("rear", "OUT")])
def test_single_camera_real_event_preview_and_reverse(tmp_path: Path, name: str, direction: str) -> None:
    video = tmp_path / "input.mp4"
    fixture_video(video)
    config = AppConfig("sqlite://", DetectorConfig(), {name: cam(name)})
    result = CameraPipeline(cam(name), config, Detector(name)).run(str(video), tmp_path / "forward")
    assert result.status == "COMPLETED"
    assert result.frames == 8
    assert [event.direction.value for event in result.events] == [direction]
    preview = cv2.imread(str(tmp_path / "forward" / f"{name}_preview.jpg"))
    assert preview is not None
    # The actual pipeline overlay changes pixels; this is not a timer-derived image.
    assert tuple(preview[0, 0]) != (100, 100, 100)
    assert json.loads((tmp_path / "forward" / f"{name}_progress.json").read_text())["counts"][direction] == 1
    reverse = CameraPipeline(cam(name), config, Detector(name, reverse=True)).run(
        str(video), tmp_path / "reverse"
    )
    assert reverse.events == []


def test_stop_releases_video_and_keeps_partial_result(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "input.mp4"
    fixture_video(video)
    output = tmp_path / "stopped"
    marker = output / "stop.request"
    config = AppConfig("sqlite://", DetectorConfig(), {"front": cam("front")})
    from busreid.video import OpenCVVideoSource

    closed = []
    original_close = OpenCVVideoSource.close

    def spy_close(self):
        closed.append(True)
        original_close(self)

    monkeypatch.setattr(OpenCVVideoSource, "close", spy_close)
    result = CameraPipeline(cam("front"), config, Detector("front", stop_file=marker)).run(
        str(video), output, stop_file=marker
    )
    assert result.status == "STOPPED"
    assert result.frames == 5
    assert len(result.events) == 1
    assert closed
    validate_video(output / "front_annotated.mp4", 5)
    assert (output / "front_events.csv").is_file()
    assert json.loads((output / "front_progress.json").read_text())["status"] == "STOPPED"


def test_windows_preview_replace_retries_temporary_lock(tmp_path: Path, monkeypatch) -> None:
    temporary = tmp_path / "preview.tmp.jpg"
    destination = tmp_path / "preview.jpg"
    temporary.write_bytes(b"image")
    original = Path.replace
    attempts = []

    def temporarily_locked(self, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("WinError 5")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    _replace_with_retry(temporary, destination)
    assert len(attempts) == 3
    assert destination.read_bytes() == b"image"


def test_source_webcam_and_session_command_isolated(tmp_path: Path, monkeypatch) -> None:
    assert resolve_source(None, None, "", "", tmp_path, webcam_index=0) == "webcam:0"
    with pytest.raises(ValueError):
        resolve_source(None, None, "", "", tmp_path, webcam_index=-1)
    called = []

    def fake_popen(command, **_kwargs):
        called.append(command)
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr("app.video_lab.subprocess.Popen", fake_popen)
    start_job(tmp_path, tmp_path, {"rear": "webcam:0"}, session_only=True)
    command = called[0]
    assert "run_count_session.py" in " ".join(command)
    assert "--rear" in command and "--front" not in command
    assert "--trip-id" not in command and "--telemetry-json" not in command
    assert "--stop-file" in command
    with pytest.raises(ValueError, match="không nhận Trip"):
        start_job(tmp_path, tmp_path, {"front": "webcam:0"}, trip_id="SIM-1", session_only=True)


def test_session_summary_has_no_trip_or_synthetic_data(tmp_path: Path, monkeypatch) -> None:
    result = SimpleNamespace(status="COMPLETED", frames=2, events=[], processing_fps=3.0)

    class FakeRunner:
        def __init__(self, _config):
            pass

        def run(self, sources, output, **_kwargs):
            assert sources == {"rear": 0}
            return SimpleNamespace(results={"rear": result}, errors={})

    monkeypatch.setattr("scripts.run_count_session.DualCameraRunner", FakeRunner)
    monkeypatch.setattr("scripts.run_count_session.load_config", lambda _: object())
    output = tmp_path / "independent"
    assert (
        session_main(["--rear", "webcam:0", "--output", str(output), "--stop-file", str(output / "stop")])
        == 0
    )
    summary = json.loads((output / "session_summary.json").read_text())
    assert summary["association"] == "INDEPENDENT_SESSION_NO_TRIP"
    assert summary["provenance"] == "REAL_VIDEO_AI"
    assert "trip_id" not in summary
    assert not list(tmp_path.glob("*.db"))


def test_one_camera_failure_keeps_other_camera_result(tmp_path: Path, monkeypatch) -> None:
    result = SimpleNamespace(status="COMPLETED", frames=4, events=[], processing_fps=2.0)

    class FakeRunner:
        def __init__(self, _config):
            pass

        def run(self, _sources, _output, **_kwargs):
            return SimpleNamespace(results={"rear": result}, errors={"front": "camera unavailable"})

    monkeypatch.setattr("scripts.run_count_session.DualCameraRunner", FakeRunner)
    monkeypatch.setattr("scripts.run_count_session.load_config", lambda _: object())
    output = tmp_path / "partial"
    assert (
        session_main(
            [
                "--front",
                "bad.mp4",
                "--rear",
                "good.mp4",
                "--output",
                str(output),
                "--stop-file",
                str(output / "stop"),
            ]
        )
        == 0
    )
    summary = json.loads((output / "session_summary.json").read_text())
    assert summary["cameras"]["rear"]["status"] == "COMPLETED"
    assert summary["errors"]["front"] == "camera unavailable"


def test_dashboard_start_stop_reset_and_no_duplicate_worker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError())
    )
    monkeypatch.setattr("app.counting_tab.new_job_dir", lambda *_a: tmp_path)
    video = tmp_path / "front.mp4"
    fixture_video(video)
    process = SimpleNamespace(poll=lambda: None)
    started = []

    def fake_start(*_args, **kwargs):
        started.append(kwargs)
        return process

    monkeypatch.setattr("app.counting_tab.start_job", fake_start)
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    assert not app.exception
    assert any("Chưa kết nối" in item.value for item in app.info)
    next(item for item in app.selectbox if item.key == "counting_mode_front").set_value("Đường dẫn MP4")
    app.run(timeout=20)
    next(item for item in app.text_input if item.key == "counting_path_front").set_value(str(video))
    next(item for item in app.button if item.label == "Bắt đầu đếm").click()
    app.run(timeout=20)
    assert not app.exception
    assert len(started) == 1 and started[0]["session_only"] is True
    assert app.session_state["ai_counting_job"]["sources"] == {"front": str(video.resolve())}
    app.run(timeout=20)
    assert len(started) == 1
    next(item for item in app.button if item.label == "Dừng đếm").click()
    app.run(timeout=20)
    assert (tmp_path / "stop.request").is_file()
    process.poll = lambda: 0
    process.returncode = 0
    app.run(timeout=20)
    next(item for item in app.button if item.label == "Đặt lại bộ đếm").click()
    app.run(timeout=20)
    assert "ai_counting_job" not in app.session_state
    assert (tmp_path / "stop.request").is_file()  # reset does not erase stored run


def test_invalid_mp4_does_not_crash_dashboard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError())
    )
    monkeypatch.setattr("app.counting_tab.new_job_dir", lambda *_a: tmp_path)
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.selectbox if item.key == "counting_mode_front").set_value("Đường dẫn MP4")
    app.run(timeout=20)
    next(item for item in app.text_input if item.key == "counting_path_front").set_value(
        str(tmp_path / "missing.mp4")
    )
    next(item for item in app.button if item.label == "Bắt đầu đếm").click()
    app.run(timeout=20)
    assert not app.exception
    assert any("Không tìm thấy MP4" in item.value for item in app.error)


def test_missing_sources_does_not_start_worker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError())
    )
    monkeypatch.setattr("app.counting_tab.new_job_dir", lambda *_a: tmp_path)
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.button if item.label == "Bắt đầu đếm").click()
    app.run(timeout=20)
    assert not app.exception
    assert any("Chọn ít nhất một video" in item.value for item in app.error)


def test_completed_counting_tab_shows_ai_result_not_unconnected_zero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError())
    )
    intermediate = tmp_path / "front_encoding.mp4"
    fixture_video(intermediate)
    make_browser_video(intermediate, tmp_path / "front_annotated.mp4", 8)
    (tmp_path / "front_progress.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "frames": 8,
                "events": 1,
                "counts": {"IN": 1, "OUT": 0},
                "processing_fps": 4.0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "front_events.csv").write_text(
        "camera,track_id,direction,frame_index,timestamp_s,confidence,crop_path,capture_time_utc,created_at\n"
        f"front,front:1,IN,4,0.4,0.9,,,{datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    app = AppTest.from_file("app/dashboard.py")
    app.session_state["ai_counting_job"] = {
        "process": SimpleNamespace(poll=lambda: 0, returncode=0),
        "output": tmp_path,
        "sources": {"front": "fixture.mp4"},
    }
    app.run(timeout=20)
    assert not app.exception
    assert len(app.get("video")) >= 1
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Tổng khách lên"] == "1"
    assert metrics["Tổng khách xuống"] == "Chưa kết nối"
    assert metrics["Tổng lượt ghi nhận"] == "1"
    assert any("IN +1" in item.value for item in app.success)


def test_failed_camera_without_frames_is_not_reported_as_zero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_a, **_k: (_ for _ in ()).throw(requests.ConnectionError())
    )
    (tmp_path / "session_summary.json").write_text(
        json.dumps({"status": "FAILED", "errors": {"front": "camera unavailable"}}), encoding="utf-8"
    )
    app = AppTest.from_file("app/dashboard.py")
    app.session_state["ai_counting_job"] = {
        "process": SimpleNamespace(poll=lambda: 1, returncode=1),
        "output": tmp_path,
        "sources": {"front": "bad.mp4"},
    }
    app.run(timeout=20)
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Tổng khách lên"] == "Chưa có frame"
    assert metrics["Tổng khách xuống"] == "Chưa kết nối"
    assert metrics["Tổng lượt ghi nhận"] == "Chưa có frame"
