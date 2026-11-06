"""Software tests with fake tracks; these do not measure model or tracking accuracy."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.counting_tab import _show_preview
from app.video_lab import start_job
from busreid.config import AppConfig, CameraConfig, DetectorConfig, RearHeadOutConfig, load_config
from busreid.rear_head_out import HeadTracker, HeadZoneCounter, RearHeadOutPipeline
from busreid.types import Detection, Direction
from busreid.video import VideoMetadata
from busreid.video_output import validate_video
from busreid.workers import DualCameraRunner
from scripts.run_count_session import main as session_main


def config(**changes) -> RearHeadOutConfig:
    settings = dict(
        roi=[(0, 0), (1, 0), (1, 1), (0, 1)],
        zone_a=[(0.1, 0.2), (0.3, 0.2), (0.3, 0.8), (0.1, 0.8)],
        zone_b=[(0.7, 0.2), (0.9, 0.2), (0.9, 0.8), (0.7, 0.8)],
        min_stable_frames=2,
    )
    settings.update(changes)
    return RearHeadOutConfig(**settings)


def detection(track_id: str, x: float, y: float = 0.5) -> Detection:
    return Detection(track_id, (round(100 * x - 3), round(100 * y - 3),
                                round(100 * x + 3), round(100 * y + 3)), 0.92)


def run_tracks(frames: list[list[tuple[str, float]]], **changes):
    counter = HeadZoneCounter(config(**changes))
    events = []
    for frame_index, tracks in enumerate(frames):
        active_track_ids = {track_id for track_id, _ in tracks}
        for track_id, x in tracks:
            event = counter.update(
                detection(track_id, x), (100, 100, 3), frame_index, frame_index / 10, active_track_ids
            )
            if event:
                events.append(event)
        counter.prune(frame_index)
    return counter, events


def path_for(track_id: str, positions: list[float]) -> list[list[tuple[str, float]]]:
    return [[(track_id, x)] for x in positions]


def test_a_transition_b_counts_once_and_uses_head_center() -> None:
    counter, events = run_tracks(path_for("head1", [.2, .2, .5, .5, .8, .8, .8, .8]))
    assert counter.total_out == 1
    assert len(events) == 1
    assert events[0].direction == "OUT" and events[0].crossing_zone == "A->transition->B"
    assert events[0].frame_index == 5 and events[0].anchor_x == 80
    assert len(counter.states["head1"].trail) == 8


@pytest.mark.parametrize("positions", [
    [.8, .8, .5, .5, .2, .2],  # reverse B -> A
    [.2] * 8,  # standing in A
    [.2, .2, .5, .5, .2, .2],  # turn back to A
    [.2, .2, .8, .8],  # no confirmed transition
    [.2, .2, .5, .5, .8],  # tracking lost before stable B
])
def test_invalid_or_unconfirmed_crossing(positions: list[float]) -> None:
    counter, events = run_tracks(path_for("head1", positions))
    assert counter.total_out == 0 and not events


def test_boundary_oscillation_and_reentry_do_not_duplicate() -> None:
    positions = [.2, .2, .5, .5, .8, .8, .69, .71, .69, .71, .8, .8]
    counter, events = run_tracks(path_for("head1", positions))
    assert counter.total_out == len(events) == 1


def test_two_heads_count_independently() -> None:
    frames = [[("one", x), ("two", x)] for x in [.2, .2, .5, .5, .8, .8]]
    counter, events = run_tracks(frames)
    assert counter.total_out == 2
    assert {event.track_id for event in events} == {"one", "two"}


def test_temporary_loss_and_unambiguous_id_change_recover_state() -> None:
    frames = path_for("old", [.2, .2, .5, .5]) + [[], []] + path_for("new", [.52, .8, .8])
    counter, events = run_tracks(frames)
    assert counter.total_out == 1 and len(events) == 1
    assert events[0].track_id == "new" and "old" not in counter.states


def test_ambiguous_id_change_during_occlusion_is_not_counted() -> None:
    frames = [
        [("one", .2), ("two", .24)], [("one", .2), ("two", .24)],
        [("one", .5), ("two", .52)], [("one", .5), ("two", .52)],
        [], [("new", .51)], [("new", .8)], [("new", .8)],
    ]
    counter, events = run_tracks(frames)
    assert counter.total_out == 0 and not events
    assert counter.states["new"].blocked


def test_new_id_does_not_steal_state_from_visible_track() -> None:
    frames = path_for("old", [.2, .2, .5, .5]) + [
        [("new", .51), ("old", .5)],
        [("new", .8), ("old", .5)],
        [("new", .8), ("old", .5)],
    ]
    counter, events = run_tracks(frames)
    assert counter.total_out == 0 and not events
    assert "old" in counter.states and "new" in counter.states


def test_stale_track_is_pruned_and_new_video_resets() -> None:
    counter, events = run_tracks(path_for("one", [.2, .2]) + [[]] * 4,
                                 state_ttl_frames=2)
    assert not events and not counter.states
    fresh, _ = run_tracks(path_for("one", [.8, .8]))
    assert fresh.total_out == 0


def test_head_tracker_calls_single_class_tracking_with_persist(monkeypatch) -> None:
    class Tensor:
        def __init__(self, data):
            self.data = np.array(data)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

        def __len__(self):
            return len(self.data)

    calls = []

    class FakeModel:
        task = "detect"
        names = {0: "head"}

        def track(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=Tensor([[10, 10, 20, 20]]), id=Tensor([7]), conf=Tensor([.8])
            ))]

    monkeypatch.setattr("busreid.rear_occupancy._weights_path", lambda _model: Path("head.pt"))
    monkeypatch.setitem(sys.modules, "ultralytics", ModuleType("ultralytics"))
    monkeypatch.setattr(sys.modules["ultralytics"], "YOLO", lambda _path: FakeModel(), raising=False)
    tracker = HeadTracker(config())
    found = tracker.track(np.zeros((100, 100, 3), dtype=np.uint8))
    assert found[0].track_id == "rear:head00007"
    assert calls[0]["persist"] and calls[0]["classes"] == [0]
    assert calls[0]["tracker"] == "bytetrack.yaml"


def video(path: Path, frames: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((100, 100, 3), 60, dtype=np.uint8))
    writer.release()


class FakeTracker:
    def __init__(self, positions):
        self.positions = iter(positions)

    def track(self, _frame):
        return [detection("head1", next(self.positions))]


def test_pipeline_outputs_events_video_summary_and_no_business_event(tmp_path: Path) -> None:
    source = tmp_path / "rear.mp4"
    video(source)
    output = tmp_path / "run"
    result = RearHeadOutPipeline(config(), FakeTracker([.2, .2, .5, .5, .8, .8])).run(source, output)
    assert result.status == "COMPLETED" and len(result.events) == 1
    assert result.frames == 6
    validate_video(output / "rear_head_out_annotated.mp4", 6)
    assert (output / "rear_head_out_preview.jpg").is_file()
    with (output / "rear_head_out_events.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["direction"] == "OUT"
    summary = json.loads((output / "rear_head_out_summary.json").read_text(encoding="utf-8"))
    assert summary["count"] == 1 and summary["evaluation"].startswith("N/A")
    assert not (output / "rear_events.csv").exists()
    assert not list(tmp_path.glob("*.db"))


def test_no_detection_frames_write_zero_events(tmp_path: Path) -> None:
    source = tmp_path / "rear.mp4"
    video(source)

    class EmptyTracker:
        def track(self, _frame):
            return []

    result = RearHeadOutPipeline(config(), EmptyTracker()).run(source, tmp_path / "empty")
    assert result.frames == 6 and not result.events


def test_rtsp_reconnect_keeps_head_event_timestamp_monotonic(tmp_path: Path, monkeypatch) -> None:
    class ReconnectingSource:
        def __init__(self, _path):
            self.generation = 0
            self.index = 0

        def open(self):
            return VideoMetadata("rtsp", 100, 100, 10.0, None, None)

        def read(self):
            if self.index == 6:
                return False, None, 0.0
            if self.index == 3:
                self.generation += 1
            timestamp = (self.index if self.index < 3 else self.index - 3) / 10
            self.index += 1
            return True, np.zeros((100, 100, 3), dtype=np.uint8), timestamp

        def close(self):
            pass

    monkeypatch.setattr("busreid.rear_head_out.OpenCVVideoSource", ReconnectingSource)
    result = RearHeadOutPipeline(config(), FakeTracker([.2, .2, .5, .5, .8, .8])).run(
        "rtsp://user:pass@example.test/live?token=secret", tmp_path
    )
    assert len(result.events) == 1
    assert result.events[0].timestamp_s > 0.2
    assert result.events[0].capture_time_utc
    summary = (tmp_path / "rear_head_out_summary.json").read_text(encoding="utf-8")
    assert "pass" not in summary and "secret" not in summary


def test_front_parallel_and_old_rear_modes_unchanged(tmp_path: Path, monkeypatch) -> None:
    calls = []

    class FakeDoorPipeline:
        def __init__(self, camera, _config):
            self.camera = camera.name

        def run(self, *_args, **_kwargs):
            calls.append(self.camera)
            return SimpleNamespace(status="COMPLETED", events=[], frames=1)

    class FakeExperiment:
        def __init__(self, _config):
            pass

        def run(self, *_args, **_kwargs):
            calls.append("head_zone")
            return SimpleNamespace(status="COMPLETED", events=[], frames=1)

    monkeypatch.setattr("busreid.workers.CameraPipeline", FakeDoorPipeline)
    monkeypatch.setattr("busreid.workers.RearHeadOutPipeline", FakeExperiment)
    cameras = {
        name: CameraConfig(name, name, direction, None, config().zone_a, config().zone_b)
        for name, direction in (("front", Direction.IN), ("rear", Direction.OUT))
    }
    runner = DualCameraRunner(AppConfig("sqlite://", DetectorConfig(), cameras,
                                        rear_head_out=config()))
    result = runner.run({"front": "front.mp4", "rear": "rear.mp4"}, tmp_path,
                        rear_mode="head_tracking_zone_out")
    assert set(result.results) == {"front", "rear"} and set(calls) == {"front", "head_zone"}
    calls.clear()
    result = runner.run({"rear": "rear.mp4"}, tmp_path, rear_mode="door_out")
    assert "rear" in result.results and calls == ["rear"]


def test_cli_mode_and_config_validation(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "rear.mp4"
    video(source)
    monkeypatch.setattr("busreid.rear_head_out.HeadTracker",
                        lambda _config: FakeTracker([.2, .2, .5, .5, .8, .8]))
    import yaml

    settings = yaml.safe_load(Path("configs/cameras.yaml").read_text(encoding="utf-8"))
    settings["rear_head_out"]["zone_a"] = config().zone_a
    settings["rear_head_out"]["zone_b"] = config().zone_b
    custom = tmp_path / "cameras.yaml"
    custom.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    output = tmp_path / "session"
    assert session_main(["--rear", str(source), "--rear-mode", "head_tracking_zone_out",
                         "--config", str(custom), "--output", str(output),
                         "--stop-file", str(output / "stop.request")]) == 0
    summary = json.loads((output / "session_summary.json").read_text(encoding="utf-8"))
    assert summary["cameras"]["rear"]["count"] == 1
    assert summary["cameras"]["rear"]["mode"] == "HEAD_TRACKING_ZONE_OUT"
    assert load_config().rear_head_out.zone_a
    invalid = tmp_path / "bad.yaml"
    invalid.write_text(Path("configs/cameras.yaml").read_text(encoding="utf-8").replace(
        "id_switch_max_distance: 0.08", "id_switch_max_distance: 2"), encoding="utf-8")
    with pytest.raises(ValueError, match="ID switch distance"):
        load_config(invalid)


def test_locked_preview_does_not_crash_dashboard(tmp_path: Path, monkeypatch) -> None:
    preview = tmp_path / "rear_preview.jpg"
    preview.write_bytes(b"image")
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(PermissionError(13, "locked")))
    assert not _show_preview(preview, "latest")


def test_dashboard_separates_experimental_head_out_from_business_total(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.ConnectionError())
    )
    (tmp_path / "rear_head_out_progress.json").write_text(json.dumps({
        "mode": "HEAD_TRACKING_ZONE_OUT", "status": "COMPLETED", "frames": 6,
        "count": 1, "processing_fps": 8.0, "model": "11n-head.pt", "tracker": "bytetrack.yaml",
    }), encoding="utf-8")
    (tmp_path / "rear_head_out_events.csv").write_text(
        "camera,track_id,direction,frame_index,timestamp_s,confidence,crossing_zone\n"
        "rear,head1,OUT,5,0.5,0.9,A->transition->B\n", encoding="utf-8"
    )
    app = AppTest.from_file("app/dashboard.py")
    app.session_state["ai_counting_job"] = {
        "process": SimpleNamespace(poll=lambda: 0, returncode=0),
        "output": tmp_path, "sources": {"rear": "fixture.mp4"},
        "rear_mode": "head_tracking_zone_out",
    }
    app.run(timeout=20)
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["REAR HEAD-OUT"] == "1"
    assert metrics["Head-OUT thử nghiệm"] == "1"
    assert metrics["Tổng khách xuống"] == "N/A – Chế độ thử nghiệm"


def test_running_dashboard_survives_locked_preview(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        requests, "request", lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.ConnectionError())
    )
    (tmp_path / "rear_head_out_progress.json").write_text(json.dumps({
        "status": "RUNNING", "frames": 2, "count": 0, "processing_fps": 1.0,
    }), encoding="utf-8")
    preview = tmp_path / "rear_head_out_preview.jpg"
    preview.write_bytes(b"pending")
    original = Path.read_bytes

    def locked_read(path):
        if path == preview:
            raise PermissionError(13, "Windows writer holds preview")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", locked_read)
    app = AppTest.from_file("app/dashboard.py")
    app.session_state["ai_counting_job"] = {
        "process": SimpleNamespace(poll=lambda: None), "output": tmp_path,
        "sources": {"rear": "fixture.mp4"}, "rear_mode": "head_tracking_zone_out",
    }
    app.run(timeout=20)
    assert not app.exception


def test_start_job_accepts_head_zone_only_in_independent_rear_session(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "cameras.yaml").write_bytes(Path("configs/cameras.yaml").read_bytes())
    commands = []
    monkeypatch.setattr("app.video_lab.subprocess.Popen",
                        lambda command, **_kwargs: commands.append(command) or SimpleNamespace(poll=lambda: None))
    start_job(tmp_path, tmp_path / "session", {"rear": "rear.mp4"},
              session_only=True, rear_mode="head_tracking_zone_out")
    assert commands[0][commands[0].index("--rear-mode") + 1] == "head_tracking_zone_out"
    with pytest.raises(ValueError, match="chỉ hỗ trợ REAR"):
        start_job(tmp_path, tmp_path / "bad", {"front": "front.mp4"},
                  session_only=True, rear_mode="head_tracking_zone_out")
