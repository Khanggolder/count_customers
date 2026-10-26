"""Software fixtures exercise video handling; FakeDetector does not measure AI accuracy."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.video_lab import camera_status, event_timeline, first_frame, resolve_source, start_job
from busreid.config import AppConfig, CameraConfig, DetectorConfig
from busreid.pipeline import CameraPipeline, OverlayOptions, PipelineCancelled
from busreid.types import Detection, Direction
from busreid.video_output import validate_video


def fixture_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.full((100, 100, 3), 100, dtype=np.uint8))
    writer.release()


class FakeDetector:
    def __init__(self, camera: str, reverse: bool = False):
        self.camera = camera
        self.reverse = reverse
        self.index = 0

    def track(self, _frame):
        y = (80 if self.index < 3 else 25) if self.reverse else (25 if self.index < 3 else 80)
        self.index += 1
        return [Detection(f"{self.camera}:track1", (40, y - 20, 60, y), 0.91)]


def camera(name: str) -> CameraConfig:
    direction = Direction.IN if name == "front" else Direction.OUT
    return CameraConfig(
        name,
        name,
        direction,
        None,
        [(0, 0), (1, 0), (1, 0.45), (0, 0.45)],
        [(0, 0.55), (1, 0.55), (1, 1), (0, 1)],
        2,
        4,
    )


def test_upload_path_rtsp_and_empty_source(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    fixture_video(source)
    uploaded = resolve_source(io.BytesIO(source.read_bytes()), "camera.mp4", "", "", tmp_path / "uploads")
    assert uploaded is not None and Path(uploaded).is_file()
    assert first_frame(uploaded) is not None
    assert resolve_source(None, None, str(source), "", tmp_path / "uploads") == str(source.resolve())
    assert resolve_source(None, None, "", "rtsp://camera/live", tmp_path / "uploads") == "rtsp://camera/live"
    assert resolve_source(None, None, "", "", tmp_path / "uploads") is None
    with pytest.raises(ValueError, match="Chỉ hỗ trợ"):
        resolve_source(io.BytesIO(b"bad"), "video.avi", "", "", tmp_path / "uploads")


@pytest.mark.parametrize("name,direction", [("front", Direction.IN), ("rear", Direction.OUT)])
def test_ai_overlay_browser_video_and_reverse_policy(tmp_path: Path, name: str, direction: Direction) -> None:
    source = tmp_path / "fixture.mp4"
    fixture_video(source)
    cam = camera(name)
    config = AppConfig("sqlite://", DetectorConfig(), {name: cam})
    output = tmp_path / "output"
    result = CameraPipeline(cam, config, FakeDetector(name)).run(
        str(source), output, overlay=OverlayOptions()
    )
    assert [item.direction for item in result.events] == [direction]
    assert result.frames == 8
    validate_video(Path(result.annotated_path), 8)
    capture = cv2.VideoCapture(result.annotated_path)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    assert "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).lower() in {"avc1", "h264"}
    assert (output / f"{name}_preview.jpg").is_file()
    progress = camera_status(output, name, 0)
    assert progress["status"] == "COMPLETED"
    assert progress["counts"][direction.value] == 1
    rows = event_timeline(output, [name])
    assert len(rows) == 1 and rows[0]["event_status"] == "UNASSIGNED"
    reverse = CameraPipeline(cam, config, FakeDetector(name, reverse=True)).run(
        str(source),
        tmp_path / "reverse",
        overlay=OverlayOptions(boxes=False, track_ids=False, zones=False, trajectory=False, events=False),
    )
    assert [event.direction.value for event in reverse.events] == ["OUT" if name == "front" else "IN"]
    assert [event.delta for event in reverse.events] == [-1 if name == "front" else 1]
    assert camera_status(tmp_path / "reverse", name, 0)["counts"][reverse.events[0].direction.value] == 1
    reverse_row = event_timeline(tmp_path / "reverse", [name])[0]
    assert reverse_row["delta"] == (-1 if name == "front" else 1)
    assert reverse_row["event_status"] == "UNASSIGNED"


def test_progress_failure_invalid_output_and_cancel(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "fixture.mp4"
    fixture_video(source)
    cam = camera("front")
    config = AppConfig("sqlite://", DetectorConfig(), {"front": cam})
    output = tmp_path / "failed"

    def broken_encoder(*_args):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr("busreid.pipeline.make_browser_video", broken_encoder)
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        CameraPipeline(cam, config, FakeDetector("front")).run(str(source), output)
    assert camera_status(output, "front", 1)["status"] == "FAILED"
    assert not (output / "front_summary.json").exists()
    with pytest.raises(RuntimeError, match="rỗng"):
        validate_video(tmp_path / "missing.mp4")

    marker = tmp_path / "cancel.request"
    marker.touch()
    with pytest.raises(PipelineCancelled):
        CameraPipeline(cam, config, FakeDetector("front")).run(
            str(source), tmp_path / "cancelled", cancel_file=marker
        )
    assert camera_status(tmp_path / "cancelled", "front", 2)["status"] == "CANCELLED"
    assert camera_status(tmp_path / "no-progress", "front", None)["status"] == "QUEUED"


def test_timeline_reads_both_cameras(tmp_path: Path) -> None:
    for camera_name, direction in (("front", "IN"), ("rear", "OUT")):
        path = tmp_path / f"{camera_name}_events.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "camera",
                    "track_id",
                    "direction",
                    "frame_index",
                    "timestamp_s",
                    "confidence",
                    "crop_path",
                    "capture_time_utc",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "camera": camera_name,
                    "track_id": f"{camera_name}:1",
                    "direction": direction,
                    "frame_index": 5,
                    "timestamp_s": 0.5,
                    "confidence": 0.8,
                    "crop_path": "",
                    "capture_time_utc": "",
                }
            )
    rows = event_timeline(tmp_path, ["front", "rear"])
    assert {(row["camera"], row["direction"]) for row in rows} == {("FRONT", "IN"), ("REAR", "OUT")}
    assert all(row["delta"] == 1 for row in rows)  # Old positive CSV stays readable as inferred.
    (tmp_path / "front_progress.json").write_text(json.dumps({"status": "RUNNING", "frames": 5}))
    assert camera_status(tmp_path, "front", None)["frames"] == 5


def test_video_lab_launch_does_not_select_synthetic_trip(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return object()

    monkeypatch.setattr("app.video_lab.subprocess.Popen", fake_popen)
    source = tmp_path / "uploaded.mp4"
    fixture_video(source)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "cameras.yaml").write_bytes(
        (Path(__file__).parents[1] / "configs" / "cameras.yaml").read_bytes()
    )
    start_job(tmp_path, tmp_path, {"front": str(source)}, hidden_layers={"boxes"})
    assert "--front" in captured["command"]
    assert "--rear" not in captured["command"]
    assert "--trip-id" not in captured["command"]
    assert "--telemetry-synchronized" not in captured["command"]
    assert "--hide-boxes" in captured["command"]
    assert "--cancel-file" in captured["command"]
