"""Synthetic motion tests verify software behavior, not bus passenger accuracy."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests
import yaml
from streamlit.testing.v1 import AppTest

from app import flow_roi_editor
from app.video_lab import start_job
from busreid.config import RearFlowOutConfig, load_config
from busreid.rear_flow_counting import FlowLineAnalyzer, RearFlowPipeline
from busreid.roi import restore_flow_geometry, save_flow_geometry, validate_flow_geometry
from scripts.evaluate_flow import evaluate_motion

ROI = [(.1, .1), (.9, .1), (.9, .9), (.1, .9)]
LINE = [(.2, .5), (.8, .5)]


def config() -> RearFlowOutConfig:
    return RearFlowOutConfig(enabled=True, roi=ROI, counting_line=LINE, out_direction=(0, 1),
                             smoothing_window=2, min_event_frames=2, max_gap_frames=2,
                             min_motion_magnitude=.7, min_active_fraction=.05)


def frames(*, reverse: bool = False, stationary: bool = False, noise: bool = False) -> list[np.ndarray]:
    random = np.random.default_rng(7)
    patch = random.integers(70, 255, (24, 24, 3), np.uint8)
    positions = list(range(15, 95, 3))
    if reverse:
        positions.reverse()
    output = []
    for top in positions:
        image = random.integers(0, 4, (120, 120, 3), np.uint8) if noise else np.zeros((120, 120, 3), np.uint8)
        y = 22 if stationary else top
        image[y:y + 24, 48:72] = patch
        output.append(image)
    return output


def analyze(images: list[np.ndarray], cfg: RearFlowOutConfig | None = None):
    analyzer = FlowLineAnalyzer(cfg or config(), images[0].shape)
    for index, frame in enumerate(images):
        analyzer.update(frame, index, index / 10)
    analyzer.finish()
    return analyzer


def test_geometry_rejects_outside_concave_roi_and_parallel_arrow() -> None:
    validate_flow_geometry(ROI, LINE, (0, 1))
    validate_flow_geometry(ROI, list(reversed(LINE)), (0, 1))
    with pytest.raises(ValueError):
        validate_flow_geometry(ROI, [(0, .5), (.8, .5)], (0, 1))
    with pytest.raises(ValueError):
        validate_flow_geometry(ROI, LINE, (1, 0))


def test_save_restore_flow_only(tmp_path: Path) -> None:
    path = tmp_path / "cameras.yaml"
    path.write_text(Path("configs/cameras.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    before = load_config(path)
    backup = save_flow_geometry(path, ROI, LINE, (0, 1))
    assert backup.exists()
    saved = load_config(path)
    assert saved.rear_flow_out.enabled and saved.rear_flow_out.roi == ROI
    assert saved.cameras == before.cameras and saved.rear_head_out == before.rear_head_out
    restore_flow_geometry(path)
    assert not load_config(path).rear_flow_out.enabled


def test_out_reverse_stationary_and_noise() -> None:
    out = analyze(frames())
    assert sum(event.direction == "OUT" for event in out.events) == 1
    assert all(event.estimated_passenger_count is None for event in out.events)
    assert len(out.slice_rows) == len(out.signed_rows) == len(frames())
    assert out.temporal_image().shape[1] == len(out.sample_x)
    reverse = analyze(frames(reverse=True))
    assert not any(event.direction == "OUT" for event in reverse.events)
    assert not analyze(frames(stationary=True)).events
    assert not analyze(frames(stationary=True, noise=True)).events


def test_two_separated_crossings_and_close_group_uncertain() -> None:
    images = frames() + [np.zeros((120, 120, 3), np.uint8) for _ in range(20)] + frames()
    result = analyze(images)
    assert sum(event.direction == "OUT" for event in result.events) == 2
    assert all(event.review_status == "REVIEW_REQUIRED" for event in result.events)


def test_crowded_occlusion_low_light_shake_and_door_motion_remain_uncalibrated() -> None:
    base = frames()
    low_light = [np.clip(frame // 5, 0, 255).astype(np.uint8) for frame in base]
    shake = [cv2.warpAffine(frame, np.float32([[1, 0, index % 3], [0, 1, 0]]), (120, 120))
             for index, frame in enumerate(base)]
    occluded = [frame.copy() for frame in base]
    for frame in occluded:
        frame[40:80, 58:70] = 0
    crowd = [frame.copy() for frame in base]
    for frame in crowd:
        frame[:, 73:97] = frame[:, 48:72]
    # A moving rectangle can model a door panel. It may produce a motion event,
    # but the software must never turn it into a passenger count.
    for clip, cfg in ((low_light, config()), (shake, replace(config(), stabilization=True)),
                      (occluded, config()), (crowd, config()), (base, config())):
        observed = analyze(clip, cfg)
        assert all(event.estimated_passenger_count is None and event.review_status == "REVIEW_REQUIRED"
                   for event in observed.events)


def test_real_pipeline_files_are_separate_from_business_events(tmp_path: Path) -> None:
    path = tmp_path / "motion.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (120, 120))
    assert writer.isOpened()
    for frame in frames():
        writer.write(frame)
    writer.release()
    result = RearFlowPipeline(config()).run(path, tmp_path / "out")
    assert result.frames == len(frames())
    assert (tmp_path / "out" / "rear_flow_temporal_slice.png").is_file()
    summary = json.loads((tmp_path / "out" / "rear_flow_summary.json").read_text())
    assert summary["estimated_out"] is None and summary["ground_truth_evaluation"] == "N/A"
    assert not (tmp_path / "out" / "rear_events.csv").exists()
    with pytest.raises(ValueError, match="Configure"):
        RearFlowPipeline(replace(config(), enabled=False)).run(path, tmp_path / "disabled")


def test_flow_editor_preview_click_and_worker_snapshot(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "cameras.yaml"
    path.write_text(Path("configs/cameras.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    old = load_config(path)
    image = flow_roi_editor._preview(np.zeros((120, 120, 3), np.uint8),
                                     {"roi": ROI, "line": LINE, "direction": (0, 1)}, path)
    assert image.size == (120, 120)
    state = {"draft": {"roi": list(ROI), "line": list(LINE), "direction": (0, 1)},
             "history": [], "revision": 0}
    flow_roi_editor._click(state, "Đầu mũi tên OUT", (.5, .1), 0)
    assert state["draft"]["direction"][1] < 0
    assert state["revision"] == 1 and len(state["history"]) == 1
    launched = []
    monkeypatch.setattr("app.video_lab.subprocess.Popen",
                        lambda command, **_kwargs: launched.append(command) or type("Job", (), {"poll": lambda _self: None})())
    start_job(tmp_path, tmp_path / "before", {"rear": "rear.mp4"},
              rear_mode="flow_line_out", session_only=True)
    save_flow_geometry(path, ROI, LINE, (0, -1))
    start_job(tmp_path, tmp_path / "after", {"rear": "rear.mp4"},
              rear_mode="flow_line_out", session_only=True)
    assert not old.rear_flow_out.enabled
    assert not load_config(tmp_path / "before" / "cameras_snapshot.yaml").rear_flow_out.enabled
    assert load_config(tmp_path / "after" / "cameras_snapshot.yaml").rear_flow_out.enabled
    assert all("flow_line_out" in command for command in launched)


def test_flow_evaluator_keeps_passenger_accuracy_na() -> None:
    event = {"direction": "OUT", "start_time_s": "1", "end_time_s": "2"}
    assert evaluate_motion([event], None)["motion_precision"] is None
    assert evaluate_motion([event], [])["motion_precision"] is None
    report = evaluate_motion([event], [{"direction": "OUT", "event_time_s": "1.5",
                                        "passenger_count": "3"}])
    assert report["motion_precision"] == report["motion_recall"] == 1.0
    assert report["passenger_mae"] is None


def test_flow_config_rejects_invalid_line(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/cameras.yaml").read_text(encoding="utf-8"))
    raw["rear_flow_out"]["counting_line"] = [[0, 0], [0, 0]]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Counting Line"):
        load_config(path)


def test_dashboard_rear_flow_editor_renders(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(requests, "request", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        requests.ConnectionError("API offline in UI fixture")))
    video = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (120, 120))
    assert writer.isOpened()
    writer.write(np.zeros((120, 120, 3), np.uint8))
    writer.release()
    monkeypatch.setattr(flow_roi_editor, "_select_source", lambda *_args: (str(video), True))
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.selectbox if item.label == "Camera").set_value("rear_flow_out").run(timeout=20)
    assert not app.exception
    assert any(button.label == "Lưu Optical Flow ROI" for button in app.button)
    assert any(button.label == "Test Optical Flow ROI" for button in app.button)
