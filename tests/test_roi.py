from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.roi_editor import _apply_click, _display_frame, read_video_frame
from busreid.config import load_config
from busreid.counting import DirectionCounter
from busreid.pipeline import CameraPipeline
from busreid.roi import (
    restore_backup,
    save_zones,
    to_normalized,
    to_pixels,
    validate_polygon,
    validate_zones,
)
from busreid.types import Detection, Direction

ZONE_A = [(0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)]
ZONE_B = [(0.6, 0.6), (0.9, 0.6), (0.9, 0.9), (0.6, 0.9)]


def config_copy(tmp_path: Path) -> Path:
    path = tmp_path / "cameras.yaml"
    path.write_bytes((Path(__file__).parents[1] / "configs" / "cameras.yaml").read_bytes())
    return path


def test_normalized_round_trip_and_outside() -> None:
    assert to_normalized((160, 120), 320, 240) == (0.5, 0.5)
    assert to_pixels((0.5, 0.5), 320, 240) == (160, 120)
    with pytest.raises(ValueError, match="ngoài frame"):
        to_normalized((320, 1), 320, 240)


@pytest.mark.parametrize(
    "a,b",
    [
        ([(0, 0), (0.5, 0)], ZONE_B),
        ([(0, 0), (1.1, 0), (0, 1)], ZONE_B),
        ([(0, 0), (1, 1), (0, 1), (1, 0)], ZONE_B),
        (ZONE_A, [(0.2, 0.2), (0.3, 0.2), (0.3, 0.3), (0.2, 0.3)]),
        (ZONE_A, [(0.3, 0.3), (0.6, 0.3), (0.6, 0.6), (0.3, 0.6)]),
    ],
)
def test_rejects_invalid_polygons(a, b) -> None:
    with pytest.raises(ValueError):
        validate_zones(a, b)


def test_save_load_backup_restore_and_policy(tmp_path: Path) -> None:
    path = config_copy(tmp_path)
    original = load_config(path)
    backup = save_zones(path, "FRONT", ZONE_A, ZONE_B)
    updated = load_config(path)
    assert backup.is_file()
    assert updated.cameras["front"].zone_a == ZONE_A
    assert updated.cameras["front"].zone_b == ZONE_B
    assert updated.cameras["front"].event_direction == Direction.IN
    assert updated.cameras["rear"] == original.cameras["rear"]
    restore_backup(path, "front")
    assert load_config(path).cameras["front"] == original.cameras["front"]


@pytest.mark.parametrize("points", [[], [(0.1, 0.1)], [(0.1, 0.1), (0.2, 0.2)]])
def test_incomplete_polygons_render_without_changing_saved_config(tmp_path: Path, points) -> None:
    path = config_copy(tmp_path)
    before = path.read_bytes()
    camera = load_config(path).cameras["front"]
    image = _display_frame(
        np.zeros((100, 100, 3), dtype=np.uint8), camera, {"a": points, "b": ZONE_B}
    )
    assert image.size == (100, 100)
    with pytest.raises(ValueError, match="ít nhất 3"):
        validate_polygon(points)
    assert path.read_bytes() == before


def test_complete_and_invalid_polygons_still_render_for_repair(tmp_path: Path) -> None:
    camera = load_config(config_copy(tmp_path)).cameras["front"]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for points in (
        [(0.1, 0.1), (0.4, 0.1), (0.1, 0.4)],
        ZONE_A,
        [(0.1, 0.1), (0.4, 0.4), (0.1, 0.4), (0.4, 0.1)],
        [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)],
    ):
        assert _display_frame(frame, camera, {"a": points, "b": ZONE_B}).size == (100, 100)
    with pytest.raises(ValueError, match="tự cắt"):
        validate_polygon([(0.1, 0.1), (0.4, 0.1), (0.1, 0.4), (0.4, 0.4)])
    with pytest.raises(ValueError, match="diện tích"):
        validate_polygon([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)])


@pytest.mark.parametrize("camera", ["front", "rear"])
def test_restore_only_selected_camera_roi_and_preserves_other_settings(tmp_path: Path, camera: str) -> None:
    path = config_copy(tmp_path)
    original = load_config(path)
    save_zones(path, camera, ZONE_A, ZONE_B)
    other = "rear" if camera == "front" else "front"
    save_zones(path, other, ZONE_A, ZONE_B)
    before_restore = load_config(path)
    recovery = restore_backup(path, camera)
    restored = load_config(path)
    assert recovery.is_file()
    assert restored.cameras[camera].zone_a == original.cameras[camera].zone_a
    assert restored.cameras[other] == before_restore.cameras[other]
    assert restored.detector == original.detector
    assert restored.cameras[camera].event_direction == original.cameras[camera].event_direction


def test_restore_uses_backup_roi_without_reverting_unrelated_camera(tmp_path: Path) -> None:
    path = config_copy(tmp_path)
    original = load_config(path)
    save_zones(path, "front", ZONE_A, ZONE_B)
    save_zones(path, "rear", ZONE_A, ZONE_B)
    front_before = load_config(path).cameras["front"]
    restore_backup(path, "rear")
    result = load_config(path)
    assert result.cameras["front"] == front_before
    assert result.cameras["rear"] == original.cameras["rear"]


def test_new_pipeline_uses_saved_roi_and_running_pipeline_keeps_snapshot(tmp_path: Path) -> None:
    path = config_copy(tmp_path)
    old_config = load_config(path)
    detector = object()
    running = CameraPipeline(old_config.cameras["front"], old_config, detector=detector)
    save_zones(path, "front", ZONE_A, ZONE_B)
    new_config = load_config(path)
    fresh = CameraPipeline(new_config.cameras["front"], new_config, detector=detector)
    assert running.counter.zone_a_normalized != ZONE_A
    assert fresh.counter.zone_a_normalized == ZONE_A
    assert fresh.counter.zone_b_normalized == ZONE_B
    assert fresh.counter.forward_direction == Direction.IN

    counter = DirectionCounter("front", ZONE_A, ZONE_B, Direction.IN, min_stable_frames=1)
    ys = [20, 20, 50, 70, 70]
    xs = [20, 20, 50, 70, 70]
    events = []
    for frame, (x, y) in enumerate(zip(xs, ys, strict=True)):
        event = counter.update(
            Detection("track-1", (x - 5, y - 10, x + 5, y), 0.9), (100, 100, 3), frame, frame / 10
        )
        if event:
            events.append(event)
    assert len(events) == 1
    assert events[0].direction == Direction.IN


def test_click_edit_undo_and_video_frame(tmp_path: Path) -> None:
    path = config_copy(tmp_path)
    camera = load_config(path).cameras["front"]
    zones = {"a": copy.deepcopy(ZONE_A), "b": copy.deepcopy(ZONE_B)}
    state = {"zones": zones, "history": [], "revision": 0}
    _apply_click(state, "a", "Thêm đỉnh", {"x": 30, "y": 30}, (100, 100), 0)
    assert state["zones"]["a"][-1] == (0.3, 0.3)
    _apply_click(state, "a", "Dời đỉnh", {"x1": 10, "y1": 10, "x2": 15, "y2": 15}, (100, 100), 2)
    assert state["zones"]["a"][0] == (0.15, 0.15)
    assert len(state["history"]) == 2
    image = _display_frame(np.zeros((100, 100, 3), dtype=np.uint8), camera, zones)
    assert image.size == (100, 100)

    video = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    assert writer.isOpened()
    for value in (20, 80, 140):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    frame, count, fps = read_video_frame(str(video), 2)
    assert count == 3
    assert fps == 5
    assert frame.mean() > 100


@pytest.mark.parametrize("label", ["a", "b"])
def test_first_and_second_click_and_cleared_zone_preview(tmp_path: Path, label: str) -> None:
    camera = load_config(config_copy(tmp_path)).cameras["front"]
    state = {"zones": {"a": copy.deepcopy(ZONE_A), "b": copy.deepcopy(ZONE_B)}, "history": [], "revision": 0}
    state["zones"][label] = []
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert _display_frame(frame, camera, state["zones"]).size == (100, 100)
    for index, coords in enumerate(((10, 10), (30, 10)), start=1):
        _apply_click(state, label, "Thêm đỉnh", {"x": coords[0], "y": coords[1]}, (100, 100), 0)
        assert len(state["zones"][label]) == index
        assert _display_frame(frame, camera, state["zones"]).size == (100, 100)
    state["zones"] = state["history"].pop()
    assert len(state["zones"][label]) == 1
