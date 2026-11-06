"""Head ROI editor software tests; fake jobs do not measure AI accuracy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests
import yaml
from streamlit.testing.v1 import AppTest

from app import roi_editor
from app.video_lab import start_job
from busreid.config import load_config
from busreid.rear_head_out import HeadZoneCounter, RearHeadOutPipeline
from busreid.roi import restore_head_out_backup, save_head_out_zones, validate_head_out_zones
from busreid.types import Detection

ROI = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
A = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.8), (0.1, 0.8)]
B = [(0.7, 0.2), (0.9, 0.2), (0.9, 0.8), (0.7, 0.8)]
NARROW = [(0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)]


def config_copy(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / "cameras.yaml"
    raw = yaml.safe_load(Path("configs/cameras.yaml").read_text(encoding="utf-8"))
    raw["rear_head_out"]["roi"] = ROI
    raw["rear_head_out"]["zone_a"] = [(.35, .2), (.45, .2), (.45, .8), (.35, .8)]
    raw["rear_head_out"]["zone_b"] = [(.55, .2), (.65, .2), (.65, .8), (.55, .8)]
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()


@pytest.mark.parametrize("roi,a,b", [
    (ROI, A, [(0.8, 0.5), (1.1, 0.5), (0.8, 0.8)]),  # outside frame/ROI
    (NARROW, [(0.0, 0.2), (0.3, 0.2), (0.3, 0.8)], B),  # outside ROI
    (ROI, A, [(0.2, 0.3), (0.4, 0.3), (0.4, 0.7), (0.2, 0.7)]),  # overlap
    (ROI, A, [(0.3, 0.2), (0.5, 0.2), (0.5, 0.8), (0.3, 0.8)]),  # no gap
    (ROI, [(0.1, 0.2), (0.3, 0.8), (0.3, 0.2), (0.1, 0.8)], B),  # self crossing
])
def test_head_polygon_rejects_invalid_or_missing_transition(roi, a, b) -> None:
    with pytest.raises(ValueError):
        validate_head_out_zones(roi, a, b)


def test_concave_roi_rejects_zone_edge_that_leaves_roi() -> None:
    concave = [(0, 0), (1, 0), (1, 1), (.7, 1), (.7, .3), (.3, .3), (.3, 1), (0, 1)]
    # All triangle vertices are inside, but its bottom edge crosses the indentation.
    crossing = [(.2, .5), (.8, .5), (.5, .2)]
    other = [(.85, .05), (.95, .05), (.95, .2), (.85, .2)]
    with pytest.raises(ValueError, match="ROI"):
        validate_head_out_zones(concave, crossing, other)


def test_head_save_restore_scoped_to_three_fields(tmp_path: Path) -> None:
    path = config_copy(tmp_path)
    original = yaml.safe_load(path.read_text(encoding="utf-8"))
    original["custom_field"] = {"kept": True}
    path.write_text(yaml.safe_dump(original, allow_unicode=True), encoding="utf-8")
    backup = save_head_out_zones(path, ROI, A, B)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert backup.is_file()
    assert saved["cameras"] == original["cameras"]
    assert saved["rear_occupancy"] == original["rear_occupancy"]
    assert saved["detector"] == original["detector"]
    assert saved["custom_field"] == original["custom_field"]
    assert {key: saved["rear_head_out"][key] for key in ("roi", "zone_a", "zone_b")} == {
        "roi": [list(p) for p in ROI], "zone_a": [list(p) for p in A],
        "zone_b": [list(p) for p in B],
    }
    assert saved["rear_head_out"]["model"] == original["rear_head_out"]["model"]

    saved["rear_head_out"]["model"] = "other-head.pt"
    saved["cameras"]["front"]["time_offset_ms"] = 123
    path.write_text(yaml.safe_dump(saved, allow_unicode=True), encoding="utf-8")
    recovery = restore_head_out_backup(path)
    restored = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert recovery.is_file()
    assert restored["rear_head_out"]["zone_a"] == original["rear_head_out"]["zone_a"]
    assert restored["rear_head_out"]["zone_b"] == original["rear_head_out"]["zone_b"]
    assert restored["rear_head_out"]["model"] == "other-head.pt"
    assert restored["cameras"]["front"]["time_offset_ms"] == 123


def test_invalid_head_save_does_not_modify_yaml(tmp_path: Path) -> None:
    path = config_copy(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_head_out_zones(path, ROI, A, A)
    assert path.read_bytes() == before
    assert not path.with_name("cameras.yaml.rear_head_out.bak").exists()


def test_click_edit_all_three_zones_and_preview(tmp_path: Path) -> None:
    configured = load_config(config_copy(tmp_path))
    zones = {"roi": copy.deepcopy(ROI), "a": copy.deepcopy(A), "b": copy.deepcopy(B)}
    state = {"zones": zones, "history": [], "revision": 0}
    for label in ("roi", "a", "b"):
        roi_editor._apply_click(state, label, "Thêm đỉnh", {"x": 45, "y": 45}, (100, 100), 0)
        assert state["zones"][label][-1] == (.45, .45)
        roi_editor._apply_click(
            state, label, "Dời đỉnh", {"x1": 10, "y1": 20, "x2": 12, "y2": 22}, (100, 100), 0
        )
        assert state["zones"][label][0] == (.12, .22)
    assert len(state["history"]) == 6
    state["zones"] = state["history"].pop()  # undo
    assert state["zones"]["b"][0] == B[0]
    state["zones"]["roi"] = []  # partial drafts remain renderable
    image = roi_editor._display_frame(
        np.zeros((100, 100, 3), dtype=np.uint8), configured.cameras["rear"],
        state["zones"], head_mode=True,
    )
    assert image.size == (100, 100)
    state["zones"] = roi_editor._configured_zones("rear_head_out", config_copy(tmp_path))
    assert state["zones"]["roi"] == ROI  # reload saved config


def test_new_job_snapshots_saved_zones_while_running_job_keeps_old(tmp_path: Path, monkeypatch) -> None:
    path = config_copy(tmp_path)
    old_config = load_config(path)
    running = HeadZoneCounter(old_config.rear_head_out)
    launched = []
    monkeypatch.setattr("app.video_lab.subprocess.Popen",
                        lambda command, **_kwargs: launched.append(command) or SimpleNamespace(poll=lambda: None))
    start_job(tmp_path, tmp_path / "old_job", {"rear": "rear.mp4"},
              rear_mode="head_tracking_zone_out", session_only=True)
    save_head_out_zones(path, ROI, A, B)
    new_config = load_config(path)
    fresh = HeadZoneCounter(new_config.rear_head_out)
    start_job(tmp_path, tmp_path / "new_job", {"rear": "rear.mp4"},
              rear_mode="head_tracking_zone_out", session_only=True)
    assert running.config.zone_a != A and fresh.config.zone_a == A
    assert load_config(tmp_path / "old_job" / "cameras_snapshot.yaml").rear_head_out.zone_a != A
    assert load_config(tmp_path / "new_job" / "cameras_snapshot.yaml").rear_head_out.zone_a == A
    assert all("--rear-mode" in command for command in launched)


def test_new_head_pipeline_uses_saved_zone_and_old_pipeline_keeps_old(tmp_path: Path) -> None:
    path = config_copy(tmp_path)
    old_config = load_config(path)

    class FakeTracks:
        def __init__(self):
            self.positions = iter([.2, .2, .5, .5, .8, .8])

        def track(self, _frame):
            x = round(next(self.positions) * 100)
            return [Detection("head1", (x - 3, 47, x + 3, 53), .9)]

    # Six software frames: A, transition, B. This is not an AI accuracy test.
    video_path = tmp_path / "six.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    save_head_out_zones(path, ROI, A, B)
    new_config = load_config(path)
    old_result = RearHeadOutPipeline(old_config.rear_head_out, FakeTracks()).run(video_path, tmp_path / "old")
    new_result = RearHeadOutPipeline(new_config.rear_head_out, FakeTracks()).run(video_path, tmp_path / "new")
    assert len(old_result.events) == 0 and len(new_result.events) == 1
    assert not (tmp_path / "new" / "rear_events.csv").exists()


def _offline(monkeypatch) -> None:
    monkeypatch.setattr(requests, "request", lambda *_a, **_kw: (_ for _ in ()).throw(requests.ConnectionError()))


def test_dashboard_head_draft_survives_switch_and_undo(tmp_path: Path, monkeypatch) -> None:
    _offline(monkeypatch)
    original_roi = load_config().rear_head_out.roi
    source = tmp_path / "rear.mp4"
    video(source)
    monkeypatch.setattr(roi_editor, "_select_source", lambda *_args: (str(source), True))
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.selectbox if item.label == "Camera").set_value("rear_head_out").run(timeout=20)
    assert not app.exception
    assert set(app.session_state["roi_editor_rear_head_out"]["zones"]) == {"roi", "a", "b"}
    next(button for button in app.button if button.label == "Xóa vùng").click().run(timeout=20)
    assert app.session_state["roi_editor_rear_head_out"]["zones"]["roi"] == []
    assert next(button for button in app.button if button.label == "Lưu ROI").disabled
    next(item for item in app.selectbox if item.label == "Camera").set_value("rear").run(timeout=20)
    assert app.session_state["roi_editor_rear"]["zones"]["a"]
    next(item for item in app.selectbox if item.label == "Camera").set_value("rear_head_out").run(timeout=20)
    assert app.session_state["roi_editor_rear_head_out"]["zones"]["roi"] == []
    next(button for button in app.button if button.label == "Hoàn tác").click().run(timeout=20)
    assert app.session_state["roi_editor_rear_head_out"]["zones"]["roi"] == original_roi
    assert not app.exception


def test_head_canvas_click_creates_roi_vertex(tmp_path: Path, monkeypatch) -> None:
    _offline(monkeypatch)
    source = tmp_path / "rear.mp4"
    video(source)
    monkeypatch.setattr(roi_editor, "_select_source", lambda *_args: (str(source), True))
    clicked = False

    def canvas(_picture, *, key, **_kwargs):
        nonlocal clicked
        if "roi_canvas_rear_head_out" in key and not clicked:
            clicked = True
            return {"x": 50, "y": 50}
        return None

    monkeypatch.setattr(roi_editor, "streamlit_image_coordinates", canvas)
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.selectbox if item.label == "Camera").set_value("rear_head_out").run(timeout=20)
    assert not app.exception and clicked
    assert app.session_state["roi_editor_rear_head_out"]["zones"]["roi"][-1] == (.5, .5)


def test_head_editor_save_and_restore_buttons_update_only_head_zones(tmp_path: Path, monkeypatch) -> None:
    _offline(monkeypatch)
    path = config_copy(tmp_path)
    source = tmp_path / "rear.mp4"
    video(source)
    monkeypatch.setattr(roi_editor, "_select_source", lambda *_args: (str(source), True))
    monkeypatch.setattr(roi_editor, "load_config", lambda _path: load_config(path))
    monkeypatch.setattr(roi_editor, "save_head_out_zones",
                        lambda _path, roi, a, b: save_head_out_zones(path, roi, a, b))
    monkeypatch.setattr(roi_editor, "restore_head_out_backup",
                        lambda _path: restore_head_out_backup(path))
    before = yaml.safe_load(path.read_text(encoding="utf-8"))
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.selectbox if item.label == "Camera").set_value("rear_head_out").run(timeout=20)
    state = app.session_state["roi_editor_rear_head_out"]
    roi_editor._mutate(state, {"roi": ROI, "a": A, "b": B})
    app.run(timeout=20)
    next(button for button in app.button if button.label == "Lưu ROI").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["roi_editor_rear_head_out"]["saved_revision"] == 1
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["rear_head_out"]["zone_a"] == [list(point) for point in A]
    assert saved["cameras"] == before["cameras"]
    next(button for button in app.button if button.label == "Khôi phục ROI Head Tracking từ bản sao lưu").click().run(
        timeout=20
    )
    assert not app.exception
    restored = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert restored["rear_head_out"]["zone_a"] == before["rear_head_out"]["zone_a"]
    assert restored["cameras"] == before["cameras"]


def test_head_roi_test_button_routes_worker_and_shows_diagnostics(tmp_path: Path, monkeypatch) -> None:
    _offline(monkeypatch)
    source = tmp_path / "rear.mp4"
    video(source)
    monkeypatch.setattr(roi_editor, "_select_source", lambda *_args: (str(source), True))
    output = tmp_path / "job"
    output.mkdir()
    monkeypatch.setattr(roi_editor, "new_job_dir", lambda *_a, **_kw: output)
    (output / "rear_head_out_progress.json").write_text(json.dumps({
        "status": "RUNNING", "frames": 10, "count": 2, "detected_heads": 12,
        "tracked_heads": 11, "stable_a_confirmations": 3,
        "transition_confirmations": 2, "stable_b_confirmations": 2, "processing_fps": 7.5,
    }), encoding="utf-8")
    preview = output / "rear_head_out_preview.jpg"
    preview.write_bytes(b"locked")
    (output / "rear_head_out_annotated.mp4").write_bytes(b"locked")
    original_read = Path.read_bytes

    def locked_read(path):
        if path == preview:
            raise PermissionError(13, "locked")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", locked_read)
    monkeypatch.setattr(roi_editor.st, "video",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError(13, "locked")))
    calls = []
    monkeypatch.setattr(roi_editor, "start_job", lambda *_args, **kwargs:
                        calls.append(kwargs) or SimpleNamespace(poll=lambda: None))
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.selectbox if item.label == "Camera").set_value("rear_head_out").run(timeout=20)
    next(button for button in app.button if button.label == "Test Head Tracking ROI").click().run(timeout=20)
    assert not app.exception
    assert calls[0]["rear_mode"] == "head_tracking_zone_out"
    assert calls[0]["session_only"] is True and calls[0]["max_frames"] == 90
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Head-OUT"] == "2"
    assert metrics["Detection"] == "12" and metrics["Track"] == "11"
    assert any("Xác nhận A: 3" in item.value for item in app.get("caption"))


def test_unsaved_head_draft_blocks_roi_test(tmp_path: Path, monkeypatch) -> None:
    _offline(monkeypatch)
    source = tmp_path / "rear.mp4"
    video(source)
    monkeypatch.setattr(roi_editor, "_select_source", lambda *_args: (str(source), True))
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    next(item for item in app.selectbox if item.label == "Camera").set_value("rear_head_out").run(timeout=20)
    next(button for button in app.button if button.label == "Xóa đỉnh đã chọn").click().run(timeout=20)
    assert app.session_state["roi_editor_rear_head_out"]["revision"] != (
        app.session_state["roi_editor_rear_head_out"]["saved_revision"]
    )
    assert next(button for button in app.button if button.label == "Test Head Tracking ROI").disabled
    assert not app.exception
