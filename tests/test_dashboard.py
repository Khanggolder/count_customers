from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import requests
from streamlit.testing.v1 import AppTest

from busreid.video_output import make_browser_video


def test_dashboard_shows_only_counting_and_roi_by_default(monkeypatch) -> None:
    monkeypatch.delenv("BUSSTOP_SHOW_ADVANCED_TABS", raising=False)

    def offline(*_args, **_kwargs):
        raise requests.ConnectionError("API offline in dashboard fixture")

    monkeypatch.setattr(requests, "request", offline)
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Đếm khách AI", "Thiết lập vùng đếm (ROI)"]
    assert any(button.label == "Bắt đầu đếm" for button in app.button)
    assert not any(button.label == "Chạy AI Counting" for button in app.button)


def test_dashboard_can_restore_advanced_tabs(monkeypatch) -> None:
    monkeypatch.setenv("BUSSTOP_SHOW_ADVANCED_TABS", "1")
    monkeypatch.setattr(requests, "request", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        requests.ConnectionError("API offline in dashboard fixture")
    ))
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Đếm khách AI",
        "Thiết lập vùng đếm (ROI)",
        "Tổng quan chuyến",
        "Hai cửa",
        "Trạm & bản đồ",
        "Sự kiện & kiểm thử",
        "Demo mẫu",
        "Cài đặt",
    ]
    assert any(button.label == "Chạy AI Counting" for button in app.button)


def test_dashboard_renders_completed_annotated_video(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BUSSTOP_SHOW_ADVANCED_TABS", "1")

    def offline(*_args, **_kwargs):
        raise requests.ConnectionError("API offline in dashboard fixture")

    monkeypatch.setattr(requests, "request", offline)
    intermediate = tmp_path / "front_encoding.mp4"
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
    writer.release()
    make_browser_video(intermediate, tmp_path / "front_annotated.mp4", 3)
    (tmp_path / "front_progress.json").write_text(
        '{"status":"COMPLETED","frames":3,"events":0,"counts":{"IN":0,"OUT":0},'
        '"elapsed_s":1.0,"processing_fps":3.0}',
        encoding="utf-8",
    )
    app = AppTest.from_file("app/dashboard.py")
    app.session_state["video_lab_job"] = {
        "process": SimpleNamespace(poll=lambda: 0),
        "output": tmp_path,
        "sources": {"front": "fixture.mp4"},
        "trip_id": None,
    }
    app.run(timeout=20)
    assert not app.exception
    assert len(app.get("video")) == 1
    assert any(metric.label == "Lượt đếm AI" for metric in app.metric)


def test_roi_editor_empty_zone_and_camera_drafts_survive_reruns(tmp_path: Path, monkeypatch) -> None:
    from app import roi_editor

    monkeypatch.setattr(requests, "request", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        requests.ConnectionError("API offline")
    ))
    video = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (100, 100))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(roi_editor, "_select_source", lambda *_args: (str(video), True))
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    assert not app.exception
    next(button for button in app.button if button.label == "Xóa vùng").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["roi_editor_front"]["zones"]["a"] == []
    assert next(button for button in app.button if button.label == "Lưu ROI").disabled
    camera = next(item for item in app.selectbox if item.label == "Camera")
    camera.set_value("rear").run(timeout=20)
    assert not app.exception
    assert app.session_state["roi_editor_front"]["zones"]["a"] == []
    assert app.session_state["roi_editor_rear"]["zones"]["a"]


def test_roi_editor_missing_mp4_shows_error_without_dashboard_exception(tmp_path: Path, monkeypatch) -> None:
    from app import roi_editor

    monkeypatch.setattr(requests, "request", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        requests.ConnectionError("API offline")
    ))
    monkeypatch.setattr(roi_editor, "_select_source", lambda *_args: (str(tmp_path / "missing.mp4"), True))
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    assert not app.exception
    assert any("Không mở/đọc được MP4" in item.value for item in app.error)
