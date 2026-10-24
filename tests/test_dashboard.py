from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import requests
from streamlit.testing.v1 import AppTest

from busreid.video_output import make_browser_video


def test_dashboard_keeps_six_tabs_and_video_lab_empty_state(monkeypatch) -> None:
    def offline(*_args, **_kwargs):
        raise requests.ConnectionError("API offline in dashboard fixture")

    monkeypatch.setattr(requests, "request", offline)
    app = AppTest.from_file("app/dashboard.py").run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Tổng quan chuyến",
        "Hai cửa",
        "Trạm & bản đồ",
        "Sự kiện & kiểm thử",
        "Demo mẫu",
        "Cài đặt",
    ]
    assert any("không cần GPS hay Trip" in item.value for item in app.info)
    assert any("Chưa chọn nguồn camera" in item.value for item in app.info)
    assert any(button.label == "Chạy AI Counting" for button in app.button)


def test_dashboard_renders_completed_annotated_video(tmp_path: Path, monkeypatch) -> None:
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
