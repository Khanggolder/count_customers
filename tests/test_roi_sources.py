"""Camera failures must not crash the ROI editor or discard an in-progress draft."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app import roi_editor


class FakeCapture:
    def __init__(self, opened=True, frame=None) -> None:
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frame is not None, self.frame

    def get(self, _prop):
        return 0

    def release(self):
        self.released = True


@pytest.mark.parametrize(
    "path,capture",
    [
        (99, FakeCapture(opened=False)),
        (0, FakeCapture()),
        (0, FakeCapture(frame=np.empty((0, 0, 3), dtype=np.uint8))),
        ("rtsp://user:secret@example.invalid/live", FakeCapture(opened=False)),
    ],
)
def test_capture_failure_releases_device_and_does_not_expose_password(monkeypatch, path, capture) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args: capture)
    with pytest.raises(ValueError) as error:
        roi_editor.read_video_frame(path)
    assert capture.released
    assert "secret" not in str(error.value)


def test_rtsp_receives_open_and_read_timeout(monkeypatch) -> None:
    calls = []
    capture = FakeCapture(frame=np.ones((4, 4, 3), dtype=np.uint8))

    def open_capture(*args):
        calls.append(args)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    frame, _, _ = roi_editor.read_video_frame("rtsp://user:secret@example.invalid/live")
    assert frame.shape == (4, 4, 3)
    assert capture.released
    assert calls[0][1] == cv2.CAP_FFMPEG
    assert calls[0][2] == [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000
    ]


def test_rtsp_read_timeout_releases_capture(monkeypatch) -> None:
    class TimedOutCapture(FakeCapture):
        def read(self):
            raise TimeoutError("network timeout")

    capture = TimedOutCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args: capture)
    with pytest.raises(TimeoutError):
        roi_editor.read_video_frame("rtsp://user:secret@example.invalid/live")
    assert capture.released


@pytest.mark.parametrize("mode", ["Webcam", "RTSP"])
@pytest.mark.parametrize("failure", [ValueError("bad"), TimeoutError("timeout")])
def test_source_error_keeps_draft_and_surfaces_vietnamese_message(monkeypatch, mode, failure) -> None:
    errors = []
    state = {"zones": {"a": [(0.1, 0.1)], "b": []}, "mode": mode, "live_frame": "old"}
    monkeypatch.setattr(roi_editor.st, "radio", lambda *_args, **_kwargs: mode)
    monkeypatch.setattr(roi_editor.st, "button", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(roi_editor.st, "number_input", lambda *_args, **_kwargs: 99)
    monkeypatch.setattr(
        roi_editor.st, "text_input", lambda *_args, **_kwargs: "rtsp://user:secret@example.invalid/live"
    )
    monkeypatch.setattr(roi_editor.st, "error", errors.append)
    monkeypatch.setattr(roi_editor.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(roi_editor, "read_video_frame", lambda *_args: (_ for _ in ()).throw(failure))
    roi_editor._select_source(Path(__file__).parents[1], "front", state)
    assert state["zones"]["a"] == [(0.1, 0.1)]
    assert state["live_frame"] == "old"
    assert errors and "secret" not in errors[0]
