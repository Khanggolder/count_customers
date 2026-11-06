"""Streamlit polygon editor for person and experimental head crossing zones."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from busreid.config import CameraConfig, load_config
from busreid.pipeline import OverlayOptions, annotate_frame
from busreid.roi import (
    restore_backup,
    restore_head_out_backup,
    save_head_out_zones,
    save_zones,
    to_normalized,
    to_pixels,
    validate_head_out_zones,
    validate_polygon,
    validate_zones,
)
from busreid.video import OpenCVVideoSource

from .counting_tab import _show_preview, _show_video
from .video_lab import camera_status, new_job_dir, save_upload, start_job

_HEAD_TARGET = "rear_head_out"
_FLOW_TARGET = "rear_flow_out"
_LABELS = {"front": "FRONT – Person Detection", "rear": "REAR – Person Detection",
           _HEAD_TARGET: "REAR – Head Tracking + Zone OUT",
           _FLOW_TARGET: "REAR – Optical Flow Counting"}
_COLORS = {"roi": (180, 255, 180), "a": (0, 200, 255), "b": (255, 180, 0)}


def read_video_frame(path: str | int, frame_index: int = 0) -> tuple[np.ndarray, int, float]:
    if isinstance(path, str) and not path.lower().startswith(("rtsp://", "rtsps://")):
        source = OpenCVVideoSource(path)
        try:
            metadata = source.open()
            source.seek(frame_index)
            ok, frame, _ = source.read()
            if not ok or frame is None or frame.size == 0:
                raise ValueError("Không đọc được frame đã chọn")
            return frame, metadata.frame_count or 0, metadata.fps
        finally:
            source.close()
    capture = (
        cv2.VideoCapture(
            path,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )
        if isinstance(path, str) and path.lower().startswith(("rtsp://", "rtsps://"))
        else cv2.VideoCapture(path)
    )
    try:
        if not capture.isOpened():
            raise ValueError("Không mở được nguồn video/camera")
        count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            raise ValueError("Không đọc được frame đã chọn")
        return frame, count, fps
    finally:
        capture.release()


def _display_frame(
    frame: np.ndarray, camera: CameraConfig, zones: dict[str, list[tuple[float, float]]],
    *, head_mode: bool = False,
) -> Image.Image:
    height, width = frame.shape[:2]
    scale = min(1.0, 800 / width)
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    height, width = frame.shape[:2]
    shaded = frame.copy()
    complete = {}
    labels = ("roi", "a", "b") if head_mode else ("a", "b")
    for label in labels:
        color = _COLORS[label]
        points = zones[label]
        try:
            validate_polygon(points)
            complete[label] = True
        except ValueError:
            complete[label] = False
        if complete[label]:
            contour = np.asarray([to_pixels(p, width, height) for p in points], dtype=np.int32)
            cv2.fillPoly(shaded, [contour], color)
    frame = cv2.addWeighted(shaded, 0.2, frame, 0.8, 0)
    if head_mode:
        for label in labels:
            if complete[label]:
                contour = np.asarray([to_pixels(p, width, height) for p in zones[label]], dtype=np.int32)
                cv2.polylines(frame, [contour], True, _COLORS[label], 2)
                center = np.mean(contour, axis=0).astype(int)
                cv2.putText(frame, "ROI" if label == "roi" else f"ZONE {label.upper()}",
                            tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLORS[label], 2)
        cv2.putText(frame, "HEAD OUT: A -> TRANSITION -> B", (8, max(20, height - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        preview_camera = copy.copy(camera)
        preview_camera.zone_a = zones["a"] if complete["a"] else []
        preview_camera.zone_b = zones["b"] if complete["b"] else []
        frame = annotate_frame(
            frame,
            [],
            preview_camera,
            {"IN": 0, "OUT": 0},
            options=OverlayOptions(boxes=False, track_ids=False, trajectory=False, events=False),
        )
    for label in labels:
        color = _COLORS[label]
        points = zones[label]
        contour = np.asarray([to_pixels(p, width, height) for p in points], dtype=np.int32)
        if len(contour) >= 2 and not complete[label]:
            cv2.polylines(frame, [contour], False, color, 2)
        for index, point in enumerate(points):
            x, y = to_pixels(point, width, height)
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.putText(frame, str(index + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if complete["a"] and complete["b"]:
        start = np.mean([to_pixels(p, width, height) for p in zones["a"]], axis=0).astype(int)
        end = np.mean([to_pixels(p, width, height) for p in zones["b"]], axis=0).astype(int)
        cv2.arrowedLine(frame, tuple(start), tuple(end), (255, 255, 255), 2, tipLength=0.18)
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _editor_state(camera: str, config_path: Path) -> dict:
    key = f"roi_editor_{camera}"
    if key not in st.session_state:
        st.session_state[key] = {
            "zones": _configured_zones(camera, config_path),
            "history": [],
            "revision": 0,
            "saved_revision": 0,
            "source": None,
            "upload_identity": None,
        }
    return st.session_state[key]


def _configured_zones(target: str, config_path: Path) -> dict[str, list[tuple[float, float]]]:
    config = load_config(config_path)
    configured = config.rear_head_out if target == _HEAD_TARGET else config.cameras[target]
    zones = {"a": list(configured.zone_a), "b": list(configured.zone_b)}
    if target == _HEAD_TARGET:
        zones = {"roi": list(configured.roi), **zones}
    return zones


def _validate_editor_zones(target: str, zones: dict[str, list[tuple[float, float]]]) -> None:
    if target == _HEAD_TARGET:
        validate_head_out_zones(zones["roi"], zones["a"], zones["b"])
    else:
        validate_zones(zones["a"], zones["b"])


def _mutate(state: dict, zones: dict[str, list[tuple[float, float]]]) -> None:
    state["history"].append(copy.deepcopy(state["zones"]))
    state["zones"] = zones
    state["revision"] += 1


def _select_source(root: Path, camera: str, state: dict) -> tuple[str | int | None, bool]:
    mode = st.radio(
        "Nguồn frame", ["Video có sẵn", "Upload MP4", "Đường dẫn MP4", "Webcam", "RTSP"],
        horizontal=True, key=f"roi_mode_{camera}",
    )
    if mode != state.get("mode"):
        state.pop("live_frame", None)
        state["mode"] = mode
    if mode == "Video có sẵn":
        paths = sorted((root / "data" / "raw").glob("*.mp4"))
        source = st.selectbox(
            "Video trong data/raw", [None, *paths], format_func=lambda p: p.name if p else "Chọn video"
        )
        return (str(source), True) if source else (None, True)
    if mode == "Upload MP4":
        upload = st.file_uploader("Tải MP4", type="mp4", key=f"roi_upload_{camera}")
        if upload is None:
            return None, True
        identity = (upload.name, upload.size, getattr(upload, "file_id", None))
        if identity != state["upload_identity"]:
            state["source"] = str(save_upload(upload, upload.name, root / "data" / "raw" / "roi_uploads"))
            state["upload_identity"] = identity
        return state["source"], True
    if mode == "Đường dẫn MP4":
        value = st.text_input("Đường dẫn MP4 trên máy chạy Streamlit", key=f"roi_path_{camera}").strip()
        path = Path(value) if value else None
        if path and (not path.is_file() or path.suffix.lower() != ".mp4"):
            st.warning("Đường dẫn phải trỏ đến file MP4 có thật.")
            return None, True
        return str(path) if path else None, True
    if mode == "Webcam":
        index = int(st.number_input("Webcam index trên máy chạy Streamlit", min_value=0, step=1))
        if st.button("Lấy frame webcam"):
            try:
                state["live_frame"] = read_video_frame(index)[0]
            except (ValueError, OSError, RuntimeError, cv2.error):
                st.error("Không mở/đọc được webcam. Kiểm tra index, quyền truy cập và camera đang dùng.")
                if "live_frame" in state:
                    st.info("Đang giữ frame cũ và bản nháp ROI; chưa lấy được frame webcam mới.")
        return index, False
    url = st.text_input("RTSP được cấp phép", type="password", key=f"roi_rtsp_{camera}")
    if st.button("Lấy frame RTSP"):
        if not url.lower().startswith(("rtsp://", "rtsps://")):
            st.error("URL phải bắt đầu bằng rtsp:// hoặc rtsps://")
        else:
            try:
                state["live_frame"] = read_video_frame(url)[0]
            except (ValueError, OSError, RuntimeError, cv2.error):
                st.error("Không kết nối/đọc được RTSP trong 5 giây. Kiểm tra URL, quyền truy cập và mạng.")
                if "live_frame" in state:
                    st.info("Đang giữ frame cũ và bản nháp ROI; chưa lấy được frame RTSP mới.")
    st.caption("Chỉ lấy một frame khi bấm nút; không lưu RTSP credential vào YAML.")
    return url if url else None, False


def _apply_click(
    state: dict, zone: str, action: str, click: dict, size: tuple[int, int], vertex: int
) -> None:
    width, height = size
    coords = click.get("x2", click.get("x")), click.get("y2", click.get("y"))
    if any(value is None for value in coords):
        return
    try:
        point = to_normalized((int(coords[0]), int(coords[1])), width, height)
    except ValueError as exc:
        st.error(str(exc))
        return
    zones = copy.deepcopy(state["zones"])
    points = zones[zone]
    if action == "Thêm đỉnh":
        points.append(point)
    elif action == "Dời đỉnh" and points:
        start = (click.get("x1"), click.get("y1"))
        nearest = vertex
        if all(value is not None for value in start):
            distances = [
                sum((a - b) ** 2 for a, b in zip(to_pixels(p, width, height), start, strict=False))
                for p in points
            ]
            if min(distances) <= 15**2:
                nearest = distances.index(min(distances))
        points[nearest] = point
    else:
        return
    _mutate(state, zones)


def render_roi_editor(root: Path) -> None:
    st.subheader("Thiết lập vùng đếm (ROI)")
    st.caption("FRONT Person: A → B = IN · REAR Person: A → B = OUT · REAR Head: A → Transition → B = OUT.")
    config_path = root / "configs" / "cameras.yaml"
    target = st.selectbox("Camera", ["front", "rear", _HEAD_TARGET, _FLOW_TARGET],
                          format_func=lambda value: _LABELS[value])
    if target == _FLOW_TARGET:
        from .flow_roi_editor import render_flow_roi_editor

        render_flow_roi_editor(root)
        return
    camera = "rear" if target == _HEAD_TARGET else target
    head_mode = target == _HEAD_TARGET
    state = _editor_state(target, config_path)
    if state["revision"] != state["saved_revision"]:
        st.warning(f"{_LABELS[target]} có chỉnh sửa ROI chưa lưu; chuyển chế độ không xóa bản nháp này.")
    source, is_mp4 = _select_source(root, target, state)
    frame = None
    frame_index = 0
    frame_count = 0
    if is_mp4 and source:
        try:
            _, frame_count, fps = read_video_frame(source)
            if source != state.get("display_source"):
                state["display_source"] = source
                st.session_state.pop(f"roi_frame_{target}", None)
            frame_index = st.slider("Frame MP4", 0, max(0, frame_count - 1), 0, key=f"roi_frame_{target}")
            frame = read_video_frame(source, frame_index)[0]
            st.caption(f"Frame {frame_index + 1}/{frame_count or '?'} · {fps:.2f} FPS")
        except (ValueError, OSError, RuntimeError, cv2.error):
            st.error("Không mở/đọc được MP4 hoặc frame đã chọn. Kiểm tra video và vị trí frame.")
    elif not is_mp4:
        frame = state.get("live_frame")
    if frame is None:
        st.info("Chọn MP4 hoặc lấy frame camera để bắt đầu vẽ ROI.")
        return
    zone = st.radio(
        "Vùng chỉnh sửa", ["roi", "a", "b"] if head_mode else ["a", "b"],
        format_func=lambda value: "ROI – vùng quan sát" if value == "roi" else f"Zone {value.upper()}",
        horizontal=True,
    )
    action = st.radio("Thao tác trên ảnh", ["Thêm đỉnh", "Dời đỉnh"], horizontal=True)
    points = state["zones"][zone]
    vertex = 0
    if points:
        vertex = st.selectbox(
            "Đỉnh đang chọn", list(range(len(points))), format_func=lambda i: f"{zone.upper()}{i + 1}"
        )
    picture = _display_frame(frame, load_config(config_path).cameras[camera], state["zones"],
                             head_mode=head_mode)
    st.caption("Click để thêm đỉnh; ở chế độ dời, kéo từ đỉnh sang vị trí mới hoặc chọn đỉnh rồi click đích.")
    click = streamlit_image_coordinates(
        picture,
        key=f"roi_canvas_{target}_{frame_index}_{state['revision']}_{zone}_{action}",
        click_and_drag=action == "Dời đỉnh",
        cursor="crosshair",
    )
    if click:
        _apply_click(state, zone, action, click, picture.size, vertex)
        st.rerun()
    buttons = st.columns(4)
    if buttons[0].button("Xóa đỉnh đã chọn", disabled=not points):
        zones = copy.deepcopy(state["zones"])
        zones[zone].pop(vertex)
        _mutate(state, zones)
        st.rerun()
    if buttons[1].button("Hoàn tác", disabled=not state["history"]):
        state["zones"] = state["history"].pop()
        state["revision"] += 1
        st.rerun()
    if buttons[2].button("Xóa vùng"):
        zones = copy.deepcopy(state["zones"])
        zones[zone] = []
        _mutate(state, zones)
        st.rerun()
    if buttons[3].button("Nạp lại cấu hình"):
        _mutate(state, _configured_zones(target, config_path))
        st.rerun()
    height, width = frame.shape[:2]
    for label in (("roi", "a", "b") if head_mode else ("a", "b")):
        rows = [
            {
                "đỉnh": i + 1,
                "pixel": to_pixels(point, width, height),
                "normalized": tuple(round(v, 5) for v in point),
            }
            for i, point in enumerate(state["zones"][label])
        ]
        st.write("ROI – vùng quan sát" if label == "roi" else f"Zone {label.upper()}")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    try:
        _validate_editor_zones(target, state["zones"])
        valid = True
        st.success("Các polygon hợp lệ, nằm đúng ROI và có vùng chuyển tiếp." if head_mode
                   else "Hai polygon hợp lệ, không tự cắt và không chồng lấn.")
    except ValueError as exc:
        valid = False
        st.warning(f"Chưa thể lưu: {exc}")
    left, right = st.columns(2)
    if left.button("Lưu ROI", type="primary", disabled=not valid):
        try:
            backup = (
                save_head_out_zones(config_path, state["zones"]["roi"],
                                    state["zones"]["a"], state["zones"]["b"])
                if head_mode else save_zones(config_path, camera, state["zones"]["a"], state["zones"]["b"])
            )
            state["saved_revision"] = state["revision"]
            st.success(f"Đã lưu. Bản sao lưu: {backup.name}. Phiên inference mới sẽ dùng ROI này.")
        except (OSError, ValueError) as exc:
            st.error(f"Không lưu được ROI: {exc}")
    if right.button("Khôi phục ROI Head Tracking từ bản sao lưu" if head_mode
                    else "Khôi phục ROI camera này từ bản sao lưu"):
        try:
            if head_mode:
                restore_head_out_backup(config_path)
            else:
                restore_backup(config_path, camera)
            _mutate(state, _configured_zones(target, config_path))
            state["saved_revision"] = state["revision"]
            st.rerun()
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.error(str(exc))
    st.markdown("#### Test Head Tracking ROI" if head_mode else "#### Test ROI trên đoạn MP4")
    st.caption("Chạy 11n-head.pt + ByteTrack + HeadZoneCounter trên nguồn đã chọn; "
               "Head-OUT là output AI, không phải ground truth." if head_mode else
               "Chạy YOLO11n + ByteTrack thật trên đoạn video đã chọn; "
               "event là output AI, không phải ground truth.")
    if state["revision"] != state["saved_revision"]:
        st.warning("Bản nháp chưa lưu. Hãy Lưu ROI trước khi Test để worker dùng đúng vùng đang hiển thị.")
    use_current = st.checkbox("Bắt đầu tại frame đang hiển thị", value=True, disabled=not is_mp4)
    start_frame = frame_index if use_current else st.number_input(
        "Frame bắt đầu (0-based)", min_value=0, max_value=max(0, frame_count - 1), value=0
    )
    limit = st.number_input("Số frame kiểm thử", min_value=1, max_value=600, value=90)
    if is_mp4 and frame_count:
        st.caption(f"Sẽ xử lý frame {start_frame}–{min(frame_count - 1, start_frame + int(limit) - 1)}.")
    job_key = f"roi_test_job_{target}"
    current_job = st.session_state.get(job_key)
    busy = bool(current_job and current_job["process"].poll() is None)
    if st.button(
        "Test Head Tracking ROI" if head_mode else "Test ROI",
        disabled=busy or not (valid and frame is not None and source is not None
                              and state["revision"] == state["saved_revision"]),
    ):
        output = new_job_dir(root, kind="ai_counting")
        try:
            process = start_job(
                root, output, {camera: f"webcam:{source}" if isinstance(source, int) else source},
                max_frames=int(limit), start_frame=int(start_frame) if is_mp4 else 0,
                session_only=True, rear_mode="head_tracking_zone_out" if head_mode else "door_out",
            )
        except (OSError, ValueError) as exc:
            st.error(f"Không bắt đầu được Test ROI: {exc}")
        else:
            st.session_state[job_key] = {
                "process": process, "output": output, "camera": camera, "head_mode": head_mode,
            }
            st.rerun()
    job = st.session_state.get(job_key)
    if job:
        if head_mode:
            progress_path = job["output"] / "rear_head_out_progress.json"
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = {"status": "QUEUED" if job["process"].poll() is None else "FAILED", "frames": 0}
            st.write(f"REAR Head Tracking: {progress['status']} · {progress.get('frames', 0)} frame")
            columns = st.columns(4)
            for column, (label, field) in zip(columns, (
                ("Head-OUT", "count"), ("Detection", "detected_heads"),
                ("Track", "tracked_heads"), ("FPS xử lý", "processing_fps"),
            ), strict=True):
                value = progress.get(field)
                column.metric(label, round(value, 2) if field == "processing_fps" and value is not None
                              else value if value is not None else "N/A")
            st.caption(
                f"Xác nhận A: {progress.get('stable_a_confirmations', 'N/A')} · "
                f"Transition: {progress.get('transition_confirmations', 'N/A')} · "
                f"B: {progress.get('stable_b_confirmations', 'N/A')}"
            )
            st.caption("Detection và Track là tổng box quan sát qua các frame, không phải số người duy nhất.")
            total_frames = progress.get("total_frames")
            if total_frames and progress["status"] == "RUNNING":
                st.progress(min(progress.get("frames", 0) / total_frames, 1.0))
            stem = "rear_head_out"
            events = job["output"] / "rear_head_out_events.csv"
        else:
            progress = camera_status(job["output"], job["camera"], job["process"].poll())
            st.write(
                f"{job['camera'].upper()}: {progress['status']} · {progress['frames']} frame · "
                f"{progress['events']} event"
            )
            stem = job["camera"]
            events = job["output"] / f"{stem}_events.csv"
        if progress.get("error"):
            st.error(str(progress["error"]))
        preview = job["output"] / f"{stem}_preview.jpg"
        if preview.is_file():
            _show_preview(preview, "Preview inference thật")
        if events.is_file():
            try:
                st.download_button("Tải crossing CSV", events.read_bytes(), events.name, "text/csv")
            except OSError:
                st.caption("CSV đang được ghi; thử lại sau.")
        video = job["output"] / f"{stem}_annotated.mp4"
        if video.is_file():
            _show_video(video)
        if job["process"].poll() is None:
            st.button("Cập nhật tiến độ")
