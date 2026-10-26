"""Streamlit polygon editor for FRONT/REAR counting zones."""

from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from busreid.config import CameraConfig, load_config
from busreid.pipeline import OverlayOptions, annotate_frame
from busreid.roi import restore_backup, save_zones, to_normalized, to_pixels, validate_polygon, validate_zones
from busreid.video import OpenCVVideoSource

from .video_lab import camera_status, new_job_dir, save_upload, start_job


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
    frame: np.ndarray, camera: CameraConfig, zones: dict[str, list[tuple[float, float]]]
) -> Image.Image:
    height, width = frame.shape[:2]
    scale = min(1.0, 800 / width)
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    height, width = frame.shape[:2]
    shaded = frame.copy()
    complete = {}
    for label, color in (("a", (0, 200, 255)), ("b", (255, 180, 0))):
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
    for label, color in (("a", (0, 200, 255)), ("b", (255, 180, 0))):
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
        configured = load_config(config_path).cameras[camera]
        st.session_state[key] = {
            "zones": {"a": list(configured.zone_a), "b": list(configured.zone_b)},
            "history": [],
            "revision": 0,
            "saved_revision": 0,
            "source": None,
            "upload_identity": None,
        }
    return st.session_state[key]


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
    st.caption(
        "FRONT A → B = IN (+1 occupancy), B → A = OUT (-1); "
        "REAR A → B = OUT (-1), B → A = IN (+1). ROI không phải ground truth."
    )
    config_path = root / "configs" / "cameras.yaml"
    camera = st.selectbox("Camera", ["front", "rear"], format_func=str.upper)
    state = _editor_state(camera, config_path)
    if state["revision"] != state["saved_revision"]:
        st.warning(f"{camera.upper()} có chỉnh sửa ROI chưa lưu; chuyển camera không xóa bản nháp này.")
    source, is_mp4 = _select_source(root, camera, state)
    frame = None
    frame_index = 0
    frame_count = 0
    if is_mp4 and source:
        try:
            _, frame_count, fps = read_video_frame(source)
            if source != state.get("display_source"):
                state["display_source"] = source
                st.session_state.pop(f"roi_frame_{camera}", None)
            frame_index = st.slider("Frame MP4", 0, max(0, frame_count - 1), 0, key=f"roi_frame_{camera}")
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
        "Vùng chỉnh sửa", ["a", "b"], format_func=lambda value: f"Zone {value.upper()}", horizontal=True
    )
    action = st.radio("Thao tác trên ảnh", ["Thêm đỉnh", "Dời đỉnh"], horizontal=True)
    points = state["zones"][zone]
    vertex = 0
    if points:
        vertex = st.selectbox(
            "Đỉnh đang chọn", list(range(len(points))), format_func=lambda i: f"{zone.upper()}{i + 1}"
        )
    picture = _display_frame(frame, load_config(config_path).cameras[camera], state["zones"])
    st.caption("Click để thêm đỉnh; ở chế độ dời, kéo từ đỉnh sang vị trí mới hoặc chọn đỉnh rồi click đích.")
    click = streamlit_image_coordinates(
        picture,
        key=f"roi_canvas_{camera}_{frame_index}_{state['revision']}_{zone}_{action}",
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
        configured = load_config(config_path).cameras[camera]
        _mutate(state, {"a": list(configured.zone_a), "b": list(configured.zone_b)})
        st.rerun()
    height, width = frame.shape[:2]
    for label in ("a", "b"):
        rows = [
            {
                "đỉnh": i + 1,
                "pixel": to_pixels(point, width, height),
                "normalized": tuple(round(v, 5) for v in point),
            }
            for i, point in enumerate(state["zones"][label])
        ]
        st.write(f"Zone {label.upper()}")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    try:
        validate_zones(state["zones"]["a"], state["zones"]["b"])
        valid = True
        st.success("Hai polygon hợp lệ, không tự cắt và không chồng lấn.")
    except ValueError as exc:
        valid = False
        st.warning(f"Chưa thể lưu: {exc}")
    left, right = st.columns(2)
    if left.button("Lưu ROI", type="primary", disabled=not valid):
        try:
            backup = save_zones(config_path, camera, state["zones"]["a"], state["zones"]["b"])
            state["saved_revision"] = state["revision"]
            st.success(f"Đã lưu. Bản sao lưu: {backup.name}. Phiên inference mới sẽ dùng ROI này.")
        except (OSError, ValueError) as exc:
            st.error(f"Không lưu được ROI: {exc}")
    if right.button("Khôi phục ROI camera này từ bản sao lưu"):
        try:
            restore_backup(config_path, camera)
            configured = load_config(config_path).cameras[camera]
            _mutate(state, {"a": list(configured.zone_a), "b": list(configured.zone_b)})
            state["saved_revision"] = state["revision"]
            st.rerun()
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.error(str(exc))
    st.markdown("#### Test ROI trên đoạn MP4")
    st.caption(
        "Chạy YOLO11n + ByteTrack thật trên đoạn video đã chọn; event là output AI, không phải ground truth."
    )
    if state["revision"] != state["saved_revision"]:
        st.info("Lưu ROI trước khi Test ROI để worker dùng đúng vùng đang hiển thị.")
    use_current = st.checkbox("Bắt đầu tại frame đang hiển thị", value=True, disabled=not is_mp4)
    start_frame = frame_index if use_current else st.number_input(
        "Frame bắt đầu (0-based)", min_value=0, max_value=max(0, frame_count - 1), value=0
    )
    limit = st.number_input("Số frame kiểm thử", min_value=1, max_value=600, value=90)
    if is_mp4 and frame_count:
        st.caption(f"Sẽ xử lý frame {start_frame}–{min(frame_count - 1, start_frame + int(limit) - 1)}.")
    current_job = st.session_state.get("roi_test_job")
    busy = bool(current_job and current_job["process"].poll() is None)
    if st.button(
        "Test ROI",
        disabled=busy or not (valid and is_mp4 and source and state["revision"] == state["saved_revision"]),
    ):
        output = new_job_dir(root, kind="ai_counting")
        process = start_job(
            root, output, {camera: source}, max_frames=int(limit), start_frame=int(start_frame), session_only=True
        )
        st.session_state["roi_test_job"] = {"process": process, "output": output, "camera": camera}
        st.rerun()
    job = st.session_state.get("roi_test_job")
    if job:
        progress = camera_status(job["output"], job["camera"], job["process"].poll())
        st.write(
            f"{job['camera'].upper()}: {progress['status']} · {progress['frames']} frame · {progress['events']} event"
        )
        preview = job["output"] / f"{job['camera']}_preview.jpg"
        if preview.is_file():
            st.image(str(preview), caption="Preview inference thật")
        events = job["output"] / f"{job['camera']}_events.csv"
        if events.is_file():
            st.download_button("Tải crossing CSV", events.read_bytes(), events.name, "text/csv")
        video = job["output"] / f"{job['camera']}_annotated.mp4"
        if video.is_file():
            st.video(str(video))
        if job["process"].poll() is None:
            st.button("Cập nhật tiến độ")
