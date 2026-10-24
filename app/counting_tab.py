"""Simple, trip-independent Streamlit view over the real counting workers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from .video_lab import camera_status, event_timeline, first_frame, new_job_dir, resolve_source, start_job

_DIRECTIONS = {
    "front": ("Cửa trước – Khách lên", "IN", "Khách đã lên"),
    "rear": ("Cửa sau – Khách xuống", "OUT", "Khách đã xuống"),
}
_STATUS = {
    "QUEUED": "Đang khởi động",
    "RUNNING": "Đang xử lý AI",
    "COMPLETED": "Hoàn tất",
    "STOPPED": "Đã dừng",
    "FAILED": "Lỗi",
}


def _source_picker(camera: str) -> tuple[object | None, str, str, int | None]:
    mode = st.selectbox(
        "Nguồn hình ảnh",
        ["Chưa kết nối", "Tải MP4 lên", "Đường dẫn MP4", "Webcam", "RTSP"],
        key=f"counting_mode_{camera}",
    )
    upload = (
        st.file_uploader("Chọn video MP4", type="mp4", key=f"counting_upload_{camera}")
        if mode == "Tải MP4 lên"
        else None
    )
    path = st.text_input("Đường dẫn MP4", key=f"counting_path_{camera}") if mode == "Đường dẫn MP4" else ""
    webcam = (
        int(st.number_input("Chỉ số webcam", min_value=0, value=0, key=f"counting_webcam_{camera}"))
        if mode == "Webcam"
        else None
    )
    rtsp = (
        st.text_input("Địa chỉ RTSP", type="password", key=f"counting_rtsp_{camera}")
        if mode == "RTSP"
        else ""
    )
    if mode == "Đường dẫn MP4" and path:
        preview = first_frame(path)
        if preview:
            st.image(preview, caption="Frame đầu của video phát lại")
        else:
            st.warning("Chưa đọc được video MP4 này.")
    elif mode == "Webcam":
        st.caption("Webcam phải gắn với máy đang chạy Streamlit.")
    elif mode == "RTSP":
        st.caption("RTSP là nguồn trực tiếp; không lưu mật khẩu trong file cấu hình.")
    elif mode == "Chưa kết nối":
        st.info("Chọn video hoặc camera để bắt đầu.")
    return upload, path, rtsp, webcam


def _summary(output: Path) -> dict:
    path = output / "session_summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _camera_progress(job: dict | None, camera: str) -> dict | None:
    if not job or camera not in job["sources"]:
        return None
    returncode = job["process"].poll()
    progress = camera_status(job["output"], camera, returncode)
    summary = _summary(job["output"])
    if camera in summary.get("errors", {}):
        progress["status"] = "FAILED"
        progress["error"] = summary["errors"][camera]
    elif returncode is not None and progress["status"] in {"QUEUED", "CANCELLED"}:
        progress["status"] = "FAILED"
    return progress


def _camera_view(job: dict | None, camera: str) -> dict | None:
    title, direction, counter_label = _DIRECTIONS[camera]
    with st.container(border=True):
        st.subheader(title)
        progress = _camera_progress(job, camera)
        if progress is None:
            st.info("Chưa kết nối · chưa có kết quả quan sát từ cửa này.")
            return None
        state = progress.get("status", "QUEUED")
        st.caption(f"{_STATUS.get(state, state)} · chỉ đếm {direction} theo A → B")
        count = int(progress.get("counts", {}).get(direction, 0))
        has_frames = int(progress.get("frames", 0)) > 0
        st.metric(counter_label, count if has_frames else "Chưa có frame")
        output = job["output"]
        recent = event_timeline(output, [camera])
        if recent and recent[-1]["created_at"]:
            try:
                event_time = datetime.fromisoformat(recent[-1]["created_at"])
                if 0 <= (datetime.now(timezone.utc) - event_time).total_seconds() <= 5:
                    st.success(f"{direction} +1 · Track {recent[-1]['track_id']}")
            except (TypeError, ValueError):
                pass
        preview = output / f"{camera}_preview.jpg"
        annotated = output / f"{camera}_annotated.mp4"
        if state in {"COMPLETED", "STOPPED"} and annotated.is_file():
            st.video(str(annotated), format="video/mp4")
            st.caption("Video phát lại đã chú thích · H.264")
        elif preview.is_file():
            st.image(preview.read_bytes(), caption="Frame AI đã xử lý · cập nhật theo chu kỳ")
        elif state == "FAILED":
            st.error("Không mở hoặc xử lý được nguồn hình ảnh này.")
        else:
            st.info("Đang chờ frame AI đầu tiên...")
        if progress.get("error"):
            st.error(str(progress["error"]))
        frames = int(progress.get("frames", 0))
        fps = float(progress.get("processing_fps", 0))
        st.caption(f"{frames} frame đã xử lý · {fps:.1f} FPS xử lý")
        total = progress.get("total_frames")
        if total and state == "RUNNING":
            st.progress(min(frames / total, 1.0))
        if state in {"COMPLETED", "STOPPED"}:
            csv_path = output / f"{camera}_events.csv"
            if csv_path.is_file():
                st.download_button(
                    "Tải CSV sự kiện",
                    csv_path.read_bytes(),
                    file_name=f"{output.name}_{camera}_events.csv",
                    mime="text/csv",
                    key=f"counting_csv_{camera}_{output.name}",
                )
        return progress


def _results(job: dict | None) -> None:
    left, right = st.columns(2)
    with left:
        front = _camera_view(job, "front")
    with right:
        rear = _camera_view(job, "rear")
    total_in = int(front.get("counts", {}).get("IN", 0)) if front and front.get("frames", 0) else None
    total_out = int(rear.get("counts", {}).get("OUT", 0)) if rear and rear.get("frames", 0) else None
    metrics = st.columns(3)
    metrics[0].metric(
        "Tổng khách lên", total_in if total_in is not None else "Chưa có frame" if front else "Chưa kết nối"
    )
    metrics[1].metric(
        "Tổng khách xuống",
        total_out if total_out is not None else "Chưa có frame" if rear else "Chưa kết nối",
    )
    metrics[2].metric(
        "Tổng lượt ghi nhận",
        (total_in or 0) + (total_out or 0)
        if total_in is not None or total_out is not None
        else "Chưa có frame",
    )
    if not job:
        return
    rows = event_timeline(job["output"], list(job["sources"]))
    if rows:
        st.markdown("#### Sự kiện đi qua cửa")
        st.dataframe(
            pd.DataFrame(rows)[["camera", "timestamp", "frame_index", "direction", "track_id"]],
            use_container_width=True,
            hide_index=True,
        )
        identities = [(row["camera"], row["track_id"]) for row in rows]
        if len(identities) != len(set(identities)):
            st.warning("Có Track ID được đếm nhiều lần; cần xem lại video/ROI trước khi kết luận số khách.")
    elif job["process"].poll() is not None:
        st.info("Chưa phát hiện lượt đi qua cửa hợp lệ.")


def render_counting_tab(root: Path) -> None:
    st.subheader("Đếm khách AI")
    st.caption("YOLO11n + ByteTrack · FRONT chỉ IN · REAR chỉ OUT · không cần Trip, GPS hoặc telemetry")
    job = st.session_state.get("ai_counting_job")
    running = bool(job and job["process"].poll() is None)
    inputs = {}
    left, right = st.columns(2)
    for column, camera in zip((left, right), ("front", "rear"), strict=True):
        with column:
            st.markdown(f"#### {_DIRECTIONS[camera][0]}")
            inputs[camera] = _source_picker(camera)

    with st.expander("Tùy chọn hiển thị và chạy", expanded=False):
        layers = {
            "boxes": "Khung người",
            "track-ids": "Track ID và độ tin cậy",
            "zones": "Vùng đếm A/B",
            "trajectory": "Quỹ đạo",
            "events": "Thông báo IN/OUT +1",
        }
        hidden = {
            key
            for key, label in layers.items()
            if not st.checkbox(label, value=True, key=f"counting_layer_{key}")
        }
        max_frames = st.number_input(
            "Giới hạn frame mỗi cửa (0 = hết MP4 / chạy đến khi dừng)",
            min_value=0,
            value=0,
            key="counting_max_frames",
        )
        st.caption("Zone A/B lấy từ cấu hình camera; hiệu chỉnh theo đúng góc quay trước khi đánh giá đếm.")

    buttons = st.columns(3)
    start = buttons[0].button("Bắt đầu đếm", type="primary", disabled=running)
    stop = buttons[1].button("Dừng đếm", disabled=not running)
    reset = buttons[2].button("Đặt lại bộ đếm", disabled=running or not job)
    if stop and job:
        (job["output"] / "stop.request").touch()
        st.warning("Đã yêu cầu dừng. Worker sẽ đóng video sau frame hiện tại.")
    if reset:
        # Results remain on disk for audit; only the visible session is cleared.
        st.session_state.pop("ai_counting_job", None)
        st.rerun()
    if start:
        output = new_job_dir(root, "ai_counting")
        try:
            sources = {}
            for camera, (upload, path, rtsp, webcam) in inputs.items():
                source = resolve_source(
                    upload, getattr(upload, "name", None), path, rtsp, output / "uploads", webcam
                )
                if source:
                    if source.lower().endswith(".mp4") and first_frame(source) is None:
                        raise ValueError(f"{camera.upper()}: không đọc được frame MP4 đầu tiên")
                    sources[camera] = source
            process = start_job(
                root,
                output,
                sources,
                max_frames=int(max_frames),
                hidden_layers=hidden,
                session_only=True,
            )
        except (OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["ai_counting_job"] = {
                "process": process,
                "output": output,
                "sources": sources,
            }
            st.rerun()

    job = st.session_state.get("ai_counting_job")
    if not job:
        st.info("Chọn ít nhất một MP4, webcam hoặc RTSP rồi nhấn “Bắt đầu đếm”.")
        _results(None)
        return
    if job["process"].poll() is None:
        st.info("Đang xử lý · với MP4 đây là video phát lại, không phải camera live.")

        @st.fragment(run_every="2s")
        def monitor() -> None:
            _results(job)
            if job["process"].poll() is not None:
                st.rerun()

        monitor()
    else:
        summary = _summary(job["output"])
        if summary.get("errors"):
            st.warning("Một hoặc nhiều camera gặp lỗi; kết quả camera còn lại vẫn được giữ.")
        elif summary.get("status") == "STOPPED":
            st.warning("Phiên đã dừng và lưu kết quả đến frame cuối được xử lý.")
        elif job["process"].returncode == 0:
            st.success("Phiên đếm hoàn tất.")
        else:
            st.error("Phiên xử lý thất bại. Kiểm tra nguồn video, model và process.log trong thư mục phiên.")
        _results(job)
        st.caption(f"Phiên độc lập: {job['output'].name} · kết quả AI, không gán Trip/trạm · accuracy N/A")
