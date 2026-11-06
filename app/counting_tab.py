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
    if camera == "rear" and job.get("rear_mode") == "head_tracking_zone_out":
        path = job["output"] / "rear_head_out_progress.json"
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            progress = {"status": "QUEUED" if returncode is None else "FAILED", "frames": 0}
    else:
        progress = camera_status(job["output"], camera, returncode)
    summary = _summary(job["output"])
    if camera in summary.get("errors", {}):
        progress["status"] = "FAILED"
        progress["error"] = summary["errors"][camera]
    elif returncode is not None and progress["status"] in {"QUEUED", "CANCELLED"}:
        progress["status"] = "FAILED"
    return progress


def _show_preview(path: Path, caption: str) -> bool:
    """A Windows writer can briefly lock a preview during atomic replacement."""
    try:
        data = path.read_bytes()
    except (OSError, PermissionError):
        return False
    if not data:
        return False
    st.image(data, caption=caption)
    return True


def _show_video(path: Path) -> bool:
    try:
        st.video(str(path), format="video/mp4")
    except OSError:
        st.caption("Video đang bị khóa tạm thời; vui lòng thử lại sau.")
        return False
    return True


def _camera_view(job: dict | None, camera: str) -> dict | None:
    title, direction, counter_label = _DIRECTIONS[camera]
    head_mode = camera == "rear" and bool(job and job.get("rear_mode") == "occupancy_headcount")
    zone_mode = camera == "rear" and bool(job and job.get("rear_mode") == "head_tracking_zone_out")
    with st.container(border=True):
        st.subheader("Camera sau – Head Detection (thử nghiệm)" if head_mode else
                     "Camera sau – Head Tracking + Zone OUT (thử nghiệm)" if zone_mode else title)
        progress = _camera_progress(job, camera)
        if progress is None:
            st.info("Chưa kết nối · chưa có kết quả quan sát từ cửa này.")
            return None
        state = progress.get("status", "QUEUED")
        if head_mode:
            st.caption(f"{_STATUS.get(state, state)} · đếm đầu người nhìn thấy trong từng frame, không đếm lượt xuống")
            frames = int(progress.get("frames", 0))
            observed = progress.get("visible_head_count") if frames and state != "FAILED" else None
            st.metric("Số khách quan sát được", observed if observed is not None else "Chưa có kết quả")
            st.caption(
                f"{frames} frame · {float(progress.get('processing_fps', 0)):.1f} FPS xử lý · "
                f"{progress.get('model', '11n-head.pt')} · confidence {progress.get('confidence', 'N/A')}"
            )
            output = job["output"]
            annotated = output / "rear_annotated.mp4"
            preview = output / "rear_preview.jpg"
            if state in {"COMPLETED", "STOPPED"} and annotated.is_file():
                _show_video(annotated)
            elif preview.is_file() and state == "RUNNING":
                _show_preview(preview, "Frame Head Detection mới nhất")
            if progress.get("error"):
                st.error(str(progress["error"]))
            total = progress.get("total_frames")
            if total and state == "RUNNING":
                st.progress(min(frames / total, 1.0))
            csv_path = output / "rear_occupancy.csv"
            if state in {"COMPLETED", "STOPPED"} and csv_path.is_file():
                st.download_button("Tải CSV occupancy", csv_path.read_bytes(),
                                   file_name=f"{output.name}_rear_occupancy.csv", mime="text/csv")
            return progress
        if zone_mode:
            frames = int(progress.get("frames", 0))
            count = progress.get("count") if frames and state != "FAILED" else None
            st.caption(f"{_STATUS.get(state, state)} · đầu đi A → vùng chuyển tiếp → B · kết quả thử nghiệm")
            st.metric("REAR HEAD-OUT", count if count is not None else "Chưa có kết quả")
            st.caption(f"{frames} frame · {float(progress.get('processing_fps', 0)):.1f} FPS xử lý · "
                       f"{progress.get('model', '11n-head.pt')} · {progress.get('tracker', 'bytetrack.yaml')}")
            output = job["output"]
            annotated = output / "rear_head_out_annotated.mp4"
            preview = output / "rear_head_out_preview.jpg"
            if state in {"COMPLETED", "STOPPED"} and annotated.is_file():
                _show_video(annotated)
            elif state == "RUNNING" and preview.is_file():
                _show_preview(preview, "Frame Head Tracking mới nhất")
            if progress.get("error"):
                st.error(str(progress["error"]))
            total = progress.get("total_frames")
            if total and state == "RUNNING":
                st.progress(min(frames / total, 1.0))
            csv_path = output / "rear_head_out_events.csv"
            if state in {"COMPLETED", "STOPPED"} and csv_path.is_file():
                try:
                    st.download_button("Tải CSV Head-OUT", csv_path.read_bytes(),
                                       file_name=f"{output.name}_rear_head_out_events.csv", mime="text/csv")
                except OSError:
                    st.caption("CSV đang được ghi; vui lòng thử lại.")
            return progress
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
            if _show_video(annotated):
                st.caption("Video phát lại đã chú thích · H.264")
        elif preview.is_file():
            _show_preview(preview, "Frame AI đã xử lý · cập nhật theo chu kỳ")
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
    head_mode = bool(job and job.get("rear_mode") == "occupancy_headcount" and "rear" in job["sources"])
    zone_mode = bool(job and job.get("rear_mode") == "head_tracking_zone_out" and "rear" in job["sources"])
    experimental = head_mode or zone_mode
    total_in = int(front.get("counts", {}).get("IN", 0)) if front and front.get("frames", 0) else None
    total_out = int(rear.get("counts", {}).get("OUT", 0)) if rear and rear.get("frames", 0) and not experimental else None
    metrics = st.columns(4 if experimental else 3)
    metrics[0].metric(
        "Tổng khách lên", total_in if total_in is not None else "Chưa có frame" if front else "Chưa kết nối"
    )
    metrics[1].metric(
        "Tổng khách xuống",
        "N/A – Chế độ thử nghiệm" if experimental else total_out if total_out is not None else "Chưa có frame" if rear else "Chưa kết nối",
    )
    metrics[2].metric(
        "Tổng lượt ghi nhận",
        "N/A – thiếu REAR-OUT" if experimental else (total_in or 0) + (total_out or 0)
        if total_in is not None or total_out is not None
        else "Chưa có frame",
    )
    if head_mode:
        visible = rear.get("visible_head_count") if rear and rear.get("frames", 0) and rear.get("status") != "FAILED" else None
        metrics[3].metric("Occupancy quan sát", visible if visible is not None else "Chưa có kết quả")
    if zone_mode:
        count = rear.get("count") if rear and rear.get("frames", 0) and rear.get("status") != "FAILED" else None
        metrics[3].metric("Head-OUT thử nghiệm", count if count is not None else "Chưa có kết quả")
    if not job:
        return
    event_cameras = [camera for camera in job["sources"] if camera != "rear" or not experimental]
    rows = event_timeline(job["output"], event_cameras)
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
    if zone_mode:
        csv_path = job["output"] / "rear_head_out_events.csv"
        if csv_path.is_file():
            try:
                head_rows = pd.read_csv(csv_path)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                head_rows = pd.DataFrame()
            if not head_rows.empty:
                st.markdown("#### Sự kiện REAR Head-OUT thử nghiệm")
                st.dataframe(head_rows, use_container_width=True, hide_index=True)


def render_counting_tab(root: Path) -> None:
    st.subheader("Đếm khách AI")
    st.caption("FRONT: YOLO11n + ByteTrack, chỉ IN · REAR mặc định chỉ OUT; có chế độ Head Detection thử nghiệm riêng · không cần Trip/GPS")
    job = st.session_state.get("ai_counting_job")
    running = bool(job and job["process"].poll() is None)
    inputs = {}
    left, right = st.columns(2)
    for column, camera in zip((left, right), ("front", "rear"), strict=True):
        with column:
            st.markdown(f"#### {_DIRECTIONS[camera][0]}")
            if camera == "rear":
                rear_mode = st.selectbox(
                    "Chế độ xử lý camera sau",
                    ["Đếm khách xuống qua cửa", "Thử nghiệm đếm khách trong xe (Head Detection)",
                     "Thử nghiệm Head Tracking + Zone OUT"],
                    key="counting_rear_mode",
                )
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
            selected_rear_mode = (
                "head_tracking_zone_out" if "rear" in sources and rear_mode == "Thử nghiệm Head Tracking + Zone OUT"
                else "occupancy_headcount" if "rear" in sources and rear_mode.startswith("Thử nghiệm")
                else "door_out"
            )
            process = start_job(
                root,
                output,
                sources,
                max_frames=int(max_frames),
                hidden_layers=hidden,
                session_only=True,
                rear_mode=selected_rear_mode,
            )
        except (OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["ai_counting_job"] = {
                "process": process,
                "output": output,
                "sources": sources,
                "rear_mode": selected_rear_mode,
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
