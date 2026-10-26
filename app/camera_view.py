from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from .video_lab import camera_status, event_timeline, first_frame, new_job_dir, resolve_source, start_job


def _source_input(camera: str, root: Path) -> tuple[object | None, str, str]:
    label = "FRONT · Boarding / IN" if camera == "front" else "REAR · Alighting / OUT"
    st.markdown(f"#### {label}")
    public_clip = root / "data" / "raw" / "pexels_bus_boarding_31337217.mp4"
    initial = "Đường dẫn MP4" if camera == "front" and public_clip.is_file() else "Không dùng"
    modes = ["Không dùng", "Upload MP4", "Đường dẫn MP4", "RTSP"]
    mode = st.selectbox("Nguồn video", modes, index=modes.index(initial), key=f"source_mode_{camera}")
    upload = st.file_uploader("Tải MP4", type="mp4", key=f"upload_{camera}") if mode == "Upload MP4" else None
    path = (
        st.text_input(
            "Đường dẫn MP4 trên máy",
            value=str(public_clip) if public_clip.is_file() else "",
            key=f"path_{camera}",
        )
        if mode == "Đường dẫn MP4"
        else ""
    )
    rtsp = st.text_input("RTSP URL", type="password", key=f"rtsp_{camera}") if mode == "RTSP" else ""
    st.caption("FRONT A → B: IN, B → A: OUT · REAR A → B: OUT, B → A: IN")
    if mode == "Không dùng":
        st.info("Chưa chọn nguồn camera này.")
    elif mode == "RTSP":
        st.info("Preview RTSP chỉ cập nhật định kỳ sau khi worker kết nối.")
    elif mode == "Đường dẫn MP4" and path:
        preview = first_frame(path)
        if preview:
            st.image(preview, caption="Frame đầu vào")
        else:
            st.warning("Không đọc được frame đầu của video đầu vào.")
    elif mode == "Upload MP4" and upload is not None:
        st.caption(f"Đã chọn {upload.name} · {upload.size / 1024 / 1024:.1f} MB")
    return upload, path, rtsp


def _camera_panel(output: Path, camera: str, active: bool, returncode: int | None) -> None:
    title = "FRONT Camera · IN" if camera == "front" else "REAR Camera · OUT"
    with st.container(border=True):
        st.markdown(f"### {title}")
        if not active:
            st.info("Không có video cho camera này.")
            return
        progress = camera_status(output, camera, returncode)
        state = progress["status"]
        st.caption(f"Trạng thái: {state} · hai chiều IN/OUT")
        count_in = progress.get("counts", {}).get("IN", 0)
        count_out = progress.get("counts", {}).get("OUT", 0)
        metrics = st.columns(4)
        metrics[0].metric("IN", count_in)
        metrics[1].metric("OUT", count_out)
        metrics[2].metric("Ròng", count_in - count_out)
        metrics[3].metric("FPS xử lý", f"{progress.get('processing_fps', 0):.1f}")
        st.caption(
            f"Thời gian xử lý {progress.get('elapsed_s', 0):.1f}s · {progress.get('events', 0)} sự kiện"
        )
        total = progress.get("total_frames")
        if total:
            st.progress(min(progress.get("frames", 0) / total, 1.0))
        input_preview = output / f"{camera}_input.jpg"
        if input_preview.is_file():
            st.image(input_preview.read_bytes(), caption="Video đầu vào · frame đầu tiên")
        preview = output / f"{camera}_preview.jpg"
        annotated = output / f"{camera}_annotated.mp4"
        if state == "COMPLETED" and annotated.is_file():
            st.video(str(annotated), format="video/mp4")
            st.caption("Video H.264 đã kiểm tra frame và codec · bấm Play để xem toàn bộ inference")
        elif preview.is_file():
            st.image(
                preview.read_bytes(), caption="Preview AI cập nhật theo chu kỳ, không phải live RTSP stream"
            )
        else:
            st.info("Đang chờ frame AI đầu tiên.")


def _render_timeline(output: Path, cameras: list[str], trip_id: str | None, api: Callable) -> None:
    st.markdown("### Timeline sự kiện AI")
    rows = event_timeline(output, cameras)
    if trip_id and rows:
        stored = api("GET", f"/trips/{trip_id}/passenger-events?limit=5000") or []
        by_key = {(row["camera"], row["track_id"], row["video_frame"]): row for row in stored}
        for row in rows:
            match = by_key.get((row["camera"], row["track_id"], row["frame_index"]))
            if match:
                row["event_status"] = match["assignment_status"]
    if not rows:
        st.info("Chưa có crossing A ↔ B từ AI.")
        return
    columns = ["timestamp", "camera", "track_id", "crossing_direction", "direction", "occupancy_delta", "frame_index", "confidence", "event_status"]
    st.dataframe(pd.DataFrame(rows)[columns], use_container_width=True, hide_index=True)
    chosen = st.selectbox(
        "Xem bằng chứng sự kiện",
        range(len(rows)),
        format_func=lambda index: f"{rows[index]['camera']} · {rows[index]['direction']} · "
        f"{rows[index]['timestamp']} · {rows[index]['track_id']}",
    )
    item = rows[chosen]
    crop = Path(item["crop_path"]) if item["crop_path"] else None
    if crop and crop.is_file():
        st.image(str(crop), caption="Crop bằng chứng từ AI")
    annotated = output / f"{item['camera'].lower()}_annotated.mp4"
    if annotated.is_file():
        st.video(str(annotated), format="video/mp4", start_time=int(item["timestamp_s"]))
        st.caption("Trình duyệt tua gần thời điểm sự kiện; frame chính xác nằm trong bảng timeline.")


def render_camera_tab(root: Path, trips: list[dict], api: Callable) -> None:
    st.subheader("AI Counting · hai cửa")
    st.caption(
        "YOLO + ByteTrack · FRONT/REAR đều đếm IN và OUT · video không có telemetry cùng nguồn giữ UNASSIGNED"
    )
    mode = st.radio("Chế độ", ["AI Video Lab", "Gắn với chuyến thật"], horizontal=True)
    trip_id = None
    telemetry_file = ""
    capture_start = ""
    synchronized = False
    if mode == "AI Video Lab":
        st.info(
            "Chạy một hoặc hai video không cần GPS hay Trip. Kết quả lưu REAL_VIDEO_AI và không gán trạm."
        )
    else:
        real_trips = [item for item in trips if item["provenance"] in {"REAL_VIDEO_AI", "REAL_OR_USER_INPUT"}]
        if not real_trips:
            st.warning("Chưa có Trip thật. Dùng AI Video Lab để xem AI trước.")
            return
        trip_id = st.selectbox("Trip thật", [item["trip_id"] for item in real_trips])
        telemetry_file = st.text_input("Telemetry JSON đồng bộ (tùy chọn)")
        capture_start = st.text_input("Capture start ISO-8601 có timezone (nếu đã xác minh)")
        synchronized = st.checkbox("Tôi đã xác minh video, telemetry và đồng hồ thuộc cùng chuyến")
    left, right = st.columns(2)
    with left:
        front = _source_input("front", root)
    with right:
        rear = _source_input("rear", root)
    with st.expander("Lớp hiển thị AI", expanded=False):
        layers = {
            "boxes": "Bounding boxes",
            "track-ids": "Track ID và confidence",
            "zones": "ROI Zones A/B",
            "trajectory": "Trajectory",
            "events": "Counting events",
        }
        hidden = {
            key for key, label in layers.items() if not st.checkbox(label, value=True, key=f"layer_{key}")
        }
    max_frames = st.number_input("Giới hạn frame mỗi camera (0 = toàn bộ)", min_value=0, value=0)
    job = st.session_state.get("video_lab_job")
    running = bool(job and job["process"].poll() is None)
    if st.button("Chạy AI Counting", type="primary", disabled=running):
        if synchronized and (not telemetry_file or not capture_start):
            st.error("Đồng bộ telemetry cần file JSON và capture start time.")
        else:
            output = new_job_dir(root)
            try:
                sources = {}
                for camera, values in (("front", front), ("rear", rear)):
                    upload, path, rtsp = values
                    source = resolve_source(
                        upload, getattr(upload, "name", None), path, rtsp, output / "uploads"
                    )
                    if source:
                        if not source.lower().startswith(("rtsp://", "rtsps://")):
                            input_frame = first_frame(source)
                            if not input_frame:
                                raise ValueError(f"{camera.upper()}: MP4 không đọc được")
                            (output / f"{camera}_input.jpg").write_bytes(input_frame)
                        sources[camera] = source
                process = start_job(
                    root,
                    output,
                    sources,
                    trip_id=trip_id,
                    telemetry_file=telemetry_file,
                    capture_start=capture_start,
                    synchronized=synchronized,
                    max_frames=max_frames,
                    hidden_layers=hidden,
                )
            except (ValueError, OSError) as exc:
                st.error(str(exc))
            else:
                st.session_state["video_lab_job"] = {
                    "process": process,
                    "output": output,
                    "sources": sources,
                    "trip_id": trip_id,
                }
                st.rerun()
    job = st.session_state.get("video_lab_job")
    if not job:
        st.info(
            "Chọn ít nhất một MP4 hoặc RTSP rồi bấm Chạy AI Counting. Video Pexels có thể tải theo README."
        )
        return

    @st.fragment(run_every="2s")
    def monitor() -> None:
        process = job["process"]
        returncode = process.poll()
        if returncode is not None:
            st.rerun()
        st.info("AI đang xử lý · preview cập nhật theo chu kỳ khoảng 15 frame.")
        if st.button("Hủy inference"):
            (job["output"] / "cancel.request").touch()
            st.warning("Đã yêu cầu hủy; worker sẽ dừng sau frame hiện tại.")
        panels = st.columns(2)
        for column, camera in zip(panels, ("front", "rear"), strict=True):
            with column:
                _camera_panel(job["output"], camera, camera in job["sources"], returncode)
        _render_timeline(job["output"], list(job["sources"]), None, api)

    returncode = job["process"].poll()
    if returncode is None:
        monitor()
        return
    if returncode == 0:
        st.success("AI inference hoàn tất; video H.264 và sự kiện đã được lưu.")
    elif returncode == 2:
        st.warning("Inference đã hủy; không nhập event từng phần vào Trip.")
    else:
        st.error(f"Inference thất bại (exit {returncode}). Xem process.log trong {job['output']}.")
    panels = st.columns(2)
    for column, camera in zip(panels, ("front", "rear"), strict=True):
        with column:
            _camera_panel(job["output"], camera, camera in job["sources"], returncode)
    _render_timeline(job["output"], list(job["sources"]), job["trip_id"] if returncode == 0 else None, api)
    summary = job["output"] / "run_summary.json"
    if summary.is_file():
        result = json.loads(summary.read_text(encoding="utf-8"))
        st.caption(
            f"Run: {result['trip_id']} · provenance {result['result_kind']} · "
            f"clock {result['clock']} · AI accuracy N/A (chưa có ground truth)"
        )
        if result["errors"]:
            st.error(f"Camera errors: {result['errors']}")
