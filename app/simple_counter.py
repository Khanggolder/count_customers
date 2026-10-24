"""Focused two-door passenger counting tab, independent of trip/stop analytics."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from .video_lab import camera_status, first_frame, new_job_dir, resolve_source, start_job


def count_for_camera(progress: dict, camera: str) -> int:
    """Only the business-approved direction contributes to each camera count."""
    direction = "IN" if camera == "front" else "OUT"
    return int(progress.get("counts", {}).get(direction, 0))


def _source_input(camera: str) -> tuple[object | None, str, str]:
    label = "Cửa trước · Khách lên (IN)" if camera == "front" else "Cửa sau · Khách xuống (OUT)"
    st.markdown(f"#### {label}")
    mode = st.selectbox(
        "Nguồn video",
        ["Không dùng", "Tải video MP4", "Đường dẫn MP4", "RTSP"],
        key=f"simple_source_{camera}",
    )
    upload = (
        st.file_uploader("Chọn MP4", type=["mp4"], key=f"simple_upload_{camera}")
        if mode == "Tải video MP4"
        else None
    )
    path = (
        st.text_input("Đường dẫn MP4", key=f"simple_path_{camera}")
        if mode == "Đường dẫn MP4"
        else ""
    )
    rtsp = (
        st.text_input("Địa chỉ RTSP", type="password", key=f"simple_rtsp_{camera}")
        if mode == "RTSP"
        else ""
    )
    return upload, path, rtsp


def _camera_output(output: Path, camera: str, active: bool, returncode: int | None) -> None:
    name = "CỬA TRƯỚC — IN" if camera == "front" else "CỬA SAU — OUT"
    with st.container(border=True):
        st.markdown(f"**{name}**")
        if not active:
            st.info("Chưa chọn nguồn video.")
            return

        progress = camera_status(output, camera, returncode)
        st.metric("Khách lên" if camera == "front" else "Khách xuống", count_for_camera(progress, camera))
        st.caption(
            f"{progress.get('status', 'QUEUED')} · "
            f"{progress.get('frames', 0)} frame · "
            f"{progress.get('processing_fps', 0):.1f} FPS"
        )
        total = progress.get("total_frames")
        if total:
            st.progress(min(progress.get("frames", 0) / total, 1.0))

        annotated = output / f"{camera}_annotated.mp4"
        preview = output / f"{camera}_preview.jpg"
        initial = output / f"{camera}_input.jpg"
        if returncode == 0 and annotated.is_file():
            st.video(str(annotated), format="video/mp4")
            st.caption("Video AI: khung người, Track ID, vùng A/B và lượt đếm.")
        elif preview.is_file():
            st.image(preview.read_bytes(), caption="Khung hình AI cập nhật định kỳ")
        elif initial.is_file():
            st.image(initial.read_bytes(), caption="Khung hình đầu vào")
        else:
            st.info("Đang chờ khung hình đầu tiên.")


def _show_results(job: dict) -> None:
    process = job["process"]
    code = process.poll()
    if code is None:
        st.info("AI đang xử lý video; khung hình và số đếm cập nhật định kỳ.")
    elif code == 0:
        st.success("Đã xử lý xong. Bấm Play trên video để xem quá trình AI nhận diện.")
    elif code == 2:
        st.warning("Đã dừng xử lý theo yêu cầu.")
    else:
        st.error("Không thể xử lý video. Xem process.log trong thư mục kết quả.")

    left, right = st.columns(2)
    for column, camera in ((left, "front"), (right, "rear")):
        with column:
            _camera_output(job["output"], camera, camera in job["sources"], code)


def render_simple_counter(root: Path) -> None:
    """Single-purpose tab: provide a source, run AI and watch IN/OUT counts."""
    st.subheader("Đếm khách lên / xuống xe buýt")
    st.caption(
        "Cửa trước chỉ đếm khách lên (IN) · Cửa sau chỉ đếm khách xuống (OUT). "
        "Không cần GPS, chuyến xe hay dữ liệu mô phỏng."
    )
    left, right = st.columns(2)
    with left:
        front = _source_input("front")
    with right:
        rear = _source_input("rear")

    job = st.session_state.get("simple_count_job")
    running = bool(job and job["process"].poll() is None)
    if st.button("Bắt đầu đếm", type="primary", disabled=running):
        output = new_job_dir(root)
        try:
            sources: dict[str, str] = {}
            for camera, values in (("front", front), ("rear", rear)):
                upload, path, rtsp = values
                source = resolve_source(
                    upload, getattr(upload, "name", None), path, rtsp, output / "uploads"
                )
                if source:
                    if not source.lower().startswith(("rtsp://", "rtsps://")):
                        frame = first_frame(source)
                        if frame is None:
                            raise ValueError(f"{camera.upper()}: không đọc được video MP4")
                        (output / f"{camera}_input.jpg").write_bytes(frame)
                    sources[camera] = source
            process = start_job(root, output, sources)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["simple_count_job"] = {
                "process": process,
                "output": output,
                "sources": sources,
            }
            st.rerun()

    job = st.session_state.get("simple_count_job")
    if job is None:
        st.info("Chọn video cho một hoặc hai cửa, sau đó bấm **Bắt đầu đếm**.")
        return

    if job["process"].poll() is None:
        @st.fragment(run_every="2s")
        def _monitor() -> None:
            if job["process"].poll() is not None:
                st.rerun()
            if st.button("Dừng đếm", key="simple_stop"):
                (job["output"] / "cancel.request").touch()
                st.warning("Đã gửi yêu cầu dừng.")
            _show_results(job)

        _monitor()
    else:
        _show_results(job)

    st.caption(
        "Kết quả lấy từ YOLO + ByteTrack trên video đã chọn; chưa có ground truth nên "
        "không khẳng định độ chính xác. Video không có GPS sẽ không được gán trạm."
    )
