"""Mouse editor for the independent REAR optical-flow ROI and counting line."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from busreid.config import load_config
from busreid.rear_flow_counting import FlowLineAnalyzer, annotate_flow
from busreid.roi import (
    restore_flow_geometry,
    save_flow_geometry,
    to_normalized,
    to_pixels,
    validate_flow_geometry,
    validate_polygon,
)

from .counting_tab import _show_preview, _show_video
from .roi_editor import _select_source, read_video_frame
from .video_lab import new_job_dir, start_job


def _geometry(config_path: Path) -> dict:
    flow = load_config(config_path).rear_flow_out
    return {"roi": list(flow.roi), "line": list(flow.counting_line),
            "direction": tuple(flow.out_direction)}


def _preview(frame: np.ndarray, draft: dict, config_path: Path) -> Image.Image:
    height, width = frame.shape[:2]
    if width > 800:
        frame = cv2.resize(frame, (800, round(height * 800 / width)))
    height, width = frame.shape[:2]
    output = frame.copy()
    try:
        validate_flow_geometry(draft["roi"], draft["line"], draft["direction"])
        flow = copy.deepcopy(load_config(config_path).rear_flow_out)
        flow.roi, flow.counting_line, flow.out_direction = draft["roi"], draft["line"], draft["direction"]
        output = annotate_flow(output, FlowLineAnalyzer(flow, output.shape), None)
    except ValueError:
        pass
    for label, points, color in (("roi", draft["roi"], (0, 220, 180)),
                                 ("line", draft["line"], (0, 200, 255))):
        contour = np.asarray([to_pixels(point, width, height) for point in points], np.int32)
        if label == "roi" and len(contour) >= 2:
            cv2.polylines(output, [contour], len(contour) >= 3, color, 2)
        if label == "line" and len(contour) == 2:
            cv2.line(output, tuple(contour[0]), tuple(contour[1]), color, 2)
        for index, point in enumerate(contour):
            cv2.circle(output, tuple(point), 5, color, -1)
            cv2.putText(output, str(index + 1), tuple(point + (6, -6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


def _click(state: dict, target: str, point: tuple[float, float], selected: int) -> None:
    state["history"].append(copy.deepcopy(state["draft"]))
    draft = state["draft"]
    if target == "Thêm đỉnh ROI":
        draft["roi"].append(point)
    elif target == "Dời đỉnh ROI" and draft["roi"]:
        draft["roi"][selected] = point
    elif target == "Điểm đầu line":
        draft["line"][0] = point
    elif target == "Điểm cuối line":
        draft["line"][1] = point
    elif target == "Đầu mũi tên OUT":
        middle = np.mean(draft["line"], axis=0)
        draft["direction"] = tuple(np.asarray(point) - middle)
    state["revision"] += 1


def render_flow_roi_editor(root: Path) -> None:
    config_path = root / "configs" / "cameras.yaml"
    st.subheader("REAR – Optical Flow Counting")
    st.caption("Vẽ ROI, đường đếm và bấm phía OUT để đặt mũi tên. Kết quả là motion events; số khách vẫn N/A khi chưa hiệu chuẩn.")
    key = "roi_editor_rear_flow_out"
    if key not in st.session_state:
        st.session_state[key] = {"draft": _geometry(config_path), "history": [],
                                 "revision": 0, "saved_revision": 0,
                                 "source": None, "upload_identity": None}
    state = st.session_state[key]
    source, is_mp4 = _select_source(root, "rear_flow_out", state)
    frame = None
    frame_index = 0
    frame_count = 0
    if source is not None and is_mp4:
        try:
            _, frame_count, fps = read_video_frame(source)
            frame_index = (st.slider("Frame MP4", 0, frame_count - 1, 0, key="flow_roi_frame")
                           if frame_count > 1 else 0)
            frame = read_video_frame(source, frame_index)[0]
            st.caption(f"Frame {frame_index + 1}/{frame_count or '?'} · {fps:.2f} FPS")
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            st.error(f"Không đọc được MP4/frame: {exc}")
    elif not is_mp4:
        frame = state.get("live_frame")
    if frame is None:
        st.info("Chọn MP4 hoặc lấy frame từ camera để chỉnh ROI và Counting Line.")
        return
    actions = ["Thêm đỉnh ROI", "Dời đỉnh ROI", "Điểm đầu line", "Điểm cuối line", "Đầu mũi tên OUT"]
    target = st.radio("Thao tác", actions, horizontal=True)
    selected = 0
    if target == "Dời đỉnh ROI" and state["draft"]["roi"]:
        selected = st.selectbox("Đỉnh ROI", range(len(state["draft"]["roi"])),
                                format_func=lambda i: f"ROI {i + 1}")
    picture = _preview(frame, state["draft"], config_path)
    click = streamlit_image_coordinates(picture, key=f"flow_canvas_{frame_index}_{state['revision']}_{target}",
                                         click_and_drag=target == "Dời đỉnh ROI", cursor="crosshair")
    if click:
        x, y = click.get("x2", click.get("x")), click.get("y2", click.get("y"))
        if x is not None and y is not None:
            try:
                point = to_normalized((int(x), int(y)), *picture.size)
                _click(state, target, point, selected)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    cols = st.columns(4)
    if cols[0].button("Xóa đỉnh ROI", disabled=not state["draft"]["roi"]):
        state["history"].append(copy.deepcopy(state["draft"]))
        state["draft"]["roi"].pop(min(selected, len(state["draft"]["roi"]) - 1))
        state["revision"] += 1
        st.rerun()
    if cols[1].button("Hoàn tác", disabled=not state["history"]):
        state["draft"] = state["history"].pop()
        state["revision"] += 1
        st.rerun()
    if cols[2].button("Xóa ROI"):
        state["history"].append(copy.deepcopy(state["draft"]))
        state["draft"]["roi"] = []
        state["revision"] += 1
        st.rerun()
    if cols[3].button("Nạp cấu hình"):
        state["draft"] = _geometry(config_path)
        state["history"].clear()
        state["revision"] += 1
        state["saved_revision"] = state["revision"]
        st.rerun()
    for label in ("roi", "line"):
        st.write(label.upper())
        st.dataframe([{"đỉnh": i + 1, "pixel": to_pixels(p, frame.shape[1], frame.shape[0]),
                       "normalized": tuple(round(v, 5) for v in p)}
                      for i, p in enumerate(state["draft"][label])], hide_index=True)
    st.write("Hướng OUT (normalized):", tuple(round(v, 5) for v in state["draft"]["direction"]))
    try:
        validate_polygon(state["draft"]["roi"])
        validate_flow_geometry(**{"roi": state["draft"]["roi"], "line": state["draft"]["line"],
                                  "direction": state["draft"]["direction"]})
        valid = True
        st.success("ROI, Counting Line và hướng OUT hợp lệ.")
    except ValueError as exc:
        valid = False
        st.warning(f"Chưa thể lưu: {exc}")
    left, right = st.columns(2)
    if left.button("Lưu Optical Flow ROI", disabled=not valid, type="primary"):
        try:
            backup = save_flow_geometry(config_path, state["draft"]["roi"],
                                        state["draft"]["line"], state["draft"]["direction"])
            state["saved_revision"] = state["revision"]
            st.success(f"Đã lưu; bản sao: {backup.name}. Phiên mới sẽ dùng cấu hình này.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    if right.button("Khôi phục từ bản sao"):
        try:
            restore_flow_geometry(config_path)
            state["draft"] = _geometry(config_path)
            state["revision"] += 1
            state["saved_revision"] = state["revision"]
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    st.markdown("#### Test Optical Flow trên đoạn video")
    limit = st.number_input("Số frame", min_value=2, max_value=1200, value=120, key="flow_roi_limit")
    job = st.session_state.get("flow_roi_test_job")
    busy = bool(job and job["process"].poll() is None)
    if st.button("Test Optical Flow ROI", disabled=busy or not valid or source is None
                 or state["revision"] != state["saved_revision"]):
        output = new_job_dir(root, "ai_counting")
        try:
            process = start_job(root, output, {"rear": f"webcam:{source}" if isinstance(source, int) else source},
                                session_only=True, rear_mode="flow_line_out", max_frames=int(limit),
                                start_frame=frame_index if is_mp4 else 0)
            st.session_state["flow_roi_test_job"] = {"process": process, "output": output}
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    if job:
        output = job["output"]
        try:
            progress = json.loads((output / "rear_flow_progress.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            progress = {"status": "QUEUED" if job["process"].poll() is None else "FAILED"}
        st.write(f"{progress['status']} · {progress.get('frames', 0)} frame · "
                 f"{progress.get('motion_events', 0)} motion events · {progress.get('processing_fps', 0):.1f} FPS")
        if progress.get("error"):
            st.error(progress["error"])
        for name in ("rear_flow_events.csv", "rear_flow_temporal_slice.png"):
            path = output / name
            if path.is_file():
                try:
                    st.download_button(f"Tải {name}", path.read_bytes(), name)
                except OSError:
                    st.caption(f"{name} đang được ghi; thử lại sau.")
        preview = output / "rear_flow_preview.jpg"
        if preview.is_file():
            _show_preview(preview, "Frame Optical Flow mới nhất")
        video = output / "rear_flow_annotated.mp4"
        if video.is_file():
            _show_video(video)
