from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cameras.yaml"
UPLOAD_DIR = ROOT / "data" / "raw" / "uploads"
RUNS_DIR = ROOT / "runs"

st.set_page_config(page_title="BusReID AI", page_icon="🚌", layout="wide")
API = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
st.title("🚌 BusReID AI — Đếm khách hai cửa")
st.caption("MVP nghiên cứu • Không nhận diện danh tính thật • Re-ID là candidate match, không phải xác suất đã hiệu chuẩn")


def api_get(path: str):
    try:
        response = requests.get(f"{API}{path}", timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.warning(f"API chưa sẵn sàng: {exc}")
        return None


def save_upload(uploaded, camera: str) -> str | None:
    if uploaded is None:
        return None
    if Path(uploaded.name).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        st.error("Chỉ nhận MP4/AVI/MOV/MKV")
        return None
    if uploaded.size > 1_500 * 1024 * 1024:
        st.error("Video vượt giới hạn 1.5 GiB")
        return None
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{camera}_{int(time.time())}_{Path(uploaded.name).name}"
    destination.write_bytes(uploaded.getbuffer())
    return str(destination)


def roi_preview(source: str, camera_config: dict) -> np.ndarray:
    capture = cv2.VideoCapture(source)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ValueError("Không decode được frame preview")
    height, width = frame.shape[:2]
    for key, color in (("zone_a", (0, 200, 255)), ("zone_b", (255, 180, 0))):
        polygon = np.asarray([(int(x * width), int(y * height)) for x, y in camera_config[key]], dtype=np.int32)
        cv2.polylines(frame, [polygon], True, color, max(2, width // 500))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


tab_sources, tab_monitor, tab_reid, tab_events, tab_dataset, tab_guide = st.tabs(
    ["Nguồn & vùng cửa", "Theo dõi batch", "Re-ID", "Sự kiện & KPI", "Dataset & đánh giá", "Hướng dẫn demo"]
)

with tab_sources:
    st.subheader("Nguồn video")
    st.info("FRONT và REAR là hai pipeline/tracker namespace độc lập. Nếu chỉ có một video, chạy single-camera; không nhân đôi file để giả lập hai cửa.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### FRONT — cửa trước / IN")
        front_upload = st.file_uploader("Tải video FRONT", type=["mp4", "avi", "mov", "mkv"], key="front")
        front_path = st.text_input("Hoặc path/RTSP FRONT", key="front_path", type="password" if "rtsp" in st.session_state.get("front_path", "") else "default")
        if front_upload:
            st.video(front_upload)
    with col2:
        st.markdown("#### REAR — cửa sau / OUT")
        rear_upload = st.file_uploader("Tải video REAR", type=["mp4", "avi", "mov", "mkv"], key="rear")
        rear_path = st.text_input("Hoặc path/RTSP REAR", key="rear_path", type="password" if "rtsp" in st.session_state.get("rear_path", "") else "default")
        if rear_upload:
            st.video(rear_upload)

    st.subheader("Cấu hình ROI normalized")
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    roi_cols = st.columns(2)
    edited = {}
    for column, camera in zip(roi_cols, ("front", "rear"), strict=True):
        with column:
            camera_cfg = raw_config["cameras"][camera]
            zone_a = st.text_area(f"{camera.upper()} vùng A", json.dumps(camera_cfg["zone_a"]), key=f"{camera}_a")
            zone_b = st.text_area(f"{camera.upper()} vùng B", json.dumps(camera_cfg["zone_b"]), key=f"{camera}_b")
            edited[camera] = (zone_a, zone_b)
    if st.button("Lưu cấu hình vùng cửa"):
        try:
            for camera, (zone_a, zone_b) in edited.items():
                values_a, values_b = json.loads(zone_a), json.loads(zone_b)
                for polygon in (values_a, values_b):
                    if len(polygon) < 3 or any(len(point) != 2 or not all(0 <= float(value) <= 1 for value in point) for point in polygon):
                        raise ValueError("Mỗi polygon cần >=3 điểm trong [0,1]")
                raw_config["cameras"][camera]["zone_a"] = values_a
                raw_config["cameras"][camera]["zone_b"] = values_b
            CONFIG_PATH.write_text(yaml.safe_dump(raw_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
            st.success("Đã lưu ROI")
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(str(exc))

    if st.button("Xem preview ROI trên frame đầu"):
        preview_cols = st.columns(2)
        for column, camera, uploaded, path_value in (
            (preview_cols[0], "front", front_upload, front_path),
            (preview_cols[1], "rear", rear_upload, rear_path),
        ):
            with column:
                source = save_upload(uploaded, f"preview_{camera}") if uploaded else (path_value or None)
                if source:
                    try:
                        latest_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
                        st.image(roi_preview(source, latest_config["cameras"][camera]), caption=f"{camera.upper()} — A vàng, B xanh")
                    except ValueError as exc:
                        st.error(str(exc))
                else:
                    st.warning(f"{camera.upper()}: NO_SOURCE")

    if st.button("Chạy inference", type="primary"):
        selected_front = save_upload(front_upload, "front") or front_path or None
        selected_rear = save_upload(rear_upload, "rear") or rear_path or None
        if not selected_front and not selected_rear:
            st.error("NO_VIDEO: hãy thêm ít nhất một nguồn hợp lệ")
        else:
            run_dir = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
            command = [sys.executable, str(ROOT / "scripts" / "run_count.py"), "--config", str(CONFIG_PATH), "--output", str(run_dir)]
            if selected_front:
                command += ["--front", selected_front]
            if selected_rear:
                command += ["--rear", selected_rear]
            with st.spinner("Đang chạy YOLO + ByteTrack + counter…"):
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                st.session_state["last_run"] = str(run_dir)
                st.success(f"Đã hoàn tất: {run_dir}")
                st.code(result.stdout)
            else:
                st.error("Inference thất bại; xem log thực bên dưới")
                st.code(result.stderr)

with tab_monitor:
    st.subheader("Kết quả inference thực")
    default_run = st.session_state.get("last_run", "")
    run_path = Path(st.text_input("Thư mục run", default_run)) if st.session_state.get("last_run", "") or default_run else None
    if run_path and run_path.is_dir():
        columns = st.columns(2)
        for column, camera in zip(columns, ("front", "rear"), strict=True):
            with column:
                st.markdown(f"#### {camera.upper()}")
                video = run_path / f"{camera}_annotated.mp4"
                summary = run_path / f"{camera}_summary.json"
                if video.exists():
                    st.video(str(video))
                else:
                    st.warning("NO_SOURCE / chưa có output")
                if summary.exists():
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    st.metric("Processing FPS đo được", f"{data['processing_fps']:.2f}")
                    st.json(data["counts"])
    else:
        st.warning("NO_VIDEO: chưa chọn/chạy thư mục kết quả")

with tab_reid:
    st.subheader("Cross-camera Person Re-ID — thử nghiệm")
    st.warning("Không có identity ground truth trên bus public: mọi ghép bus-unpaired phải mang nhãn UNVERIFIED hoặc UNKNOWN. Similarity không phải xác suất.")
    sessions = api_get("/sessions")
    if sessions:
        session_id = st.selectbox("Session", [item["id"] for item in sessions], key="reid_session")
        selected_session = next(item for item in sessions if item["id"] == session_id)
        matches = api_get(f"/sessions/{session_id}/matches") or []
        st.dataframe(pd.DataFrame(matches), use_container_width=True)
        if matches:
            event_rows = api_get(f"/sessions/{session_id}/events") or []
            event_map = {item["id"]: item for item in event_rows}
            st.markdown("#### Gallery candidate")
            for item in matches:
                in_event = event_map.get(item.get("in_event_id"))
                out_event = event_map.get(item.get("out_event_id"))
                cols = st.columns([1, 1, 2])
                for col, event, label in ((cols[0], in_event, "IN"), (cols[1], out_event, "OUT")):
                    with col:
                        st.caption(label)
                        crop = Path(event["crop_path"]) if event and event.get("crop_path") else None
                        if crop and crop.is_file():
                            st.image(str(crop))
                        else:
                            st.text("NO_CROP")
                cols[2].write({"status": item["status"], "similarity": item["similarity"], "reason": item["reason"]})
            selected = st.selectbox("Match để review", [item["id"] for item in matches])
            review_options = ["CONFIRMED", "REJECTED", "UNKNOWN"] if selected_session["mode"] == "PAIRED" else ["REJECTED", "UNKNOWN"]
            status = st.selectbox("Quyết định reviewer", review_options)
            note = st.text_input("Ghi chú reviewer")
            if st.button("Lưu review"):
                response = requests.patch(f"{API}/matches/{selected}", json={"status": status, "reviewer_note": note or None}, timeout=5)
                st.success("Đã lưu") if response.ok else st.error(response.text)
    else:
        st.info("Chưa có session/match trong database")

with tab_events:
    st.subheader("Sự kiện & thống kê")
    sessions = api_get("/sessions")
    if sessions:
        session_id = st.selectbox("Session", [item["id"] for item in sessions], key="event_session")
        stats = api_get(f"/sessions/{session_id}/stats")
        events = api_get(f"/sessions/{session_id}/events") or []
        if stats:
            col1, col2, col3 = st.columns(3)
            col1.metric("IN", stats["in"])
            col2.metric("OUT", stats["out"])
            col3.metric("Occupancy", stats["occupancy"] if stats["occupancy"] is not None else "N/A")
            if stats["warning"]:
                st.warning(stats["warning"])
        frame = pd.DataFrame(events)
        st.dataframe(frame, use_container_width=True)
        st.download_button("Xuất JSON", data=requests.get(f"{API}/sessions/{session_id}/export?format=json", timeout=5).content, file_name=f"session_{session_id}.json")
        st.download_button("Xuất CSV", data=requests.get(f"{API}/sessions/{session_id}/export?format=csv", timeout=5).content, file_name=f"session_{session_id}.csv")

with tab_dataset:
    st.subheader("Dataset & đánh giá")
    st.markdown("- **BusOccupancyCounter:** video public để smoke test; quyền tái sử dụng video chưa được repo xác nhận.\n- **PCDS:** camera cửa bus, count labels, CC BY-NC-SA 3.0; tải thủ công từ nguồn tác giả.\n- **iLIDS-VID:** benchmark Re-ID kỹ thuật, không phải bối cảnh cửa bus.")
    reports = sorted(RUNS_DIR.glob("**/*evaluation*.json"))
    if reports:
        selected = st.selectbox("Báo cáo", reports, format_func=str)
        st.json(json.loads(selected.read_text(encoding="utf-8")))
    else:
        st.info("Metric: NA — chưa có prediction và ground truth chung được đánh giá")

with tab_guide:
    st.subheader("Kịch bản trình diễn")
    st.markdown("""
1. Khởi động FastAPI và dashboard theo README.
2. Tải video bus công khai bằng script có xác nhận, kiểm tra frame/provenance.
3. Chạy **một camera** nếu chỉ có một nguồn; không tuyên bố occupancy hay cross-camera identity.
4. Khi có hai nguồn thật cùng chuyến, cấu hình ROI riêng và chạy hai pipeline.
5. Re-ID bus không có ground truth chỉ hiển thị `candidate match — unverified`/`UNKNOWN`.
6. Benchmark iLIDS-VID được báo cáo riêng là kỹ thuật, không phải hiệu năng trên bus.
""")

device = api_get("/device")
if device:
    st.sidebar.json(device)

