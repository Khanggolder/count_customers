from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from app.camera_view import render_camera_tab
from app.counting_tab import render_counting_tab
from app.roi_editor import render_roi_editor

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "cameras.yaml"
DEFAULT_API = os.getenv("BUSSTOP_API_URL", "http://127.0.0.1:8000")
SHOW_ADVANCED_TABS = os.getenv("BUSSTOP_SHOW_ADVANCED_TABS", "0").lower() in {"1", "true", "yes"}

st.set_page_config(page_title="BusStop Flow AI", page_icon="🚌", layout="wide")
st.title("BusStop Flow AI — Lưu lượng hành khách theo trạm")
st.caption("MVP nghiên cứu cho một chuyến xe • Không nhận diện cá nhân • Không tối ưu lịch xe")
api_url = st.sidebar.text_input("API URL", DEFAULT_API).rstrip("/") if SHOW_ADVANCED_TABS else DEFAULT_API


def api(method: str, path: str, raw: bool = False, quiet: bool = False, **kwargs):
    try:
        response = requests.request(method, f"{api_url}{path}", timeout=15, **kwargs)
        response.raise_for_status()
        if not raw and response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.content
    except requests.RequestException as exc:
        if not quiet:
            st.error(f"API chưa sẵn sàng: {exc}")
        return None


health = api("GET", "/health", quiet=True) if SHOW_ADVANCED_TABS else None
if SHOW_ADVANCED_TABS:
    if health:
        st.sidebar.success("API hoạt động")
    else:
        st.sidebar.warning("API offline · AI Video Lab vẫn chạy được")

trips = api("GET", "/trips") if health else []
trip_ids = [item["trip_id"] for item in trips or []]
selected_trip = st.sidebar.selectbox("Chuyến", trip_ids) if trip_ids else None

tab_labels = ["Đếm khách AI", "Thiết lập vùng đếm (ROI)"]
if SHOW_ADVANCED_TABS:
    tab_labels.extend(["Tổng quan chuyến", "Hai cửa", "Trạm & bản đồ", "Sự kiện & kiểm thử", "Demo mẫu", "Cài đặt"])
tabs = st.tabs(tab_labels)
counting_tab, roi_tab = tabs[:2]

with counting_tab:
    render_counting_tab(ROOT)

with roi_tab:
    render_roi_editor(ROOT)

if not SHOW_ADVANCED_TABS:
    st.stop()

overview_tab, cameras_tab, stops_tab, events_tab, demo_tab, settings_tab = tabs[2:]

with overview_tab:
    if not selected_trip:
        st.info("Chưa có chuyến. Mở tab Demo mẫu và tạo chuyến mô phỏng.")
    else:
        trip = next(item for item in trips if item["trip_id"] == selected_trip)
        flow = api("GET", f"/trips/{selected_trip}/flow") or []
        provenance = trip["provenance"]
        if provenance == "SIMULATED":
            st.error("DỮ LIỆU MÔ PHỎNG — không phải kết quả AI từ camera thật")
        else:
            st.warning("REAL VIDEO AI — chỉ là model output; cần ground truth để báo accuracy")
        total_in = sum(row["total_in"] for row in flow)
        total_out = sum(row["total_out"] for row in flow)
        assigned_stops = sum(row["visit_status"] == "CONFIRMED" for row in flow)
        cols = st.columns(4)
        cols[0].metric("Khách lên · cửa trước", total_in)
        cols[1].metric("Khách xuống · cửa sau", total_out)
        cols[2].metric("Trạm xác nhận", f"{assigned_stops}/{len(flow)}")
        cols[3].metric("Ghép nguồn", trip["mode"])
        if flow:
            chart_data = pd.DataFrame(flow)
            chart = px.bar(
                chart_data,
                x="stop_name",
                y=["boarding_count", "alighting_count"],
                barmode="group",
                labels={"value": "Lượt", "stop_name": "Trạm", "variable": "Chiều"},
            )
            st.plotly_chart(chart, use_container_width=True)
            st.dataframe(
                chart_data[
                    [
                        "sequence",
                        "stop_name",
                        "front_in",
                        "rear_out",
                        "boarding_count",
                        "alighting_count",
                        "onboard_after_stop",
                        "visit_status",
                        "warning",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

with cameras_tab:
    render_camera_tab(ROOT, trips or [], api)

with stops_tab:
    if selected_trip:
        trip = next(item for item in trips if item["trip_id"] == selected_trip)
        stops = api("GET", f"/routes/{trip['route_id']}/stops") or []
        visits = api("GET", f"/trips/{selected_trip}/stop-visits") or []
        if stops:
            frame = pd.DataFrame(stops)
            st.map(frame.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
            st.dataframe(
                frame[["sequence", "stop_id", "name", "radius_m", "coordinate_source", "verified"]],
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("#### Timeline StopVisit và bằng chứng")
        st.dataframe(pd.DataFrame(visits), use_container_width=True, hide_index=True)
        if any(not item["verified"] for item in stops):
            st.warning(
                "Các tọa độ đang hiển thị chưa được xác minh; scenario mặc định dùng hình học tổng hợp."
            )
    else:
        st.info("Chọn một chuyến để xem trạm.")

with events_tab:
    if selected_trip:
        events = api("GET", f"/trips/{selected_trip}/passenger-events?limit=5000") or []
        invalid_history = sum(not row.get("policy_valid", True) for row in events)
        if invalid_history:
            st.warning(
                f"{invalid_history} event lịch sử sai chiều: giữ để kiểm toán, không tính vào lưu lượng/occupancy."
            )
        statuses = ["ALL", "AUTO_ASSIGNED", "MANUAL_ASSIGNED", "REVIEW_REQUIRED", "UNASSIGNED"]
        selected_status = st.selectbox("Trạng thái gán trạm", statuses)
        shown = (
            events
            if selected_status == "ALL"
            else [row for row in events if row["assignment_status"] == selected_status]
        )
        st.dataframe(pd.DataFrame(shown), use_container_width=True, hide_index=True)
        st.caption(
            "Accuracy video: N/A nếu không có ground truth đúng clip. Scenario synthetic chỉ kiểm thử logic, không phải AI accuracy."
        )
        if events:
            unassigned = [
                item for item in events if item["assignment_status"] in {"UNASSIGNED", "REVIEW_REQUIRED"}
            ]
            if unassigned:
                st.markdown("#### Review event chưa gán")
                event_id = st.selectbox("Event", [item["id"] for item in unassigned])
                visits = api("GET", f"/trips/{selected_trip}/stop-visits") or []
                visit_options = [None] + [item["stop_visit_id"] for item in visits]
                visit_id = st.selectbox("StopVisit mới", visit_options)
                reason = st.text_input("Lý do", "Xác nhận thủ công từ người vận hành")
                if st.button("Lưu hiệu chỉnh"):
                    result = api(
                        "PATCH",
                        f"/passenger-events/{event_id}/assignment",
                        json={"stop_visit_id": visit_id, "actor": "dashboard_operator", "reason": reason},
                    )
                    if result:
                        st.success("Đã lưu và tạo audit trail")
                        st.rerun()
        col1, col2 = st.columns(2)
        csv_data = api("GET", f"/trips/{selected_trip}/exports?format=csv", raw=True)
        json_data = api("GET", f"/trips/{selected_trip}/exports?format=json", raw=True)
        if csv_data:
            col1.download_button("Xuất CSV theo trạm", csv_data, f"{selected_trip}_flow.csv", "text/csv")
        if json_data:
            col2.download_button(
                "Xuất JSON theo trạm", json_data, f"{selected_trip}_flow.json", "application/json"
            )
    else:
        st.info("Chọn một chuyến để xem event.")

with demo_tab:
    st.error("DỮ LIỆU MÔ PHỎNG / SIMULATED EVENTS")
    st.write(
        "Kịch bản tám trạm có GPS, tốc độ và cảm biến cửa. Chỉ FRONT-IN và REAR-OUT là sự kiện khách hợp lệ. Đây là dữ liệu mô phỏng, không phải kết quả AI."
    )
    if st.button("Tạo hoặc replay chuyến mẫu", type="primary"):
        result = api("POST", "/demo/generate")
        if result:
            st.success(f"Hoàn tất: {json.dumps(result, ensure_ascii=False)}")
            st.rerun()
    if st.button("Tạo kịch bản đối chứng occupancy âm"):
        result = api("POST", "/demo/generate?kind=negative_occupancy")
        if result:
            st.warning(f"Đã tạo kịch bản đối chứng: {result['trip_id']}")
            st.rerun()
    st.markdown(
        "Các ca đối chứng: dừng đèn đỏ gần trạm nhưng cửa đóng; tốc độ 0 ngoài geofence; cửa mở ngoài trạm; GPS mất; event ngoài StopVisit. Chuyển động sai chiều không sinh PassengerEvent."
    )

with settings_tab:
    st.subheader("Nguồn, ROI và thiết bị")
    device = api("GET", "/device") if health else None
    st.json(device or {"device": "UNKNOWN"})
    st.code(CONFIG_PATH.read_text(encoding="utf-8"), language="yaml")
    st.markdown(
        "- MP4: truyền path vào `scripts/run_count.py`.\n- RTSP: dùng cùng tham số; credential chỉ cấp cục bộ và luôn được redaction khỏi DB/summary.\n- Clock offset và ROI cấu hình riêng cho từng camera.\n- OCR overlay là adapter tùy chọn, không dùng trong demo mặc định."
    )
