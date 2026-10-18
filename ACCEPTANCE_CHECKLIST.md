# Acceptance checklist

## Hợp đồng đầu ra

1. **DONE — tuyến/trip/trạm:** scenario tạo route và 8 trạm có thứ tự; API `/routes`, `/routes/{id}/stops`, `/trips`. Tên/thứ tự có nguồn, tọa độ synthetic và `verified=false`.
2. **DONE — hai cửa/cả hai chiều:** YOLO11n person + ByteTrack + normalized ROI A/B, anchor cấu hình; FRONT/REAR có detector/tracker/worker riêng. Unit test hai hướng và camera namespace; clip Pexels đã chạy GPU.
3. **DONE — event:** schema lưu UUID, camera/direction/track/frame/PTS/time/source/confidence/assignment/evidence/provenance; debounce, stable transition, reversal, ID-switch recovery và TTL có test. `event_uuid` unique.
4. **DONE — StopVisit/assignment:** geofence + speed + dwell + door + sequence; stale/missing data và negative controls có test; cửa sổ auto-assign chỉ nhận `CONFIRMED`, còn lại `UNASSIGNED`; manual audit hoạt động.
5. **DONE — aggregate:** API/JSON/CSV có front/rear × IN/OUT, total, arrival/departure/evidence/warning và occupancy có điều kiện. Scenario cố ý chứng minh cảnh báo occupancy âm.
6. **DONE — dashboard MVP:** 6 tab tiếng Việt: tổng quan/chart, hai cửa/ROI, trạm/map/evidence, event/review/export, one-click demo, settings/GPU/provenance. Streamlit AppTest render đủ 6 tab không exception.
7. **DONE — offline + real-video separation:** one-click deterministic demo không cần network/weights; `SIMULATED_EVENTS` tách `REAL_VIDEO_AI`. Pexels chạy riêng `UNPAIRED`, không ghép vào GPS synthetic.
8. **DONE/PARTIAL — Windows/CUDA:** cài và chạy trên Windows Python 3.11, RTX 3050/CUDA 13.0; CPU fallback có. API/UI local chạy. **PARTIAL** chỉ ở RTSP vật lý vì không có camera endpoint để soak-test.

## Nguồn ngoài và validation

- **DONE:** Pexels clip/license/checksum/metadata/frame review; YOLO+ByteTrack inference thật, artifact annotated.
- **BLOCKED:** PCDS labeled videos chỉ còn BaiYun và không tải hợp lệ trong phiên; AI accuracy là `N/A`.
- **BLOCKED:** GPS/cửa/two-camera thật cùng chuyến chưa được stakeholder cung cấp; occupancy thật và đồng bộ RTSP chưa kiểm chứng.
- **PARTIAL:** endpoint chính thức RouteFinding trả HTTP 403; dùng thứ tự trạm nguồn bên thứ ba nhưng tọa độ scenario được gắn synthetic/unverified.

## Bằng chứng

- Tests: `33 passed`, `0 warning`, coverage `86%`; Ruff và pip check sạch.
- Demo: `runs/demo/flow_by_stop.json`, 8 StopVisit, 57 event, 24 IN/32 OUT.
- Real video: `runs/pexels_real_video/run_summary.json`, 178 frame, GPU 6.0114 FPS, 69.2075 MiB peak tensor VRAM, no GT.
- Tài liệu: `README.md`, `ARCHITECTURE.md`, `BUSINESS_RULES.md`, `DATA_SOURCES.md`, `LIMITATIONS.md`, `MIGRATION_REPORT.md`, `TEST_REPORT.md`.
