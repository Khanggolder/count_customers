# Migration report

Ngày 2026-10-19, clone thành công `https://github.com/Khanggolder/count_customers.git` vào thư mục mới `busstop-flow-ai`, nhánh `main`; không dùng fallback repo sạch.

Đã reuse: `OpenCVVideoSource` MP4/webcam/RTSP, redaction URI, YOLO person, ByteTrack, ROI counter, crop selector, annotated MP4, FastAPI/SQLite/Streamlit và parser PCDS.

Đã thay đổi:

- Session/Re-ID schema → Route/Stop/Trip/Telemetry/StopVisit/PassengerEvent/Audit/Run.
- Pipeline tuần tự FRONT rồi REAR → worker song song, cách ly lỗi.
- Event chỉ có relative timestamp → source PTS + optional capture anchor/offset + assignment status.
- Dashboard batch/Re-ID → sáu tab tiếng Việt theo chuyến/trạm/hai cửa/map/event/demo/settings.
- Thêm deterministic scenario, negative controls, idempotent replay, export theo trạm và manual audit.
- Xóa module/script/dependency/checkpoint metadata Re-ID khỏi working tree; lịch sử vẫn còn trong Git.

Không force-push, không xóa remote và chưa push thay đổi refactor này.
