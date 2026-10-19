# Kiến trúc

## Luồng offline deterministic

`scenario.json → replay_scenario → telemetry/state machine → StopVisit → event assignment → SQLite → API/Streamlit/export`

Scenario và ground truth nằm cùng artifact nhưng ở hai nhánh riêng. Passenger event có `provenance=SIMULATED_EVENTS`; hệ thống không gọi chúng là prediction.

## Luồng video

`MP4/RTSP FRONT + MP4/RTSP REAR → ThreadPoolExecutor (2 worker độc lập) → YOLO person → ByteTrack → ROI A/B counter → PassengerEvent → SQLite`

Mỗi camera tạo một `PersonDetector`, ByteTrack persistent namespace và `DirectionCounter`. State có debounce, stable transition, ID-switch recovery thận trọng, TTL và crop buffer hữu hạn. Worker lỗi được ghi riêng; worker còn lại vẫn hoàn tất.

Video file dùng PTS/fps và chỉ có absolute capture time khi người vận hành truyền `recording_start_time`. RTSP reconnect giữ generation clock monotonic. Source URI được redaction trước khi lưu.

## Nhận biết dừng trạm

`MOVING → APPROACHING → DWELL_CANDIDATE → STOP_CONFIRMED → DEPARTING → CLOSED` được hiện thực dưới dạng state batch theo thứ tự trạm. Xác nhận cần đồng thời geofence, tốc độ thấp, dwell và cửa mở. Thiếu door signal chỉ tạo `UNCERTAIN`; cửa đóng không tạo serving stop. GPS stale/missing bị loại.

Event thật chỉ auto-assign khi source cùng trip/cửa/provenance, clock capture có neo và được xác nhận đồng bộ, StopVisit `CONFIRMED` trong cửa sổ thời gian, và mẫu cửa tương ứng mở trong 2 giây trước event. Thiếu bằng chứng cửa thành `REVIEW_REQUIRED`; thiếu đồng bộ/trạm vẫn `UNASSIGNED`. Chỉnh tay tạo `ReviewAudit`.

## Dữ liệu

SQLAlchemy/SQLite gồm Route, Stop, Trip, CameraSource, TelemetrySample, StopVisit, PassengerEvent, ReviewAudit và Run. `event_uuid` unique bảo đảm replay idempotent; index bao phủ trip/time và trip/stop.

Occupancy chỉ tính khi `mode ∈ {PAIRED_VERIFIED, SYNTHETIC_PAIRED}` và `coverage_complete=true`. Số âm không bị clamp mà tạo cảnh báo.

## Triển khai

FastAPI và Streamlit mặc định local-only, chưa có auth/TLS. SQLite thích hợp MVP một xe mẫu; production nên dùng PostgreSQL, message broker, object storage, health/metrics và quản lý secret tập trung.
