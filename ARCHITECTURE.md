# Kiến trúc hiện hành

`MP4/RTSP FRONT + MP4/RTSP REAR → worker độc lập → YOLO person → ByteTrack → Polygon A/B → CrossingEvent → PassengerEvent/SQLite → StopVisit assignment → API/Streamlit/export`

Mỗi worker có detector, tracker, ROI, SourceClock, namespace track và generation riêng. MP4 dùng PTS/fps; absolute capture time chỉ đáng tin khi có `recording_start_time`. RTSP reconnect giữ timebase monotonic; source URI được redaction trước khi lưu. ROI được snapshot khi worker bắt đầu, không đổi giữa phiên. Chiều A→B và B→A đều tạo event sau stable transition; mapping camera/crossing → IN/OUT/occupancy delta tập trung tại `busreid/business.py`. CSV ghi cả ba trường; `delta` tương thích nghĩa là occupancy delta.

`PassengerEvent` lưu trip/source/camera/provenance, crossing, passenger direction, occupancy delta, track, frame, source PTS, capture/processing timestamp, StopVisit và trạng thái assignment/direction. API từ chối event thiếu hoặc mâu thuẫn. SQLite cũ nâng cấp qua `scripts/migrate_bidirectional.py` sau backup; ambiguous legacy giữ `REVIEW_REQUIRED`, không suy đoán reverse từ signed delta cũ.

Telemetry GPS/tốc độ/cửa → StopVisit state machine → assignment. GPS missing/stale không tự đóng stop; thiếu door evidence giữ review. Assignment kiểm tra nguồn và thời gian đồng bộ. Flow truy vấn bốn ô cửa/hướng, tính total IN/OUT và occupancy khi paired coverage đủ. Negative occupancy không bị clamp. Một camera/coverage không đủ cho occupancy `N/A`.

`scenario.py` v3 tạo dữ liệu `SIMULATED_EVENTS` và ground truth synthetic riêng, 8 trạm và bốn hướng, kịch bản âm/incomplete camera tách biệt. Integration test hai MP4 fixture dùng FakeDetector để kiểm tra phần mềm, không đo độ chính xác YOLO. Video công khai thiếu telemetry cùng nguồn vẫn `UNASSIGNED`.

FastAPI và Streamlit mặc định local-only, chưa có auth/TLS/retention. SQLite và worker cục bộ là kiến trúc MVP, không phải triển khai fleet production.
