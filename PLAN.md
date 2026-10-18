# Kế hoạch refactor BusStop Flow AI

Ngày bắt đầu: 2026-10-19. Repository nguồn đã clone thành công từ
`https://github.com/Khanggolder/count_customers.git`, nhánh `main`.

- [x] Chuyển mô hình dữ liệu từ session/Re-ID sang Route, Stop, Trip, telemetry,
  StopVisit, PassengerEvent và audit; giữ SQLite và FastAPI.
- [x] Thêm stop-state machine GPS + tốc độ + cửa, cửa sổ gán event và tổng hợp
  theo trạm/cửa/chiều; mọi event thiếu bằng chứng giữ `UNASSIGNED`.
- [x] Tạo scenario một chuyến 8 trạm deterministic, có các ca đối chứng và ground
  truth riêng; replay idempotent, không cần mạng hoặc model weight.
- [x] Giữ YOLO person + ByteTrack + ROI A/B, sửa lifecycle/TTL và chạy FRONT/REAR
  bằng hai worker độc lập; hỗ trợ MP4/RTSP, redaction credential và clock offset.
- [x] Thay dashboard bằng giao diện tiếng Việt theo chuyến/trạm/hai cửa/map/event,
  export và hiệu chỉnh assignment có audit trail; bỏ Re-ID khỏi MVP/UI/runtime.
- [x] Viết và chạy unit/API/E2E tests, Ruff, demo offline; thử CUDA/video công khai
  hợp lệ nếu môi trường và giấy phép cho phép; báo cáo số đo thật, không suy diễn.
- [x] Hoàn thiện tài liệu Windows, kiến trúc, quy tắc nghiệp vụ, nguồn dữ liệu,
  hạn chế, migration và checklist acceptance.

Nguyên tắc dữ liệu: `SIMULATED_EVENTS` và `REAL_VIDEO_AI` luôn tách riêng; video
công khai không có telemetry đồng bộ không được coi là cùng chuyến mô phỏng.
