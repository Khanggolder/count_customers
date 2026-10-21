# Kế hoạch refactor đã hoàn tất (2026-10-19)

Tài liệu lịch sử: chuyển BusReID AI sang BusStop Flow AI trên nhánh
`feature/busstop-flow-ai`. Các hạng mục đã thực hiện gồm mô hình
Route/Stop/Trip/Telemetry/StopVisit/PassengerEvent, state machine dừng trạm,
hai worker camera FRONT/REAR, dashboard, API, scenario mô phỏng và kiểm thử.

Quy tắc hiện hành là **FRONT-IN / REAR-OUT**. Dữ liệu `SIMULATED_EVENTS` và
`REAL_VIDEO_AI` luôn tách biệt; video công khai không có telemetry cùng nguồn
không được xem là bằng chứng của chuyến mô phỏng.

Trạng thái và giới hạn hiện tại được duy trì trong `README.md`,
`ACCEPTANCE_CHECKLIST.md` và `LIMITATIONS.md` ở thư mục gốc.
