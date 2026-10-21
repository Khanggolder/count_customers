# Lịch sử chuyển đổi từ BusReID AI

Ngày 2026-10-19, repository `count_customers` được refactor trên nhánh
`feature/busstop-flow-ai`. Nhánh này đã được push lên GitHub; tài liệu này
không mô tả trạng thái deploy hay merge vào `main`.

Đã tái sử dụng OpenCV MP4/RTSP, YOLO person, ByteTrack, ROI counter,
annotated MP4, FastAPI/SQLite/Streamlit và parser PCDS. Session/Re-ID schema
được thay bằng Route/Stop/Trip/Telemetry/StopVisit/PassengerEvent/ReviewAudit.
Pipeline chạy FRONT và REAR độc lập; timestamp capture và processing được phân
biệt. Các module, script và dependency Re-ID đã bị loại khỏi MVP.

Đổi schema **không có migration tự động** từ SQLite của bản cũ. Cần sao lưu DB
cũ và sử dụng DB mới cho phiên bản BusStop Flow AI; xem `README.md` ở gốc repo.
