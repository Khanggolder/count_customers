# Acceptance checklist — 2026-10-21

## Đã kiểm thử bằng thực thi

- [x] Hai pipeline MP4 → crossing event → SQLite → telemetry → StopVisit → assignment → flow. Integration test `test_end_to_end.py` dùng video fixture và `FakeDetector`; đây là **software integration**, không phải đo độ chính xác YOLO.
- [x] Trip/source/camera/provenance không khớp bị từ chối; AI thật không nhập vào synthetic trip; telemetry synthetic không nhập trip thật.
- [x] GPS missing/stale khi xe dừng và cửa mở không xác nhận rời trạm. RTSP reconnect/timebase được kiểm thử với `VideoCapture` giả; event có capture time riêng processing time.
- [x] Auto-assign chỉ khi nguồn đã xác nhận đồng bộ, StopVisit confirmed và cửa tương ứng có mẫu mở gần event. Thiếu chứng cứ cửa: `REVIEW_REQUIRED`. Video công khai không GPS cùng nguồn: `UNASSIGNED`.
- [x] Counting chỉ phát FRONT-IN/REAR-OUT; reverse không chiếm cooldown. API từ chối cặp sai với 422; bản ghi lịch sử sai chiều được giữ nhưng không vào flow/export/occupancy.
- [x] Demo schema v2 có 8 StopVisit, 44 event synthetic (43 gán trạm, 1 UNASSIGNED), 18 FRONT-IN/25 REAR-OUT, initial occupancy 8, cuối chuyến 1 và không âm. Kịch bản đối chứng âm tách riêng qua `kind=negative_occupancy`; v1 không bị xóa.
- [x] Có tool gán nhãn video cục bộ và evaluator MAE + precision/recall theo timestamp; không có ground truth thì `N/A`. Metric evaluator có unit test, thao tác GUI gán nhãn chưa được kiểm thử tự động.
- [x] Streamlit có form khởi chạy inference cho trip thật, hiển thị progress file và annotated MP4. AppTest render 6 tab, 0 exception; chưa có browser E2E cho nút chạy này.
- [x] Cleanup giữ test hồi quy đạt: 48 passed, coverage 88%, Ruff và `pip check` đạt; baseline trước cleanup là 50 passed (bỏ hai test của module occupancy không dùng trong runtime). Xem `TEST_REPORT.md`.
- [x] YOLO+ByteTrack chạy lại 30 frame video Pexels công khai trên RTX 3050: 0 crossing, không có ground truth, không tự gán trạm.

## Chưa kiểm chứng, không nhận là hoàn thành thực địa

- [ ] GitHub Actions CI mới đã chạy xanh trên remote; chỉ xác nhận sau workflow run thật.
- [ ] Hai camera thật cùng một chuyến với GPS/cửa thật và ground truth gán nhãn: chưa có dữ liệu.
- [ ] RTSP vật lý, reconnect nhiều giờ, sai số đồng hồ thiết bị, tải đồng thời dài hạn: chưa có endpoint để thử. Test hiện dùng fake capture.
- [ ] AI accuracy/MAE/precision/recall trên video cửa xe thật có nhãn: `N/A`.
- [ ] Nút dashboard inference và GUI labeling đã được kiểm thử tương tác người dùng trên máy triển khai: chưa; chỉ có logic/test và smoke CLI.
- [ ] Occupancy thật: không hiển thị nếu coverage/ghép cặp chưa xác nhận.
