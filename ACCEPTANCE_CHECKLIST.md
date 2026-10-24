# Acceptance checklist — 2026-10-24

## Đã kiểm thử bằng thực thi

- [x] Hai pipeline MP4 → crossing event → SQLite → telemetry → StopVisit → assignment → flow. Integration test `test_end_to_end.py` dùng video fixture và `FakeDetector`; đây là **software integration**, không phải đo độ chính xác YOLO.
- [x] Trip/source/camera/provenance không khớp bị từ chối; AI thật không nhập vào synthetic trip; telemetry synthetic không nhập trip thật.
- [x] GPS missing/stale khi xe dừng và cửa mở không xác nhận rời trạm. RTSP reconnect/timebase được kiểm thử với `VideoCapture` giả; event có capture time riêng processing time.
- [x] Auto-assign chỉ khi nguồn đã xác nhận đồng bộ, StopVisit confirmed và cửa tương ứng có mẫu mở gần event. Thiếu chứng cứ cửa: `REVIEW_REQUIRED`. Video công khai không GPS cùng nguồn: `UNASSIGNED`.
- [x] Counting chỉ phát FRONT-IN/REAR-OUT; reverse không chiếm cooldown. API từ chối cặp sai với 422; bản ghi lịch sử sai chiều được giữ nhưng không vào flow/export/occupancy.
- [x] Demo schema v2 có 8 StopVisit, 44 event synthetic (43 gán trạm, 1 UNASSIGNED), 18 FRONT-IN/25 REAR-OUT, initial occupancy 8, cuối chuyến 1 và không âm. Kịch bản đối chứng âm tách riêng qua `kind=negative_occupancy`; v1 không bị xóa.
- [x] Có tool gán nhãn video cục bộ và evaluator MAE + precision/recall theo timestamp; không có ground truth thì `N/A`. Metric evaluator có unit test, thao tác GUI gán nhãn chưa được kiểm thử tự động.
- [x] AI Video Lab nhận upload/đường dẫn MP4 hoặc RTSP cho một/hai camera, chạy độc lập với Trip synthetic. Preview JPEG, frame/FPS/counter, timeline, cancel marker và video annotated H.264 có test phần mềm; AppTest render sáu tab và video output, 0 exception.
- [x] Output H.264 được kiểm tra frame/codec trước COMPLETED; thiếu encoder hoặc video lỗi thành FAILED. Fixture dùng FakeDetector, không đo AI accuracy.
- [x] Nhánh AI Video Lab: 56 passed, coverage 87%, Ruff và `pip check` đạt. Cleanup trước đó đạt 48 passed/88%; xem `TEST_REPORT.md`.
- [x] GitHub Actions Windows/Python 3.11 đã chạy xanh tại commit `6f4aec9`; run được liên kết trong `TEST_REPORT.md`.
- [x] YOLO+ByteTrack chạy 40 frame Pexels trên RTX 3050 (2.8972 FPS), thêm smoke hai worker 10+10 frame cùng clip: 0 crossing, không có ground truth, không tự gán trạm. Hai worker cùng clip không phải hai camera thật.

## Chưa kiểm chứng, không nhận là hoàn thành thực địa

- [ ] Hai camera thật cùng một chuyến với GPS/cửa thật và ground truth gán nhãn: chưa có dữ liệu.
- [ ] RTSP vật lý, reconnect nhiều giờ, sai số đồng hồ thiết bị, tải đồng thời dài hạn: chưa có endpoint để thử. Test hiện dùng fake capture.
- [ ] AI accuracy/MAE/precision/recall trên video cửa xe thật có nhãn: `N/A`.
- [ ] Browser E2E upload → chạy → phát video trên máy triển khai và GUI labeling chưa được xác nhận; AppTest và helper tests không thay thế kiểm thử trình duyệt.
- [ ] Occupancy thật: không hiển thị nếu coverage/ghép cặp chưa xác nhận.
