# Hạn chế hiện tại

- Chưa có hai video front/rear cùng một chuyến và không có identity ground truth bus. Vì vậy chưa đo được Re-ID accuracy/Rank-1/mAP trên bus; matching bus chỉ `UNVERIFIED`/`UNKNOWN`.
- Chưa tải PCDS do nguồn Google Drive đã bị gỡ và nguồn BaiYun có thể cần quyền/khu vực. Chưa có count ground truth chung nên MAE/precision/recall thực tế là `NA`.
- Clip tĩnh dùng counting có watermark Shutterstock và quyền footage riêng chưa rõ; chỉ dùng nội bộ. Clip Pexels có license rõ nhưng góc camera di chuyển, không đại diện camera onboard cố định.
- Run thực hiện trên clip GitHub đếm 1 IN nhưng clip không có ground truth; đây là prediction quan sát được, không phải accuracy. Nhiều người bị bỏ sót do clip bắt đầu giữa crossing, che khuất và ROI chưa hiệu chỉnh từ dữ liệu có nhãn.
- Dashboard chạy batch đồng bộ. Nút dừng giữa run/realtime WebSocket chưa có; RTSP core có reconnect nhưng chưa soak-test với camera vật lý.
- `POST /sessions/{id}/start` quản lý lifecycle; worker inference hiện được kích hoạt qua dashboard/CLI, chưa có queue nền phân tán.
- Occupancy chỉ có ý nghĩa khi hai nguồn cùng chuyến và offset thời gian đúng. Hệ thống không clamp số âm để tránh che lỗi.
- Peak VRAM báo cáo là tensor memory do PyTorch ghi nhận, không phải toàn bộ VRAM process/driver.
- SQLite và file crop phù hợp MVP local, chưa có encryption/retention scheduler/authentication. Không triển khai public nếu chưa bảo vệ RTSP credentials và dữ liệu hình ảnh.
- Không có nhận diện khuôn mặt, GPS, trạm dừng, thanh toán hay danh tính thật.

