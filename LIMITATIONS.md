# Hạn chế

- Chưa có hai video FRONT/REAR cố định trên cùng xe, cùng chuyến và đồng bộ với GPS/cửa. Kiến trúc hai worker đã test bằng fake pipeline; chưa soak-test RTSP vật lý.
- Clip Pexels có license rõ nhưng camera ngoài xe chuyển động, không đúng hình học camera gắn cao tại cửa. Kết quả 0 crossing phản ánh ROI/góc quay không phù hợp, không phải accuracy.
- PCDS chưa tải được từ BaiYun; không có count ground truth đúng clip. `MAE`, precision và recall AI là `N/A`.
- Tọa độ scenario là synthetic. Tên/thứ tự tám trạm lấy từ nguồn bên thứ ba; endpoint chính thức trả 403, nên route/stops được đánh `verified=false`.
- Door signal trong demo là simulated structured telemetry. Chưa xác minh camera của stakeholder có door sensor/API; OCR/CV door-state chỉ là hướng mở rộng.
- ID-switch recovery của counter chỉ chuyển state khi có đúng một ứng viên gần trong grace window; cảnh đông/che khuất vẫn cần calibration và ground truth tại hiện trường.
- Cancellation API hiện ghi `CANCEL_REQUESTED`; chưa nối qua process manager bền vững sau restart. CLI worker có isolation nhưng chưa có distributed queue/WebSocket progress.
- Timestamp video không có capture metadata được ghi rõ `PROCESSING_TIME_NOT_CAPTURE_TIME`; không được diễn giải thành giờ quay.
- Dashboard/API chưa có authentication, TLS, RBAC hoặc retention policy. Chỉ bind localhost cho MVP; không public Internet.
- SQLite và crop local phù hợp demo, chưa phù hợp fleet nhiều xe. Cần PostgreSQL/object storage/message broker khi production.
- Occupancy âm trong scenario cuối chuyến là ca kiểm thử cảnh báo; hệ thống cố ý không clamp để tránh che lỗi initial occupancy/missing event.
- Không có Person Re-ID, nhận diện khuôn mặt/danh tính, OD cá nhân, dự báo hoặc tối ưu lịch xe.
