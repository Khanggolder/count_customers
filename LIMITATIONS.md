# Hạn chế

- Tab **Đếm khách AI** đã chạy trên hai MP4 `in.mp4`/`out.mp4` do người dùng đặt tại máy; chưa có nhãn khách thực tế hay bằng chứng hai clip đồng bộ cùng chuyến. FRONT sinh hai event cho cùng một Track ID (frame 20/65), nên con số hiện tại cần review và không được coi là ground truth.
- Webcam và RTSP vật lý chưa được thử. Khi Streamlit restart, tiến trình phiên đang chạy không có cơ chế khôi phục điều khiển; file kết quả còn trong `runs/ai_counting/`. Dừng xử lý được kiểm thử qua marker/worker trên MP4, không phải nút bấm trong browser thật.

- Chưa có hai video FRONT/REAR cố định trên cùng xe, cùng chuyến và đồng bộ với GPS/cửa. Kiến trúc hai worker đã test bằng fake pipeline; chưa soak-test RTSP vật lý.
- Clip Pexels có license rõ nhưng camera ngoài xe chuyển động, không đúng hình học camera gắn cao tại cửa. Kết quả 0 crossing phản ánh ROI/góc quay không phù hợp, không phải accuracy.
- PCDS chưa tải được từ BaiYun; không có count ground truth đúng clip. `MAE`, precision và recall AI là `N/A`.
- Tọa độ scenario là synthetic. Tên/thứ tự tám trạm lấy từ nguồn bên thứ ba; endpoint chính thức trả 403, nên route/stops được đánh `verified=false`.
- Door signal trong demo là simulated structured telemetry. Chưa xác minh camera của stakeholder có door sensor/API; OCR/CV door-state chỉ là hướng mở rộng.
- ID-switch recovery của counter chỉ chuyển state khi có đúng một ứng viên gần trong grace window; cảnh đông/che khuất vẫn cần calibration và ground truth tại hiện trường.
- Quy tắc FRONT-IN/REAR-OUT là giả định vận hành bắt buộc; nếu thực tế hành khách đi sai cửa, chuyển động đó không được tính là lưu lượng hợp lệ và cần quy trình vận hành/review riêng. Chưa đo tỷ lệ vi phạm ở hiện trường.
- SQLite cũ có thể còn bản ghi sai chiều: chúng được giữ để kiểm toán, không được đếm trong flow/export mới. Chưa có công cụ migration gắn nhãn hàng loạt cho dữ liệu lịch sử.
- Nút hủy AI Video Lab ghi marker cục bộ để worker dừng giữa các frame; cancellation API hiện ghi `CANCEL_REQUESTED` và chưa nối qua process manager bền vững sau restart. Chưa có distributed queue/WebSocket progress; preview chỉ là JPEG cập nhật định kỳ.
- Dashboard job nằm trong session Streamlit; restart Streamlit không tự khôi phục điều khiển job đang chạy. Upload mặc định giới hạn 200 MB; file lớn dùng đường dẫn MP4 local. RTSP vật lý chưa được xác nhận trên dashboard.
- Timestamp video không có capture metadata được ghi rõ `PROCESSING_TIME_NOT_CAPTURE_TIME`; không được diễn giải thành giờ quay.
- Dashboard/API chưa có authentication, TLS, RBAC hoặc retention policy. Chỉ bind localhost cho MVP; không public Internet.
- SQLite và crop local phù hợp demo, chưa phù hợp fleet nhiều xe. Cần PostgreSQL/object storage/message broker khi production.
- Demo chính có occupancy hợp lệ; scenario đối chứng `negative_occupancy` riêng mới tạo cảnh báo âm. Hệ thống cố ý không clamp để tránh che lỗi initial occupancy/missing event.
- Không có Person Re-ID, nhận diện khuôn mặt/danh tính, OD cá nhân, dự báo hoặc tối ưu lịch xe.
