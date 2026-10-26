# Hạn chế của MVP

- Chưa có hai video FRONT/REAR gắn trên cùng xe/chuyến và telemetry GPS/cửa cùng nguồn kèm ground truth. Integration test dùng MP4 fixture và FakeDetector chỉ chứng minh luồng phần mềm. Public Pexels không có GPS cùng nguồn nên event vẫn UNASSIGNED. AI accuracy, MAE/precision/recall trên camera xe thật là N/A.
- Clip local `in.mp4`/`out.mp4` chưa chứng minh cùng một chuyến hoặc được gán nhãn. Số AI thấy được trên dashboard không phải ground truth hay occupancy thật. Chỉ một camera hoặc coverage chưa xác minh thì occupancy N/A dù IN/OUT của camera đó vẫn hiển thị.
- ROI và tracker cần hiệu chỉnh tại hiện trường. Jitter, che khuất, ID switch đông người và người đi sát ranh giới có thể gây sai. Unit test bao phủ state machine nhưng không xác nhận hiệu năng CV trong môi trường thực.
- RTSP vật lý, reconnect lâu dài, clock drift thực tế, webcam và door sensor/API xe thật chưa được kiểm chứng. RTSP fake capture test không thay thế thử nghiệm hiện trường. MP4 không có recording start time đáng tin thì không suy ra giờ capture từ giờ xử lý.
- ROI Editor có test helper/AppTest, chưa xác minh click/kéo bằng browser thật trên mọi cấu hình. Worker lấy snapshot ROI lúc bắt đầu; session Streamlit restart không phục hồi điều khiển job đang chạy. Video/crop local trong `runs/` cần quản lý dung lượng riêng.
- Dữ liệu scenario v3 là synthetic, tọa độ stop là giả lập; tên/thứ tự trạm tham khảo nguồn bên thứ ba chưa xác minh chính thức. Demo có 20 IN/27 OUT và occupancy cuối 1; scenario âm riêng cố ý không clamp. Incomplete-camera scenario giữ occupancy N/A.
- Migration SQLite giữ toàn bộ event cũ và tạo backup. `INFERRED_LEGACY` chỉ dựa trên policy cũ FRONT-IN/REAR-OUT, không phải xác minh từng video. Bản ghi thiếu crossing đáng tin là `REVIEW_REQUIRED`; không suy `delta=-1` cũ thành OUT. Backup phải được người vận hành kiểm tra và lưu an toàn.
- Dashboard/API chưa có authentication, TLS, RBAC hoặc retention policy; chỉ bind localhost trong MVP. SQLite và worker cục bộ chưa phù hợp triển khai nhiều xe/đa người dùng.
- Không có Person Re-ID, danh tính/khuôn mặt, OD cá nhân, dự báo hay tối ưu lịch xe.
