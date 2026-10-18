# Quy tắc nghiệp vụ

1. FRONT và REAR đều đếm cả `IN` và `OUT`; không mặc định cửa trước chỉ lên hoặc cửa sau chỉ xuống.
2. Mỗi track/camera/generation chỉ phát một event cho một transition ổn định; đứng tại ngưỡng, tiến tới rồi quay lại và rung vùng không được tính thêm.
3. StopVisit phục vụ khách cần GPS gần trạm, tốc độ thấp đủ lâu và bằng chứng cửa mở. Tốc độ 0 riêng lẻ, kẹt xe hoặc đèn đỏ không đủ.
4. Thiếu GPS/cửa tạo trạng thái thiếu bằng chứng; không suy đoán thành stop xác nhận.
5. Event ngoài cửa sổ StopVisit là `UNASSIGNED`; sửa tay bắt buộc actor + reason và lưu trước/sau.
6. `event_uuid` unique; replay cùng scenario không tăng số event hoặc telemetry.
7. Occupancy chỉ có ý nghĩa với nguồn cùng chuyến, đồng bộ, đủ coverage và initial occupancy hợp lệ. Kết quả âm phải cảnh báo.
8. `SIMULATED_EVENTS`, `REAL_VIDEO_AI` và ground truth không được trộn để tạo accuracy.
9. Public video không có telemetry cùng clip không được gắn vào chuyến synthetic như bằng chứng thật.
10. Hệ thống không làm Re-ID, khuôn mặt, danh tính, OD cá nhân, dự báo hay tối ưu lịch xe.
