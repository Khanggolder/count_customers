# Quy tắc nghiệp vụ

1. Cửa trước (FRONT) bắt buộc lên xe, chỉ đếm `IN`; cửa sau (REAR) bắt buộc xuống xe, chỉ đếm `OUT`. Đơn vị vận tải phải duy trì giả định vận hành này. FRONT-OUT/REAR-IN không phải PassengerEvent hợp lệ, không tính flow/occupancy.
2. Mỗi track/camera/generation chỉ phát một event cho một transition ổn định; đứng tại ngưỡng, tiến tới rồi quay lại và rung vùng không được tính thêm.
3. StopVisit phục vụ khách cần GPS gần trạm, tốc độ thấp đủ lâu và bằng chứng cửa mở. Tốc độ 0 riêng lẻ, kẹt xe hoặc đèn đỏ không đủ.
4. Thiếu GPS/cửa tạo trạng thái thiếu bằng chứng; không suy đoán thành stop xác nhận.
5. Event ngoài cửa sổ StopVisit là `UNASSIGNED`; trong cửa sổ nhưng thiếu bằng chứng cửa là `REVIEW_REQUIRED`. Sửa tay bắt buộc actor + reason và lưu trước/sau.
6. `event_uuid` unique; replay cùng scenario không tăng số event hoặc telemetry.
7. `Onboard_after = Onboard_before + FRONT-IN - REAR-OUT`. Occupancy chỉ có ý nghĩa với hai nguồn cùng chuyến đã xác nhận ghép cặp, đủ coverage và initial occupancy hợp lệ. Kết quả âm phải cảnh báo.
8. `SIMULATED_EVENTS`, `REAL_VIDEO_AI` và ground truth không được trộn để tạo accuracy.
9. Public video không có telemetry cùng clip không được gắn vào chuyến synthetic như bằng chứng thật.
10. Hệ thống không làm Re-ID, khuôn mặt, danh tính, OD cá nhân, dự báo hay tối ưu lịch xe.
11. Dữ liệu cũ sai chiều được giữ để kiểm toán nhưng bị loại khỏi flow/CSV/JSON; API tạo event và thao tác gán trạm từ chối cặp camera/hướng sai.
