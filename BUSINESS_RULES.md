# Quy tắc nghiệp vụ — đếm hai chiều

| Camera | Crossing | Passenger direction | Occupancy delta |
| --- | --- | --- | ---: |
| FRONT | A_TO_B | IN | +1 |
| FRONT | B_TO_A | OUT | -1 |
| REAR | A_TO_B | OUT | -1 |
| REAR | B_TO_A | IN | +1 |

`delta` còn xuất trong CSV để tương thích, nhưng luôn có nghĩa là **occupancy delta**, không phải số lượt IN hoặc OUT ròng. Mỗi crossing hoàn tất tạo một event; tiếp cận rồi quay đầu trước zone B không tạo event. Debounce, stable transition, TTL và ID recovery tránh đếm lặp do jitter/đổi track. Return crossing đã hoàn tất được đếm đúng chiều mới.

Flow mỗi StopVisit giữ riêng `front_in`, `front_out`, `rear_in`, `rear_out`. `total_in = front_in + rear_in`; `total_out = front_out + rear_out`; `onboard_after = onboard_before + total_in - total_out`. Không clamp occupancy âm: đánh dấu `NEGATIVE_OCCUPANCY_REVIEW_REQUIRED`. Chỉ công bố occupancy khi hai camera cùng chuyến được xác nhận ghép cặp, coverage đầy đủ và initial occupancy có cơ sở. Một camera hoặc coverage thiếu → occupancy `N/A`.

StopVisit cần GPS trong geofence, tốc độ thấp đủ dwell và bằng chứng cửa mở. GPS missing/stale không được xác nhận rời trạm sai. Event muốn auto-assign cần nguồn thuộc Trip, camera/cửa/provenance hợp lệ, capture clock có neo đồng bộ, StopVisit confirmed và mẫu cửa tương ứng đang mở gần event. Thiếu door evidence → `REVIEW_REQUIRED`; thiếu đồng bộ/StopVisit → `UNASSIGNED`. Sửa tay cần actor, reason và audit trail.

API/CLI/replay/manual review cùng kiểm tra ánh xạ bốn trường hợp. Source/Trip hoặc provenance mâu thuẫn bị từ chối; `event_uuid` unique và replay idempotent. Event lịch sử chỉ suy ra được từ policy cũ mang `INFERRED_LEGACY`; event thiếu bằng chứng hướng mang `REVIEW_REQUIRED`, giữ để kiểm toán nhưng không tính vào flow cho tới khi xác minh. Không trộn `SIMULATED_EVENTS`, `REAL_VIDEO_AI` và ground truth để công bố accuracy.

Không có Person Re-ID, nhận dạng danh tính/khuôn mặt, OD cá nhân, dự báo hoặc tối ưu lịch xe.
