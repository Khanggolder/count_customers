# Acceptance checklist — bidirectional MVP (2026-10-26)

## Đã kiểm tra bằng test phần mềm trên máy

- [x] FRONT A→B = IN +1, B→A = OUT -1; REAR A→B = OUT -1, B→A = IN +1. IN/OUT là số lượt độc lập; `delta` tương thích nghĩa là occupancy delta.
- [x] Debounce, stable transition, boundary jitter, đổi Track ID và crossing sau khi quay lại có unit tests; cùng chiều và ngược chiều liên tiếp không bị triệt tiêu.
- [x] Hai MP4 fixture qua FakeDetector → event bốn tổ hợp → SQLite → telemetry đồng bộ → StopVisit → flow. Đây là **software integration**, không phải đo AI accuracy.
- [x] API chấp nhận bốn tổ hợp, từ chối camera/source/trip/provenance hoặc crossing/direction/delta mâu thuẫn; UUID/replay idempotent. Manual direction verification tạo audit.
- [x] SQLite migration sao lưu trước, suy ra forward event từ policy cũ, giữ ambiguous legacy ở `REVIEW_REQUIRED`; không xóa bản ghi. Test có backup và chạy lại.
- [x] Flow/export có front_in, front_out, rear_in, rear_out, total_in, total_out, occupancy. Demo v3: 8 StopVisit, 48 event synthetic (47 assigned), 20 IN, 27 OUT, initial 8 → final 1, không âm. Scenario âm và camera thiếu tách riêng; camera thiếu → occupancy N/A.
- [x] GPS missing/stale không gây departure giả; door evidence/clock/provenance được kiểm tra trước auto-assignment. Video công khai không GPS cùng nguồn giữ UNASSIGNED.
- [x] Dashboard và ROI Editor có AppTest/helper tests; preview/counter/timeline dùng IN, OUT và occupancy delta; lưu ROI chỉ áp dụng phiên worker mới.
- [x] Evaluator có MAE/accuracy IN, OUT riêng, net MAE, timed-event precision/recall. Test 10 IN/10 OUT so với 5 IN/5 OUT cho net MAE = 0 nhưng hai MAE đếm = 5. Thiếu nhãn → N/A.
- [x] `pytest`, `ruff check .`, `pip check` chạy trên local; kết quả thực tế ở `TEST_REPORT.md`.

## Chưa xác minh ngoài test phần mềm

- [ ] Cặp camera vật lý trên cùng xe/chuyến với GPS/door sensor thật và ground truth: chưa có. Không công bố accuracy AI hoặc occupancy thật.
- [ ] RTSP vật lý/reconnect dài hạn, lệch đồng hồ thực tế, mật độ đông/occlusion: chưa thử; RTSP fake capture có test.
- [ ] Browser E2E thao tác người dùng upload → inference → video/ROI click/drag: AppTest không thay thế browser test.
- [ ] Bảo mật production: chưa có auth/TLS/RBAC/retention; chỉ chạy local MVP.
