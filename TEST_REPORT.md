# Báo cáo kiểm thử BusStop Flow AI

## Cleanup trước merge — 2026-10-21

Trên Windows/Python 3.11.0, baseline của commit `d185153385365e59b58f412bb0256371ce3b1806` trước cleanup:

| Kiểm tra | Kết quả baseline |
| --- | --- |
| `pytest --cov=busreid --cov=backend -W error` | 50 passed, coverage 88% |
| `ruff check .` | All checks passed |
| `pip check` | No broken requirements found |

Sau khi bỏ `busreid/occupancy.py` và hai test chỉ dành riêng cho module này (không được runtime gọi), chạy lại cùng lệnh: **48 passed, coverage 88%, Ruff pass, pip check pass**. Các bài test occupancy nghiệp vụ trong scenario/API và flow vẫn giữ nguyên. GitHub Actions Windows/Python 3.11 tại commit `6f4aec9338138760c8282d7bfe2d8752daeba861` đã [hoàn tất success](https://github.com/Khanggolder/count_customers/actions/runs/39624228424): cài dependency, `pip check`, Ruff và pytest đều qua. Đây là kết quả CI của commit được nêu, không phải phép đo AI accuracy.

Smoke cleanup với DB riêng `runs/cleanup_smoke/flow.db` (không động vào DB người dùng): chạy `scripts/run_demo.py` hai lần cho cùng scenario, cả hai trả 8 StopVisit, 44 event và `idempotent=true`. JSON flow có 8 trạm, tổng 18 FRONT-IN / 25 REAR-OUT, occupancy `[12, 13, 12, 15, 12, 11, 9, 1]`. Streamlit `AppTest` render 6 tab, 0 exception khi API không kết nối; đây không phải browser E2E.

Test hồi quy bao gồm FRONT-IN/REAR-OUT, reverse crossing không phát event, không đếm trùng, StopVisit cần GPS/tốc độ/cửa, loại dừng giả do kẹt xe/đèn đỏ, provenance synthetic/real, assignment và review, replay idempotent, API validation/export, RTSP reconnect/timebase qua fake capture. Integration test hai MP4 fixture dùng **FakeDetector** để kiểm thử luồng phần mềm từ event đến flow theo trạm; đây **không phải** phép đo AI accuracy.

## Demo và inference đã đo ở lần phát triển trước

Tại commit `d185153` (trước cleanup này), demo synthetic schema v2 đã chạy trên SQLite mặc định: trip `SIM-R01-v2-20261019-0800`, 8 StopVisit, 44 event synthetic (43 gán trạm, 1 `UNASSIGNED`). Cleanup xác nhận lại cùng kết quả trên DB riêng như trên. Kịch bản âm được tách riêng. Đây là dữ liệu mô phỏng deterministic, không phải dự đoán YOLO.

Smoke YOLO11n + ByteTrack tại lần phát triển trước đã chạy 30 frame clip Pexels 31337217 trên RTX 3050: 3.1434 processing FPS, 0 crossing với ROI mặc định. Lần chạy toàn clip 178 frame đạt 6.0114 processing FPS, cũng 0 crossing. Clip quay từ ngoài xe, không phải camera cố định tại cửa và không có telemetry đồng bộ. Kết quả không gán trạm; 0 crossing **không** chứng minh accuracy. Không chạy lại inference GPU trong cleanup này.

## Chưa kiểm chứng

- Chưa có ground truth video cửa xe thật: MAE/precision/recall AI = `N/A`.
- Chưa thử hai camera vật lý cùng xe với GPS/cảm biến cửa đồng bộ, hoặc RTSP soak-test dài hạn.
- Chưa kiểm thử browser E2E cho thao tác inference dashboard; Streamlit AppTest/smoke trước đó không thay thế được kiểm thử này.
- Mỗi commit mới cần kiểm tra lại CI riêng; kết quả run trên áp dụng cho `6f4aec9`.
