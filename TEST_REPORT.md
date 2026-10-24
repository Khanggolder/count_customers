# Báo cáo kiểm thử BusStop Flow AI

## Hoàn thiện ROI PR #4 — 2026-10-24

Windows/Python 3.11: `pytest --cov=busreid --cov=backend -W error` **104 passed**, combined coverage **87%**; Ruff và `pip check` đạt. Test mới dùng FakeDetector/MP4 fixture để xác nhận seek từ frame tuyệt đối, giới hạn đoạn xử lý, timestamp/CSV/video annotated, FRONT-IN/REAR-OUT, không đếm chiều ngược; FakeDetector chỉ kiểm thử phần mềm, không đo AI accuracy. Test thêm polygon 0–n đỉnh, lỗi webcam/RTSP qua fake capture, giữ draft khi lỗi, backup/restore riêng FRONT/REAR, snapshot YAML cho worker và Streamlit AppTest xóa vùng/chuyển camera/MP4 thiếu không crash. Browser click/kéo thật **chưa kiểm chứng**: IAB không khả dụng, Windows Computer Use lỗi native pipe sau retry/reset. Checklist thủ công nằm trong README.

Smoke YOLO11n + ByteTrack thật với `data/raw/in.mp4` local, RTX 3050 Laptop GPU: start frame 450, max 15 → COMPLETED, 15 frame annotated H.264, `next_frame_index=465`, 4.22 processing FPS, 0 crossing. Thử frame 10–69 (60 frame) → COMPLETED, 0 crossing. Hai đoạn không có nhãn ground truth; 0 là kết quả model trong đoạn đó, không phải độ chính xác. `runs/roi_seek_smoke_20261024/` và `runs/roi_event_smoke_20261024/` bị ignore, không commit.

## ROI polygon editor — 2026-10-24 (`feature/roi-polygon-editor`)

Windows/Python 3.11: **77 passed** (`pytest -q -o addopts= --disable-warnings`), `ruff check .` và `pip check` đạt. Test mới xác nhận tọa độ pixel ↔ normalized, loại polygon thiếu đỉnh/tự cắt/ngoài frame/chồng lấn, lưu/tải và khôi phục backup YAML, giữ FRONT-IN/REAR-OUT, `CameraPipeline` mới dùng ROI vừa lưu còn instance đang chạy giữ ROI cũ, thao tác click/dời đỉnh, đọc frame MP4 và dashboard vẫn render các tab cũ.

Smoke YOLO11n + ByteTrack thật: `data/raw/in.mp4` local, FRONT 20 frame, exit 0, 0 event, 4.25 processing FPS, có H.264/CSV/preview/progress tại `runs/roi_smoke_20261024/`. Đây chỉ là kiểm tra đường chạy inference ngắn với ROI hiện tại; **không** đo accuracy và **chưa** kiểm thử thao tác click/kéo thả trong browser, webcam/RTSP vật lý hoặc quy trình hiệu chỉnh trên video có nhãn.

## Tab Đếm khách AI — 2026-10-24 (nhánh `feature/ai-counting-tab`)

Tab mới dùng phiên độc lập `runs/ai_counting/`, không tạo Trip/telemetry/StopVisit. Windows/Python 3.11: **68 passed**, `ruff check .` và `pip check` đều đạt. Test phần mềm gồm AppTest empty/start/stop/reset/rerun/completed/MP4 lỗi hoặc thiếu nguồn, FRONT/REAR một nguồn, preview annotated, chiều ngược, stop giải phóng nguồn và giữ H.264/CSV từng phần, cô lập lỗi camera, webcam source và session không có Trip. Fixture dùng **FakeDetector**, không phải AI accuracy. Kiểm thử trình duyệt tương tác và camera webcam/RTSP vật lý vẫn chưa thực hiện.

Chạy YOLO11n + ByteTrack thật trên RTX 3050 Laptop GPU với hai MP4 local `data/raw/in.mp4` (320×240, 770 frame, 25 FPS) và `data/raw/out.mp4` (320×240, 414 frame, 25 FPS). Đây là hai video người dùng cung cấp trên máy; không commit và chưa xác minh cùng xe/chuyến hay nhãn ground truth. ROI trong `configs/cameras.yaml` được ước lượng từ frame đầu và chưa hiệu chỉnh theo nhãn.

| Run | FRONT | REAR | Output |
| --- | --- | --- | --- |
| 120 frame mỗi cửa | 2 IN, 21.96 processing FPS | 1 OUT, 22.65 processing FPS | 2 video H.264, 2 CSV |
| Toàn clip sau sửa khóa preview Windows | 2 IN / 770 frame, 47.84 processing FPS | 1 OUT / 414 frame, 40.16 processing FPS | 2 video H.264, 2 CSV, không lỗi worker |

Số đếm trên là **crossing event của model**, không phải số khách đã đối chiếu. Hai event FRONT ở frame 20 và 65 cùng Track ID; có nguy cơ đếm lặp do quay lại/dao động vùng, cần gán nhãn video và hiệu chỉnh ROI/debounce. Accuracy, MAE, precision/recall là `N/A`. Lượt chạy toàn clip đầu gặp `WinError 5` khi thay JPEG preview của FRONT nhưng REAR vẫn hoàn tất; sau khi thêm retry và cho preview lỗi không làm dừng inference, lượt chạy lại hoàn tất hai cửa.

Smoke cô lập lỗi thật: FRONT trỏ đến MP4 không tồn tại, REAR xử lý 10 frame `out.mp4`, tạo output H.264 và `session_summary.json` báo riêng lỗi FRONT. Không có event ở 10 frame REAR này.

Không diễn giải các số trên thành đánh giá độ chính xác AI.

## AI Video Lab — 2026-10-24 (nhánh `feature/ai-video-lab`)

Windows/Python 3.11.0: `pytest --cov=busreid --cov=backend -W error` **56 passed**, combined coverage **87%**; `ruff check .` và `pip check` đều đạt. Test mới gồm upload MP4, chọn path/RTSP, empty state, dashboard sáu tab và video component, FRONT-IN/REAR-OUT, reverse không đếm, progress/cancel/failure, kiểm tra output H.264, timeline. Các video fixture dùng **FakeDetector**; chúng chỉ kiểm thử phần mềm, không đo accuracy AI. Streamlit AppTest xác nhận video element render; không phải browser E2E.

Inference YOLO11n + ByteTrack thật trên RTX 3050 với clip Pexels 31337217 tại máy này:

| Run | Frames | Processing FPS | Events | Output |
| --- | ---: | ---: | ---: | --- |
| FRONT, 40 frame | 40 | 2.8972 | 0 | H.264 mở được, 40 frame |
| FRONT, 15 frame sau chỉnh overlay | 15 | 2.5062 | 0 | H.264 mở được; preview 1280 px |
| FRONT + REAR, cùng clip chỉ để smoke hai worker | 10 + 10 | 2.0531 / 2.0812 | 0 + 0 | Hai output H.264, không lỗi worker |

Hai worker trong smoke cùng dùng **một** clip, nên đây không phải hai camera vật lý và không xác nhận ghép chuyến. Clip quay ngoài xe, ROI mặc định không phù hợp để benchmark crossing. Không có nhãn cùng clip, accuracy/MAE/precision/recall AI đều `N/A`; không có telemetry nên các run giữ `UNASSIGNED` và flow trạm rỗng. Run 40 frame đầu đã thêm một Trip video-only vào SQLite mặc định; các run sau dùng DB riêng trong `runs/`.

Smoke từ helper upload của dashboard: đã sao chép clip vào `runs/video_lab/<run-id>/uploads/`, chạy subprocess YOLO trên 3 frame với DB riêng, exit `0`, có `front_annotated.mp4` H.264 và `run_summary.json`; provenance `REAL_VIDEO_AI`, mode `UNPAIRED`, 0 event. Đây là xác nhận đường chạy upload → worker, chưa phải click/upload trong trình duyệt.

Kiểm thử trình duyệt tương tác chưa thực hiện được: công cụ Computer Use của phiên này lỗi native pipe sau retry và reset. Cần kiểm tra thủ công upload → Start → preview → Play trên `http://localhost:8501`.

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
