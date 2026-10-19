# Báo cáo kiểm thử

## Refactor quy tắc FRONT-IN / REAR-OUT — 2026-10-20

Chạy lại trên Windows/Python 3.11 tại commit đang chuẩn bị: `pytest --cov=busreid --cov=backend -W error` **50 passed, 88% coverage**, 0 warning; `ruff check .` **All checks passed**; `pip check` **No broken requirements found**.

`scripts/run_demo.py` đã chạy hai lần trên cùng SQLite: cùng trip `SIM-R01-v2-20261019-0800`, 8 StopVisit, 44 event synthetic (43 gán trạm + 1 UNASSIGNED), không tăng bản ghi khi replay. Tổng flow FRONT-IN **18**, REAR-OUT **25**; occupancy ban đầu 8, sau từng trạm **12, 13, 12, 15, 12, 11, 9, 1**, không âm. Kịch bản âm được test riêng, không đưa vào demo chính. Các test mới xác nhận reverse không phát event/không chiếm cooldown, API 422 khi sai cặp, dữ liệu lịch sử sai chiều giữ nguyên trong DB nhưng không vào flow/CSV/JSON và không được manual-assign, scenario v2 chỉ chứa cặp hợp lệ. Hai MP4 fixture + FakeDetector tiếp tục kiểm thử nối phần mềm end-to-end, **không phải AI accuracy**.

Streamlit `AppTest` sau refactor: 6 tab, 0 exception ở trạng thái API chưa kết nối. Không chạy lại YOLO với video công khai trong lần refactor này; số liệu GPU/video ở phần lịch sử bên dưới thuộc commit trước và không được diễn giải là kết quả accuracy của phiên bản mới. Chưa có ground truth clip cửa xe phù hợp: AI MAE/precision/recall = **N/A**. Dashboard chưa được kiểm thử browser E2E sau refactor.

## Kết quả lịch sử ở commit 63379fb (trước refactor; không phải kết quả hiện tại)

Ngày chạy lại: 2026-10-20, Windows, Python 3.11.0.

## Môi trường

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB; driver 610.78.
- PyTorch `2.13.0+cu130`; `torch.cuda.is_available() == True`; CUDA runtime 13.0.
- Ultralytics `8.3.57`, OpenCV `4.10.0.84`.
- `pip check`: `No broken requirements found`.
- FFmpeg/ffprobe CLI: không cài; OpenCV decode/encode MP4 đã chạy.

## Automated tests và lint

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=busreid --cov=backend
.\.venv\Scripts\python.exe -m ruff check .
```

Kết quả regression cuối với `-W error`: **44 passed**, **0 failed**, **0 warning**, 6.23 giây; combined statement coverage **88%**. Ruff: **All checks passed**. `pip check`: không có dependency hỏng.

Integration mới dùng video fixture MP4 hai cửa + `FakeDetector` + telemetry SENSOR + trip thật + StopVisit + bảng flow. Test này xác minh luồng phần mềm, **không đo AI accuracy**. Test API chặn real/synthetic mixing; test GPS stale/missing; test RTSP reconnect bằng fake `VideoCapture`; test event thiếu bằng chứng cửa thành `REVIEW_REQUIRED`; test evaluator `N/A`/MAE/precision/recall.

Streamlit AppTest sau thay đổi: 6 tab, 0 exception ở trạng thái chưa kết nối API. Luồng bấm chạy inference trên browser thật chưa được E2E.

Phạm vi test: A→B/B→A, threshold/reverse/oscillation/multiple people/camera scope, ID switch, TTL; stop giả do tốc độ cao/kẹt xe/đèn đỏ, door missing, GPS missing, thứ tự/repeated stop; assignment/grace/UNASSIGNED/manual audit; clock offset/reconnect monotonic; replay idempotent; hai worker đồng thời và lỗi một camera; aggregate/occupancy âm; API lifecycle/validation/export/run cancel; MP4 pipeline fixture.

## Demo synthetic thực thi

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py
```

- 8 StopVisit `CONFIRMED`.
- 57 event synthetic: 56 event gán trạm + 1 event control `UNASSIGNED`.
- Tổng theo 8 trạm: 24 IN, 32 OUT; có FRONT OUT và REAR IN.
- Demo chính initial occupancy 8; cuối chuyến 0, không có occupancy âm. Kịch bản `negative_occupancy` độc lập dùng initial occupancy 0 để thử cảnh báo.
- Replay lần hai không tăng row/event.
- Đây là test nghiệp vụ deterministic, **không phải AI accuracy**.

## Inference video thật

Smoke mới sau thay đổi: Pexels 31337217, 30 frame trên RTX 3050, 3.1434 processing FPS, 69.2075 MiB peak tensor VRAM, 0 crossing, 0 StopVisit, 0 assigned, `UNPAIRED`, không có GPS cùng nguồn. `run_summary.json` có flow rỗng; ground truth/accuracy vẫn `N/A`. Ultralytics không ghi được settings/cache trong AppData do sandbox quyền hạn nhưng inference hoàn tất exit 0.

Input Pexels 31337217: 3840×2160, 30 FPS, 178 frame, 5.933 s. Command chạy toàn bộ 178 frame với YOLO11n + ByteTrack trên RTX 3050.

- Processing FPS: **6.0114**.
- Peak tensor VRAM: **69.2075 MiB**.
- Model output: **0 crossing event** với ROI FRONT mặc định.
- Annotated output: `runs/pexels_real_video/front_annotated.mp4`.
- Mode `UNPAIRED`, coverage false, event time policy `PROCESSING_TIME_NOT_CAPTURE_TIME`.
- Ground truth: không có; `video_model_validation=NO_GROUND_TRUTH_NA`. Không báo MAE/precision/recall.

## API/UI smoke

- Uvicorn port 8010: `/health` 200; `/trips/SIM-R01-20261019-0800/flow` trả 8 dòng, tổng 24 IN/32 OUT; CSV export 200, 1160 bytes.
- Streamlit headless port 8510: `/` HTTP 200 và trả shell Streamlit.
- Streamlit `AppTest` với API thật: render đủ 6 tab, 4 dataframe, 2 download button, **0 uncaught exception**. Smoke này phát hiện và đã sửa hai lỗi thật: PyArrow/NumPy không tương thích và JSON export bị truyền list thay vì bytes.

## Integration chưa chạy

- RTSP camera vật lý: không có endpoint/camera, nên chưa kiểm chứng reconnect/soak ngoài unit test.
- PCDS real labeled benchmark: dataset không tải được hợp lệ trong phiên; metric `N/A`.
- Hai camera cùng chuyến + GPS thật: không có dữ liệu stakeholder; occupancy thật chưa kiểm chứng.
