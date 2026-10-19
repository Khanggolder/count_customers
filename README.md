# BusStop Flow AI

MVP end-to-end đếm hành khách `IN/OUT` ở hai cửa và tổng hợp theo từng lần xe phục vụ trạm. Hệ thống giữ YOLO11n, ByteTrack, MP4/RTSP, FastAPI, SQLite và Streamlit; Person Re-ID đã được loại khỏi runtime/UI.

Demo mặc định chạy hoàn toàn offline sau khi cài dependency. Nó tạo một chuyến deterministic 8 trạm với GPS/tốc độ/cửa và event hai chiều ở hai cửa. Mọi màn hình/API đều gắn nhãn `SIMULATED`; đây là kiểm thử nghiệp vụ, không phải độ chính xác AI.

## Cài đặt Windows 11

Yêu cầu: Python 3.11, Git; NVIDIA driver mới nếu dùng GPU. Từ thư mục repository:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1 -Cuda
```

Không có NVIDIA hoặc muốn CPU fallback:

```powershell
.\scripts\setup_windows.ps1
```

Script tạo `.venv`, cài dependency, chạy `pip check` và in trạng thái CUDA. Bản đã kiểm tra trên máy này dùng PyTorch `2.13.0+cu130`; nếu driver khác, chọn wheel phù hợp từ tài liệu PyTorch chính thức.

## Demo một lệnh

```powershell
.\scripts\run_demo.ps1
```

Hoặc CLI tương đương:

```powershell
.\.venv\Scripts\python.exe scripts\run_demo.py
```

Kết quả:

- `data/sample/generated/scenario.json`: scenario và ground truth synthetic tách riêng.
- `data/busstop_flow.db`: SQLite chứa route, trip, telemetry, StopVisit, event và audit.
- `runs/demo/flow_by_stop.json`: tổng hợp 8 trạm.

Chạy API và dashboard trong hai PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\dashboard.py
```

Mở dashboard tại <http://localhost:8501> và Swagger tại <http://localhost:8000/docs>.

## Chạy video thật

Tải clip Pexels đã xác minh license (không commit video vào Git):

```powershell
.\.venv\Scripts\python.exe scripts\fetch_demo_assets.py --source pexels-bus-boarding --yes
```

Chạy một camera. Chỉ truyền `--recording-start-time` khi biết thời điểm capture thật:

```powershell
.\.venv\Scripts\python.exe scripts\run_count.py `
  --front data\raw\pexels_bus_boarding_31337217.mp4 `
  --config configs\cameras.yaml `
  --output runs\pexels_real_video `
  --trip-id PEXELS-31337217
```

Hai file/camera chạy đồng thời bằng hai worker:

```powershell
.\.venv\Scripts\python.exe scripts\run_count.py `
  --front D:\video\front.mp4 --rear D:\video\rear.mp4 `
  --recording-start-time 2026-10-19T08:00:00+07:00 `
  --output runs\trip-real --trip-id TRIP-REAL-001
```

Mặc định là `UNPAIRED`, event không có telemetry đồng bộ giữ `UNASSIGNED`, occupancy là `N/A`. Chỉ dùng `--paired-verified` khi hai camera thật sự cùng chuyến, đủ coverage và đã hiệu chỉnh clock; cờ này vẫn không tự bịa StopVisit.

## RTSP

Thay path MP4 bằng `rtsp://...`. Mỗi camera có worker, tracker, ROI, clock offset và generation riêng; lỗi một camera không khóa camera còn lại. Không commit URL có credential. URI ghi DB/run summary được redaction, nhưng vẫn nên cấp qua secret local và chỉ bind API vào localhost trong MVP.

## Hiệu chỉnh

Sửa `configs/cameras.yaml` cho từng góc quay:

- `zone_a`, `zone_b`: polygon normalized; A→B mang hướng `event_direction`, B→A là chiều ngược lại.
- `anchor_mode`: `bottom_center` hoặc `center`.
- `min_stable_frames`, `debounce_frames`, `state_ttl_frames`: chống rung/trùng và giới hạn bộ nhớ state.
- `time_offset_ms`: offset clock riêng camera.

Ngưỡng stop mặc định trong `StopVisitDetector` (`speed<=3 km/h`, radius theo stop, dwell 3 giây) là giá trị nghiên cứu, không phải tiêu chuẩn khai thác.

## End-to-end trên trip hiện có

Tạo trip thật qua `POST /trips` với một route/stop đã có; không dùng trip `SIMULATED`. Cung cấp hai MP4/RTSP của **cùng xe/chuyến**, thời điểm bắt đầu capture có timezone và telemetry JSON gồm các mẫu `event_time_utc`, `latitude`, `longitude`, `speed_kmh`, `door_front`, `door_rear`, `source` (`SENSOR`/`CSV`/`JSON`, không phải `SIMULATED`). Đồng hồ video/telemetry phải được kiểm chứng độc lập. Ví dụ:

```powershell
.\.venv\Scripts\python.exe scripts\run_count.py `
  --trip-id TRIP-REAL-001 --front D:\video\front.mp4 --rear D:\video\rear.mp4 `
  --recording-start-time 2026-10-20T08:00:00+07:00 `
  --telemetry-json D:\telemetry\trip-001.json --telemetry-synchronized `
  --output runs\trip-001
```

Lệnh dùng trip hiện có, tạo/kiểm tra `CameraSource`, chạy hai pipeline, nhập telemetry idempotent, phát hiện StopVisit, gán event và ghi `run_summary.json` + video annotated. `--telemetry-synchronized` là lời xác nhận của người vận hành, **không** phải thuật toán tự chứng minh đồng bộ. Event chỉ `AUTO_ASSIGNED` khi nguồn thuộc đúng trip/cửa, capture clock có neo thời gian, StopVisit confirmed và mẫu cửa tương ứng đang mở trong 2 giây trước event. Thiếu bằng chứng cửa thành `REVIEW_REQUIRED`; thiếu GPS/telemetry cùng nguồn hoặc không xác nhận đồng bộ thì `UNASSIGNED`. Public Pexels không có GPS cùng nguồn nên vẫn `UNASSIGNED`. Không dùng `--paired-verified` để ngụ ý coverage đầy đủ nếu chưa kiểm chứng; occupancy thật vẫn `N/A` khi `coverage_complete=false`.

Dashboard tab **Hai cửa** cho chọn trip thật, FRONT/REAR, telemetry, capture start, xem progress theo frame và video annotated sau khi xong. Đây là subprocess cục bộ Windows; không dùng làm dịch vụ inference nhiều người dùng. Các endpoint `/runs` hiện chỉ lưu metadata, chưa phải job queue chạy inference. `runs/` và video không được commit.

Demo chính: `POST /demo/generate` có 8 trạm, 57 event synthetic, occupancy ban đầu 8 và không âm. Đối chứng lỗi riêng: `POST /demo/generate?kind=negative_occupancy`, occupancy ban đầu 0 và có cảnh báo âm. Cả hai đều không phải AI output.

Gán nhãn thủ công và đánh giá:

```powershell
.\.venv\Scripts\python.exe scripts\label_video.py --video D:\video\front.mp4 --output runs\front_gt.csv
.\.venv\Scripts\python.exe scripts\evaluate.py --pred runs\trip-001\front_events.csv --gt runs\front_gt.csv --output runs\front_eval.json
```

Trong cửa sổ nhãn: `I` = IN, `O` = OUT, `Space` = tạm dừng, `Q` = lưu. Với event timestamps, evaluator ghép 1-1 trong tolerance 1 giây để tính precision/recall, đồng thời tính MAE đếm theo clip/chiều. Không có nhãn thì tất cả metric là `N/A` (`null` trong JSON). Các test dùng `FakeDetector` chỉ chứng minh phần mềm nối đúng, không đo AI accuracy.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=busreid --cov=backend
.\.venv\Scripts\python.exe -m ruff check .
```

Xem kết quả đo thật trong [TEST_REPORT.md](TEST_REPORT.md), nguồn dữ liệu trong [DATA_SOURCES.md](DATA_SOURCES.md) và giới hạn trong [LIMITATIONS.md](LIMITATIONS.md).
