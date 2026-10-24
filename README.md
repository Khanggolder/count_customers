# BusStop Flow AI

MVP end-to-end đếm khách **lên ở cửa trước (FRONT-IN)** và **xuống ở cửa sau (REAR-OUT)**, rồi tổng hợp theo từng lần xe phục vụ trạm. Đây là giả định vận hành bắt buộc; đơn vị vận tải phải duy trì quy tắc một chiều này khi triển khai. Hệ thống giữ YOLO11n, ByteTrack, MP4/RTSP, FastAPI, SQLite và Streamlit; không có Person Re-ID.

Demo mặc định chạy offline sau khi cài dependency. Scenario v2 tạo chuyến 8 trạm với GPS/tốc độ/cửa và chỉ có event FRONT-IN, REAR-OUT. Mọi màn hình/API đều gắn nhãn `SIMULATED`; đây là kiểm thử nghiệp vụ, không phải độ chính xác AI.

## Bài toán và phạm vi

Mỗi cửa xe có một camera. YOLO11n phát hiện người, ByteTrack theo dõi và bộ đếm ROI chỉ phát sự kiện khi track đi qua vùng theo chiều hợp lệ. Telemetry GPS/tốc độ/cửa xác định `StopVisit`; sự kiện được gán vào trạm khi có đủ bằng chứng thời gian và nguồn. FastAPI/SQLAlchemy/SQLite lưu và cung cấp dữ liệu, Streamlit hiển thị flow theo chuyến/trạm. Sơ đồ và điều kiện gán chi tiết ở [ARCHITECTURE.md](ARCHITECTURE.md) và [BUSINESS_RULES.md](BUSINESS_RULES.md).

MVP không nhận diện danh tính, Re-ID, dự báo hay điều hành xe. Nó chưa được xác nhận trên hai camera và cảm biến của xe thật; không triển khai công khai trên Internet.

## Cấu trúc chính

| Vị trí | Vai trò |
| --- | --- |
| `busreid/` | Namespace kế thừa cho counting, video, telemetry/stop, assignment và scenario; giữ tên để tránh regression trước merge |
| `backend/` | FastAPI, schema và SQLite |
| `app/` | Dashboard Streamlit |
| `configs/` | ROI, camera và detector |
| `scripts/` | Cài đặt, chạy demo/inference, tải clip hợp lệ, gán nhãn và đánh giá |
| `tests/` | Unit/API/software integration; MP4 fixture dùng FakeDetector không đo AI accuracy |
| `docs/archive/` | Tài liệu lịch sử refactor, không phải đặc tả vận hành hiện hành |

`data/raw/`, `data/sample/generated/` và `runs/` chứa dữ liệu/đầu ra local, không commit vào Git.

## Cài đặt Windows 11

Yêu cầu: Windows 11, Python 3.11, Git; NVIDIA driver tương thích nếu dùng GPU. Clone nhánh MVP và vào thư mục repository:

```powershell
git clone --branch main https://github.com/Khanggolder/count_customers.git busstop-flow-ai
cd busstop-flow-ai
```

Sau đó cài đặt:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1 -Cuda
```

Không có NVIDIA hoặc muốn CPU fallback:

```powershell
.\scripts\setup_windows.ps1
```

Script tạo `.venv`, cài dependency, chạy `pip check` và in trạng thái CUDA. `-Cuda` cài wheel PyTorch CUDA 13.0; chỉ dùng khi driver hỗ trợ. Nếu cài đặt hoặc `torch.cuda.is_available()` lỗi, chạy CPU fallback không có `-Cuda`, hoặc chọn wheel tương thích driver theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/) rồi chạy `scripts/check_environment.py`. Không cần GPU để chạy test và demo synthetic.

`requirements.txt` cài project từ `pyproject.toml` cùng công cụ test; hai file có vai trò khác nhau. Model weight và video không được đóng gói trong Git.

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

### Thiết lập vùng đếm (ROI) trên Dashboard

Không cần Roboflow hay sửa YAML bằng tay. Chạy Streamlit như phần trên, mở tab **Thiết lập vùng đếm (ROI)** (không cần API):

1. Chọn **FRONT** hoặc **REAR**. Chọn MP4 trong `data/raw`, upload MP4, nhập đường dẫn MP4, hoặc bấm lấy một frame từ webcam/RTSP được cấp phép. Với MP4, dùng thanh trượt để chọn frame bất kỳ. Camera trực tiếp chỉ dùng để lấy frame hiệu chỉnh; URL RTSP không được ghi vào YAML.
2. Chọn Zone A hoặc B và **Thêm đỉnh**; click các góc trên ảnh theo thứ tự quanh vùng. Chọn **Dời đỉnh** rồi kéo đỉnh tới vị trí mới, hoặc chọn đỉnh trong danh sách và click đích. Có nút xóa đỉnh, hoàn tác, xóa cả vùng và nạp lại cấu hình hiện tại. Bảng dưới ảnh hiển thị tọa độ pixel của frame gốc và normalized `[0, 1]`.
3. Chỉ khi hai polygon có ít nhất ba đỉnh, nằm trong frame, không tự cắt và không chồng lấn, nút **Lưu ROI** mới khả dụng. Bản lưu cập nhật đúng camera trong `configs/cameras.yaml`, tạo `configs/cameras.yaml.bak` để khôi phục (file backup không commit). Nút **Khôi phục bản sao lưu** phục hồi cấu hình trước lần lưu gần nhất. Hãy kiểm tra chiều A → B trên mũi tên: FRONT = IN, REAR = OUT; không đổi chiều nghiệp vụ bằng cách vẽ ngược.
4. Bấm **Test ROI** với MP4 và giới hạn số frame (mặc định 90) sau khi lưu. Worker dùng YOLO11n + ByteTrack thật; xem progress, frame annotated và tải CSV crossing. Bấm cập nhật tiến độ nếu job đang chạy. Sau đó sang tab **Đếm khách AI**, chọn video và bắt đầu phiên inference mới; phiên này đọc ngay ROI vừa lưu. Worker đã chạy giữ ROI được nạp lúc khởi động, không đổi giữa chừng.

Ảnh preview vùng dùng cùng `annotate_frame` của pipeline đếm, cộng màu phủ A/B và mũi tên để dễ hiệu chỉnh. Event trong **Test ROI** là output AI trên đoạn MP4, không phải ground truth hay chỉ số accuracy. Video/camera chưa có nhãn thủ công thì accuracy vẫn **N/A**. Nên chọn frame nơi cửa, lối đi và hướng chuyển động đều rõ; tránh vùng A/B chạm nhau hoặc chứa người đứng yên lâu.

### Xem AI Counting trực quan

Tab **Đếm khách AI** là đường demo ngắn nhất và không cần FastAPI, Trip hay GPS:

1. Chạy `.\.venv\Scripts\python.exe -m streamlit run app\dashboard.py` từ gốc repository.
2. Mở `http://localhost:8501` → **Đếm khách AI**. Chọn MP4 upload/đường dẫn, webcam trên máy chạy Streamlit hoặc RTSP cho cửa trước/sau; có thể chỉ chọn một cửa. `data/raw/in.mp4` và `data/raw/out.mp4` là video cục bộ nếu đã có; chúng không nằm trong Git.
3. Nhấn **Bắt đầu đếm**. Preview là frame YOLO + ByteTrack đã xử lý (box, Track ID/confidence, điểm neo, Zone A/B, hướng và số lượt), cập nhật theo frame thực. MP4 là **video phát lại**, không phải camera live. Chỉ crossing A→B tạo FRONT-IN hoặc REAR-OUT; bộ đếm không tăng theo thời gian.
4. **Dừng đếm** yêu cầu worker kết thúc sau frame hiện tại, giải phóng nguồn và xuất video H.264 của phần đã xử lý. Sau khi worker dừng, **Đặt lại bộ đếm** chỉ xóa phiên trên màn hình, không xóa file. Kết thúc tự nhiên hiển thị cùng loại kết quả.

Kết quả ở `runs/ai_counting/<session-id>/`: `front/rear_preview.jpg`, `front/rear_annotated.mp4`, `front/rear_events.csv`, `front/rear_progress.json`, `session_summary.json`. Mỗi camera có trạng thái và lỗi riêng; camera không chọn hiển thị “Chưa kết nối”, không coi 0 là quan sát. Phiên này **không tạo Trip, telemetry, StopVisit hoặc gán trạm**. Số liệu là output AI thực trên video được chọn, không phải độ chính xác; accuracy là N/A khi chưa có ground truth. ROI lấy từ `configs/cameras.yaml`; có thể hiệu chỉnh ngay trong tab **Thiết lập vùng đếm (ROI)**. Không có webcam/RTSP vật lý trong test tự động.

Tab **Hai cửa** dưới đây vẫn giữ luồng cũ cho AI Video Lab và Trip đã xác minh:

Trong tab **Hai cửa**, chọn **AI Video Lab**. Với mỗi camera có thể upload MP4, chọn đường dẫn MP4 trên máy hoặc nhập RTSP. Chỉ cần một nguồn FRONT hoặc REAR là chạy được; không cần tạo Trip hay GPS. Bấm **Chạy AI Counting**, xem preview annotated cập nhật khoảng mỗi 15 frame, frame/FPS/thời gian và bộ đếm của từng camera. Nút **Hủy inference** tạo yêu cầu dừng giữa các frame. Sau khi hoàn tất, video H.264 có thể phát bằng trình duyệt và timeline cho phép xem crop bằng chứng hoặc tua gần thời điểm event. Các lớp box, Track ID/confidence, ROI, trajectory và event có thể bật/tắt trước khi chạy.

Để thử ngay sau khi tải clip Pexels:

1. Khởi động API và Streamlit theo hai lệnh phía trên, mở `http://localhost:8501`.
2. Vào **Hai cửa → AI Video Lab**. FRONT tự điền đường dẫn clip Pexels local, REAR chọn **Không dùng**.
3. Đặt giới hạn 40 frame để chạy thử nhanh, bấm **Chạy AI Counting** và theo dõi preview, FPS, trạng thái.
4. Khi báo COMPLETED, bấm Play trên video annotated. Thử upload MP4 khác hoặc thêm REAR để so sánh hai camera.

AI Video Lab vẫn chạy khi FastAPI offline; khi đó sidebar báo API offline và các tab phụ thuộc Trip không tải dữ liệu.

Nếu đã tải clip Pexels ở mục dưới, đường dẫn FRONT được chọn sẵn. Clip này quay ngoài xe và không có camera REAR hay nhãn crossing; 0 lượt tại ROI mặc định là kết quả AI thực của clip thử, **không** phải phép đo accuracy. Để trình diễn crossing có ý nghĩa cần MP4 đúng góc cửa và hiệu chỉnh `configs/cameras.yaml` cho chính video đó. Chế độ **Gắn với chuyến thật** giữ luồng telemetry/StopVisit cũ, chỉ cho chọn Trip không phải `SIMULATED`. Kết quả AI Video Lab được lưu là `REAL_VIDEO_AI`, `UNPAIRED`, `UNASSIGNED` khi thiếu telemetry đồng bộ; không ghép vào chuyến mô phỏng tám trạm.

File output nằm trong `runs/video_lab/<run-id>/`: `front/rear_preview.jpg`, `front/rear_annotated.mp4`, `front/rear_events.csv`, progress JSON, crop bằng chứng và `run_summary.json`. Preview là ảnh định kỳ, **không phải RTSP live streaming**. Dashboard dùng `st.fragment` để cập nhật phần job mà không nạp lại model mỗi lần rerun; YOLO được nạp một lần trong mỗi worker camera của run. Upload Streamlit mặc định tối đa 200 MB; với file lớn, chọn đường dẫn MP4 trên máy.

Video annotated được chuyển mã H.264/yuv420p/faststart bằng FFmpeg do `imageio-ffmpeg` cung cấp (hoặc `ffmpeg` trên PATH), sau đó kiểm tra codec, frame và khả năng mở. Nếu thiếu FFmpeg hoặc chuyển mã lỗi, job báo FAILED. Clip nguồn không có audio nên output không có AAC. `imageio-ffmpeg` là wrapper BSD; binary FFmpeg dùng `libx264` có điều kiện GPL riêng theo [FFmpeg Legal](https://ffmpeg.org/legal.html). Binary và video không được commit vào repository.

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

RTSP vật lý và telemetry từ xe thật chưa được kiểm thử. Không nhập credential vào issue, PR, log hay file được Git theo dõi.

## Hiệu chỉnh

Sửa `configs/cameras.yaml` cho từng góc quay:

- `zone_a`, `zone_b`: polygon normalized; A→B là chiều hợp lệ của từng camera (FRONT-IN, REAR-OUT). B→A bị bỏ khỏi PassengerEvent và không chiếm cooldown. Cấu hình sai camera/hướng bị từ chối.
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

Dashboard tab **Hai cửa** cho chọn trip thật, FRONT/REAR, telemetry, capture start, xem preview/progress và video annotated H.264 sau khi xong. Đây là subprocess cục bộ Windows; không dùng làm dịch vụ inference nhiều người dùng. Các endpoint `/runs` hiện chỉ lưu metadata, chưa phải job queue chạy inference. `runs/` và video không được commit.

Demo chính: `POST /demo/generate` có 8 trạm, 44 event synthetic (43 gán trạm, 1 UNASSIGNED), 18 khách lên/25 khách xuống, occupancy ban đầu 8 và cuối chuyến 1. Đối chứng lỗi riêng: `POST /demo/generate?kind=negative_occupancy`, occupancy ban đầu 0 và có cảnh báo âm. Cả hai đều không phải AI output. Schema v2 dùng trip ID `SIM-R01-v2-20261019-0800`; dữ liệu scenario v1 trong SQLite không bị xóa, replay v1 bị từ chối. Flow/export mới loại các event lịch sử FRONT-OUT/REAR-IN; cột cũ `front_out`/`rear_in` còn trong API để tương thích nhưng luôn 0 trong thống kê hợp lệ.

Gán nhãn thủ công và đánh giá:

```powershell
.\.venv\Scripts\python.exe scripts\label_video.py --video D:\video\front.mp4 --output runs\front_gt.csv
.\.venv\Scripts\python.exe scripts\evaluate.py --pred runs\trip-001\front_events.csv --gt runs\front_gt.csv --output runs\front_eval.json
```

Trong cửa sổ nhãn: `I` = IN, `O` = OUT, `Space` = tạm dừng, `Q` = lưu. Với event timestamps, evaluator ghép 1-1 trong tolerance 1 giây để tính precision/recall, đồng thời tính MAE đếm theo clip/chiều. Không có nhãn thì tất cả metric là `N/A` (`null` trong JSON). Các test dùng `FakeDetector` chỉ chứng minh phần mềm nối đúng, không đo AI accuracy.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=busreid --cov=backend -W error
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```

CI Windows/Python 3.11 chạy các lệnh tương tự, không cần GPU/video/credential. Xem kết quả đo theo commit trong [TEST_REPORT.md](TEST_REPORT.md), nguồn dữ liệu trong [DATA_SOURCES.md](DATA_SOURCES.md), giới hạn trong [LIMITATIONS.md](LIMITATIONS.md) và trạng thái nghiệm thu trong [ACCEPTANCE_CHECKLIST.md](ACCEPTANCE_CHECKLIST.md).

## Dữ liệu cũ và hướng tiếp theo

Schema SQLite của phiên bản BusReID trên `main` **không có migration tự động**. Trước khi chuyển môi trường, sao lưu file DB cũ ở vị trí riêng; không trỏ `BUSSTOP_DATABASE_URL` của bản này vào DB cũ. Tạo DB mới bằng `scripts/run_demo.py` hoặc qua API, chỉ nhập dữ liệu thật sau khi kiểm tra provenance và đồng bộ nguồn. Không xóa DB cũ để lấy chỗ.

Ưu tiên tiếp theo: thu thập hai video camera cửa cùng chuyến với GPS/cửa và nhãn thủ công, hiệu chỉnh ROI/clock, đo accuracy, rồi thử RTSP dài hạn và bảo mật trước pilot. Chưa có license riêng được khai báo cho source repository; thư viện và clip công khai tuân theo license của từng nguồn ghi tại [DATA_SOURCES.md](DATA_SOURCES.md).
