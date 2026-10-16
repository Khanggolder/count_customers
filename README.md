# BusReID AI

MVP end-to-end đếm hành khách lên ở cửa trước, xuống ở cửa sau bằng hai pipeline camera độc lập; lưu sự kiện vào SQLite; cung cấp REST API, dashboard tiếng Việt và Person Re-ID thử nghiệm với cơ chế `UNKNOWN`.

> Không nhận diện danh tính thật. Re-ID chỉ tạo mã hành khách tạm trong session. Với hai video không có identity ground truth, kết quả luôn là `UNVERIFIED` hoặc `UNKNOWN`, không được diễn giải là độ chính xác nhận dạng trên xe buýt.

## Chức năng đã triển khai

- `VideoSource` chung cho MP4, webcam index và RTSP; RTSP có timeout/reconnect và che credential trong log.
- YOLO11n class `person` + ByteTrack; mỗi camera có model/tracker namespace riêng (`front:*`, `rear:*`).
- Counter A→B/B→A bằng bottom-center, stable-frame, hysteresis qua vùng trung gian và debounce.
- Crop selector theo kích thước/độ nét; lưu crop tại crossing.
- OSNet x0.25 checkpoint MSMT17 thật, batch embedding 512-D và L2 normalization.
- Hungarian one-to-one, time constraint `OUT > IN`, ambiguity margin, reject threshold, `UNKNOWN`, `UNVERIFIED`, human review.
- Session/event/source/match/calibration/evaluation models trong SQLite; CSV/JSON export.
- FastAPI + Swagger và Streamlit dashboard tiếng Việt.
- Parser ZIP/folder PCDS, count evaluation và scripts tải asset/model có checksum.

## Cài đặt Windows 10/11

Yêu cầu: Python 3.11, driver NVIDIA mới nếu dùng GPU. FFmpeg CLI là tùy chọn; OpenCV wheel đã đọc/ghi MP4 trực tiếp.

```powershell
Set-Location C:\Users\ad\Downloads\codepython\count_customers
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1 -Cuda
.\.venv\Scripts\python.exe scripts\check_environment.py
```

`-Cuda` cài wheel PyTorch CUDA 13.0 chính thức. Nếu driver/GPU không phù hợp, bỏ cờ để chạy CPU. Lần đầu có thể cần khoảng 2 GB tải xuống.

Linux/CPU:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install Cython==3.1.4 gdown==5.2.0 yacs==0.1.8 tensorboard==2.20.0 chardet==5.2.0
python -m pip install --no-build-isolation -r requirements-reid.txt
```

## Tải asset hợp lệ

```powershell
.\.venv\Scripts\python.exe scripts\fetch_demo_assets.py --source pexels-bus-boarding --yes
.\.venv\Scripts\python.exe scripts\fetch_demo_assets.py --source github-bus-counter --yes
.\.venv\Scripts\python.exe scripts\fetch_reid_model.py --yes
```

Pexels có license rõ nhưng camera ngoài xe di chuyển, chỉ phù hợp detector/tracker smoke test. Clip GitHub có góc tĩnh phù hợp counting hơn nhưng mang watermark Shutterstock và quyền video riêng chưa rõ, chỉ dùng kiểm chứng nội bộ. Xem [DATA_SOURCES.md](DATA_SOURCES.md).

PCDS phải tải hợp lệ từ nguồn tác giả (BaiYun key `2s31`), sau đó:

```powershell
.\.venv\Scripts\python.exe scripts\import_pcds.py --input D:\datasets\PCDS.zip --output data\processed
```

## Chạy inference

Single camera (occupancy luôn `N/A`):

```powershell
.\.venv\Scripts\python.exe scripts\run_count.py `
  --front data\raw\bus_occupancy_counter_input.mp4 `
  --config configs\cameras.yaml `
  --output runs\front_demo
```

Hai camera độc lập:

```powershell
.\.venv\Scripts\python.exe scripts\run_count.py `
  --front D:\video\front.mp4 --rear D:\video\rear.mp4 `
  --config configs\cameras.yaml --output runs\trip01
```

Mặc định hai nguồn là `UNPAIRED`. Chỉ thêm `--paired` khi chắc chắn chúng là hai camera cùng chuyến/timeline; nếu không, occupancy và confirmed identity bị vô hiệu.

Mỗi run tạo annotated MP4, crops, event CSV, summary JSON và một session SQLite. ROI A/B dùng tọa độ normalized trong `configs/cameras.yaml`; phải hiệu chỉnh theo từng camera.

## Chạy API và dashboard

Mở hai PowerShell:

```powershell
# Terminal 1
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload

# Terminal 2
.\.venv\Scripts\python.exe -m streamlit run app\dashboard.py
```

- Dashboard: <http://localhost:8501>
- Swagger: <http://localhost:8000/docs>

Dashboard cho upload/path/RTSP, cấu hình ROI, chạy batch thật, xem annotated video/KPI, event, export và review Re-ID. `NO_VIDEO`, `NO_SOURCE` và `UNPAIRED` được hiển thị rõ.

## Re-ID

Sau khi session có crop IN/OUT:

```powershell
.\.venv\Scripts\python.exe scripts\run_reid.py --session 2 --mode bus-unpaired
```

Chế độ:

- `bus-unpaired`: candidate chỉ `UNVERIFIED`/`UNKNOWN`.
- `paired-gt`: chỉ dùng khi có identity ground truth thực.
- `technical-benchmark`: dành cho dataset Re-ID có nhãn như iLIDS-VID, không phải KPI bus.

Threshold cosine mặc định `0.72`, ambiguity margin `0.05`; similarity không phải xác suất đã hiệu chuẩn.

## Đánh giá và test

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --pred runs\trip01\front_events.csv --gt data\processed\pcds_manifest.csv
.\.venv\Scripts\python.exe -m pytest --cov=busreid --cov=backend
.\.venv\Scripts\python.exe -m ruff check .
```

Count MAE chỉ xuất khi prediction/GT có cùng clip và direction. Rank-1/mAP Re-ID giữ `NA` khi không có identity ground truth.

## Đổi sang RTSP

Core không đổi: truyền URL `rtsp://...` vào `OpenCVVideoSource`/dashboard thay MP4. Trong triển khai xe thật nên lưu credential bằng secret/environment, dùng một worker cho mỗi camera, đồng bộ clock và đặt camera calibration riêng; URL log đã được redaction nhưng URI trong database local vẫn có thể chứa credential nên phải bảo vệ file DB.

Kết quả chạy thực của máy hiện tại nằm trong [TEST_REPORT.md](TEST_REPORT.md); hạn chế ở [LIMITATIONS.md](LIMITATIONS.md).

