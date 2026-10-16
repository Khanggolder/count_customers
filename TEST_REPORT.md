# Báo cáo kiểm thử thực tế

Ngày chạy: 2026-10-17, Windows, Python 3.11.0, NVIDIA GeForce RTX 3050 Laptop GPU 4 GB, driver 610.78.

## Môi trường

- PyTorch `2.14.1+cu130`, CUDA runtime 13.0, `torch.cuda.is_available() == True`.
- Ultralytics `8.3.57`, OpenCV `4.10.0.84`, torchreid `1.4.0` từ commit official `f8cd150...`.
- FFmpeg/ffprobe CLI: không cài. OpenCV decode/encode MP4 đã test pass.

## Automated tests

Lệnh:

```text
.\.venv\Scripts\python.exe -m pytest --cov=busreid --cov=backend --cov-report=term-missing
```

Kết quả cuối: **24 passed**, 1 warning, 4,21 giây; combined statement coverage **83%**. Test gồm crossing state machine, đứng ngưỡng, đảo hướng, debounce, multi-person, namespace camera, track mất; matching threshold/time/one-to-one/ambiguity/UNKNOWN; occupancy âm/unpaired; quality crop; MP4 decode; parser PCDS (gồm format per-clip chính thức); API/SQLite/export; pipeline video với fixture mang nhãn synthetic.

Lệnh lint:

```text
.\.venv\Scripts\python.exe -m ruff check .
```

Kết quả: **All checks passed**.

Evaluator CLI cũng chạy pass với GT fixture mang nhãn `SYNTHETIC_EVALUATOR_SMOKE` (1 cặp direction, MAE 0.0). Con số này chỉ kiểm tra phép tính/định dạng, không phải metric dataset thật; PCDS MAE vẫn `NA`.

## Inference thật — clip cửa trước GitHub

Output: `runs/github_bus_e2e/` và SQLite session `1`.

- Input: 900×474, 25 FPS, 281 frame, 11,24 giây.
- YOLO11n + ByteTrack + counter, RTX 3050: **41,21 processing FPS**.
- Peak tensor VRAM: **69,21 MiB**.
- Prediction: **1 IN, 0 OUT**; event `front:track00015`, frame `233`, timestamp `9,32 s`, detector confidence `0,698`.
- Event/crop không hardcode; sinh từ pipeline thật. Không báo accuracy do video không có GT.
- Annotated MP4: `runs/github_bus_e2e/front_annotated.mp4`.
- Event CSV: `runs/github_bus_e2e/front_events.csv`.

## Hai pipeline thật, nguồn UNPAIRED

Output: `runs/two_camera_unpaired/`, SQLite session `2`.

- FRONT clip GitHub: 281 frame, 1 event, 43,52 FPS, peak 69,21 MiB.
- REAR dùng clip Pexels khác nguồn: 178 frame 4K, 0 event, 8,02 FPS, peak 80,04 MiB.
- Mode: `UNPAIRED`; occupancy `N/A`; không confirmed identity. Đây chỉ chứng minh hai detector/tracker/counter namespace độc lập, không mô phỏng hai camera cùng chuyến.

## Re-ID thật

- Checkpoint OSNet x0.25 MSMT17 chính thức load thành công; classifier train-time bị bỏ do shape khác, backbone weights được load.
- Crop thật từ event frame 233 → embedding 512-D, L2 norm 1.0 trên CUDA.
- 50 lượt encode batch=1: **58,07 ảnh/giây**, peak tensor VRAM **12,36 MiB**.
- Similarity crop với bản flip trong CPU smoke trước đó: `0,9840`; đây chỉ là invariance smoke test, **không phải accuracy**.
- `run_reid --session 2 --mode bus-unpaired` trả `[]` vì không có event OUT; hệ thống không bịa candidate.

## API/UI smoke

- Uvicorn khởi động thành công; `/health`, `/device`, `/sessions`, `/sessions/1/stats`, `/sessions/1/events`, `/sessions/1/export?format=csv` đều trả HTTP 200.
- Session 1 API: IN=1, OUT=0, occupancy=`null`, warning=`UNPAIRED`.
- CSV export 170 bytes chứa đúng một event thật, `synthetic=False`.
- Streamlit khởi động tại port 8501; `/` HTTP 200 và `/_stcore/health` trả `ok`.

