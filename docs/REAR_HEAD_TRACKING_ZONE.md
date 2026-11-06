# REAR Head Tracking + Zone OUT (thử nghiệm)

Chế độ `head_tracking_zone_out` đếm **lượt đầu đi qua cửa sau theo chiều A → B**, không ước lượng số người trong xe. Nó chạy trong phiên AI độc lập, xuất file riêng và **không ghi** `PassengerEvent`, Trip, SQLite hoặc thống kê theo trạm. FRONT-IN và hai chế độ REAR cũ (`door_out`, `occupancy_headcount`) giữ nguyên. Số OUT thử nghiệm được hiển thị riêng và không cộng vào tổng nghiệp vụ.

## Model, tracking và điều kiện đếm

Checkpoint mặc định là `11n-head.pt` của [wesjos/YOLO-head-detection](https://huggingface.co/wesjos/YOLO-head-detection), một class head, license AGPL-3.0. Lần đầu chạy, worker tải checkpoint vào `runs/models/` (Git bỏ qua); có thể đặt file `.pt` tin cậy ở máy và cấu hình `rear_head_out.model`. Model được nạp một lần cho mỗi worker, không nạp mỗi frame. Ultralytics `model.track(..., persist=True, classes=[0])` dùng ByteTrack mặc định hoặc BoT-SORT qua `rear_head_out.tracker`. `device: auto` có thể dùng CUDA khi môi trường hỗ trợ; `device: cpu` dùng CPU. Confidence, IoU và `imgsz` chỉnh trong `configs/cameras.yaml`.

Mỗi Track ID dùng **tâm bounding box đầu người** làm anchor. Bộ đếm yêu cầu: ổn định trong A ít nhất `min_stable_frames`, ổn định trong vùng chuyển tiếp ngoài A/B, rồi ổn định trong B. Một lần như vậy tạo đúng một event OUT. Đứng yên, A → A, B → A, nhảy thẳng A → B không có vùng chuyển tiếp xác nhận, hoặc mất track trước B đều không tạo event. Mỗi track có trạng thái riêng; track cũ bị xóa sau `state_ttl_frames`. ID mới chỉ thừa kế trạng thái cũ khi trong `id_switch_grace_frames` có **đúng một** track gần hơn `id_switch_max_distance` theo tọa độ normalized. Khi có nhiều ứng viên, track mới bị chặn đếm để tránh tự suy diễn danh tính. Điều này ưu tiên hạn chế đếm sai nhưng có thể bỏ sót khi đông người, che khuất hoặc ID switch.

## Hiệu chỉnh ROI và Zone

Sửa riêng khối `rear_head_out` trong `configs/cameras.yaml`:

```yaml
rear_head_out:
  roi: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
  zone_a: [[0.10, 0.20], [0.30, 0.20], [0.30, 0.80], [0.10, 0.80]]
  zone_b: [[0.70, 0.20], [0.90, 0.20], [0.90, 0.80], [0.70, 0.80]]
```

Các số trong ví dụ chỉ minh họa cú pháp, **không phải** vị trí mặc định phù hợp cho mọi camera. Tọa độ normalized bằng `x_pixel / chiều_rộng`, `y_pixel / chiều_cao`. Mặc định trong YAML lấy zone cửa REAR hiện có làm điểm bắt đầu, nhưng zone này trước đây được hiệu chỉnh cho chân người, nên cần kiểm tra lại **tâm đầu** trên video thực. A phải nằm phía trong xe, B phía ngoài hoặc phía hành khách đang đi xuống. Để có vùng chuyển tiếp, A và B cần cách nhau đủ xa cho ít nhất `min_stable_frames` trong mỗi vùng. Polygon phải hợp lệ, không tự cắt/chồng lấn, nằm trong `[0,1]`.

Tab **Thiết lập vùng đếm (ROI)** hiện chỉ sửa zone của pipeline person FRONT/REAR; nó chưa sửa khối `rear_head_out`. Với thử nghiệm này, chỉnh YAML và chạy video ngắn để xem ROI/A/B trực tiếp trên video annotated. Mỗi phiên dashboard chụp snapshot YAML lúc bắt đầu nên không đổi ROI của worker đang chạy. Overlay có thể bật/tắt box, Track ID, quỹ đạo, zone và thông báo OUT trong phần tùy chọn của tab **Đếm khách AI**.

## Chạy trên Windows

Từ thư mục `busstop-flow-ai` sau khi cài dependencies theo README:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\dashboard.py
```

Trong tab **Đếm khách AI**, chọn REAR → **Thử nghiệm Head Tracking + Zone OUT**, chọn MP4/RTSP được phép rồi bấm **Bắt đầu đếm**. Có thể chọn thêm FRONT cùng phiên. Dashboard hiển thị preview khi xử lý, video H.264 sau khi hoàn tất, bộ đếm `REAR HEAD-OUT` và nút tải CSV. Nếu Windows tạm khóa JPEG preview khi worker thay ảnh, dashboard bỏ qua lần đọc đó và cập nhật ở chu kỳ kế tiếp.

CLI không cần Streamlit:

```powershell
.\.venv\Scripts\python.exe scripts\run_count_session.py --rear "C:\path\to\rear.mp4" --rear-mode head_tracking_zone_out --config configs\cameras.yaml --output runs\head_zone_trial --stop-file runs\head_zone_trial\stop.request --max-frames 200
```

Có thể thêm `--front "C:\path\to\front.mp4"`. Bỏ `--rear-mode` để dùng REAR person `door_out` mặc định. `scripts/run_count.py` nghiệp vụ không hỗ trợ mode thử nghiệm. Với RTSP, `timestamp_s` được giữ tăng đơn điệu qua reconnect; `capture_time_utc` là xấp xỉ dựa trên lúc mở stream. MP4 không có mốc quay UTC được xác minh nên `capture_time_utc` để trống.

## Output, chẩn đoán và giới hạn

Trong thư mục output có `rear_head_out_events.csv`, `rear_head_out_annotated.mp4`, `rear_head_out_summary.json`, `rear_head_out_progress.json`, `rear_head_out_preview.jpg` và `session_summary.json`. CSV có camera, Track ID, OUT, frame index, timestamp nguồn, confidence, `crossing_zone`, tọa độ anchor và thời gian UTC nếu là RTSP. Summary có số frame, FPS xử lý, model/tracker/device, status, snapshot cấu hình và số event. `diagnostics` tách tổng detection, tổng box có Track ID và số lần xác nhận A/vùng chuyển tiếp/B; mỗi tổng là lượt quan sát theo frame, **không phải số hành khách**. Ví dụ detection > 0 nhưng stable A = 0 thường gợi ý vùng chưa trúng tâm đầu. Nếu detection = 0 cần xem model, ngưỡng và góc quay; nếu detection > tracked, xem tracker/che khuất; nếu đã qua A và transition nhưng không ra event, xem B và điều kiện ổn định.

Smoke test cuối 40 frame trên hai video cửa local (không đưa video vào Git) chạy FRONT và REAR đồng thời: REAR dùng `cuda:0`, 6.86 FPS xử lý, 40 detection và 40 box có ID, nhưng không có xác nhận A hay event; FRONT 9.15 FPS, 0 event. FPS thay đổi theo tải máy; lần chạy trước trên cùng 40 frame cho 8.28 và 11.81 FPS. Đây là kiểm tra đường chạy bằng model thật trên một đoạn ngắn, **không đo accuracy**. Video góc cửa không có nhãn người đi qua cùng ID; phải gán nhãn thủ công các crossing và kiểm tra toàn clip trước khi kết luận precision/recall. ID switch, người chen nhau, đầu quá nhỏ hoặc bị che khuất vẫn có thể gây bỏ sót hay đếm sai. Unit/integration test dùng FakeTracker để kiểm tra phần mềm, không chứng minh độ chính xác AI.

Kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```
