# Thử nghiệm REAR occupancy bằng Head Detection

Đây là một chế độ **thử nghiệm trong phiên đếm độc lập** (`scripts/run_count_session.py`). FRONT vẫn dùng YOLO11n + ByteTrack + Zone A/B để đếm IN. REAR ở chế độ mặc định `door_out` vẫn dùng pipeline cũ để đếm OUT. Chỉ khi chọn `occupancy_headcount`, REAR chuyển sang phát hiện đầu người trong **từng frame**; số này không cộng dồn, không phải số lượt xuống xe, không tạo `PassengerEvent`, không ghi SQLite và không tham gia thống kê theo trạm. Hệ thống chỉ đếm đầu nhìn thấy được, không biết người bị che hoàn toàn hoặc ngoài góc quay.

## Model và license

Checkpoint mặc định là [`wesjos/YOLO-head-detection` – `11n-head.pt`](https://huggingface.co/wesjos/YOLO-head-detection), model Ultralytics YOLO11 một class (`0` = head). Trang model công bố license **AGPL-3.0**; cần đánh giá nghĩa vụ license trước khi dùng trong dịch vụ hoặc sản phẩm thương mại. Checkpoint là PyTorch pickle: chỉ dùng nguồn tin cậy. Không upload video/ảnh lên Hugging Face; việc tải chỉ lấy weights về `runs/models/11n-head.pt` một lần. Thư mục `runs/` và mọi `.pt` bị Git ignore. Nếu máy offline, tải checkpoint theo cách được cấp phép rồi đặt ở đường dẫn local và cấu hình `rear_occupancy.model` trong `configs/cameras.yaml`. Lỗi tải hoặc nạp model sẽ báo FAILED, không đưa ra số 0 như một phép đo thành công.

Model đã được thử nạp với Ultralytics 8.3.57 trên môi trường phát triển; cấu hình `device: auto` chọn CUDA nếu thư viện và GPU khả dụng, ngược lại dùng CPU. Có thể đặt `device: cpu` hoặc `device: '0'` cho GPU đầu tiên. `confidence`, `iou`, `imgsz` cũng nằm trong `rear_occupancy` của YAML. Mỗi worker REAR nạp checkpoint một lần, không nạp lại mỗi frame. Không dùng COCO person detector hoặc ByteTrack trong chế độ này.

## ROI khoang xe

`rear_occupancy.roi` là polygon normalized `[0,1]` bao vùng khoang cần quan sát; `rear_occupancy.ignore` là danh sách polygon loại trừ kính/cửa/đường phố. Chỉ box có **tâm** nằm trong ROI và ngoài tất cả vùng ignore mới được tính. Mặc định ROI toàn frame chỉ phục vụ thử ban đầu; cần hiệu chỉnh trên chính video góc nhìn khoang xe trước khi diễn giải kết quả. Các zone A/B cửa hiện có hoàn toàn độc lập và không bị thay đổi. Polygon phải có ít nhất ba đỉnh, không tự cắt, không ra ngoài `[0,1]`.

`smoothing_window: 1` nghĩa là không làm mượt. Nếu tăng lên, CSV vẫn lưu `visible_head_count` thô và `smoothed_count` là median của cửa sổ gần nhất; không được suy diễn chênh lệch hai frame thành lượt IN/OUT. Một phiên Streamlit chụp snapshot YAML khi bấm Bắt đầu nên không thay ROI của worker đang chạy.

## Chạy Streamlit

Từ thư mục repository trên Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\dashboard.py
```

Vào **Đếm khách AI**, tại cột REAR chọn **Thử nghiệm đếm khách trong xe (Head Detection)**, chọn MP4 REAR rồi bấm **Bắt đầu đếm**. Có thể chỉ chạy REAR, chỉ FRONT hoặc cả hai. Preview hiển thị box head, ROI và số đầu quan sát được; khi hoàn tất xem video H.264 và tải CSV. FRONT vẫn hiển thị IN. REAR-OUT ở phần tổng hợp là `N/A – Chế độ thử nghiệm`; occupancy được hiển thị riêng. Lỗi REAR không làm hỏng kết quả FRONT.

## Chạy CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_count_session.py --rear "C:\path\to\rear.mp4" --rear-mode occupancy_headcount --output runs\ai_counting\head_trial --stop-file runs\ai_counting\head_trial\stop.request --max-frames 200
```

Có thể thêm `--front "C:\path\to\front.mp4"`. Bỏ `--rear-mode` để chạy REAR `door_out` như trước. Lệnh nghiệp vụ `scripts/run_count.py` không hỗ trợ mode thử nghiệm và không bị thay đổi. File `stop.request` yêu cầu dừng sau frame hiện tại; MP4 hết thì tự hoàn tất.

## Output và diễn giải

Trong `runs/ai_counting/<session-id>/` có `rear_preview.jpg`, `rear_annotated.mp4`, `rear_occupancy.csv`, `rear_progress.json`, `rear_occupancy_summary.json` và `session_summary.json`. CSV chứa `frame_index,timestamp_s,visible_head_count,smoothed_count`. Summary chứa mode, tên model, số frame, FPS xử lý, count cuối, mean/min/max của **các frame đã xử lý** và nguồn đã che credentials nếu là RTSP. Không có `rear_events.csv` trong mode này. `count` của REAR trong session summary là `null`, không phải REAR-OUT.

Head Detection dễ sai khi đầu nhỏ, che khuất, rung/nhòe, ánh sáng yếu hoặc có người ngoài xe qua kính. Góc camera cửa sau không đại diện cho toàn khoang. Để đánh giá, lấy mẫu frame theo một quy tắc cố định, gán nhãn thủ công số đầu **nhìn thấy** trong từng frame, so sánh với CSV bằng MAE và sai số theo từng frame; tách riêng đánh giá false-positive/false-negative box nếu có nhãn box. Không báo accuracy khi chưa có ground truth. Smoke test trên clip cửa sau chỉ xác nhận model và luồng output chạy được, không xác nhận độ chính xác cho góc nhìn khoang xe.
