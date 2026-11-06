# REAR Head Tracking + Zone OUT (thử nghiệm)

Chế độ `head_tracking_zone_out` đếm **lượt đầu đi qua cửa sau theo chiều A → B**, không ước lượng số người trong xe. Nó chạy trong phiên AI độc lập, xuất file riêng và **không ghi** `PassengerEvent`, Trip, SQLite hoặc thống kê theo trạm. FRONT-IN và hai chế độ REAR cũ (`door_out`, `occupancy_headcount`) giữ nguyên. Số OUT thử nghiệm được hiển thị riêng và không cộng vào tổng nghiệp vụ.

## Model, tracking và điều kiện đếm

Checkpoint mặc định là `11n-head.pt` của [wesjos/YOLO-head-detection](https://huggingface.co/wesjos/YOLO-head-detection), một class head, license AGPL-3.0. Lần đầu chạy, worker tải checkpoint vào `runs/models/` (Git bỏ qua); có thể đặt file `.pt` tin cậy ở máy và cấu hình `rear_head_out.model`. Model được nạp một lần cho mỗi worker, không nạp mỗi frame. Mỗi frame được crop theo bounding rectangle của `rear_head_out.roi`, rồi mask polygon trước khi gọi `model.predict(..., classes=[0])`. Detection có tâm ngoài polygon bị loại **trước** khi đưa vào ByteTrack hoặc BoT-SORT; tracker sống xuyên suốt phiên và chỉ nhận chuỗi crop cùng kích thước. Box được cộng offset crop để trở về tọa độ frame gốc. Video annotated vẫn giữ toàn frame, nhưng chỉ vẽ box, ID và quỹ đạo của track có tâm trong ROI. `device: auto` có thể dùng CUDA khi môi trường hỗ trợ; `device: cpu` dùng CPU. Confidence, IoU và `imgsz` chỉnh trong `configs/cameras.yaml`.

Mỗi Track ID dùng **tâm bounding box đầu người** làm anchor. Bộ đếm yêu cầu: ổn định trong A ít nhất `min_stable_frames`, ổn định trong vùng chuyển tiếp **bên trong ROI** và ngoài A/B, rồi ổn định trong B. Một lần như vậy tạo đúng một event OUT. Ra khỏi ROI hoặc vắng mặt trong một frame sẽ hủy crossing đang chờ; không suy diễn đoạn đường chưa quan sát. Đứng yên, A → A, B → A, nhảy thẳng A → B không có vùng chuyển tiếp xác nhận, hoặc mất track trước B đều không tạo event. Mỗi track có trạng thái riêng; track cũ bị xóa sau `state_ttl_frames`. Cấu hình ID switch cũ vẫn được đọc để tương thích nhưng sau khoảng mất track không thể tiếp tục một crossing đang chờ. Điều này ưu tiên hạn chế đếm sai nhưng có thể bỏ sót khi đông người, che khuất hoặc ID switch.

## Hiệu chỉnh ROI và Zone bằng chuột

1. Chạy Streamlit theo lệnh bên dưới và mở tab **Thiết lập vùng đếm (ROI)**.
2. Trong **Camera**, chọn **REAR – Head Tracking + Zone OUT**. Chọn video trong `data/raw`, upload MP4, nhập đường dẫn MP4 hoặc lấy một frame webcam/RTSP được cấp phép. Với MP4, kéo slider đến frame thấy rõ đầu, cửa và hướng di chuyển.
3. Chọn **ROI – vùng quan sát**, **Zone A** hoặc **Zone B**. Chỉ vùng đang chọn nhận thao tác; cả ba vùng luôn được vẽ cùng lúc: ROI xanh lá, A vàng, B xanh dương, mũi tên A → B trắng. Click để thêm đỉnh; chọn **Dời đỉnh** rồi kéo đỉnh hoặc chọn đỉnh và click vị trí mới. Có nút **Xóa đỉnh đã chọn**, **Hoàn tác**, **Xóa vùng**, **Nạp lại cấu hình**. Bảng dưới ảnh hiển thị tọa độ pixel và normalized của từng đỉnh.
4. Vẽ A ở phía trong xe, B ở phía ngoài/hướng xuống xe. Để một khoảng trống giữa hai vùng cho đầu được xác nhận tại **Transition**. Cả A và B phải nằm hoàn toàn trong ROI. Nút **Lưu ROI** chỉ khả dụng khi ba polygon hợp lệ, không tự cắt, không chồng lấn và A/B không chạm nhau. Khi lưu, chỉ `rear_head_out.roi`, `rear_head_out.zone_a`, `rear_head_out.zone_b` được cập nhật; FRONT, REAR Person, occupancy, model/tracker và các trường YAML khác được giữ. Bản sao lưu riêng là `configs/cameras.yaml.rear_head_out.bak`; nút **Khôi phục ROI Head Tracking từ bản sao lưu** chỉ phục hồi ba vùng này.
5. Chọn số frame rồi bấm **Test Head Tracking ROI**. Với MP4 có thể bắt đầu ngay từ frame đang hiển thị hoặc nhập frame khác. Webcam/RTSP bắt đầu ở frame hiện tại của luồng và chạy đến giới hạn frame hoặc khi nguồn kết thúc. Test dùng checkpoint head + ByteTrack thật, hiển thị preview/video annotated, `Head-OUT`, số detection/track theo frame, số xác nhận A/Transition/B, FPS và tải CSV. Nếu có bản nháp chưa lưu, nút Test bị khóa và hiện cảnh báo; nhấn **Lưu ROI** trước.
6. Quay lại tab **Đếm khách AI**, chọn chế độ REAR Head Tracking và chạy inference mới. Phiên mới đọc ba vùng đã lưu; mỗi worker giữ snapshot YAML tại lúc khởi chạy. Bản nháp của FRONT Person, REAR Person và REAR Head được giữ riêng khi đổi lựa chọn.

Test ROI và Head-OUT là output AI, **không phải ground truth**. Nếu video không có crossing hợp lệ, kết quả 0 vẫn có thể đúng với vùng đã vẽ; xem video và các số xác nhận để hiệu chỉnh.

### Tùy chọn cấu hình bằng YAML

Giao diện phía trên không yêu cầu sửa YAML. Nếu cần tự động hóa, có thể sửa riêng khối `rear_head_out` trong `configs/cameras.yaml`:

```yaml
rear_head_out:
  roi: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
  zone_a: [[0.10, 0.20], [0.30, 0.20], [0.30, 0.80], [0.10, 0.80]]
  zone_b: [[0.70, 0.20], [0.90, 0.20], [0.90, 0.80], [0.70, 0.80]]
```

Các số trong ví dụ chỉ minh họa cú pháp, **không phải** vị trí mặc định phù hợp cho mọi camera. Tọa độ normalized bằng `x_pixel / chiều_rộng`, `y_pixel / chiều_cao`. Mặc định lấy zone cửa REAR Person hiện có làm điểm bắt đầu, nhưng zone này trước đây được hiệu chỉnh cho chân người, nên cần kiểm tra lại **tâm đầu** trên video thực. Overlay box, Track ID, quỹ đạo, zone và thông báo OUT có thể bật/tắt trong tab **Đếm khách AI**.

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

Trong thư mục output có `rear_head_out_events.csv`, `rear_head_out_annotated.mp4`, `rear_head_out_summary.json`, `rear_head_out_progress.json`, `rear_head_out_preview.jpg` và `session_summary.json`. CSV có camera, Track ID, OUT, frame index, timestamp nguồn, confidence, `crossing_zone`, tọa độ anchor và thời gian UTC nếu là RTSP. Summary có số frame, FPS xử lý, model/tracker/device, status, snapshot cấu hình và số event. `diagnostics` tách tổng detection **có tâm trong ROI**, tổng box có Track ID, thời gian `model.predict` (`inference_s`), kích thước crop và số lần xác nhận A/vùng chuyển tiếp/B; mỗi tổng box là lượt quan sát theo frame, **không phải số hành khách**. Ví dụ detection > 0 nhưng stable A = 0 thường gợi ý vùng chưa trúng tâm đầu. Nếu detection = 0 cần xem model, ngưỡng và góc quay; nếu detection > tracked, xem tracker/che khuất; nếu đã qua A và transition nhưng không ra event, xem B và điều kiện ổn định.

Smoke test cuối 40 frame trên hai video cửa local (không đưa video vào Git) chạy FRONT và REAR đồng thời: REAR dùng `cuda:0`, 6.86 FPS xử lý, 40 detection và 40 box có ID, nhưng không có xác nhận A hay event; FRONT 9.15 FPS, 0 event. FPS thay đổi theo tải máy; lần chạy trước trên cùng 40 frame cho 8.28 và 11.81 FPS. Đây là kiểm tra đường chạy bằng model thật trên một đoạn ngắn, **không đo accuracy**. Video góc cửa không có nhãn người đi qua cùng ID; phải gán nhãn thủ công các crossing và kiểm tra toàn clip trước khi kết luận precision/recall. ID switch, người chen nhau, đầu quá nhỏ hoặc bị che khuất vẫn có thể gây bỏ sót hay đếm sai. Unit/integration test dùng FakeTracker để kiểm tra phần mềm, không chứng minh độ chính xác AI.

Sau khi thêm trình chỉnh vùng, smoke test lại 40 frame REAR với cấu hình vùng mặc định chưa hiệu chỉnh: `cuda:0`, 4.53 FPS, 40 detection, 40 box có ID, 0 xác nhận A và 0 OUT. FPS giảm trong lần chạy này do tải máy khác; đây không phải benchmark. Hãy vẽ lại A/B bằng giao diện trên frame có đầu người đi qua cửa rồi dùng **Test Head Tracking ROI** để xem sự kiện trên chính video của bạn.

### Đối chứng crop/mask trên 40 frame

Ngày 06/11/2026, cùng một MP4 cửa sau local 320×240 và `11n-head.pt` trên `cuda:0`:

| Phương án | ROI đầu vào | FPS toàn pipeline | Detection / track có ID | OUT |
| --- | --- | ---: | ---: | ---: |
| Code trước sửa, YOLO toàn frame | 320×240 | 9.93 | 40 / 40 | 0 |
| Code mới, ROI do người dùng vừa vẽ | crop 29×46, mask polygon | 8.94 | 0 / 0 | 0 |
| Code mới, ROI toàn frame để đối chứng tracker | 320×240 | 8.47 | 40 / 40 | 0 |

Hai lượt code mới cuối cùng ghi tổng thời gian `model.predict` lần lượt 4.220 và 4.456 giây cho 40 frame; baseline cũ chỉ có thời gian toàn pipeline (4.028 giây), không ghi riêng inference. Hai lượt code mới trước đó cho 10.79 và 10.49 FPS với tổng `model.predict` 3.481 và 3.574 giây. Các số này là smoke test tuần tự, chịu ảnh hưởng cache GPU/tải máy, không chứng minh tốc độ tăng. ROI người dùng trên clip này rất nhỏ và không chứa tâm đầu được model phát hiện; cần hiệu chỉnh trên frame có chuyển động thật. Không có ground truth nên precision, recall và accuracy đều N/A. Khi RTSP đổi độ phân giải giữa phiên, worker báo lỗi rõ ràng để tránh dùng tracker với crop có kích thước thay đổi; hãy khởi động phiên mới sau khi chỉnh nguồn.

Kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
```
