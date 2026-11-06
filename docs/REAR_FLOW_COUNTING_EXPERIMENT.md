# REAR Optical Flow + Counting Line (thử nghiệm)

Nhánh `experiment/rear-optical-flow-counting` thêm pipeline độc lập cho camera sau. Pipeline **không dùng YOLO, Head Detection, ByteTrack hay Track ID**. Nó **không ghi PassengerEvent/SQLite** và không thay thế luật FRONT-IN/REAR-OUT hiện hành.

## Cách hoạt động

1. Người dùng vẽ ROI, hai đầu Counting Line và bấm vào phía OUT để xác định chiều. Tọa độ lưu normalized trong `rear_flow_out` của `configs/cameras.yaml`. Worker lấy snapshot YAML lúc bắt đầu.
2. Mỗi frame chuyển grayscale, tùy chọn Gaussian Blur/CLAHE. Farnebäck dense optical flow chỉ tính trên bounding crop của ROI, ngoài polygon được mask đen.
3. Dọc Counting Line, mỗi pixel mẫu có vector flow. Thành phần theo mũi tên OUT hoặc ngược lại được lọc theo độ lớn. Tỷ lệ pixel chuyển động trên line được làm mượt theo cửa sổ thời gian; các đoạn hoạt động có hysteresis, gap và debounce tạo **motion crossing events**.
4. Ảnh `rear_flow_temporal_slice.png`: nửa trên là cường độ xám dọc Counting Line theo thời gian, nửa dưới là flow có dấu (màu). Trục ngang là vị trí trên line, trục dọc là frame. `rear_flow_diagnostics.json` chứa tỷ lệ chuyển động theo frame.
5. Mỗi event có frame/thời gian đầu và cuối, chiều, cường độ, thời lượng và `REVIEW_REQUIRED`. `estimated_passenger_count` luôn null, vì chưa có mô hình hiệu chuẩn/ground truth. Một motion blob có thể gồm nhiều người hoặc chuyển động cửa.

Optional stabilization ước lượng dịch chuyển toàn cục bằng `phaseCorrelate` trên phần nền **ngoài** ROI và trừ dịch chuyển đó khỏi flow. Khi nền ngoài ROI dưới 20% frame hoặc tín hiệu yếu, bước này bỏ qua. Mặc định tắt; chưa có benchmark chứng minh cải thiện trên xe buýt.

Thiết kế temporal slice lấy cảm hứng từ [Ma & Chan, CVPR 2013](https://openaccess.thecvf.com/content_cvpr_2013/html/Ma_Crossing_the_Line_2013_CVPR_paper.html). Bài đó dùng hồi quy và integer programming đã huấn luyện để suy số người từ slice; prototype này **không** có các bước ấy. [Liu et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2455_ECCV_2020_paper.php) là hướng People Flow dựa trên mạng học sâu, chưa triển khai. Paper [Spatio-Temporal Optical Flow Analysis, AVSS 2010](https://doi.org/10.1109/AVSS.2010.29) là tài liệu liên quan, không phải code được sao chép.

## Chạy trên Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\dashboard.py
```

Tab **Thiết lập vùng đếm (ROI)** → Camera **REAR – Optical Flow Counting** → chọn MP4 hoặc lấy frame webcam/RTSP được cấp phép → chọn frame → vẽ ROI bằng click, đặt hai đầu line, bấm phía OUT để đặt mũi tên → **Lưu Optical Flow ROI**. Nút lưu tạo `cameras.yaml.rear_flow_out.bak`. **Test Optical Flow ROI** chạy đoạn ngắn bằng cấu hình đã lưu. Sau đó tab **Đếm khách AI** → chế độ **Thử nghiệm Optical Flow + Line Counting** → chọn REAR video và chạy. Có thể chạy cùng FRONT; kết quả REAR vẫn ở khu vực thử nghiệm, không cộng vào tổng hành khách.

CLI (dùng cấu hình đã lưu, `enabled: true`):

```powershell
.\.venv\Scripts\python.exe scripts\run_count_session.py --rear data\raw\out_test.mp4 --rear-mode flow_line_out --output runs\flow_trial --max-frames 180 --stop-file runs\flow_trial\stop.request
```

Output trong session: `rear_flow_events.csv`, `rear_flow_summary.json`, `rear_flow_progress.json`, `rear_flow_annotated.mp4`, `rear_flow_temporal_slice.png`, `rear_flow_diagnostics.json`. `session_summary.json` có `count: null` và `motion_events` riêng. URL RTSP được redact trong summary; credential không lưu trong YAML.

## Gán nhãn và đánh giá

```powershell
.\.venv\Scripts\python.exe scripts\label_flow.py --video data\raw\out_test.mp4 --output runs\flow_labels.csv
.\.venv\Scripts\python.exe scripts\evaluate_flow.py --events runs\flow_trial\rear_flow_events.csv --labels runs\flow_labels.csv --output runs\flow_evaluation.json
```

Nhấn `O`/`I` trong cửa sổ gán nhãn để ghi thời điểm; nhập số người **quan sát được** hoặc bỏ trống nếu không rõ. Nhãn có `video_id,event_time_s,direction,passenger_count,notes`. Evaluator chỉ so khớp **thời điểm motion event** theo dung sai thời gian, trả precision/recall cho motion. MAE và accuracy số khách luôn `N/A` vì pipeline chưa dự đoán số người. Không có nhãn thì mọi metric đánh giá là `N/A`. Muốn so sánh YOLO + ByteTrack phải gán nhãn cùng clip, cùng khoảng thời gian và chuẩn hóa event window trước.

## Kết quả đã chạy tại máy phát triển

- Kiểm thử logic tổng hợp: một đối tượng đi OUT tạo một event, chiều ngược không tạo OUT, đứng yên/nhiễu nhẹ không tạo event, hai lượt cách nhau tạo hai event. Đây là **kiểm thử phần mềm**, không phải AI accuracy.
- Smoke test video thật sẵn có tại máy `data/raw/out_test.mp4` (camera trong xe nhìn về cửa sau, 1280×720, 18.01 FPS): 180 frame đầu, ROI thử nghiệm `(.32,.05)-(.73,.7)`, line ngang y=.43, OUT hướng lên; 9.86 s, 18.25 FPS xử lý; 7 motion events, trong đó 4 theo hướng OUT. Số khách OUT và accuracy: **N/A**. ROI/line này chỉ để smoke test, chưa được hiệu chuẩn; kết quả có thể chứa chuyển động cửa, camera hoặc hành khách không đi qua cửa. Cấu hình này ở `runs/flow_real_smoke_config.yaml`, không commit vì là thử nghiệm theo clip. Nguồn phát hành và giấy phép của file MP4 cục bộ chưa được xác minh, nên không phân phối lại.
- Test/quality gates: chạy `pytest -q`, `ruff check .`, `pip check`; ghi kết quả thực tại báo cáo commit.

## Hạn chế và quyết định tiếp theo

Camera cabin hiện tại nhìn chéo, lối đi đông và cửa xa, nên nhiều người có thể thành một vùng chuyển động; người đứng che khuất hoàn toàn không tạo tín hiệu có thể suy ra. Cửa mở/đóng, bóng và rung camera có thể tạo event. Mọi event cần review; không cộng trực tiếp vào occupancy hoặc flow theo trạm. Chỉ cân nhắc mô hình People Flow sau khi có nhãn thật đủ nhiều để so sánh cùng clip. Nếu vẫn không tách được nhóm đông, cần xem lại vị trí camera nhìn từ trên xuống hoặc depth camera.

## Checklist kiểm thử

- [x] Pipeline không cần YOLO/tracker, có ROI crop/mask, hướng OUT rõ ràng, temporal slice và file diagnostics.
- [x] Motion event tách khỏi passenger count; không ghi dữ liệu nghiệp vụ.
- [x] Editor lưu/khôi phục cấu hình và worker snapshot; chế độ cũ còn chạy qua regression suite.
- [x] Smoke test trên MP4 thật; có FPS và output kiểm tra được.
- [ ] Độ chính xác số khách trên video thật: chưa có ground truth hoặc ước lượng số khách.
- [ ] Đánh giá đông người, che khuất, rung camera, cửa mở/đóng bằng nhãn thực: chưa thực hiện.
- [ ] RTSP vật lý và hiệu quả stabilization trên xe thật: chưa kiểm chứng.
