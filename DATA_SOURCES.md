# Nguồn dữ liệu và model

## Asset đã tải và dùng thực tế

### BusOccupancyCounter `bus_input.mp4`

- Trang nguồn: https://github.com/sanjarbek1030/BusOccupancyCounter
- File local: `data/raw/bus_occupancy_counter_input.mp4`
- SHA-256: `96d8b1b32b47acfe50f17145c74002a91b22b9367e5b0cc7fc8fc940f3214717`
- 1.985.897 bytes; 900×474; 25 FPS; 281 frame; 11,24 giây.
- Kiểm tra trực quan: cảnh tĩnh ngoài cửa trước xe buýt, nhiều người lần lượt bước lên; phù hợp smoke test crossing theo phương ngang.
- Quyền: README repository hiện ghi MIT, nhưng video hiển thị watermark Shutterstock. License code/repository không tự động cấp quyền cho stock footage. Chỉ dùng kiểm chứng nghiên cứu nội bộ; không đóng gói/redistribute/public demo asset này.
- Kết quả model là prediction, không phải ground truth. `bus_output.mp4` của upstream không được dùng làm GT.

### Pexels video 31337217

- Trang clip: https://www.pexels.com/video/busy-city-bus-stop-with-passengers-boarding-31337217/
- Tác giả: Shariar Tawsif.
- License: https://www.pexels.com/license/ (free to use; không cần attribution nhưng dự án vẫn ghi nguồn; tuân thủ các điều cấm trên trang license).
- File local: `data/raw/pexels_bus_boarding_31337217.mp4`
- SHA-256: `2155814b2d7cd1620b7c924b1098d926f485907ea1a193238a60b8d536fc708d`
- 16.906.273 bytes; 3840×2160; 30 FPS; 178 frame; 5,93 giây.
- Kiểm tra trực quan: hành khách đang lên cửa xe buýt công cộng màu hồng; camera ngoài đường và cửa xe dịch chuyển trong frame. Đây không đại diện camera cố định gắn trên xe, vì vậy không dùng số crossing của clip làm KPI counting.
- Đã chạy YOLO/ByteTrack thật; 0 crossing với ROI cấu hình camera cố định, đúng là không đủ cơ sở để báo count.

Hai clip trên là hai cảnh/chuyến khác nhau. Run hai camera mang nhãn `UNPAIRED`; tuyệt đối không dùng để đánh giá cross-camera identity hoặc occupancy.

## Dataset đã xác minh nhưng chưa tải

### PCDS

- https://github.com/shijieS/people-counting-dataset
- Tác giả công bố 5.464 cặp depth/RGB tại cửa xe buýt, camera gắn trên cao; `label.txt` chứa count entering/exiting.
- License: CC BY-NC-SA 3.0; bắt buộc attribution, phi thương mại, share-alike.
- Google Drive đã bị gỡ; README chỉ còn BaiYun key `2s31`. Workspace không bypass đăng nhập/giới hạn địa lý, nên dataset chưa được tải và không có MAE PCDS.
- Dataset không xác nhận identity pair giữa front/back; không dùng đánh giá Person Re-ID.

### iLIDS-VID

- https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html
- Chỉ dành benchmark Re-ID kỹ thuật có identity đa camera; không phải bối cảnh cửa bus.
- Chưa tải trong lần triển khai này; Rank-1/mAP là `NA`.

## Checkpoint đã tải

### YOLO11n

- Nguồn tải tự động chính thức của Ultralytics: release assets `v8.3.0`.
- File: `yolo11n.pt`, 5.613.764 bytes.
- SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- COCO pretrained; chỉ inference class `person`.

### OSNet x0.25 MSMT17

- Model zoo chính thức: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- Google Drive file ID: `1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e`.
- File: `models/osnet_x0_25_msmt17.pt`, 9.336.983 bytes.
- SHA-256: `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`.
- Huấn luyện MSMT17 `combineall`; đây là checkpoint Person Re-ID, không phải ImageNet-only.

