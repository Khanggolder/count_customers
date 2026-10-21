# Nguồn dữ liệu

Ngày rà soát: 2026-10-19.

## Chuyến mẫu và tuyến/trạm

Scenario dùng tên tám trạm liên tiếp của hướng đi Bến xe Chợ Lớn trên trang tham khảo tuyến 01: Bến Thành B → Trường Ernst Thalmann → KTX Trần Hưng Đạo → Rạp Hưng Đạo → Bệnh viện Răng Hàm Mặt → Trần Đình Xu → Tổng Cty Samco → Chợ Nanci.

- Nguồn tên/thứ tự: https://xe-buyt.com/ben-thanh-ben-xe-cho-lon.html
- Nguồn chính thức được thử: https://buyttphcm.com.vn/RouteFinding, nhưng trả HTTP 403 trong phiên kiểm chứng.
- Vì chưa lấy được tọa độ chính thức, mọi tọa độ/geofence trong scenario là hình học tổng hợp và `verified=false`. File ghi rõ `SYNTHETIC_ROUTE_GEOMETRY_NOT_REAL_STOP_COORDINATES`.
- Đây không phải dữ liệu vận hành, lịch chạy hay GPS thật của tuyến 01.

## Video Pexels đã tải và chạy thật

- Clip: https://www.pexels.com/video/busy-city-bus-stop-with-passengers-boarding-31337217/
- Tác giả: Shariar Tawsif.
- License: https://www.pexels.com/license/ — trang license cho phép tải/dùng miễn phí; dự án vẫn giữ attribution và không commit binary.
- File local ignored: `data/raw/pexels_bus_boarding_31337217.mp4`.
- SHA-256: `2155814b2d7cd1620b7c924b1098d926f485907ea1a193238a60b8d536fc708d`.
- 16,906,273 bytes; 3840×2160; 30 FPS; 178 frame; 5.933 s.
- Review frame 90: xe buýt, cửa mở, một hành khách tại cửa và nhân viên điều tiết; camera ở ngoài xe và chuyển động. Chỉ dùng detector/tracker smoke, không dùng làm benchmark crossing hoặc chuyến hai camera.

## PCDS — đúng góc cửa nhưng chưa tải được

- https://github.com/shijieS/people-counting-dataset
- README công bố 5,464 cặp RGB/depth quay trên cao tại cửa xe buýt và count enter/exit theo clip; license CC BY-NC-SA 3.0.
- Google Drive đã bị gỡ, chỉ còn BaiYun key `2s31`. Phiên này không bypass tài khoản/khu vực, vì vậy không có file PCDS local và không có MAE/precision/recall PCDS.

## Telemetry repositories được rà soát nhưng không dùng

- https://github.com/N2T-vn/hcmc_bus_tracking mô tả schema `vehicle,datetime,x,y,speed,heading,ignition,aircon,door_up,door_down,driver`, nhưng CSV phải tải từ shared source riêng và repository không hiển thị license. README cũng nói ứng dụng không dùng `door_up/down`. Không nhập dữ liệu này.
- https://github.com/huynguyentuank22/BDC-Data-Hackathon-2025 không cung cấp dataset/license rõ ràng qua nội dung truy cập được trong phiên. Không nhập dữ liệu này.

## Model

- YOLO11n COCO person weight do Ultralytics tự tải từ official release asset `v8.3.0`; file `yolo11n.pt` bị gitignore.
- Không có model Re-ID trong MVP.

`video_model_validation=NO_GROUND_TRUTH_NA`: không có metric AI thật nào được suy ra từ scenario hoặc từ clip Pexels.
