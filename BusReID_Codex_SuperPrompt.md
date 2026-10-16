# SUPER PROMPT FOR CODEX — BusReID AI (MVP end-to-end)

Bạn là Senior Computer Vision Engineer, ML Engineer, Backend Engineer, Frontend Engineer và QA Engineer của dự án. Hãy **trực tiếp xây dựng, chạy thử, kiểm thử và hoàn thiện source code** trong workspace hiện tại; không chỉ viết kế hoạch, pseudo-code, giao diện giả hoặc scaffold. Nếu workspace trống, tạo project mới `busreid-ai`. Nếu có repo, khảo sát trước để tránh ghi đè. Chủ động thực hiện các bước hợp lý, không hỏi người dùng những điều đã được xác định trong prompt.

## 0. BỐI CẢNH THỰC TẾ VÀ MỤC TIÊU

Nhóm sinh viên đang làm MVP đề xuất cho Trung tâm Quản lý Giao thông công cộng TP.HCM. Xe buýt có **cửa lên phía trước** và **cửa xuống phía sau**, mỗi cửa một camera gắn cao, nhìn chéo xuống lối đi. Hiện **không có thiết bị camera, video tự quay, GPS hay quyền truy cập camera của Trung tâm**. Nhóm có laptop/PC NVIDIA GPU. Nhóm muốn demo sản phẩm từ **video công khai có thật** trước; nếu Trung tâm đồng ý mới được lựa chọn một xe và tích hợp với camera hiện hữu qua RTSP/thiết bị ghi phù hợp.

Mục tiêu MVP:
1. Phát hiện người và tracking độc lập trên luồng cửa trước/cửa sau.
2. Đếm boarding (IN, cửa trước) và alighting (OUT, cửa sau) chính xác theo hướng vượt qua vùng cửa. Không nhầm detection mỗi frame với lượt người.
3. Cho phép thử nghiệm **Cross-Camera Person Re-Identification** (Re-ID): một hành khách xuất hiện lúc lên cửa trước có thể được ghép với người lúc xuống cửa sau, có ngưỡng từ chối UNKNOWN. Đây là *tính năng thí nghiệm*, không được tuyên bố đã nhận dạng chính xác hành khách trên xe thực tế nếu dữ liệu thiếu nhãn cùng người ở hai camera.
4. Hiển thị hai video, overlay bounding box, tracking ID, hướng/cửa, IN/OUT, occupancy ước tính, danh sách sự kiện, cặp ghép Re-ID và các trường hợp UNKNOWN, bảng KPI thực nghiệm.
5. Thiết kế đầu vào video có thể đổi từ MP4 sang RTSP trong tương lai mà không viết lại core AI.

**KHÔNG làm trong MVP này:** GPS, bản đồ/trạm dừng, dự báo kẹt xe, nhận diện khuôn mặt, nhận diện danh tính thật, thanh toán/trợ giá, Cloud production, tích hợp camera Trung tâm khi chưa có quyền. Chỉ tạo API/kiến trúc mở rộng nếu không cản trở hoàn thiện core.

## 1. RÀNG BUỘC KHÔNG ĐƯỢC VI PHẠM

- **Không tự quay video; người dùng sẽ không đi quay.** Sử dụng video/dataset công khai với xuất xứ và giấy phép được kiểm tra; hỗ trợ upload thủ công nếu bên ngoài cung cấp video sau này.
- **Không giả mạo kết quả:** tuyệt đối không hardcode số IN/OUT, ID match, confidence, FPS hay các chỉ số như thể được AI dự đoán. Synthetic fixtures được phép để unit test và demo UI nhưng phải có nhãn `SYNTHETIC/DEMO`, tách khỏi kết quả thực nghiệm.
- **Không giả định video hai cửa lấy từ các nguồn khác nhau có cùng người.** Nếu không có cặp video đồng bộ và ground-truth identity, UI chỉ được hiển thị `candidate match — unverified`, không công bố Re-ID accuracy/OD từng người trên bus.
- **Dữ liệu học thuật đúng bối cảnh:** ưu tiên video camera cửa xe buýt, tránh mặc định lấy video người đi bộ đường phố/cửa siêu thị làm bằng chứng hiệu quả trên bus.
- **Dùng checkpoint thực có thể tải/kiểm chứng**. Nếu không tải được weights, nêu lỗi rõ và cung cấp chế độ kiểm tra pipeline bằng detections giả lập *được gắn nhãn*, không giả làm inference thật.
- Không vượt qua hệ thống đăng nhập, CAPTCHA, quyền truy cập data, DRM hoặc tự động tải nội dung không được phép. Không commit video/dataset/weights lớn hoặc private data vào Git.
- Tất cả ID chỉ là mã tạm theo một phiên/chuyến, không cố nhận diện cá nhân ngoài xe, không dùng face recognition.
- Ưu tiên hoàn thiện hoạt động end-to-end hơn việc tạo nhiều trang dashboard không có chức năng.

## 2. NGUỒN VIDEO VÀ DATASET CẦN KIỂM TRA, TÍCH HỢP

### A. Video thử ngay: BusOccupancyCounter
- Project: https://github.com/sanjarbek1030/BusOccupancyCounter
- File đầu vào có thật trong repo: https://github.com/sanjarbek1030/BusOccupancyCounter/blob/main/bus_input.mp4
- Nguồn raw ứng viên: https://raw.githubusercontent.com/sanjarbek1030/BusOccupancyCounter/main/bus_input.mp4
- Repo có `bus_output.mp4` (video kết quả) và code tham khảo. **KHÔNG lấy bus_output làm ground truth**. Trước khi đưa video vào demo, kiểm tra `ffprobe`/OpenCV, trích khung hình và xác thực nội dung thật sự cho thấy khu vực người lên/xuống xe; ghi nhận nguồn và điều kiện license. Quyền đối với video riêng lẻ không đương nhiên giống license mã nguồn.
- Viết `scripts/fetch_demo_assets.py` có lệnh tải tự nguyện nguồn này, retry, timeout, progress, kiểm tra định dạng và checksum sau tải, không tự tải dữ liệu kích thước lớn khi chưa xác nhận. Nếu tệp không phù hợp ngữ cảnh, hiển thị thông báo và KHÔNG dùng để báo cáo kết quả.

### B. Dataset chính để đếm khách xe buýt: PCDS
- https://github.com/shijieS/people-counting-dataset
- 5.464 cặp video RGB/Depth từ camera gắn phía trên cửa xe buýt có góc nghiêng; thư mục tên dạng `BUS_DATETIME_front` và `BUS_DATETIME_back`; file `label.txt` chứa tổng Entering/Exiting trên từng clip. Đây là dữ liệu cực sát bài toán đếm người ở cửa. Giấy phép **CC BY-NC-SA 3.0**; bắt buộc ghi nguồn/citation, chỉ dùng nghiên cứu phi thương mại theo điều kiện.
- Repo thông báo Google Drive cũ đã xóa. Link dữ liệu BaiYun từ README, mật khẩu `2s31`. Nếu bị hạn chế địa lý hoặc phải đăng nhập, KHÔNG bypass; hướng dẫn người dùng tải hợp lệ và chỉ định thư mục local. Hỗ trợ nhập `.zip`/folder riêng, parser cho phân cấp `front/back`, trích `label.txt`; không giả định RGB và depth luôn đồng bộ.
- **KHÔNG** coi hai clip front/back là cùng hành khách chỉ vì tên bus/timestamp tương tự. Dataset có count labels, không có xác nhận identity pairs xuyên cửa.

### C. Dataset bổ trợ mô hình Re-ID có nhãn danh tính ở nhiều camera
- iLIDS-VID: https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html
- Chỉ sử dụng làm **technical benchmark riêng** cho đặc trưng/matching giữa camera. Đây **không phải** video cửa xe buýt, phải ghi rõ trong UI và báo cáo. Nếu link yêu cầu quyền, hướng dẫn thủ công; không tự nhận đã đánh giá nếu chưa truy cập được.
- Có thể dùng data khác nếu kiểm tra được nguồn, nhãn, giấy phép và đưa vào `data/SOURCES.md`.

### D. Tùy chọn có kiểm soát: PAMELA-UANDES
- https://videodatasets.org/PAMELA-UANDES/ground_truth.html
- Video thí nghiệm boarding/alighting ở cửa phương tiện, annotation tracking; yêu cầu tài khoản được tác giả cung cấp cho nghiên cứu. Không tải nếu chưa được cấp quyền.

### Khi không tải được dữ liệu
- APP phải khởi động được với trạng thái `NO_VIDEO` kèm chỉ dẫn thêm video hợp lệ.
- Có thể tạo *synthetic metadata events* để test API/database/dashboard và *synthetic tracks* để unit-test state machine. Gắn `SYNTHETIC`, KHÔNG tạo video AI giả mang nhãn video bus thật.
- Nếu chỉ có một video bus: vẫn cho phép chạy single-camera counting mode và hiển thị cửa còn lại `NO_SOURCE`. Không dùng cùng một file giả làm hai camera đồng bộ nếu muốn đo Re-ID.

## 3. STACK ƯU TIÊN

- Python 3.11; PyTorch CUDA khi có, CPU fallback nếu không.
- Ultralytics YOLO pretrained hỗ trợ detect class `person`; OpenCV để đọc/ghi video, ByteTrack hoặc BoT-SORT local tracking.
- Re-ID: Torchreid OSNet pretrained *trên Person Re-ID*, không dùng nhầm model ImageNet-only. Cài và tải trọng số từ nguồn chính thức được kiểm chứng; cosine normalized embeddings + scipy Hungarian assignment, reject threshold.
- Backend: FastAPI + Pydantic + SQLite (SQLAlchemy được ưu tiên) để chạy local; API docs Swagger.
- UI: Streamlit cho MVP nhanh, tiếng Việt, có trang cấu hình camera, theo dõi phiên và kết quả. Nếu Streamlit cản trở hiển thị liên tục hai video, dùng FastAPI MJPEG/WebSocket và frontend React/Vite nhẹ nhưng vẫn giữ API đầy đủ; chọn một phương án và làm chạy được, không nửa nọ nửa kia.
- pytest, ruff, type hints cốt lõi, logging và cấu hình YAML/.env; `requirements.txt` hoặc `pyproject.toml` pin phiên bản tương thích.
- Tính tương thích Windows 10/11 là ưu tiên; hướng dẫn PowerShell cài môi trường, GPU, chạy app, đồng thời có lệnh cho Linux.

## 4. CẤU TRÚC KIẾN TRÚC BẮT BUỘC

- `VideoSource` interface: MP4, USB/webcam index, RTSP (có reconnect và timeout), timebase từ FPS/timestamp; per-camera configuration.
- `PersonDetector`: detect class person và confidence/NMS tunable.
- `SingleCameraTracker`: mỗi camera namespace riêng (`front:track001`, `rear:track001`) không đụng ID.
- `DirectionCounter`: ROI/polygon hai vùng hoặc hai vạch và state machine cho từng track: OUTSIDE -> TRANSITION -> INSIDE; state tối thiểu 2–3 frame, hysteresis và debounce. Kết quả event IN tại cửa front và OUT tại cửa rear; có thể cấu hình chiều ngược nếu thước/góc camera khác. Dựa trên bottom-center của box hoặc centroid cấu hình, tránh đếm mỗi frame, tránh tăng số khi chỉ đứng ở ngưỡng, xử lý đi qua rồi quay lại rõ ràng.
- `QualitySelector`: chọn nhiều crop rõ và ít bị che khuất trên tracklet lúc xác nhận crossing, tránh crop người ngoài ROI, có tiêu chí kích thước/sharpness; tách crop phục vụ re-id và không trữ lâu.
- `ReIDEncoder`: embedding OSNet; L2 normalization; batch processing, checkpoint loading báo lỗi rõ.
- `MatchingEngine`: chỉ so OUT với tập IN đang ONBOARD trong cùng session; score similarity, ràng buộc time (`t_out > t_in`), one-to-one Hungarian với reject/dummy; xử lý nhiều OUT gần đồng thời; giữ trạng thái PENDING/CONFIRMED/UNKNOWN và log lý do. Không ghép 1 OUT cho nhiều IN. Cho phép human review xác nhận/bác bỏ match.
- `SessionManager`: `session_id`, nguồn front/rear, offset đồng bộ thời gian, bắt đầu/kết thúc, reset counters/ID, pause/resume processing, lưu events. Nếu video không tương ứng phải cho chế độ UNPAIRED và disable confirmed identity matching.
- `OccupancyEstimator`: tính IN - OUT khi hai luồng tương ứng cùng chuyến, xuất phát với số khách ban đầu có thể cấu hình; không lấy `max(0,...)` để giấu lỗi thiếu dữ liệu, hiển thị cảnh báo khi âm/không đồng bộ.
- `Persistence`: lưu session, source metadata, count events, temporary track ID, match records, metrics, camera configurations vào SQLite. CSV/JSON export, tùy chọn video đã annotate và kết quả chạy.
- `Evaluation`: so counts ground truth theo clip/direction (PCDS labels), event-level precision/recall nếu có frame labels, Re-ID Rank-1/mAP và match precision/coverage chỉ khi có identity ground truth; không tạo metrics giả.

## 5. GIAO DIỆN MVP BẮT BUỘC

UI tiếng Việt, gọn, sử dụng được ngay. Có ít nhất:

1. **Nguồn video:** FRONT/REAR upload MP4 hoặc chọn path/RTSP an toàn; preview frame, trạng thái video, fps/resolution/duration, source provenance; nút `Chạy thử 1 camera` hoặc `Chạy 2 camera`. Luồng thiếu hoặc chưa được kiểm tra phải hiện trạng thái rõ.
2. **Cấu hình vùng cửa:** xem snapshot mỗi camera và vẽ/đặt vùng A/B bằng tọa độ normalized; chọn chiều di chuyển tương ứng IN/OUT, ghi YAML/DB, xem preview overlay. Đừng hardcode ROI cho tất cả góc quay.
3. **Theo dõi:** hai khung video có box/track ID/ROI/đường đếm/overlay, trạng thái GPU/CPU, số IN, OUT, occupancy nếu đủ điều kiện, fps và progress. Có stop, reset và chọn chế độ xử lý batch/realtime playback; tránh giao diện đứng im.
4. **Re-ID:** gallery ảnh crop theo mã `P001` tạm, kết quả candidate với score, trạng thái CONFIRMED/UNKNOWN/UNVERIFIED, cho phép reviewer sửa. Nêu rõ confidence similarity KHÔNG là xác suất đã calibration.
5. **Sự kiện & thống kê:** bảng timestamp, camera, hướng, track ID, session ID, candidate passenger ID, lý do reject; đồ thị timeline, xuất CSV/JSON, danh sách lỗi.
6. **Dataset & Đánh giá:** danh sách nguồn đã tải/nhập, licensing, file label, tổng clip, bảng GT vs predicted, MAE/error, metric NA nếu thiếu GT; xem báo cáo thí nghiệm.
7. **Hướng dẫn demo:** chế độ buýt với camera public, chế độ technical Re-ID benchmark, nhãn rõ sự khác nhau. Không hiển thị mock realtime như kết quả live nếu đang replay.

## 6. QUY TRÌNH TỰ ĐỘNG HÓA / CLI

Bổ sung các lệnh chạy thật (tên có thể thay đổi nếu nhất quán):

- `python scripts/check_environment.py` – Python/CUDA/GPU/ffmpeg/model availability.
- `python scripts/fetch_demo_assets.py --source github-bus-counter` – tải hợp lệ video public nhỏ nếu nguồn phù hợp, xác nhận decode qua OpenCV.
- `python scripts/import_pcds.py --input <local_dir_or_zip> --output data/processed/` – parse cấu trúc bus/front/back + label counts, manifest CSV.
- `python scripts/run_count.py --front <mp4> --rear <mp4> --config configs/cameras.yaml --output runs/test01` – chạy batch, số đếm và annotated MP4/events CSV; nếu một file không có thì single-cam mode.
- `python scripts/run_reid.py --session <session_id> --mode bus-unpaired|paired-gt|technical-benchmark` – chỉ xác nhận đúng loại kết quả theo dữ liệu.
- `python scripts/evaluate.py --pred <...> --gt <...>` – tạo bảng và JSON metric thật.
- `uvicorn backend.main:app --reload` và `streamlit run app/dashboard.py` (hoặc hướng dẫn tương đương nếu chọn UI khác).
- PowerShell quickstart `scripts/setup_windows.ps1` nếu khả thi; không hardcode đường dẫn GPU.

## 7. DATABASE / API HỢP ĐỒNG

Entities tối thiểu: `Session`, `CameraSource`, `CountEvent`, `PassengerTrack`, `ReIDMatch`, `CameraCalibration`, `EvaluationReport`.

API tối thiểu:
- `GET /health`, `GET /device`
- `POST /sessions`, `GET /sessions`, `GET /sessions/{id}`
- `POST /sessions/{id}/start`, `/stop`
- `GET /sessions/{id}/events`, `/stats`, `/matches`
- `PATCH /matches/{id}` cho human verification
- `GET /sessions/{id}/export?format=csv|json`

API triển khai thật, có validation và message lỗi; file upload giới hạn kích thước/đuôi và không đưa RTSP credentials vào logs. Không expose nguồn data riêng nếu triển khai public.

## 8. TESTS VÀ CHẤT LƯỢNG

Viết và chạy:
- Unit test counting: outside->inside = 1, đứng ở ngưỡng = 0, đổi hướng, dao động, nhiều người, tracker ID collision giữa hai camera, track mất giữa cửa.
- Unit test matching: cùng người match đúng trong fixture, non-match bị từ chối, `OUT` trước `IN` không ghép, one-to-one, hai ứng viên áo giống nhau trả UNKNOWN khi không chắc, unpaired video không công bố identity accuracy.
- Unit test session/state/event persistence, occupancy edge cases, CSV/JSON exports, PCDS label parser.
- End-to-end smoke test với video public sau khi kiểm tra và có thể tải; nếu không có Internet, chạy synthetic tracks smoke test rồi ghi rõ chưa xác minh model/video thật.
- Dành bước đo FPS và VRAM trên GPU có thật; nếu không có GPU lúc Codex chạy, hiện `not tested on CUDA`, không tự bịa benchmark.
- `README.md` tiếng Việt, `DATA_SOURCES.md` trích nguồn/giấy phép, `LIMITATIONS.md`, `TEST_REPORT.md` với lệnh đã thực sự chạy và kết quả thực tế.

## 9. CHẤP NHẬN NGHIỆM THU

Chỉ gọi là hoàn thành MVP khi:
- Ứng dụng khởi động được theo README trên một môi trường tiêu chuẩn; có thể tải/nhập MP4 và cấu hình 2 camera.
- Chạy YOLO detection+tracking thật trên ít nhất một video tải được, xuất annotated video và sự kiện IN/OUT thật, hoặc ghi rõ chưa đạt do không tải được assets/mô hình.
- Có 2 engine đếm độc lập, sự kiện có timestamp và không đếm trùng từng frame.
- Module OSNet Re-ID chạy thật với checkpoint Person Re-ID; ghép bằng matching + ngưỡng và UNKNOWN; chạy unit tests ground-truth identity fixtures. Không tự tuyên bố Re-ID bus validated nếu chưa có identity ground truth.
- Dashboard hiển thị nguồn thật (hoặc NO_SOURCE minh bạch), counter, events, review và export thật.
- Có tests tự động chạy và báo cáo kết quả; ít nhất các tests core nêu trên phải pass. Mọi module chưa hoàn thành cần được ghi rõ, không đặt fake placeholders/"TODO production" rồi kết luận complete.

## 10. CÁCH BẠN PHẢI LÀM VIỆC (BẮT BUỘC)

1. Kiểm tra workspace và thiết bị có sẵn. Lập `PLAN.md` theo workstream, nhưng **bắt đầu code ngay**, không dừng ở tài liệu.
2. Triển khai incrementally: (a) environment/video IO, (b) two-camera counting, (c) event/session DB, (d) Re-ID, (e) matching/unknown, (f) UI, (g) tests/evaluation.
3. Sau mỗi module, tự chạy lệnh/test và sửa lỗi; không chờ user chọn framework/trang trí UI. Dùng lựa chọn mặc định như spec nếu cần.
4. Với nguồn dữ liệu, xác minh link, quyền truy cập, định dạng và nội dung. Nếu link chết, tìm nguồn công khai **có thể kiểm chứng** khác cùng bối cảnh cửa xe buýt, ghi nguồn rõ ràng. Không nói video từ cùng chuyến khi không có chứng cứ.
5. Trường hợp môi trường thực thi không cho cài đặt GPU, download hoặc chạy inference, **vẫn viết code và tests đầy đủ nhất có thể**, ghi chính xác phần nào đã chạy và phần nào chưa.
6. Trước khi kết thúc, cung cấp: đường dẫn source code, cách chạy Windows từng bước, nguồn video/dataset thực sự dùng, screenshot hoặc mô tả demo đã kiểm chứng, lệnh test + log pass/fail, những hạn chế còn lại và đường dẫn export thử nghiệm.
7. Không chuyển hướng sang dự báo, GPS, bản đồ, quản lý tài chính hay quá nhiều tính năng ngoài MVP.

Hãy **bắt đầu tạo và triển khai source code ngay trong repository hiện tại**, sau đó tự kiểm tra toàn bộ pipeline. Đừng chỉ giải thích phải làm như thế nào.
