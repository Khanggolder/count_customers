# Kế hoạch triển khai BusReID AI MVP

Trạng thái được cập nhật theo kết quả chạy thật, không dùng số liệu hardcode.

- [x] Môi trường, cấu hình, nguồn MP4/RTSP và kiểm tra metadata video.
- [x] Hai pipeline detector/tracker/counter độc lập FRONT và REAR.
- [x] Session, event, occupancy, SQLite, export CSV/JSON và REST API.
- [x] OSNet Person Re-ID, quality crop, matching one-to-one và UNKNOWN.
- [x] Dashboard Streamlit tiếng Việt, cấu hình nguồn/ROI, theo dõi và review.
- [x] Parser/evaluation PCDS, tải và xác minh hai video demo công khai (PCDS chưa tải do nguồn giới hạn; ghi rõ trong báo cáo).
- [x] Unit/integration/E2E tests; benchmark thật và báo cáo giới hạn.

Thứ tự ưu tiên: counting end-to-end trước, Re-ID thử nghiệm sau, dashboard cuối.

