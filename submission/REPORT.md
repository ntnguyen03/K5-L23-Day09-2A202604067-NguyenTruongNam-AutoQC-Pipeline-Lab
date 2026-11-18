# Báo cáo Day 09 — Nguyễn Trường Nam, 2A202604067

## 1. Kết luận

**Hold.** Intake nhận 205/224 ảnh và loại 19. Pipeline xuất 205 ảnh dưới dạng bản nháp có manifest, với 0 lỗi bắt buộc còn lại. Sau khi mô phỏng duyệt 80 ảnh, còn 64/205 ảnh có cảnh báo chưa duyệt (31,22%), vượt ngưỡng Hold 10%.

## 2. Lỗi và phân việc

Số dưới đây là số **issue** trong `02_error_list_v1.csv`, gồm 36 lỗi bắt buộc và 404 cảnh báo, trên lô đầu vào 224 ảnh với 205 ảnh qua intake.

| Mã lỗi | Số issue | Xử lý |
|---|---:|---|
| `CORRUPT_IMAGE`, `BAD_SIZE`, `DUP_IMAGE_EXACT`, `DUP_IMAGE_NEAR`, `LEAK_TEST` | 2, 3, 8, 2, 4 | Máy loại ảnh lỗi, quá nhỏ, trùng hoặc rò rỉ từ intake. |
| `BAD_FORMAT`, `BAD_CLASS_ID`, `BOX_ZERO`, `DUP_BOX` | 3, 6, 4, 7 | Máy bỏ dòng vi phạm quy tắc. |
| `BOX_OUT_OF_IMAGE`, `BAD_CLASS_NAME` | 8, 2 | Máy cắt box hoặc chuẩn hóa tên lớp. Hai issue tên lớp được xử lý bằng một lần sửa YAML. |
| `MISSING_LABEL_FILE` | 6 | Chuyển người gán vì thiếu file nhãn chưa xác nhận được đây là ảnh nền. |
| `BOX_HUGE`, `BOX_TINY`, `EMPTY_LABEL`, `CLASS_RARE` | 3, 6, 12, 4 | Người xem box hoặc ảnh bị cờ. `CLASS_RARE` là cảnh báo để phân tích ở mức bộ dữ liệu. |
| `POSSIBLE_FALSE_POS`, `POSSIBLE_LOOSE_BOX`, `POSSIBLE_MISSING`, `POSSIBLE_WRONG_CLASS` | 29, 60, 267, 23 | Người đối chiếu ảnh với nhãn vì dự đoán của model cũng có thể sai. |

Hàng đợi có 149 ảnh. Điểm cộng dồn theo trọng số mã lỗi và độ tin cậy khi có, cộng thêm điểm cho các ca đã sửa box hoặc cần gán nhãn. `img_0160` đứng đầu (109,727 điểm) với cờ thiếu nhãn và nhầm lớp. Trong 80 ảnh đã duyệt mô phỏng, 74 ảnh có ít nhất một cờ `POSSIBLE_*`, theo `06_review_log.csv`. Đây là số ảnh được xem, không phải số lỗi ngữ nghĩa được xác nhận.

## 3. Calibration

| Tham số | Ban đầu → chọn | Precision / recall trước → sau | Lý do |
|---|---|---|---|
| `duplicates.phash_dist` | 2 → 4 | Gần trùng 1,00 / 0,50 → 1,00 / 0,67 | Bắt 4/6 thay vì 3/6 ca, không tăng FP trên calib. Mức 4 là ngưỡng nhỏ nhất có kết quả này. |
| `second_opinion.conf` | 0,25 → 0,10 | Box lỏng 0,29 / 0,40 → 0,38 / 0,60. Thiếu nhãn 0,50 / 1,00 → 0,40 / 1,00. Nhầm lớp giữ 0,56 / 0,83. | Ưu tiên bắt thêm ca box lỏng (3/5 thay vì 2/5), chấp nhận nhiều báo động thiếu nhãn cần duyệt hơn. |

Precision 1,00 của pHash chỉ dựa trên 4 ca gần trùng được gắn cờ trên calib ở ngưỡng 4. Trên raw, hai cặp gần trùng ghi ở intake (`img_0076/img_0042` và `img_0143/img_0037`) đều có khoảng cách 0. Lợi ích tăng recall từ 2 lên 4 đo được trên calib chưa xuất hiện ở hai cặp raw này.

## 4. Cửa chặn và điểm mù

Phép thử `--budget 0` với báo cáo và export riêng cho **Rework**, mã thoát **20**. Báo cáo v3 còn 6 lỗi thiếu file nhãn, ví dụ `img_0031`. `Test-Path work/exp_nb/clean_v1/manifest.json` trả `False`.

- pHash ở ngưỡng đã chọn bỏ sót 2/6 ca gần trùng được chèn trên calib. Ảnh biến đổi mạnh có thể còn lọt vào lô.
- So sánh với model lớn bỏ sót 2/5 ca box lỏng được chèn trên calib ở ngưỡng đã chọn. Box đúng cú pháp vẫn có thể sai hình học so với vật thể mà không được đưa vào hàng đợi.

## 5. Train raw vs clean

YOLO26n, Colab GPU, 20 epoch, batch 16, ảnh 640 px, seed 0 và 1. Hai biến thể dùng chung validation và test. Số ảnh trong bảng là số file ảnh đã dựng trước bước lọc của Ultralytics.

| Biến thể | Ảnh đầu vào | mAP50 | mAP50–95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| raw | 224 | 0,726 ± 0,000 | 0,512 ± 0,004 | 0,764 ± 0,061 | 0,649 ± 0,028 |
| clean | 205 | 0,682 ± 0,047 | 0,469 ± 0,025 | 0,718 ± 0,111 | 0,603 ± 0,012 |

Clean thấp hơn raw ở cả hai seed. Chênh mAP50 trung bình là −0,0435, nhỏ hơn độ lệch chuẩn 0,0473 của clean. Với hai seed, chưa đủ bằng chứng để kết luận việc làm sạch làm giảm khả năng tổng quát. Hai giả thuyết cần kiểm là raw được lợi từ 4 ảnh trùng test và clean vẫn còn 64 ảnh có cảnh báo chưa duyệt. Có thể đánh giá lại hai biến thể trên cùng tập test đã loại ca trùng, đối chiếu các cờ còn lại bằng mắt, rồi chạy thêm seed. Xem [phân tích theo lớp và cách kiểm](results/ANALYSIS.md).


## 6. Quyết định và ghi chú AI

Giữ **Hold**. Với mẫu số 205 và 0 lỗi bắt buộc, cần giảm số ảnh còn cảnh báo chưa duyệt xuống tối đa 20. Nếu lô và cảnh báo giữ nguyên, cần xử lý ít nhất 44 trong 64 ảnh còn cờ rồi kiểm lại.

- Prompt chính: Viết bốn module Python theo docstring và test có sẵn, đọc nhãn YOLO và cache dự đoán, xuất `Issue`, log sửa và quyết định gate, chỉ sửa bản làm việc khi autofix.
- Kiểm tra bản sinh bằng 79 unit test/e2e, lỗi chèn trong calib và phép thử ngân sách 0 trên bản làm việc riêng.
- Một lỗi trong hướng dẫn AI: lệnh kiểm Python qua PATH gọi alias WindowsApps và không chạy trên máy thử. Dùng trực tiếp Python 3.13 trong môi trường ảo để cài và chạy kiểm.
