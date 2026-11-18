# Phân tích train Lab 09 — raw và clean

Thí nghiệm dùng YOLO26n trên Colab GPU, 20 epoch, ảnh 640 px, batch 16, seed 0 và 1. Hai biến thể dùng chung tập validation 40 ảnh và tập test 100 ảnh. Nguồn số liệu là bốn JSON kết quả, bảng [comparison.md](comparison.md) và báo cáo QC.

## 1. Chênh lệch so với độ dao động giữa seed

mAP50 trung bình của raw là **0,72575**, của clean là **0,68225**. Chênh lệch clean − raw là **−0,0435** (−4,35 điểm phần trăm). Chênh lệch theo seed lần lượt là −0,0768 và −0,0102. Độ lệch chuẩn mẫu giữa hai seed là khoảng 0,00021 với raw và 0,04731 với clean. Raw dẫn ở cả hai lượt quan sát, nhưng mức giảm trung bình của clean nhỏ hơn độ dao động giữa seed của chính clean. Với hai seed và khả năng rò rỉ trong raw, chưa đủ bằng chứng để kết luận hiệu quả tổng quát của việc làm sạch. Nên chạy thêm seed và kiểm lại trên tập đánh giá độc lập.

## 2. Số ảnh raw thực sự dùng khi train

JSON ghi 224 ảnh raw và 205 ảnh clean. Trường train_images đếm file trong thư mục ảnh đã dựng trước khi Ultralytics lọc. Báo cáo intake xác nhận có hai ảnh hỏng trong lô gốc, và raw còn có các dòng nhãn sai cú pháp, ID sai hoặc box tràn. Để xác định chính xác mẫu số train thực tế, cần lấy dòng quét train của Ultralytics, phân biệt mục corrupt với backgrounds vì ảnh nền hợp lệ vẫn có thể được dùng. Bốn JSON và bảng tổng hợp hiện có không lưu số đó.

## 3. Rò rỉ từ tập test

01_intake.csv ghi bốn cặp LEAK_TEST: img_0049/test_033, img_0167/test_044, img_0174/test_085 và img_0221/test_009. Bốn ảnh raw này được loại khỏi clean. Đây là 4/100 ảnh test có bản tương ứng trong đầu vào raw. Mức ảnh hưởng lên mAP50 chưa được đo, và cần kiểm xem Ultralytics có giữ từng ảnh tương ứng khi train raw hay không.

Một phép kiểm là đánh giá lại cả hai checkpoint trên cùng 96 ảnh test sau khi bỏ bốn ảnh trùng, rồi so chênh lệch với kết quả trên 100 ảnh. Chênh lệch khi đổi tập test còn phụ thuộc độ khó của bốn ảnh bị bỏ. Để kiểm riêng ảnh hưởng của rò rỉ khi train, có thể chạy thêm biến thể raw bỏ đúng bốn ảnh bị rò rỉ, giữ các cấu hình còn lại và đánh giá các biến thể trên cùng một tập độc lập với dữ liệu train.

## 4. AP50 theo lớp

| Lớp | Raw trung bình | Clean trung bình | Clean − raw |
|---|---:|---:|---:|
| car | 0,7375 | 0,7352 | −0,0024 |
| person | 0,8221 | 0,8252 | +0,0032 |
| truck | 0,6485 | 0,5144 | −0,1341 |
| bus | 0,6949 | 0,6544 | −0,0406 |

Person tăng nhiều nhất về trung bình, khoảng 0,32 điểm phần trăm, nhưng tăng ở seed 0 (+0,0143) và giảm ở seed 1 (−0,0080). Một giả thuyết là duyệt các ca POSSIBLE_MISSING hoặc POSSIBLE_LOOSE_BOX giúp một số box person đầy đủ hoặc chặt hơn. Báo cáo v1 có 267 cờ thiếu nhãn và 60 cờ box lỏng trên toàn lô. Để kiểm giả thuyết theo lớp, cần đối chiếu các ảnh đã duyệt trong 06_review_log.csv với nhãn raw và clean, đếm riêng box person được thêm hoặc chỉnh. Truck giảm mạnh nhất. Báo cáo class_stats.csv chỉ có 41 box truck so với 585 box person. Có thể kiểm xem tập ảnh bị loại hoặc các lần sửa nhãn có làm mất mẫu truck hữu ích hay không bằng cách đếm theo lớp trước và sau từng pha.

## 5. Chất lượng của clean

Hàng đợi có 149 ảnh, vượt ngân sách duyệt 80 ảnh. Sau duyệt, gate còn 64/205 ảnh chưa duyệt có cảnh báo (31,22%) và giữ Hold. Báo cáo v3 vẫn ghi các cờ thiếu nhãn, box lỏng, sai lớp và dương tính giả có thể xảy ra. Các cờ này cần người xác minh vì model kiểm cũng có thể sai. Tập clean mới là bản đã qua QC và duyệt một phần. Để có mốc tham chiếu chất lượng cao hơn, cần duyệt đầy đủ hoặc so với biến thể oracle trên cùng danh sách ảnh và cùng seed.
