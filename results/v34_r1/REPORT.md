# V34-R1 — tổng hợp kiểm chứng Video Swin trên 1.024 video

Ngày tổng hợp: 01:03 13/10/2026 (UTC+7)

Kết luận theo giao thức đã chốt: **ĐẠT** tiêu chí kiểm chứng AR với Video Swin, checkpoint 303401, trên cả H.264 và H.265.

## Bốn phiên Kaggle

Các tên seed 0–3 trong notebook là bốn phần dữ liệu riêng, mỗi phần 256 video. Mỗi phần đều đánh giá cả ba checkpoint 303401/303402/303403. Gộp dự đoán và rate theo video rồi tính lại đường rate–accuracy; không lấy trung bình BD-rate của bốn phần.

| Phần | Tài khoản | Video | Trạng thái | Thời gian xử lý thực | BD H.264 của phần (%) | BD H.265 của phần (%) |
|---|---|---:|---|---:|---:|---:|
| 0 | [hieusunday0412](https://www.kaggle.com/code/hieusunday0412/v34-swin-independent-s0-282448b0) | 256 | COMPLETE | 4h 32m | -16.49 | -12.04 |
| 1 | [dieulinhh](https://www.kaggle.com/code/dieulinhh/v34-swin-independent-s1-282448b0) | 256 | COMPLETE | 5h 05m | -20.47 | -11.18 |
| 2 | [ngynanhthuw](https://www.kaggle.com/code/ngynanhthuw/v34-swin-independent-s2-282448b0) | 256 | COMPLETE | 5h 10m | -10.34 | -10.01 |
| 3 | [dngbolm](https://www.kaggle.com/code/dngbolm/v34-swin-independent-s3-282448b0) | 256 | COMPLETE | 4h 47m | -10.92 | -11.66 |

BD trong bảng từng phần dùng checkpoint chính 303401 và Video Swin. Đây là số mô tả, không thay thế kết quả trên 1.024 video. Thời gian tính từ lúc mã bắt đầu xử lý đến hoàn tất, không gồm thời gian hàng đợi.

## Kết quả gộp — Video Swin

| Codec | Checkpoint seed | BD-rate (%) | PCHIP BD-rate (%) | BD-quality (điểm %) | Khoảng tin cậy BD-rate | Mẫu bootstrap hữu hạn | Point gate |
|---|---:|---:|---:|---:|---|---:|---|
| h264 | 303401 | -14.46 | -16.13 | 4.13 | 97.5%: [-17.19; -11.68] | 5000/5000 | Đạt |
| h264 | 303402 | -14.42 | -16.61 | 4.20 | 95%: [-16.83; -12.01] | 5000/5000 | Đạt |
| h264 | 303403 | -14.67 | -16.55 | 4.25 | 95%: [-17.10; -12.12] | 5000/5000 | Đạt |
| h265 | 303401 | -11.41 | -12.72 | 4.85 | 97.5%: [-13.42; -9.43] | 5000/5000 | Đạt |
| h265 | 303402 | -11.62 | -13.01 | 4.98 | 95%: [-13.37; -9.88] | 5000/5000 | Đạt |
| h265 | 303403 | -11.36 | -12.73 | 4.85 | 95%: [-13.14; -9.61] | 5000/5000 | Đạt |

Checkpoint 303401 được chọn trước khi xem tập mới. Hai khoảng tin cậy chính 97,5% được điều chỉnh Bonferroni để có mức bao phủ chung 95% trên hai codec. Các checkpoint còn lại và R3D dùng khoảng mô tả 95%.

BD-rate âm biểu thị giảm rate ở cùng mức accuracy trên khoảng chất lượng chung của hai đường. Ngưỡng −10% là tiêu chí của ước lượng điểm; tiêu chí thống kê đã chốt là cận trên khoảng tin cậy dưới 0, không phải chứng minh mức tiết kiệm tối thiểu 10% trong toàn bộ khoảng tin cậy.

## Accuracy và rate tại cùng QP — Video Swin, checkpoint 303401

| Codec | QP | Anchor bpp | Adaptive bpp | Anchor accuracy (%) | Adaptive accuracy (%) | Δ accuracy (điểm %) | Δ rate (%) | Corrected | Damaged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h264 | 30 | 0.386702 | 0.337214 | 64.65 | 65.33 | 0.68 | -12.80 | 24 | 17 |
| h264 | 35 | 0.220151 | 0.198585 | 60.25 | 61.91 | 1.66 | -9.80 | 24 | 7 |
| h264 | 40 | 0.131425 | 0.123074 | 51.46 | 54.30 | 2.83 | -6.35 | 35 | 6 |
| h264 | 45 | 0.084846 | 0.081551 | 36.52 | 41.80 | 5.27 | -3.88 | 63 | 9 |
| h264 | 50 | 0.059342 | 0.058212 | 18.75 | 22.66 | 3.91 | -1.90 | 42 | 2 |
| h265 | 30 | 0.419638 | 0.370387 | 64.16 | 64.94 | 0.78 | -11.74 | 22 | 14 |
| h265 | 35 | 0.257748 | 0.236723 | 57.91 | 60.55 | 2.64 | -8.16 | 36 | 9 |
| h265 | 40 | 0.170652 | 0.162032 | 47.75 | 50.98 | 3.22 | -5.05 | 43 | 10 |
| h265 | 45 | 0.122456 | 0.119577 | 31.15 | 37.11 | 5.96 | -2.35 | 70 | 9 |
| h265 | 50 | 0.098157 | 0.097301 | 10.16 | 15.43 | 5.27 | -0.87 | 55 | 1 |

Corrected: anchor sai, adaptive đúng; damaged: anchor đúng, adaptive sai. Δ rate tính từ tỷ lệ rate trung bình của hai nhánh, có tính common paid header.

## Kiểm tra chuyển sang R3D-18 — cùng packet

| Codec | Checkpoint seed | BD-rate (%) | PCHIP BD-rate (%) | BD-quality (điểm %) | CI 95% | Point gate mô tả |
|---|---:|---:|---:|---:|---|---|
| h264 | 303401 | 1.25 | 2.36 | -0.41 | [-2.10; 4.44] | Không đạt |
| h264 | 303402 | 2.39 | 3.51 | -0.67 | [-0.90; 5.55] | Không đạt |
| h264 | 303403 | 2.02 | 3.15 | -0.59 | [-1.29; 5.19] | Không đạt |
| h265 | 303401 | 2.11 | 2.23 | -0.65 | [-0.74; 5.13] | Không đạt |
| h265 | 303402 | 2.43 | 2.14 | -0.63 | [-0.47; 5.48] | Không đạt |
| h265 | 303403 | 2.11 | 2.15 | -0.61 | [-0.77; 5.12] | Không đạt |

## Đối chứng lọc cố định

| Codec | Backbone | Fixed so với anchor: BD-rate (%) | PCHIP (%) | BD-quality (điểm %) |
|---|---|---:|---:|---:|
| h264 | swin3d_t | -1.86 | -1.79 | 0.37 |
| h264 | r3d_18 | 4.68 | 6.12 | -1.21 |
| h265 | swin3d_t | -1.51 | -0.77 | 0.51 |
| h265 | r3d_18 | 3.64 | 4.33 | -1.36 |

Đây là đối chứng mô tả của nhánh fixed đã có trong giao thức. Chưa tính khoảng tin cậy cho so sánh adaptive trực tiếp với fixed; không kết luận superiority từ chênh lệch hai BD-rate so với anchor.

## Tỷ lệ lựa chọn identity

| Codec | Checkpoint seed | Identity | Tổng clip × QP | Tỷ lệ identity (%) |
|---|---:|---:|---:|---:|
| h264 | 303401 | 1888 | 5120 | 36.88 |
| h265 | 303401 | 1855 | 5120 | 36.23 |
| h264 | 303402 | 1906 | 5120 | 37.23 |
| h265 | 303402 | 1855 | 5120 | 36.23 |
| h264 | 303403 | 1891 | 5120 | 36.93 |
| h265 | 303403 | 1845 | 5120 | 36.04 |

## Phạm vi và kiểm tra

- 1.024 ID video và 1.024 pixel hash khác nhau; bốn phần đúng membership và có đủ 112.640 dòng grid.
- Protocol revision 2, checkpoint hashes, state trước/sau và task-model hashes khớp; không train hoặc chỉnh ngưỡng trên tập này.
- 5.000 paired-video bootstrap cho mỗi phép so sánh; cùng chỉ số video được lấy lại mẫu qua toàn bộ QP.
- Chỉ xác minh endpoint AR trên official K400 val dùng JPEG provider, 16 frame stride 2, RGB 128×128 và QP 30/35/40/45/50. Không phải kết quả full-video multi-view K400 chuẩn.
- Chưa có GT object tracking hay object detection độc lập trong lượt này. OT source-consistency guard không thay thế HOTA/IDF1/MOTA; không suy ra hiệu quả OT/OD từ bảng AR.
- Nhịp decode gốc của provider chưa được kiểm chứng; independence nhắm tới adaptive-controller development và official backbone TRAIN. Backbone có thể từng được lựa chọn bằng published validation performance.

Dữ liệu đầy đủ: [independent_summary.json](independent_summary.json), [merge-audit.json](merge-audit.json), [shard-summary.json](shard-summary.json).
