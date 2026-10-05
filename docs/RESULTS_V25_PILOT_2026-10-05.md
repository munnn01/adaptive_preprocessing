# V25 pilot: audit độc lập, DEV16

Pilot `qktttttttttt/v25-ranking-pilot-ar-s302101` COMPLETE ngày 2026-10-05, GPU Tesla T4, exit 0. Code chạy tại commit `0bfcead429dcde08d0e67cb0589216aa7a357131`. Có đủ 80 nhóm đo TRAIN trên 80 source, 1.500 bước replay và 160 điểm DEV: 16 clip × 5 QP × 2 codec, với 7 arm/điểm. Bootstrap bằng 0. Không có TEST độc lập trong pilot này.

Kiểm tra commit/module/config/checkpoint/NPZ/measurement hash, tách source và source pixel giữa TRAIN/DEV, lựa chọn ít byte nhất trong đúng tập đề xuất và giữ guard hiện hành đều PASS. Tất cả **160 anchor** trùng V24 về source SHA, stream SHA và coded bytes. Tất cả **160 controls streams** trùng V24 adaptive, nên đóng góp byte thêm của V25 được ghép cặp trực tiếp, không phải biến thiên bản build codec.

Archive SHA-256: `0e9bfaeb71d4dbec1b4f168403ad990f4658838628dca4793623c9df8cbb6ad6`.

## Đóng góp thực đo

Learned ranking được chọn **18/160 điểm**: H.264 8/80, H.265 10/80. Trong đó **15 điểm dùng filter không đổi resolution**, tạo thêm 4.045 byte tiết kiệm trên controls; 3 điểm resample tạo thêm 862 byte. Tổng thêm 4.907 byte, tương ứng **0,5847 điểm phần trăm** tính theo tổng anchor bytes của pilot. Network học xếp hạng một bank filter cố định; kết quả này không chứng minh CNN học được hệ số lọc pixel.

Ở QP≥40, policy tiết kiệm thêm 1.898 byte trên controls, **0,6224 điểm phần trăm** của 304.962 anchor bytes. Nó thu được 33,26% phần khả năng tiết kiệm thêm mà full-bank teacher oracle đo được. Oracle chỉ là upper bound theo encoder teachers, không phải bảo đảm accuracy của evaluator độc lập.

Mọi tỷ lệ trong bảng dùng đầy đủ coded bytes, gồm headers, trên cùng QP. Dấu dương của cột tăng thêm nghĩa là learned dùng ít byte hơn comparator. Group-static là top3 riêng từng codec/QP, đóng băng theo mean guarded actual savings đo trên TRAIN; có cùng ngân sách ba đề xuất như learned. Comparator này chưa có đánh giá quality đầy đủ: 3/160 chosen streams chưa trùng một arm được evaluator chấm. Toàn bộ 10 nhóm byte/group-static trùng kết quả của auditor riêng do parent cập nhật.

| Codec | QP | Adaptive tiết kiệm (%) | Thêm trên controls (pp) | Thêm trên group-static (pp) | Learned chọn | Filter không resample | r2plus gap với anchor (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H.264 | 30 | 24,983 | 0,797 | 0,075 | 3 | 3 | 0 |
| H.264 | 35 | 18,771 | 0,190 | −1,493 | 1 | 1 | −6,25 |
| H.264 | 40 | 11,768 | 1,408 | 0,914 | 4 | 3 | +6,25 |
| H.264 | 45 | 3,775 | 0 | −0,868 | 0 | 0 | +6,25 |
| H.264 | 50 | 0,970 | 0 | 0 | 0 | 0 | 0 |
| H.265 | 30 | 23,013 | 0,127 | −2,313 | 1 | 1 | 0 |
| H.265 | 35 | 11,422 | 1,225 | −0,051 | 3 | 2 | +6,25 |
| H.265 | 40 | 8,641 | 0,803 | 0,509 | 2 | 1 | 0 |
| H.265 | 45 | 1,479 | 0,685 | 0,098 | 3 | 3 | 0 |
| H.265 | 50 | 0,171 | 0,171 | 0,171 | 1 | 1 | 0 |

H.265 QP45 có thêm ba filter không resample: `core_joint`, `uniform_detail_soft`, `core_temporal_soft`. QP50 chọn thêm `core_residual_soft`. H.264 QP45/50 chưa có đóng góp learned trên controls.

Trong toàn pilot, learned dùng **nhiều hơn** global-static 4.518 byte và group-static 4.593 byte. Ở riêng QP≥40, learned dùng ít hơn group-static 722 byte. Vì vậy chưa chứng minh được neural policy tốt hơn policy TRAIN-only đơn giản trên toàn dải QP.

r2plus accuracy của adaptive so controls không đổi tại mọi codec/QP; r3d gap so anchor bằng 0 ở mọi nhóm. Tuy nhiên r2plus H.264 QP35 vẫn giảm 6,25pp với anchor, giống controls/V24. Không đạt gate giảm accuracy tối đa 0,5pp. Một clip trong DEV16 đã tương ứng 6,25pp, nên không suy diễn kết luận ổn định từ pilot.

## TRAIN khả thi và giới hạn

Bank có ít nhất một action đạt guard hiện hành và tiết kiệm ≥1% ở **30/80 nhóm**. Cả 30 nhóm đều có một filter khả thi không đổi resolution. Có **184 action trials khả thi**, trong đó **157 không resample**, trên 1.360 nonidentity trials. Learned top3 phủ 29/30 nhóm khả thi trên chính TRAIN, so global-static top3 là 15/30. Đây là resubstitution; teacher-safety accuracy 100% trên TRAIN không phải độ chính xác dự báo trên DEV.

| Codec/QP | Nhóm TRAIN | Nhóm có action khả thi | Trials khả thi | Oracle mean savings (%) |
|---|---:|---:|---:|---:|
| H.264/30 | 3 | 2 | 16 | 12,478 |
| H.264/35 | 4 | 4 | 28 | 16,519 |
| H.264/40 | 8 | 4 | 28 | 7,599 |
| H.264/45 | 10 | 6 | 25 | 6,403 |
| H.264/50 | 10 | 1 | 6 | 0,474 |
| H.265/30 | 5 | 5 | 34 | 14,566 |
| H.265/35 | 4 | 3 | 32 | 9,382 |
| H.265/40 | 6 | 3 | 11 | 3,956 |
| H.265/45 | 10 | 2 | 4 | 0,962 |
| H.265/50 | 20 | 0 | 0 | 0 |

H.265 QP50 TRAIN không tìm được action khả thi trong 20 nhóm, dù DEV có một điểm learned thắng. H.26445/50 có thêm capacity của bank nhưng learned chưa thu được; training nhỏ và overfitting vẫn là vấn đề. Không so trực tiếp tỷ lệ feasible với V24 TRAIN vì source/QP sampling và số action khác nhau.

**Chưa xác nhận mục tiêu BD-rate <−10%**. Không sử dụng BD từ DEV16/zero bootstrap để xác nhận mục tiêu; phải chờ DEV đầy đủ và TEST độc lập đã đăng ký.

Các bằng chứng máy đọc nằm trong `audit.json` và `independent.json`. Script độc lập `independent_compare.py` chứa quy tắc group-static TRAIN-only, kiểm tra pairing V24 và phân rã đóng góp theo family. Bundle gốc ở `archive_only/`; bản giải nén ở `extracted/outputs/v25-ranking-pilot-ar-s302101/`.
