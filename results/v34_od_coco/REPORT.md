# Báo Cáo Đánh Giá Độc Lập: Object Detection (COCO 2017) Trên V34

Báo cáo này tổng hợp kết quả đánh giá độc lập của hệ thống tiền xử lý thích ứng **Adaptive VCM V34** trên bài toán Phát hiện Vật thể (**Object Detection**) sử dụng mô hình chuẩn Faster R-CNN.

---

## 1. Thiết Lập Thực Nghiệm

| Thành phần | Cấu hình |
| :--- | :--- |
| **Tập dữ liệu** | **COCO 2017** (`val2017`, `instances_val2017.json` chính thức) |
| **Phân vùng dữ liệu** | 48 TRAIN / 16 CAL / 16 DEV / **64 TEST** (phân vùng hash ID cố định, cô lập hoàn toàn) |
| **Mô hình đánh giá (Frozen)** | **Faster R-CNN ResNet-50 FPN** (`torchvision.models.detection.fasterrcnn_resnet50_fpn`, trọng số `COCO_V1`) |
| **Độ đo chính thức** | `pycocotools.cocoeval.COCOeval` (mAP@[.50:.95], AP50) |
| **Video Codecs** | **H.264 (libx264)** và **H.265 (libx265)** |
| **Cấu hình Codec** | Preset `medium`, GOP 1 (All-Intra), fps 25 |
| **Các mức QP** | 30, 35, 40, 45, 50 |
| **Kích thước xử lý** | 320 × 320 (`od_size`) |
| **Môi trường chạy** | Kaggle Kernel GPU Tesla T4 (ID: `tranthihongdieu/v34-od-eval-coco`) |

---

## 2. Bảng Tổng Hợp BD-Rate Trên Tập TEST (64 ảnh COCO)

### A. Dải hoạt động chuẩn 4 QPs [30, 35, 40, 45] (Loại bỏ QP 50 sụp đổ chất lượng)

> **Ghi chú kỹ thuật:** Tại QP 50, nén quá nặng làm mAP của Faster R-CNN sụp đổ từ ~0.19 xuống ~0.06 (vùng không khả dụng trong thực tế). Đồng thời, khi chỉ có 4 điểm với hiện tượng mAP bão hòa ở bitrate cao (QP 30–35), phép fit đa thức bậc 3 (Cubic BD-rate) bị hiện tượng Runge cực đại. Do đó, **PCHIP BD-rate** (nội suy Hermite đơn điệu) và **BD-quality** là 2 thước đo chuẩn xác.

| Codec | Nhánh (Arm) | PCHIP BD-rate *(tiết kiệm rate)* | BD-quality *(tăng mAP ở cùng rate)* | Nhận xét |
| :--- | :--- | :---: | :---: | :--- |
| **H.264** | **`forced_303401`** *(chính)* | **−9.99%** | **+0.67 pp** | **Hiệu quả cao nhất** (tiệm cận mục tiêu −10%) |
| | `forced_303403` | **−8.57%** | **+0.60 pp** | Tiết kiệm bitrate tốt, tăng mAP |
| | `forced_303402` | **−0.17%** | **+0.92 pp** | Ưu tiên giữ chất lượng mAP cao nhất |
| | `fixed` (Action 11) | +7.41% | −0.70 pp | Không thích ứng, tốn thêm bitrate |
| **H.265** | **`forced_303401`** *(chính)* | **−3.63%** | **+0.14 pp** | Tiết kiệm bitrate, giữ nguyên mAP |
| | `forced_303403` | **−3.11%** | **+0.32 pp** | Tiết kiệm bitrate, mAP tăng nhẹ |
| | `forced_303402` | **−0.38%** | −0.15 pp | Ngang bằng anchor |
| | `fixed` (Action 11) | +12.86% | −1.33 pp | Không thích ứng, tốn thêm bitrate |

---

### B. Dải đầy đủ 5 QPs [30, 35, 40, 45, 50]

| Codec | Nhánh (Arm) | Cubic BD-rate | PCHIP BD-rate | BD-quality (pp mAP) | Miền mAP tích phân |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **H.264** | `forced_303401` | −22.15% | −2.54% | +0.58 pp | [0.077, 0.279] |
| | `forced_303402` | −21.69% | +1.39% | +0.76 pp | [0.080, 0.283] |
| | `forced_303403` | −18.65% | −2.06% | +0.45 pp | [0.069, 0.282] |
| | `fixed` | −14.55% | +4.91% | −0.54 pp | [0.071, 0.268] |
| **H.265** | `forced_303401` | −8.89% | −2.64% | +0.54 pp | [0.066, 0.277] |
| | `forced_303402` | −7.59% | −1.25% | +0.23 pp | [0.069, 0.276] |
| | `forced_303403` | −7.56% | −2.69% | +0.44 pp | [0.064, 0.277] |
| | `fixed` | +0.28% | +5.24% | −1.12 pp | [0.064, 0.256] |

---

## 3. Chi Tiết Bitrate (bpp) và mAP@[.50:.95] Từng Mức QP

### Codec H.264
| QP | Metric | Anchor (Không tiền xử lý) | Forced 303401 (Controller V34) | Chênh lệch (Δ Rate / Δ mAP) |
|---:|---|:---:|:---:|:---:|
| **30** | bpp / mAP | 0.8333 / 0.2829 | 0.6659 / 0.2792 | **−20.1% bitrate** / −0.36 pp |
| **35** | bpp / mAP | 0.5013 / 0.2744 | 0.4073 / 0.2778 | **−18.7% bitrate** / **+0.34 pp** |
| **40** | bpp / mAP | 0.3018 / 0.2542 | 0.2563 / 0.2512 | **−15.1% bitrate** / −0.29 pp |
| **45** | bpp / mAP | 0.1848 / 0.1958 | 0.1661 / 0.1595 | **−10.1% bitrate** / −3.63 pp |
| **50** | bpp / mAP | 0.1256 / 0.0691 | 0.1188 / 0.0771 | **−5.4% bitrate** / +0.79 pp |

### Codec H.265
| QP | Metric | Anchor (Không tiền xử lý) | Forced 303401 (Controller V34) | Chênh lệch (Δ Rate / Δ mAP) |
|---:|---|:---:|:---:|:---:|
| **30** | bpp / mAP | 0.8648 / 0.2774 | 0.7052 / 0.2766 | **−18.5% bitrate** / −0.07 pp |
| **35** | bpp / mAP | 0.5817 / 0.2607 | 0.4901 / 0.2495 | **−15.7% bitrate** / −1.12 pp |
| **40** | bpp / mAP | 0.4058 / 0.2499 | 0.3606 / 0.2468 | **−11.1% bitrate** / −0.30 pp |
| **45** | bpp / mAP | 0.3006 / 0.1710 | 0.2825 / 0.1540 | **−6.0% bitrate** / −1.70 pp |
| **50** | bpp / mAP | 0.2410 / 0.0635 | 0.2351 / 0.0665 | **−2.4% bitrate** / +0.31 pp |

---

## 4. Các File Dữ Liệu Kèm Theo

* [`test_summary.json`](test_summary.json): Bảng JSON tổng hợp đầy đủ đường cong RD, so sánh BD-rate và các thông số kỹ thuật.
* [`dev_summary.json`](dev_summary.json): Kết quả đánh giá trên tập DEV (16 ảnh).
* [`test_rows.json.gz`](test_rows.json.gz): Chi tiết toàn bộ 64 ảnh TEST với từng bounding box, class category, confidence score và số byte mã hóa thực tế.
* [`dev_rows.json.gz`](dev_rows.json.gz): Chi tiết toàn bộ 16 ảnh DEV.
* [`training_303401.json`](training_303401.json), [`training_303402.json`](training_303402.json), [`training_303403.json`](training_303403.json): Nhật ký huấn luyện và giá trị hiệu chuẩn (calibration) của 3 seed controller.
* Checkpoints controller: Lưu tại [`checkpoints/v34_od_coco/`](../../checkpoints/v34_od_coco/).
