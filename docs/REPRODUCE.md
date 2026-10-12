# Tái lập V34-R1

Chạy từ root repository. Kết quả được công bố ngày 13/10/2026 từ evaluated source commit `282448b0a7e456b2dbea11b310a14be8891de5ea`; controller được tạo từ commit `df1fa0dd712e5d6c7046a24b18cdd1a306a2de59`. Đây là ID provenance của source local lúc chạy, không phải branch cũ được công bố trên GitHub. Hash từng runtime file và checkpoint nằm trong [publication manifest](../results/v34_r1/publication_manifest.json).

## Cài đặt

```bash
python -m pip install -e ".[dev,tracking]"
python -m pytest -q
```

Chọn torch/torchvision tương thích nhau theo [PyTorch official installation](https://pytorch.org/get-started/locally/) khi dùng GPU. FFmpeg phải hỗ trợ `libx264` và `libx265` cho inference/training. `supervision==0.27.0` là phiên bản ByteTrack đã khóa. TrackEval là dependency ngoài dành cho phép đánh giá GT tracking; hai integration test tự skip nếu không khai báo `V34_TRACKEVAL` trỏ đến checkout TrackEval.

## Tái tính kết quả đã đo — không cần GPU hoặc dataset

```bash
python scripts/reproduce_results.py --verify-only
python scripts/reproduce_results.py --out output/reproduced_v34
```

Bốn archive `test_rows.json.gz` chứa **toàn bộ** 112.640 observations, gồm anchor, fixed, forced/calibrated/selected cho mỗi controller. Giải nén giữ nguyên byte gốc và SHA-256. Metadata gồm cohorts, complete status, selection, frozen protocol, evaluator hashes và inventory.

Script kiểm tra runtime/asset hashes, giải nén vào thư mục output trống, kiểm tra grid hoàn chỉnh và gộp theo video. Sau đó evaluator thực hiện đủ 5.000 paired bootstrap cho mỗi trong 12 so sánh và kiểm tra summary tái tạo bằng summary đã công bố. Cần RAM cho toàn bộ bảng JSON; bộ dependency cho bước này vẫn có torch do helper được import, nhưng không tải task weights hay dùng GPU.

Lệnh merge tương đương sau khi giải nén:

```bash
python -m adaptive_vcm.v34.holdout merge --protocol results/v34_r1/protocol/protocol.json --paths output/reproduced_v34/shards/shard0 output/reproduced_v34/shards/shard1 output/reproduced_v34/shards/shard2 output/reproduced_v34/shards/shard3 --out output/merged_v34
```

Không hạ số bootstrap để tạo kết quả công bố. `independent_summary.json` giữ nguyên: `comparison.screen_passes` / `comparison.target_confirmed` là placeholder của curve helper khi chưa nhận CI, trong khi CI nằm ở `bootstrap`. Quyết định toàn nghiên cứu là **top-level `primary_confirmed`** trong hàm `holdout.merge`.

## Chạy lại frozen inference trên JPEG provider

Dataset nguồn: [jizeyong/kinetics400-rgb256-f10-val-data](https://www.kaggle.com/datasets/jizeyong/kinetics400-rgb256-f10-val-data). Không kèm JPEG hoặc video trong repository. Mount/download dataset bằng quyền truy cập của bạn. Đặt `DATASET_ROOT` ở thư mục chứa các folder YouTube ID với `img_*.jpg`.

```bash
python -m adaptive_vcm.v34.holdout evaluate --root DATASET_ROOT --protocol results/v34_r1/protocol/protocol.json --annotations results/v34_r1/protocol/annotations.csv --exclusions results/v34_r1/protocol/exclusions.json --checkpoints checkpoints/v34_r1 --shard 0 --out output/new_shard0
```

Lặp cho shard 1, 2, 3 với thư mục output riêng và sau đó merge đủ bốn. Mỗi shard đánh giá cả ba controller; không train hoặc recalibrate. First run tải frozen torchvision Swin/R3D/detector weights. Cần GPU cho candidate search thực tế. Bốn phiên đã hoàn tất mất khoảng **4 giờ 32 phút đến 5 giờ 10 phút/shard** từ lúc xử lý bắt đầu; đây không phải cam kết runtime trên máy khác và không gồm queue.

Giao thức revision 2 lọc folder có ≥32 JPEG **trước** hash ranking, chọn 1.024 ID official val bằng uniform deterministic hash. Không lọc theo confidence/accuracy, không padding hoặc thay video đã chọn. Muốn tái tạo đúng membership phải có cùng inventory provider; `selection.json` đã công bố giúp phát hiện drift.

Label mapping lấy official CVDF K400 annotations và category order của `Swin3D_T_Weights.KINETICS400_V1`; xem provenance trong protocol. Source frame cadence chưa được xác minh, không xem 25 fps của codec là nhịp thật của provider. Không đưa nhãn TEST vào sender guard.

## Huấn luyện lại controller — một thí nghiệm mới

Runtime giữ cả `ar_ot` và `ar_od` API; frozen checkpoints công bố là **AR+OT R1**. Tái huấn luyện sẽ tạo checkpoint mới, không thay cho chứng cứ của checkpoint đã khóa.

```bash
python -m adaptive_vcm.v34.runner --help
python -m adaptive_vcm.v34.runner --experiment ar_ot --config configs/v34_arot_r1.json --ar-root K400_FRAME_OR_VIDEO_ROOT --mot-root MOT20_ROOT --trackeval TRACKEVAL_CHECKOUT --origin TRAINING_ORIGIN_JSON --out output/new_training
```

`TRAINING_ORIGIN_JSON` là manifest provenance của source snapshot của lần train mới. `MOT20_ROOT` phải chứa `MOT20-*/seqinfo.ini`, `img1/` và `gt/gt.txt` theo adapter. AR adapter dùng cấu trúc video/frame và category label theo `adaptive_vcm.data.ar_plan`; kiểm tra source plan trước khi chạy dài. Các domain K400/MOT20 không có cùng-video joint GT; mask/provenance tách AR GT khỏi source track consistency và OT GT khỏi source AR consistency.

CLI runner có cấu hình cơ sở mặc định `configs/v34.json` để tương thích test. **Luôn truyền `--config configs/v34_arot_r1.json`** cho R1, hoặc dùng frozen independent protocol như lệnh evaluate ở trên.

## Checkpoint và dữ liệu kết quả

| Seed | File | Byte | SHA-256 |
|---|---|---:|---|
| 303401 | [controller_303401.pth](../checkpoints/v34_r1/controller_303401.pth) | 63.923 | `c4622dbd701df88444344ec908263ccebcbbd28da40c9812c1b86749169c1577` |
| 303402 | [controller_303402.pth](../checkpoints/v34_r1/controller_303402.pth) | 63.923 | `5186940d2d687325cea6caba2cde0a69fa6182bba92df949e39753d53f4f77cc` |
| 303403 | [controller_303403.pth](../checkpoints/v34_r1/controller_303403.pth) | 63.923 | `d8c78f2bb1fc5a20883b5d81013555fb1ec66d489cf08648ae0533c8e4efaf4b` |

Ba checkpoint LAST, cùng architecture CNN+GRU 8.661 tham số. Video Swin là task weights torchvision riêng, không phải ba checkpoint này. State hashes và training metadata: [descriptions.json](../checkpoints/v34_r1/descriptions.json).

Dataset/video và pretrained task weights có điều khoản của nhà cung cấp; tải từ nguồn gốc. Bản công bố này cung cấp mã, controller và số đo phục vụ tái lập, không phân phối lại các asset đó.
