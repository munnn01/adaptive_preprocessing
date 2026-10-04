# Adaptive Video Preprocessing for VCM

Bộ preprocessing V22 cho **Action Recognition (AR)** và **Object Detection (OD)**,
phát triển từ bài học V1–V21 và tài liệu [IEEE TCSVT, IEEE TIP, CVPR cùng các
bài trong LAB](docs/RESEARCH_DESIGN.md). Mục tiêu: **BD-rate < −10% so với từng
H.264 và H.265**, theo Top-1 của AR và COCO mAP của OD.

**Trạng thái:** có code và kiểm thử; chưa có kết quả task thực nghiệm V22 để
tuyên bố đạt mục tiêu. Kết quả cũ và con số từ bài báo không phải kết quả V22.

## Bộ adaptive preprocessor

- [Learned multiscale blend](adaptive_vcm/learned.py): mạng nhỏ dự đoán mức
  blend và hỗn hợp Gaussian0.7/1.5/3.0, điều kiện theo QP/codec, có bảo vệ
  semantic và gate theo chuyển động/cắt cảnh.
- [Preprocessing và controls](adaptive_vcm/preprocessing.py): giảm chi tiết
  nền cho OD; giữ saliency/chuyển động cho AR; resize/blur controls từ V1/V2.
- [Chọn stream](adaptive_vcm/selection.py): chỉ nhận byte thật và tín hiệu
  teacher; fallback identity khi không có biến đổi đủ an toàn/tiết kiệm.
- [Train](adaptive_vcm/train.py): real-codec forward, STE backward, task regret
  và rate prior được hiệu chỉnh bằng bitrate đo. [Thiết kế](docs/RESEARCH_DESIGN.md)
  giải thích rõ giới hạn của rate prior và feature loss OD.

Decoder dùng H.264/H.265 và mạng tác vụ thông thường. Không cần neural
postprocessor hay truyền mask. Thử nhiều candidate tăng chi phí encoder;
runner ghi thời gian chuẩn bị source và toàn bộ candidate search.

## Chạy local

Python>=3.10, PyTorch/TorchVision, OpenCV, SciPy, pycocotools, FFmpeg có x264/x265:

```powershell
cd D:\STUDY\LAB\hope
& D:\STUDY\AI\envs\ten_env\python.exe -m pytest tests -q --basetemp=outputs\pytest
```

Linux/Kaggle có thể cài `pip install -e .`; không cần cài lại torch nếu runtime
đã cung cấp. CPU Windows/Conda tự tìm FFmpeg trong `Library/bin` khi thiếu PATH.

```bash
python -m adaptive_vcm.train --task ar --root /data/kineticscleaned \
  --count 512 --steps 1000 --seed 302001 --out outputs/ar_train
python -m adaptive_vcm.evaluate --task ar --root /data/kineticscleaned \
  --checkpoint outputs/ar_train/preprocessor_last.pth \
  --count 128 --split dev --codecs h264 h265 --bootstrap 2000 --out outputs/ar_eval
python -m adaptive_vcm.train --task od --root /data/coco/val2017 \
  --annotations /data/coco/annotations/instances_val2017.json \
  --count 512 --steps 1000 --out outputs/od_train
python -m adaptive_vcm.evaluate --task od --root /data/coco/val2017 \
  --annotations /data/coco/annotations/instances_val2017.json \
  --checkpoint outputs/od_train/preprocessor_last.pth \
  --count 100 --split dev --codecs h264 h265 --bootstrap 200 --out outputs/od_eval
```

Bỏ `--checkpoint` để chạy nhánh preprocessing không huấn luyện. TRAIN/DEV/TEST
tách theo hash source; COCO val2017 trong pilot cũng chia hash trước khi chọn
ảnh. Đây không phải tuyên bố đánh giá toàn COCO hay dữ liệu holdout mới.

Output: manifest mã/config/source, log huấn luyện, final checkpoint,
JSONL từng source/QP/arm, audit candidate selection, curve/CI/decision.
`--save-streams` giữ các elementary streams đã chọn. BD-rate dùng pixel nguồn
làm mẫu số ngay cả khi candidate giảm độ phân giải. Runner từ chối thư mục
output đã có dữ liệu để tránh trộn bằng chứng.

## Kaggle từ pool.json

Token chỉ đọc từ file ngoài repo và truyền qua environment subprocess.
Notebook private, free T4, code clone đúng full commit SHA; không chứa token.

```powershell
$pythonVCM = 'D:\STUDY\AI\envs\ten_env\python.exe'
$commitVCM = git rev-parse HEAD
& $pythonVCM scripts/kaggle_runner.py prepare --account qktttttttttt \
  --slug v22-adaptive-ar-s302001 --task ar --mode learned --commit $commitVCM \
  --count 128 --steps 1000 --bootstrap 2000
& $pythonVCM scripts/kaggle_runner.py submit --account qktttttttttt \
  --slug v22-adaptive-ar-s302001 --pool D:\STUDY\LAB\pool.json
& $pythonVCM scripts/kaggle_runner.py status --account qktttttttttt \
  --slug v22-adaptive-ar-s302001 --pool D:\STUDY\LAB\pool.json
```

PowerShell: viết mỗi lệnh trên một dòng hoặc dùng backtick để nối dòng
(dấu `\` trong ví dụ chỉ giúp trình bày). OD dùng `--task od --count 100
--bootstrap 200` và tài khoản có COCO2017. `download` thu output theo cùng
handle. Không có lịch tự chạy/check ngầm.

Gate báo cáo rõ `point_passes`, `screen_passes`, và `target_confirmed=false`.
Cần kết quả thực nghiệm độc lập trước khi kết luận đã đạt <−10% cho cả OD/AR
và hai codec.
