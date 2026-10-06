# Adaptive Video Preprocessing for VCM

Bộ preprocessing V22–V29 cho **Action Recognition (AR)** và **Object Detection (OD)**,
phát triển từ bài học V1–V21 và tài liệu [IEEE TCSVT, IEEE TIP, CVPR cùng các
bài trong LAB](docs/RESEARCH_DESIGN.md). Mục tiêu: **BD-rate < −10% so với từng
H.264 và H.265**, theo Top-1 của AR và COCO mAP của OD.

**V29, 2026-10-06:** ba nhánh A/B/C cải thiện pipeline hiện có: giám sát trực tiếp
expert/mức lọc từ byte thật; thử lọc nền static/dynamic theo task; hiệu chỉnh
điều kiện nhận learned trên TRAIN tách khỏi tập fit. Vòng đầu TRAIN32/DEV32
cho cả AR/OD, đủ hai codec và năm QP; sau đó chạy riêng AR128 rồi OD100.
Giữ ba đề xuất neural, tọa độ RGB và bộ đánh giá độc lập. Xem
[giao thức và giới hạn tài liệu SSVC](docs/V29_PROTOCOL.md). Toàn bộ 359 test đã
pass và review đã xử lý các lỗi phát hiện. Chưa có BD-rate V29.

V28 full đã được kiểm toán: AR Top-1 **−6,758% / −2,333%**, OD mAP
**−9,594% / −6,642%** (H.264 / H.265). Learned chọn132/1280 điểm AR và168/1000
điểm OD; mục tiêu bốn trường hợp cùng dưới−10% chưa đạt.

**V28, 2026-10-06:** learned preprocessor dự đoán mức lọc và hỗn hợp bốn expert
theo từng pixel, dùng bảo vệ semantic/chuyển động và nền ổn định theo đoạn video.
AR/OD có checkpoint riêng, giữ H.264/H.265 và tọa độ ảnh gốc. Train đủ lưới
source × codec × QP bằng byte thật; đánh giá ba đề xuất neural và đo phần tiết
kiệm thêm so với controls. V27 vẫn là thí nghiệm riêng. **306 kiểm thử đã qua;
chưa có kết quả V28 xác nhận mục tiêu BD-rate.** Xem [protocol V28](docs/V28_PROTOCOL.md),
[kiểm chứng](docs/V28_VERIFICATION.md) và [đọc MoCrop/tài liệu AR](docs/LITERATURE_V28.md).

**Trạng thái 2026-10-05:** V24 AR đã hoàn tất và được kiểm toán. Guard hết
mâu thuẫn source/anchor, nhưng learned vẫn **0/1.280** và chọn identity trên
mọi điểm DEV. AR/r2plus1d_18: BD-rate **−5,21% / −2,41%**; tại QP50 chỉ tiết
kiệm **0,6325% / 0,1006%** byte (H.264 / H.265). Toàn bộ thay đổi đến từ controls.
OD V23/ResNet50: **−11,87% / −7,63%**, learned **100/1.000**.
**Mục tiêu chung chưa đạt.**

**V24 tập trung AR:** sửa guard mâu thuẫn source/anchor, dùng cùng guard trong
train/eval; học chọn profile preprocessing bằng nhãn từ byte codec thật trên
TRAIN. Các profile mạnh và temporal/DC phục vụ QP40/45/50; identity vẫn hợp lệ.
Lượt Kaggle V24 và đối chứng chỉ sửa guard đã COMPLETE; hai nhánh có kết quả
giống nhau. Xem [thiết kế V24](docs/V24_AR_DESIGN.md).

**V25 là nhánh thử nghiệm AR:** [17 mức lọc mới](docs/V25_BANK.md),
[học xếp hạng từ mọi phép đo TRAIN](docs/V25_POLICY.md), và
[đối chứng policy cùng ngân sách top-3](docs/V25_DESIGN.md).
Kết quả ảnh tổng hợp chỉ kiểm tra khả năng giảm byte và tính đúng của pipeline;
chưa xác nhận cải thiện trên video thật hay BD-rate. Nhánh này chỉ được công
nhận tốt hơn sau DEV và TEST độc lập.

## Bộ adaptive preprocessor

- [V28 spatial learned preprocessor](adaptive_vcm/motion_learned.py),
  [motion support](adaptive_vcm/motion_support.py) và [trainer](adaptive_vcm/train_motion.py):
  học map liên tục từ target TRAIN khả thi qua codec thật; QP>=40 có trọng số2.
  [Selection](adaptive_vcm/motion_selection.py) giữ guard nghiêm ngặt và tách static/oracle khỏi primary.
- [V25 rank policy](adaptive_vcm/ranking.py) và [trainer](adaptive_vcm/train_ranking.py):
  dự đoán độ an toàn teacher và log(byte/anchor), học minibatch từ mọi mức lọc;
  [bank](adaptive_vcm/task_bank.py) có shrinkage chi tiết, chroma, temporal và resize kết hợp.
- [V24 AR profile policy](adaptive_vcm/profiles.py) và [trainer](adaptive_vcm/train_profiles.py):
  học chọn profile khả thi theo byte thật, giữ identity fallback và teacher guard.
- [V23 rate-aware policy](adaptive_vcm/rateaware.py) và [trainer measured-codec](adaptive_vcm/train_rateaware.py):
  7 control, không dùng rate prior của V22; tăng budget học ở QP cao.
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

V25 pilot DEV16 đã audit: learned được chọn18/160 điểm, gồm15 filter không
resample; tiết kiệm thêm4907 byte trên controls ghép cặp với V24. Policy vẫn
thua TRAIN group-static trên toàn pilot và chưa đạt quality gate. Xem
[báo cáo V25 pilot](docs/RESULTS_V25_PILOT_2026-10-05.md).
Run [V25 DEV128 đã hoàn tất](docs/RESULTS_V25_FULL_2026-10-05.md): learned
131/1.280 điểm, nhưng BD-rate held-out chỉ −4,12% H.264 / −3,00% H.265 và
quality gate chưa đạt. Chưa xác nhận mục tiêu BD-rate hoặc TEST.

V26 thử nghiệm bổ sung16 mức lọc luma/DC/exposure giữ nguyên resolution, và
policy41D học byte tiết kiệm **thêm trên controls** bằng TRAIN CV tách source.
Bank có33 action ngoài identity; policy vẫn chỉ đề xuất3. Xem
[thiết kế và cách chạy V26](docs/V26_DESIGN.md). Kết quả TRAIN/synthetic không
được dùng để xác nhận accuracy hay mục tiêu BD-rate.
[V26 pilot đã audit](docs/RESULTS_V26_PILOT_2026-10-05.md): learned 27/160
điểm và TRAIN thêm 3 nhóm khả thi, nhưng tiết kiệm tại QP cao chưa tăng đều;
chỉ 7 lựa chọn là lọc thuần. Nhãn policy nay trừ byte nguyên, với 160 test
đạt và review Superpowers đã đóng hai lỗi P2. Xem
[tài liệu nghiên cứu nguồn gốc](docs/LITERATURE_V26.md).

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
