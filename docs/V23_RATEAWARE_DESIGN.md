# V23: learned preprocessing tối ưu theo byte thật, ưu tiên QP cao

V22 chọn learned chỉ 5/1.280 điểm AR và 0/1.000 điểm OD. Hash stream learned
giống identity ở 997/1.280 điểm AR và 976/980 điểm OD có thử learned. V23
giải quyết độ mạnh của biến đổi, cách học và phép đo đóng góp từng nhánh.
Mục tiêu BD-rate < −10% cho AR/OD trên từng codec vẫn chưa được xác nhận.

## Từ bằng chứng V22 đến thay đổi

V22 khởi tạo sigmoid strength tại −3, tương đương 0,0474 trước protection và
motion. AR còn dùng mask chuyển động lấy max theo cả clip rồi tiếp tục giảm
strength theo motion trong model. Những phép nhân này có thể làm mức edit
hiệu dụng rất nhỏ. OD chỉ có các sigma learned≤3, trong khi controls có sigma
4/8/12 và thường được chọn. Đây là các hạn chế trực tiếp của parameterization;
chưa có ablation chứng minh từng nguyên nhân của thất bại V22.

V23 dùng policy nhỏ dự đoán **7 control**: strength, 5 trọng số expert và
temporal strength. Policy nhận RGB, motion, protection, QP và codec; có 6.919
tham số ở width24. Policy lấy context của cả clip, nên đây là preprocessing
offline. Renderer tái dùng frame nguồn đã lọc trước đó theo cơ chế causal.

Bank AR: Gaussian0,7/1,5/3/6; bank OD: Gaussian2/4/8/12. Expert thứ năm dùng
block average ở QP<45 (grid4 dưới QP40, grid8 từ QP40) và trung bình màu/DC
theo frame ở QP≥45. Expert DC cho phép loại bỏ mạnh chi tiết nền khi các
hệ số tần số cao vốn đã bị quantize nhiều. Đây là giả thuyết kỹ thuật của V23,
không phải đảm bảo bitrate hay chất lượng.

Prior strength tăng theo QP30→50: AR khoảng0,38→0,73, OD0,62→0,92, trước
protection/motion. Neural controls có thể tăng hoặc giảm strength; không ép
mọi source phải có tiết kiệm. Motion giảm edit ít hơn ở QP cao. AR dùng
semantic map bình phương, giữ core≥0,95 bằng đúng pixel nguồn; motion được
áp dụng theo từng frame trong renderer thay vì hard-protect cả motion tube.
OD giữ mask box/halo như V22. Mọi output là convex blend, không thêm RGB
residual và không đổi decoder, QP encoder hay mẫu số pixel của bitrate.

## Học bằng phép đo codec

Trainer mới dùng objective:

`L(c) = rate_credit(c) + dual * normalized_teacher_violation(c)`.

Rate credit bằng `log(bytes(c)/bytes(identity))` khi probe thỏa guard; nếu
probe vi phạm thì chỉ ghi nhận overhead dương, không thưởng tiết kiệm byte.
Như vậy bỏ gần hết background để tiết kiệm rất nhiều nhưng làm hỏng task
cũng không có objective tốt hơn identity. Margin vượt slack được chia cho
slack trước khi tính penalty; dual không giảm để chấp nhận thêm vi phạm.

Teacher violation gồm phần vượt slack của teacher tệ nhất và penalty khi
không giữ decision. Source/anchor/trial được đo qua codec thật và frozen
teacher. Không dùng true label, mAP/Top-1, evaluator hoặc mẫu DEV/TEST để học.
Tỷ lệ byte được chuẩn hóa theo source tại cùng QP: cùng mức tiết kiệm10% tạo
tín hiệu rate tương đương ở stream1KB và100KB, tránh rate-gradient chỉ phụ
thuộc giá trị bpp tuyệt đối vốn nhỏ ở QP cao.

Mỗi bước đo center và hai perturbation `c±0,5*Δ`, với Δ Rademacher. Slope
được ước lượng bằng `(L_plus−L_minus)/(2ε)*Δ` rồi backpropagate qua policy.
Đây là SPSA ở không gian 7 control, không phải entropy model hay pixel STE.
Byte codec và quyết định teacher rời rạc; estimator có nhiễu và plateau.
Không có bảo đảm hội tụ hoặc suy ra tăng accuracy từ slope này.

Cứ10 bước thêm3 probe mạnh: expert1, expert3 và expert4. Chỉ target thật sự
tiết kiệm≥1% byte và thỏa mọi teacher guard mới được dùng cho imitation loss
weight0,2. Target là control từ TRAIN probes của renderer; không dùng controls
V22 làm ground truth. Identity được giữ khi không có probe đủ điều kiện.
Numerical value của gradient carrier không được báo như measured RD loss.

1.000 bước, width24, Adam1e−3, ε0,5, dual ban đầu2 và cập nhật theo codec/QP,
seed302001; khởi tạo mới, lưu FINAL-LAST. Sau vòng đầu phủ tất cả groups,
QP30/35/40/45/50 có sampling weight1/1/2/3/3 trên mỗi codec. Khoảng80% budget
đi vào QP≥40; evaluation vẫn dùng đủ5 QP để giữ phép so sánh với V22.

## Đánh giá đóng góp và chất lượng

`--ablate-learned` tái dùng các stream đã encode, chấm cùng source/QP:

- `adaptive`: composite selector, như V22, nhưng với V23 guard.
- `controls`: selector trên controls, loại candidate learned.
- `learned_guarded`: chỉ identity và learned, vẫn áp dụng mọi guard.
- `learned_raw`: stream learned trước fallback; có thể tăng byte hoặc giảm task.

Primary adaptive curves có paired bootstrap. Component curves có cubic BD,
PCHIP và quality gaps nhưng không có CI; `screen_passes` của component không
thể true vì chưa có bootstrap. Không coi tần suất được chọn là bằng chứng
causal và không thay tie-break để ưu tiên learned. `high_qp_diagnostics` ghi
byte change và Top-1/mAP gap từng arm tại QP40/45/50.

AR V23 yêu cầu giữ argmax của anchor kể cả khi confidence thấp, cùng với
guard source/confidence và KL đã có. Đổi guard áp dụng cả controls và learned;
vì vậy so sánh V22/V23 là so sánh recipe, không phải ablation riêng architecture.
Teacher agreement vẫn không bảo đảm giữ chất lượng trên evaluator khác.
Gate cuối giữ worst-QP gap≥−0,5pp và BD-rate<−10% trên từng codec; không dùng
việc tiết kiệm byte để bỏ qua quality gate.

## Tài liệu khoa học

Lu et al. hỗ trợ hướng preprocessing có điều kiện quantization và proxy cho
codec truyền thống; V23 chuyển sang đo codec thật khi cập nhật policy, thay
vì tái tạo kiến trúc/loss của bài. [Preprocessing Enhanced Image Compression
for Machine Vision](https://arxiv.org/abs/2206.05650); journal metadata và
các bài LAB/IEEE TIP/CVPR liên quan nằm trong [thiết kế V22](RESEARCH_DESIGN.md).

Zhao et al. dùng neural preprocessing và differentiable virtual codec cho
video machine vision, rồi kiểm tra bằng codec chuẩn. Nguồn arXiv được nộp
năm2025 và tác giả ghi accepted DCC2024; không suy ra đây là bài journal2025.
[Author preprint](https://arxiv.org/abs/2512.15331). Không dùng mức tiết kiệm
được báo trong bài làm kết quả V23.

Cơ sở cho estimator hai perturbation: Spall, IEEE Transactions on Automatic
Control37(3),1992,332–341, DOI10.1109/9.119632.
[Publisher](https://ieeexplore.ieee.org/document/119632/),
[author institution full text](https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf).
Việc áp dụng SPSA vào renderer VCM này là lựa chọn triển khai của dự án.

## Kiểm thử và trạng thái

64 pytest pass trên Windows CPU, gồm real H.264/H.265, train/evaluate AR/OD
fixtures, task/config checkpoint compatibility, semantic core, scene cuts,
SPSA trên quadratic có gradient biết trước, high-QP byte saving và component
pairing. Fixtures không đo accuracy của pretrained task networks.

`scripts/rateaware_codec_probe.py` dùng nền texture tổng hợp và ROI màu phẳng
ở AR16×128×128, OD1×320×320. Đây là **policy khởi tạo**, không có task analyzer.
Output đánh dấu `trained=false`, `quality_evaluated=false`, `target_confirmed=false`.
Số liệu này chỉ chứng minh có edit ảnh hưởng byte thật. Nó không chứng minh
BD-rate, Top-1, mAP hoặc hiệu quả trên dữ liệu tự nhiên.

Lượt Kaggle kế tiếp giữ512 TRAIN source/task, 1.000 bước, AR128 DEV clip và
OD100 DEV ảnh với hai codec/five QP, để so sánh đúng cohort với V22. OD vẫn
là pilot ảnh đơn; hiệu quả OD video cần dataset/video annotations riêng.

Chạy:

```bash
python -m adaptive_vcm.train_rateaware --task ar --root /data/kineticscleaned \
  --count 512 --steps 1000 --seed 302001 --out outputs/v23_ar/train
python -m adaptive_vcm.evaluate --task ar --root /data/kineticscleaned \
  --config configs/v23_screen.json --checkpoint outputs/v23_ar/train/preprocessor_last.pth \
  --count 128 --split dev --codecs h264 h265 --bootstrap 2000 --ablate-learned --out outputs/v23_ar/eval
```

OD dùng `--task od`, COCO image root và `--annotations`, count100/bootstrap200.
Kaggle runner dùng `--recipe v23 --mode learned` và immutable full commit SHA.
