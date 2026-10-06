# V29: semantic guidance within the existing RGB preprocessing pipeline

User-approved bounded design, 2026-10-06. Baseline commit:
`53d720780f444d98562ebdf99fbcc512d3b57501` (V28). V27/V28 are retained.

## Experiments and execution order

All runs evaluate both H.264 and H.265 at QP30/35/40/45/50. First screen:
three independent variants, each TRAIN32 and DEV32 for AR and OD, eight fixed
epochs, width12, seed302901, final-LAST checkpoint, 2,000 paired bootstrap draws.
Each private notebook runs AR then OD sequentially. Accounts are assigned in
pool insertion order, starting with nguyenhoanglan1232, vtk269, wagur124705,
subject to verified quota and successful scheduling. Credentials stay local.

After auditing the screen, run AR TRAIN128/DEV128 separately, followed by OD
TRAIN100/DEV100. These counts describe the existing development protocol:
128 sampled video clips and100 single COCO images, not the complete underlying
datasets. A larger DEV includes previously observed development examples and
is exploratory. A held-out TEST is necessary for a confirmatory claim.

The variants use the same four named experts, twelve reference profiles and
exactly three primary neural scales(.5/1/1.5). Original controls remain fixed.
Reference profiles and TRAIN staticK3 are audit comparisons, excluded from the
primary pool. A changed RGB transform always receives fresh actual-codec
TRAIN measurements; old bytes/guard outcomes are not reused as labels.

| Variant | Compared with A | Question |
|---|---|---|
|A|Direct measured expert/strength/identity supervision; editable RGB loss and capped marginal-byte utility weights|Can the existing neural renderer learn useful strong profiles more reliably?|
|B|Stationary editable-background DC and causal expert-logit smoothing for AR; separate luma/chroma background treatment for single-frame OD|Does this task-specific static/dynamic treatment improve actual rate–accuracy?|
|C|Fit75% of TRAIN sources; calibrate on the disjoint25% using post-fit actual neural streams and encoder teachers|Does non-worsening teacher admission reduce quality damage from learned replacements?|

For all V29 profiles and neural outputs, use the same protection/motion alpha
attenuation. Hard semantic cores remain pixel-exact before compression. OD has
no fabricated temporal cues. Flow/cuts remain the existing V28 source RGB
proxy. B changes a combined renderer treatment; it does not isolate each
subcomponent's causal effect.

C never uses DEV, ground-truth labels or the independent evaluators to fit
the model/admission policy. Per codec/QP, restrict calibration neural actions
to the original strict guards, at least1% anchor saving and strict extra
saving beyond the original control winner. Among actions whose maximum
teacher-relative distance is nonpositive, use its75th percentile as the
admission threshold. If none exists, disable learned admission for that group.
Inference retains the original guard and additionally enforces this policy
for learned actions only. Audit the same C weights with the policy removed
(`policy_unrestricted`), so policy effects can be separated from model fitting.
The32-source screen gives C24 fit and8 calibration sources, versus32 fit
sources in A/B; across-variant results therefore include this data allocation
difference. Teacher stability does not guarantee independent Top1/mAP.

## Locked measurements

Standard RGB bitstreams and decoding; no receiver side channels or adapted
analyzers. Full preprocessing/selection/probe times are recorded. Elementary
stream bytes include codec headers and use original source pixels for bpp.
AR frozen encoder teachers are r3d_18/mc3_18; primary independent analyzer is
r2plus1d_18. The reported r3d_18 curve overlaps the teacher role. OD encoder
teacher is MobileNet and independent analyzer is ResNet50, COCO mAP@.50:.95.

Report four separate BD-rates: AR/OD × H.264/H.265. The objective is strict
BD-rate<-10% in each cell, with the existing independent quality-gap and
uncertainty gates. Report canonical cubic and PCHIP, actual rate–accuracy
points, common measured quality intervals, failed overlap/plateaus and paired
uncertainty. Do not extrapolate to make a curve pass. QP40/45/50 diagnostics
include actual extra bytes beyond controls/staticK3, selected neural counts,
independent quality deltas and policy effects. Counts alone are not gains.

## Literature grounding and limits

Local SSVC paper: *Semantically Video Coding: Instill Static-Dynamic Clues into
Structured Bitstream for AI Tasks*, arXiv2201.10162v2,9May2022,21pages,
SHA256`ddcd70d862c0a3c4b89e2276229df6a9500d773f581987e57fb945498e870b92`.
Published successor: JVCIR93,103816,May2023,
https://doi.org/10.1016/j.jvcir.2023.103816.

SSVC separates object/appearance, flow and predictive feature residual streams.
Its OD result uses losslessly transmitted source detector IDs/boxes in the
header; its UCF101 AR uses a trained two-stream TSN/ResNet152 and selected
I-frame/flow transport. Those protocols differ from decoded-RGB re-inference
with frozen independent analyzers. Their published savings do not predict
V29 BD-rate. B is an encoder-side engineering hypothesis inspired by
complementary static/dynamic cues, not an implementation of the SSVC codec.

Lu etal., *Preprocessing Enhanced Image Compression for Machine Vision*
(https://arxiv.org/abs/2206.05650), supports QP-aware standard-codec
preprocessing and frozen downstream objectives. Its trained BPG proxy and
multi-stage training are not reproduced by V29's actual-codec distillation;
V29 has no differentiable learned codec or asserted codec gradient.

Ge etal., *Task-Aware Encoder Control for Deep Video Compression*,CVPR2024
(https://arxiv.org/abs/2404.04848), modifies DVC modes/GoP structures.
Sun etal., *Embedding Compression Distortion in Video Coding for Machines*
(https://arxiv.org/abs/2503.21469), transmits distortion representation and
embeds it in downstream models. Both are useful context, but their rate gains
do not establish pure standard-RGB preprocessor gains. Prior MoCrop/CoViAR/
MM-ViT literature boundaries remain documented in `LITERATURE_V28.md`.

## Why these changes

Audited V28 full AR: BD-6.758%/-2.333%(H.264/H.265), learned132/1280,
extra24,330bytes beyond controls; high-QP extra4,693bytes. AR TRAIN contains
223 positive,363 no-byte-margin identity and694 guard-only identity targets.
Positive RGB loss hardly decreases across four epochs; the existing local
soft mixture receives no direct expert/strength supervision.

Audited V28 full OD: BD-9.594%/-6.642%, learned168/1000, extra131,254bytes;
high-QP extra25,052bytes. DC wins300/492 positive TRAIN targets,247 at full
strength. OD learned replacements can add independent mAP deficits despite
passing the source-detector guard. Some AR quality losses already occur in
unchanged controls; learned-only calibration cannot remove that limitation.
No V29 improvement is asserted before completed actual-bitstream evaluation.

## Implementation verification

On 2026-10-06, the complete project suite passed: 359 tests, exit0,
470.68 seconds with the local `ten_env` Python. The 53 V29 tests cover
renderer behavior, teacher-only training, disjoint calibration, checkpoint
loading, admission selection and executable joint notebook generation.
Regression tests first reproduced the missing C policy integrity/count
checks and incorrect V29 CLI defaults, then passed after their fixes.
The independent code review and focused recheck found no remaining material
issue. `compileall` and `git diff --check` also passed. These checks establish
implementation behavior; GPU completion and independent BD-rate still
require actual experiment outputs.
