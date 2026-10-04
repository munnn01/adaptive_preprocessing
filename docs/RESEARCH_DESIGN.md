# Adaptive Video Preprocessing — V22

Registered 2026-10-04 (Asia/Bangkok), before this implementation has any new
Kinetics/COCO task result. The software targets **BD-rate < −10% against each
of H.264 and H.265 separately**. AR uses Top-1; OD uses COCO mAP@[.50:.95].

## Mechanism and central research question

Can a task-protected, multiscale source blend remove expensive background
detail and stationary temporal activity while preserving decoder task outputs?
This question remains open. Gaussian attenuation, convex pixel bounds, and
encoder-teacher agreement are not proofs of codec savings or task accuracy.

The shared preprocessing operation is

`x_pre[t] = x[t] + alpha[t] * (sum_k w_k[t] * G_sigma_k(x[t]) - x[t])`.

`w` is a softmax over sigma 0.7/1.5/3.0; `alpha` is conditioned on source
content, motion, semantic protection, QP, and codec identity. A small learned
gate produces these maps. Spatial averaging regularizes map transitions;
causal averaging reuses the previous gate only away from motion and cuts.
Semantic core pixels have alpha=0. The output is a convex source/low-pass
blend. There is no learned RGB residual or trained decoder component.

The training-free branch uses fixed registered blend strengths, source
saliency/motion tubes for AR, and detector boxes plus an outward block-aligned
context halo for OD. The AR protection tube looks over the entire input clip;
only the temporal pixel/gate blending is causal. This is an offline clip
preprocessor, not a claimed zero-lookahead streaming implementation.

The encoder tests actual x264/x265 bitstreams. An eligible transformed stream
must cost at least 1% fewer bytes than codec-only and satisfy every registered
encoder task guard. Identity is the fallback. No labels, correctness flags,
or decoder-evaluator outputs enter candidate selection. Encoder analysis and
all trial encode/decode operations have a real compute cost, which is recorded.

## Evidence to implementation

| Primary source, verified 2026-10-04 | What the source supports | Implementation and limits |
|---|---|---|
| Lu et al., **IEEE TCSVT 34(12), 2024**, [publisher](https://ieeexplore.ieee.org/document/10632166/), DOI 10.1109/TCSVT.2024.3441049; local `2206.05650v1.pdf` is the earlier preprint | Quantization-adaptive neural preprocessing before a standard codec and task-oriented optimization | QP conditioning, teacher task preservation, real decoded forward pass. Our architecture/loss are different; its image-coding gains do not establish H.264/H.265 video or AR performance. |
| Talebi et al., **IEEE Transactions on Image Processing, 2021**, [author institution](https://research.google/pubs/better-compression-with-deep-pre-editing/) | Learn edits before compression under rate and image-quality objectives | Rate-dependent editing before the codec. Replace perceptual objectives with machine task constraints. JPEG results are not VCM evidence. |
| Chadha & Andreopoulos, **CVPR 2021**, [CVF paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chadha_Deep_Perceptual_Preprocessing_for_Video_Coding_CVPR_2021_paper.html) | A preprocessing-only video framework with a motion-aware rate objective | Add temporal activity and motion/cut gating. Their perceptual rate-quality result is not a Top-1/mAP result. |
| Fischer et al., **ICASSP 2022**, [author preprint](https://arxiv.org/abs/2203.05944) | Source saliency can inform machine-oriented coding | Source detector ROI protection; independently score a different detector. This implementation does not reproduce their VVC encoder control. |
| Bae & Rhee, **ITC-CSCC 2025**, local full text `Dual-Region_Preprocessing_for_Machine-Friendly_JPEG_Compression.pdf`, DOI 10.1109/ITC-CSCC66376.2025.11137672; [official program](https://itc-cscc2025.org/2025/download/ITC-CSCC_2025_Program_Book_20250625.pdf) | ROI/NROI preprocessing can change the task/rate tradeoff; their method also blurs ROI | Background-only suppression is our initial safer arm; ROI cores remain exact. Their JPEG/YOLO finding does not prove transfer to HEVC/Faster R-CNN. |
| Otsuki & Nitta, **IEVC 2026**, local full text `Gaussian_Filtering_to_Improve_Object_Detection_Accuracy_in_Coded_Video.pdf`, DOI 10.1109/IEVC69170.2026.11508261; [official program](https://www.iieej.org/wordpress/wp-content/uploads/2025/12/IEVC2026_Program_260223.pdf) | Filter type, position and QP affect detection; proposed combination is averaging **pre-filter** plus Gaussian **post-filter** | Motivate low-pass/QP ablations and strict task guards. The Gaussian post-filter contribution is excluded here because this project focuses on preprocessing. Do not attribute combined-method accuracy gains to our pre-filter. |
| Eimon et al., **IEEE MIPR 2025**, local full text `ROI-Packing_Efficient_Region-Based_Compression_for_Machine_Vision.pdf`, DOI 10.1109/MIPR67560.2025.00044 | Removing irrelevant regions can improve machine-oriented image coding | Preserve object context while simplifying background. We do not pack/reconstruct ROIs or use the source's reported savings as our own. |
| Różek et al., local full text `Video_Coding_for_Machines_using_Object_Analysis_and_Standard_Video_Codecs.pdf` | Encoder object analysis and background simplification before standard coding | Detector-guided preprocessing and actual-stream evaluation. Retargeting and coordinate side information are omitted. Venue details are not inferred from title. |
| Güleryüz et al., [Sandwiched Compression preprint](https://arxiv.org/abs/2402.05887), local PDF | Neural wrappers can repurpose standard codecs, with important proxy/complexity considerations | Motivate codec-aware preprocessing and measured validation. The paper uses pre/post wrappers; this project measures preprocessing alone. |

Access scope: the six relevant LAB text extractions were read for method details;
publisher/author/official conference pages were used for external verification.
The journal articles above are distinguished from conference papers and preprints.
The research-lookup Parallel CLI was unavailable, so primary-source web retrieval
and local full-text extracts supplied this bounded engineering evidence matrix.
No external paper is presented as a result of this project. Licensed LAB PDFs
are not copied into this public repository.

## Lessons retained from V1–V21

- V1/V2: retain source resizing and blur transforms as explicit controls, a
  label-free encoder policy, original-pixel normalization, and paired sources.
  This new selector is not a rerun of frozen V2-C.
- V3–V12: keep real H.264/H.265, QP conditioning and independent task evaluation;
  avoid assuming a differentiable proxy predicts final bitrate.
- V13–V18: retain explicit task-regret guards, separate codec/QP duals,
  fixed budgets, final-LAST checkpoints, and exclusion of holdout labels.
- V19/V20: stop increasing the learned additive-residual filter width as the
  sole mechanism. Historical H.264 sigma2 was +4.10% BD-rate; sigma3 was
  −1.65% and failed the worst-QP task guard. These are old development results.
- V21: extend the bounded source-to-lowpass parameterization to a learned
  multiscale mixture with semantic protection and motion/cut-aware gate reuse.
  V21's design had no confirmed real-codec task result.

Original local references: `D:/STUDY/LAB/111` HEAD 8a862d2 and
`D:/STUDY/LAB/pre_processor` HEAD 58d2714. Their working trees are not modified.
The canonical polynomial BD implementation is copied from `111/src/metrics/bd_rate.py`
(SHA256 5584556d55138ee72e18fc84bb63137e0a56e47b7de8dba5f2da5df7d8ebd88f).
COCO evaluation/bootstrap derives from `111/src/metrics/detection.py`
(source SHA256 d58d68f08e64ec2c6156433082c90a32254cf4a8dae1686aec09e2344eb0240e),
with a local import and empty `info` compatibility field. All new preprocessing
and learned-gate code is implemented here; no historical checkpoint is uploaded.

## Fixed screen and training protocol

`configs/v22_screen.json` freezes QPs 30/35/40/45/50, medium preset,
16 frames at stride2, AR128px, OD320px, transforms and selection thresholds.
H.264 and H.265 share source IDs and frame sampling. All rates count actual
elementary-stream bytes including headers, divided by original T*H*W pixels.

AR encoder teachers are frozen r3d_18/mc3_18. Decoder evaluations use
r2plus1d_18 (not consulted by this policy) and r3d_18 (on-teacher diagnostic).
All three model families have historical exposure in earlier research;
current teacher separation does not create a historically untouched analyzer.
OD masks/guards use frozen Faster R-CNN MobileNet; scoring uses frozen
Faster R-CNN ResNet50 with full COCO AP over the selected image population.
COCO is a single-frame OD pilot, not validation of temporally sustained video OD.

Mount-independent canonical MD5 source partitions enforce disjoint TRAIN,
DEV and TEST IDs within this implementation. A label-independent SHA256 order
selects the subset before inference. No fresh-population independence is claimed
for reused Kineticscleaned or COCO2017. Missing/corrupt selected sources fail
the run instead of becoming zero frames or being silently omitted.

Learned runs initialize fresh. Default recipe: width24, Adam2e-4,1000 steps,
rate_weight0.5, paired raw/edited real-codec forward passes, identity STE
backward, worst-teacher positive-part task regret, separate codec/QP duals.
The rate term is a real-bit-calibrated spatial/temporal activity prior. It is
not a codec entropy estimator. OD trains frozen encoder backbone-feature
preservation; this is a surrogate and not differentiable mAP. Both limitations
are recorded in the training manifest. No development best-checkpoint selection.

Report canonical cubic BD-rate for historical comparability and PCHIP as
an explicit sensitivity readout, alongside same-QP rate and quality gaps.
Undefined/non-overlapping curves cannot pass. A screen passes only when both
BD fits are strictly <−10%, BD quality is nonnegative, worst-QP quality gap
is ≥−0.5pp, actual same-QP rate does not increase, and paired bootstrap upper
95% bound is <0 with at least90% finite draws. AR bootstrap resamples sources
jointly across QPs/arms; OD repeats COCO images with new IDs and recomputes AP.

Screen success is not confirmation. `target_confirmed` stays false until an
immutable candidate passes a separately registered confirmation on untouched
sources for both tasks and codecs. These experiments are adapted development
benchmarks and do not claim full MPEG CTC conformance.

## First GPU experiment

Two private free-T4 notebooks: an AR learned run and an OD learned run,
each seed302001 and1000 steps on512 TRAIN sources. Screen AR128 DEV clips
and OD100 DEV images; both codecs and five QPs. AR2000 and OD200 paired
bootstrap draws. These are feasibility/development measurements with limited
precision. All budgets and the immutable Git commit are recorded before submission.

Before subsequent tuning, inspect actual-bit overhead, teacher regret, alpha
maps and chosen-candidate counts. If learned_blend is never selected, the
learned component has not demonstrated value over the explicit controls. A
passing composite selection is not evidence that any particular component
caused the gain; that requires registered ablations on the same sources.
