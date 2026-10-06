# V28 adaptive preprocessing: shared OD/AR architecture

V27 remains a separate immutable experiment. V28 changes the encoder preprocessor and its learning procedure; it keeps the standard H.264/H.265 bitstream, original frame coordinates, RGB decoder, and frozen task evaluators.

## What changes

AR source Farneback forward/backward flow supplies a camera-translation-compensated motion-density proxy. Photometric reliability checks and scene boundaries prevent unrelated frames from producing spurious motion regions. A deterministic sum/mean region prior protects moving content together with source-teacher appearance saliency. No crop/zoom is applied. OD bypasses motion for its single-frame benchmark and retains every source-detector box/halo. Unknown foreground falls back to identity.

A small CNN predicts per-pixel strength and a four-expert mixture conditioned on RGB, support, residual motion, codec and QP. AR and OD use the same architecture with separate trained checkpoints and task-specific Gaussian experts. The other experts are block lowpass and a segment-consistent editable-background DC color. Hard protection=1 pixels remain exactly source pixels before encoding. Lossy decoded pixels can still change, so the original strict decoded-teacher guard remains necessary.

Twelve fixed profiles are reference measurements for TRAIN and the audit oracle. The learned renderer is continuous and spatial; it is not the V27 policy ranking those profiles. Inference measures three registered strength-scaled neural renderings (.5,1,1.5). The main arm sees only those three plus unchanged controls. Same-K TRAIN static profiles and all twelve reference profiles are separate component arms. Oracle performance must never be reported as deployable learned performance.

## Learning and measurement

Each TRAIN source visits both codecs and all QPs30/35/40/45/50. Cache source RGB and support once, measure actual elementary-stream bytes (including headers) and decoded frozen-teacher guards for all controls/profiles, then choose a reference target only if it is feasible and strictly saves additional bytes beyond the guarded control winner. Otherwise the target is source identity. No source ground-truth task label, independent evaluator or DEV selection feeds optimization.

Fixed-epoch Adam minimizes RGB imitation of these measured feasible targets; QP>=40 examples receive weight2. Final-LAST is the only exported checkpoint. This is supervised spatial imitation, not a differentiable codec, a codec-gradient estimator, or a proven direct minimizer of BD-rate. A feasible reference target can yield an infeasible learned rendering; inference rechecks actual decoded teachers and byte savings. All-identity training is recorded rather than rebranded as a learned success.

Checkpoint loading validates task/config/profile registry, complete grid size, optimizer steps, TRAIN ID partition/fingerprint, source pixel hashes, finite model weights before and after dtype conversion, and all ten static portfolios. V28 evaluation requires this checkpoint before creating outputs or initializing data/analyzers. Neural outputs must remain finite before image-byte conversion. Evaluation rejects TRAIN/evaluation pixel overlap. Saved hashes support a later artifact audit; hashes alone do not establish independent task quality.

Use independent r2plus1d_18 Top-1 for primary AR transfer, with r3d_18 additionally reported as a teacher/evaluator overlap. OD primary quality is independent ResNet50 COCO mAP@[.50:.95], not Top-1. Report each codec separately, original-source bpp, same-QP quality gaps, BD-rate/PCHIP sensitivity/paired uncertainty, learned extra bytes versus controls, and same-K static comparison. The existing 128 AR DEV clips and 100 OD single images remain development screens; TEST is reserved. No 1280/1280 feasible learned-win or <-10% BD-rate guarantee is made.

## Run

```powershell
& D:/STUDY/AI/envs/ten_env/python.exe -m adaptive_vcm.train_motion --task ar --root <KINETICS_ROOT> --config configs/v28_screen.json --count 128 --epochs 4 --width 12 --seed 302301 --out outputs/v28-ar/train
& D:/STUDY/AI/envs/ten_env/python.exe -m adaptive_vcm.evaluate --task ar --root <KINETICS_ROOT> --config configs/v28_screen.json --checkpoint outputs/v28-ar/train/preprocessor_last.pth --count 128 --split dev --codecs h264 h265 --ablate-learned --bootstrap 2000 --out outputs/v28-ar/eval
```

For OD use `--task od --root <COCO_VAL2017_ROOT> --annotations <instances_val2017.json>`, separate output/checkpoint, and DEV count100. Plans select only TRAIN/DEV partitions by source ID; annotations locate images and provide evaluation labels, not training target boxes. Source detector predictions form the encoder protection.

`scripts/kaggle_runner.py --recipe v28` supports `--task ar`, `od`, or `both`. Joint pilots execute two independent task runs sequentially in one private GPU notebook with distinct checkout/output paths and both dataset mounts. Supply an immutable full commit SHA, `--train-count N --measurements N*10 --epochs 4 --width 12`. Prefer baoancut. Existing V27/qk jobs are never cancelled or edited by this recipe. Prepared payloads are not evidence of executed GPU work.

## Literature scope

Read all six local PDFs and all relevant source/scripts/notebook cells in `D:/STUDY/LAB/AR/code_mocrop`. The local repository is clean at38c60e618da211db3144173b4c322bfdfa959427. Local MoCrop v1 and current arXiv v2 differ; neither establishes VCM bitrate or COCO improvements. See `LITERATURE_V28.md` for publication status, pages, immutable hashes, protocol limitations and primary sources. V28 implements an original geometry-preserving adaptation, not a drop-in reproduction of MoCrop.
