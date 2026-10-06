# V28 motion and semantic spatial preprocessing

User intent: preserve V27 and develop encoder preprocessing for both AR and OD, informed by the six local AR papers and `AR/code_mocrop`. Target remains BD-rate strictly below -10% separately against H.264 and H.265 with retained task quality. That target is an experimental objective, not a promised result.

## Decision and alternatives

Keep source RGB -> encoder-only preprocessing -> standard codec -> standard RGB decode -> frozen task evaluator. Motion-guided in-place suppression preserves coordinates, all objects and the existing comparison. Direct crop/resize changes the field of view and is unsuitable as the shared OD/AR primary arm. A compressed-domain backbone would require new paired baselines and is a separate study.

MoCrop supplies a density-guided region hypothesis, not evidence of bitrate savings. Its local implementation consumes a precomputed density file, omits extraction/MD/MCS, and scores rectangular sums. Our implementation is original, inspired by the papers; Farneback flow is explicitly an RGB optical-flow proxy, not codec motion vectors.

## Components and contracts

`motion_support.build_motion_support(clip, protection, task)` accepts uint8 RGB THWC plus finite [0,1] semantic/box HW protection and returns protection/motion THW, cuts T, metadata. AR subtracts median global translation from forward/backward-consistent flow, aggregates residual density within scene segments, and adds a deterministic density-guided region. Preserve source-exact semantic cores. No stale motion tube crosses a cut; unreliable/no-motion cases explicitly report semantic fallback. OD requires T=1, bypasses flow, preserves every detector box/halo; unknown foreground is identity at selection.

`motion_learned.MotionAwarePreprocessor(width=12, task)` predicts spatial strength and a four-expert mixture from RGB, protection, motion, QP and codec (seven input channels). Four full-resolution experts: mild Gaussian, strong Gaussian, block lowpass, segment-consistent background DC. DC averages editable source background within each cut-delimited segment, never across a cut. Rendering is convex in [0,1], same THWC geometry, source exact at protection=1. Temporal strength smoothing resets at cuts. Three registered learned proposals scale the predicted strength by .5,1,1.5 (clamped). These are learned pixel maps, not V27's fixed-bank ranking policy.

Twelve fixed reference profiles: each expert with strength .4,.75,1. They create teacher-measured training targets and a same-K=3 TRAIN static comparator. They are not all available to primary inference. Original controls retain the exact old recipe and protection.

`train_motion` collects a complete TRAIN source x codec x QP grid, QPs [30,35,40,45,50], both codecs. Source inputs/support are cached once; actual elementary-stream bytes and decoded teacher guards label all controls and twelve profiles. Only a feasible profile that strictly saves additional bytes beyond guarded controls becomes an imitation target; otherwise target is source identity. Train a fresh spatial renderer by RGB imitation, fixed optimizer/epochs/final-LAST, QP>=40 weighted twice. No codec STE, surrogate entropy, evaluator feedback, ground-truth training labels, or DEV checkpoint selection. Target masks/profiles and hashes are recorded, not claimed to guarantee learned inference feasibility. Static K=3 portfolios use actual marginal byte labels on TRAIN, separate codec/QP groups.

Primary selection measures original controls plus three learned proposals with the existing strict teacher guards and >=1% savings requirement. Static K=3 and full-profile oracle are component audits; oracle trials must not leak into primary selection. AR uses r3d_18/mc3_18 guards and independent r2plus1d_18 scoring; OD uses MobileNet detector guard and independent ResNet50 COCO mAP. Identity/unknown-object fallback stays explicit. All bytes include headers, denominator remains original source pixels, no decoder metadata/side stream.

Checkpoint schema `adaptive-vcm-motion-v7`; load requires matching task, training config, completed grid, TRAIN identity provenance, registered profiles/static orders, measurement hash, finite compatible model state. V22-V27 loading and pipelines remain compatible.

## Evidence and admission

Meaningful TDD covers camera translation, local motion, cuts, static clips, odd dimensions, one-frame OD, multiple/small boxes, exact core retention, gradients, real high-QP H.264/H.265 streams, complete-grid collection, target eligibility, static/primary isolation, and corrupted checkpoints. Independent Superpowers review follows integrated full-suite verification.

TRAIN guides fitting; existing reused DEV screens diagnose capacity, proposal coverage, learned extra bytes and independent task quality separately by task/codec/QP. Held-out TEST is reserved for a later admission gate and is not inspected during implementation. Publishing a research branch is not scientific promotion. No claim of 1280 feasible learned wins or target success without actual evidence.

## Operational constraints

New worktrees/branch and new Kaggle slugs only; leave V27 jobs/checkouts untouched. Python ten_env, existing Torch/OpenCV/FFmpeg; no old MoCrop dependency downgrade or execution of imported training scripts. Save final code in hope and push named V28 branch. Prepare pinned private baoancut Kaggle payloads for both tasks; full-run deployment can follow verified pilots and available compute, without cancelling older jobs.
