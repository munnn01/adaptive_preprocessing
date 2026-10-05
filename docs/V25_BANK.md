# V25 hypothesis: more mild preprocessing opportunities at high QP

This branch tests one hypothesis: preserving source low-frequency appearance
while suppressing fine detail, chroma noise or stationary temporal noise creates
more encoder-teacher-feasible byte-saving choices than V24's coarse/DC actions.
The shared AR guard, standard codec, task networks and quality gates are not
changed. Real TRAIN feasibility, DEV Top-1 and BD-rate are currently **null**;
synthetic encoder results cannot establish this hypothesis on action videos.

The V24 audit found feasible oracle edits at only 12/279 TRAIN QP50 points
(4.30%), with mean oracle byte saving 0.317%. At DEV QP50, 112/128 H.264 and
124/128 H.265 points fell back to identity. These observations motivate broader
capacity at modest source distortion; forcing edits would not resolve them.

## Fixed bank

`adaptive_vcm.task_bank.build_task_bank(clip, protection, qp)` accepts source
uint8 RGB `[T,H,W,3]`, encoder semantic protection `[H,W]` in `[0,1]` and integer
QP0–51. It returns 18 `Candidate` objects in `ACTION_NAMES` order. Identity is
first. Rendering is deterministic and has no label, evaluator or decoded
prediction input. Different actions can coincide on a particular source, such
as a static frame or full semantic protection; this is not forced away.

All QP controls use `q = clip((QP-30)/20, 0, 1)`. Definitions are fixed before
the next DEV experiment. Uniform actions may edit semantic regions; they must
pass the same real-codec teacher guard. Core actions retain pixels with
protection exactly1 before coding and use a feathered blend elsewhere.

| Actions | Definition |
|---|---|
| identity | Exact source copy |
| resample120/104/88_denoise | Area resize at .9375/.8125/.6875 after mild spatial and stationary temporal denoising |
| resample104/88_detailq | Area resize at .8125/.6875 after soft/medium detail shrinkage |
| uniform_detail_soft/medium | Gaussian residual soft dead zone: sigma .8/1.2; thresholds 2+6q / 4+12q RGB levels |
| uniform_temporal_soft/medium | Detail shrinkage plus motion-gated previous-source blend .20+.15q / .35+.20q |
| uniform_chroma_soft | Keep luma; blur chroma at sigma1+q and blend .30+.30q |
| core_spatial_soft/medium | Gaussian blend with sigma .8/1.2 and strength .15+.20q / .30+.25q, protected core exact |
| core_temporal_soft/medium | Protected variants of temporal detail shrinkage |
| core_residual_soft/medium | Protected variants of detail shrinkage |
| core_joint | Protected medium spatial plus stationary temporal blend |

Resampling dimensions are even and at least2. The five resampling combinations
change spatial dimensions; all others retain source dimensions. Every action
retains frame count. The rate denominator must remain original `T*H*W`, and the
entire elementary-stream bytes, including headers, must be counted. Output
resolutions120/104/88 are literal for128 input, but ratios apply to other source
sizes. They differ from V24 controls' area112/area96 and include a new denoising
or residual transformation.

Residual shrinkage retains the local average plus soft-thresholded residual;
each channel moves toward the average by at most its threshold. Larger edges
retain a residual rather than being erased wholesale. This is pixel
preprocessing, not changing codec quantization, QP, GOP or standard syntax.

Temporal reuse references the previous **source** frame, never recursive
filtered state. Per-pixel source RGB change gates reuse down to zero at12
levels. A frame-wide mean change of32 levels resets reuse at a scene cut.
Frame0 has no reference. Motion and semantic protection remain source-based.

## Evidence and limits

`tests/test_task_bank.py` checks input immutability, exact identity/core,
determinism, scene-cut reset, finite valid protection/QP, residual movement
bounds, distinct output from existing analytic controls on a generic fixture,
frame/dimension validity, six H.264/H.265 QP40/45/50 real-codec opportunities
and original-pixel normalization. Executed locally: **19 passed**, no skips.

The actual-byte test uses synthetic8-frame128x192 two-pixel random texture
with a static protected rectangle. A smaller64x96 version is header dominated
at H.265 QP50 and showed no >=1% bank saving; rendering alone cannot overcome
an elementary-stream header floor. That limitation is retained explicitly.

`scripts/task_bank_probe.py` compares all18 new actions with nine V24 actions
on two synthetic16x128x128 cases at each codec/QP. It saves actual bytes,
source edit MAE and decoded MAE. A source-edit MAE<=10 slice is a diagnostic
of modest editing only; it is never used by selection and is not teacher
feasibility, Top-1 or BD-rate. See `outputs/task_bank_synthetic_probe.json`.

No TRAIN labels or real video cohort were evaluated locally. The new bank
must be tested by the parent's real-codec TRAIN recipe, with unchanged
encoder teachers and guards, before any increase in feasible actions can be
claimed. DEV components must distinguish a learned policy from a full-bank
oracle; higher count of renamed controls is not evidence of learning.

## Primary literature consulted

Lu et al., [Preprocessing Enhanced Image Compression for Machine Vision](https://arxiv.org/abs/2206.05650)
supports placing QP-adaptive semantic suppression before a conventional
non-differentiable codec. The present fixed residual/chroma/temporal bank is an
engineering hypothesis, not a reproduction of their neural model or gains.

The LAB author paper *Video Coding for Machines using Object Analysis and
Standard Video Codecs* describes simplifying background and retargeting before
a standard codec. Its local text was consulted at
`D:/STUDY/LAB/bao_1/papers/_text/Video_Coding_for_Machines_using_Object_Analysis_and_Standard_Video_Codecs.txt`.
Its reported VVC results concern object analysis; they are not AR evidence.

The LAB paper *Gaussian Filtering to Improve Object Detection Accuracy in
Coded Video* was also read locally. Its combined encoder/decoder filtering
experiment does not establish benefits for an encoder-only AR preprocessor;
no decoder post-filter or reported gain from that paper is imported here.

## Executed synthetic comparison

The probe finished successfully on2026-10-05. For the two-pixel block-texture
source, best actual byte changes among source-edit-MAE<=10 actions were:

| Codec/QP | New bank | V24 bank | New / V24 >=1% saving mild actions |
|---|---:|---:|---:|
| H.264/40 | -19.02% | -6.48% | 10 / 1 |
| H.264/45 | -29.11% | -7.11% | 9 / 1 |
| H.264/50 | -34.08% | -14.00% | 8 / 1 |
| H.265/40 | -17.69% | -6.15% | 10 / 1 |
| H.265/45 | -20.78% | -6.49% | 9 / 1 |
| H.265/50 | -11.82% | -4.04% | 8 / 1 |

For one-pixel texture H.265/QP50, neither bank had an action saving>=1%.
Across both synthetic sources and all six codec/QP settings,95/204 new
nonidentity actions and10/96 V24 nonidentity actions saved>=1% within this
pixel-edit slice. This is a capacity diagnostic with unequal bank sizes;
it establishes neither task feasibility nor a fair policy-compute advantage.
The paired experiment must include equal-candidate-budget comparisons.
