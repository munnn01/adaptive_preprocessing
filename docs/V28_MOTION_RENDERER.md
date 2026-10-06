# V28 support and renderer component

This original source-RGB adaptation uses Farneback forward/backward consistency,
median translation subtraction, and source-warp photometric checks. It does not
extract codec motion vectors or reproduce MoCrop's entire training pipeline.
Scene segments reset density aggregation. Deterministic grid regions score
normalized residual-density sum (0.6) and mean (0.4). Uniform, inconsistent,
strong-camera, and negligible-residual segments retain semantic support with an
explicit fallback reason. OD requires one frame, bypasses optical flow, and
retains the entire supplied box/core/halo map. `cuts[0]` is always true.

`MotionAwarePreprocessor(width=12, task='ar')` predicts alpha and four expert
mixture maps from source RGB, protection, motion, QP, and codec. The four experts
are mild Gaussian (AR sigma 0.8; OD sigma 2), strong Gaussian (AR sigma 2.5; OD
sigma 8), eight-pixel block lowpass, and background DC. Odd block geometry is
replicate-padded to the block grid and cropped back after filtering. DC averages
editable source pixels within each segment; its encoder lookahead is distinct
from causal alpha smoothing. Full protection and zero strength return exact
identity. The model schema is `adaptive-vcm-motion-v7`; its candidate name is
`learned_motion`. This component does not load checkpoints or select proposals.

`profile_candidates(clip, support, task, qp)` returns twelve registered profiles
in `PROFILE_NAMES` order: each expert at strengths 0.4, 0.75, and 1. QP validates
the measurement condition; reference strengths remain fixed across QPs. The
source shape and exact protection=1 pixels remain unchanged. `validate_support`
validates finite THW support and boolean segment starts for cached inputs.

## Bounded component evidence

39 focused tests passed on CPU with existing ten_env Torch/OpenCV/FFmpeg. The
RED stage had 31 failures because the new modules did not exist; subsequent
RED regressions reproduced equal-histogram cuts, incoherent noisy flow, odd
block-grid distortion, and absent task-specific OD Gaussian strength. Tests
cover camera pans, local motion, cuts, uniform fallback, one-frame OD with
multiple small boxes, odd/small geometry, invalid inputs, exact cores, live
codec/QP conditioning, causal/reset alpha, real network gradients, and actual
elementary-stream bytes.

The following synthetic AR fixture contains six 64x80 frames with independent
noisy background and a fixed colored protected foreground. Measurements compare
source identity with `motion_background_dc_100`, using unchanged StandardCodec
with preset ultrafast, one thread, and RGB decoding. All elementary-stream
headers are included. The foreground color-dominance proxy remains true in all
four decoded outputs; this proxy is not action-recognition accuracy or detection
mAP. Foreground MSE is measured on an interior protected window.

| Codec/QP | Source bytes | DC bytes | Byte saving | Source core MSE | DC core MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H.264 / 45 | 2115 | 692 | 67.28% | 26.924 | 40.500 |
| H.264 / 50 | 960 | 687 | 28.44% | 49.722 | 119.000 |
| H.265 / 45 | 3064 | 2492 | 18.67% | 169.171 | 57.199 |
| H.265 / 50 | 2532 | 2473 | 2.33% | 218.993 | 121.782 |

The equal-weight mean synthetic byte saving is 29.18%; this is component
`dev_score` evidence from a constructed byte/color-proxy fixture, not DEV data,
real task quality, additional savings beyond guarded controls, learned-checkpoint
performance, 1280 feasible learned wins, or a BD-rate result. H.264's increased
decoded foreground MSE demonstrates why exact encoder-side cores still require
actual decoded task guards. H.265 headers dominate this tiny QP50 fixture.

No DEV/TEST inputs, old jobs, V27 pipeline files, or MoCrop training scripts were
used. Real task admission and the strictly-below-minus-ten-percent BD-rate
objective remain the responsibility of the integrated measured experiment.
