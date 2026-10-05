# Fixed hypothesis: bounded luma/DC stabilization adds high-QP capacity

This isolated branch starts at `dd97ff7` and preserves every V25 action. It
tests one fixed hypothesis: bounded changes to codec-visible low-frequency
luma and temporal frame brightness offer extra actual-byte opportunities at
QP45/50 while retaining fine details and exact semantic cores. No geometry,
frame count, codec QP/GOP/header settings, minimum1% saving or teacher guard is
changed. Real AR feasibility, Top-1 and BD-rate remain unmeasured here.

## What actual V25 TRAIN measurements show

The completed V25 pilot contains80 source/group measurements. At H.265/QP50,
226/340 nonidentity trials saved>=1% but only1/340 passed both encoder teachers;
none met both conditions. At H.264/QP50,139/170 saved>=1%,11/170 passed the guard,
and6 met both. This limits the hypothesis: safety is the dominant bottleneck for
the whole bank, so byte-saving capacity by itself cannot establish improvement.

At QP50, `core_residual_soft` saved on average1.277% H.264 /0.885% H.265;
`core_residual_medium` saved3.080% /1.579%. Neither was feasible on any TRAIN
point at that QP. H.265 had no teacher-safe point for either. These are pilot
TRAIN diagnostics, not DEV task results or causal proof that residual removal
is the sole failure mechanism.

Input evidence:
`D:/STUDY/LAB/bao_1/output/adaptive_v25_results/pilot_audit/extracted/outputs/v25-ranking-pilot-ar-s302101/train/measurements.jsonl`.
No TEST sources or labels were inspected.

## API and fixed action order

`adaptive_vcm.stabilized_bank.build_stabilized_bank(clip, protection, qp)`
returns the frozen identity+17 V25 actions followed by16 additional actions.
The original arrays and naming/order are preserved exactly. The additive
function `build_stabilization_actions` returns only the16 new candidates.
`ACTION_NAMES` has34 total entries, and `STABILIZED_ACTION_NAMES` is:

```python
(
    'uniform_dc_soft', 'uniform_dc_medium', 'uniform_dc_strong',
    'core_dc_soft', 'core_dc_medium', 'core_dc_strong',
    'uniform_exposure_soft', 'uniform_exposure_medium', 'uniform_exposure_strong',
    'core_exposure_soft', 'core_exposure_medium', 'core_exposure_strong',
    'uniform_dc_tiny', 'core_dc_tiny', 'uniform_exposure_tiny', 'core_exposure_tiny',
)
```

Every new action has original uint8RGB `[T,H,W,3]` geometry. The protection
input is encoder semantic `[H,W]` in `[0,1]`. Core actions multiply shifts by
`1-protection` and retain protection==1 source-exact before encoding. Uniform
actions may change protected areas and remain subject to the same guard.
Neither function receives a label, teacher output or held-out evaluator result.

Define `q=clip((QP-30)/20,0,1)` and `g=.35+.65q`. RGB luma uses weights
.299/.587/.114. DC actions move pixels by
`g*a*(frame_mean_luma - Gaussian_sigma4_luma)` with `a=.20/.35/.50`
for soft/medium/strong. Adding the same scalar to each RGB channel changes
luma while preserving channel differences except at range clipping. This
damps a low-frequency field, keeping the high-frequency source residual;
it does not flatten source pixels to a frame mean.

Exposure actions move by `g*a*(window_median_frame_luma-current_frame_luma)`
with `a=.35/.65/1.0`. The median window is at most five source frames, two
on either side, entirely within a scene. This is an offline encoder operator.
Cuts are detected by mean absolute source RGB change>=32. A source-difference
motion gate removes median global brightness changes first, then falls to
zero at local12-level movement. The window cannot cross cuts and no recursive
filtered state is retained. DC actions are frame-local; exposure actions are
bounded source temporal corrections.

Before rounding, shifts are capped at `[4,8,12]*(.5+.5q)` levels per channel.
`max_pixel_change(qp,strength)` includes the integer rounding bound:

| QP | Soft | Medium | Strong |
|---:|---:|---:|---:|
| 30 | 2 | 4 | 6 |
| 40 | 3 | 6 | 9 |
| 45 | 4 | 7 | 11 |
| 50 | 4 | 8 | 12 |

Four tiny backoff actions were appended after the original12 before any real
DEV feedback: DC amount.05 and exposure amount.15, both scaled by the same `g`.
Their cap is `.5+.5q` before rounding, and at most1 level per channel after
rounding for every QP. Source cores and scene-cut/motion logic are unchanged.
The old12 action indexes and outputs remain unchanged; these extra four tests
the near-identity zone without relaxing the>=1% saving gate or forcing edits.

All byte accounting must use original `T*H*W`, including elementary stream
headers. The existing shared encoder-teacher guard and>=1% actual-byte saving
remain authoritative. Added options are not automatically selected, and a
bank oracle must be distinguished from equal-proposal-budget learned ranking.

## Local verification and synthetic scope

Executed with `D:/STUDY/AI/envs/ten_env/python.exe`:

```powershell
python -m pytest tests/test_stabilized_bank.py -q
```

**30 passed**, no skips (initial12-action version:24 passed). Checks cover per-channel movement at QP0/30/40/45/50/51,
exact protected pixels, unchanged V25 prefix, deterministic independent arrays,
scene-cut isolation, chroma/edge retention for uniform exposure without clipping,
motion gating after exposure normalization, invalid inputs and actual codec
geometry/rate accounting for both codecs at QP40/45/50.

`outputs/stabilized_probe.py` compares original17 V25 actions and new12 actions
on the two original16x128x128 static texture cases plus a16x128x128 low-frequency
exposure-flicker case. Actual H.264/H.265 medium at25fps and QP40/45/50 are used.
Both banks use the same source-edit RGB MAE<=10 diagnostic slice. This is a pixel
edit slice, not an action-teacher guard. Results are in
`outputs/stabilized_probe.json`, with source/stream hashes and all action bytes.
Teacher feasibility, Top-1, BD-rate and real DEV scores are explicitly null.

## Primary literature and inference limits

Xie et al., [Enhanced Motion Compensated Temporal Filter for VVenC](https://ieeexplore.ieee.org/document/10572000/),
IEEE Transactions on Circuits and Systems for Video Technology34(11),2024,
DOI10.1109/TCSVT.2024.3393721, describes temporal source filtering to reduce
prediction residuals. Its publication and abstract were checked against the
[authors' university record](https://scholars.cityu.edu.hk/en/publications/enhanced-motion-compensated-temporal-filter-for-vvenc/).
That work uses motion compensation and VVenC; the bounded median-luma correction
here is a different engineering hypothesis, and its reported gains are not
transferred to AR or H.264/H.265.

Lu et al., [Preprocessing Enhanced Image Compression for Machine Vision](https://arxiv.org/abs/2206.05650)
supports QP-adaptive semantic source preprocessing before a conventional codec.
It does not establish that these fixed luma/DC actions preserve action accuracy.
The semantic-core and actual-teacher guard requirements therefore remain intact.

## Executed synthetic result,2026-10-05

All18 source/codec/QP probe points completed. On the explicit low-frequency
exposure-flicker source, best actual-byte changes in the same MAE<=10 slice are:

| Codec/QP | Frozen V25 bank | New stabilization only | Old / new >=1% mild actions |
|---|---:|---:|---:|
| H.264/40 | -14.84% | -25.45% | 10 / 9 |
| H.264/45 | -2.55% | -5.78% | 4 / 6 |
| H.264/50 | -1.27% | -1.48% | 1 / 2 |
| H.265/40 | -0.60% | -2.98% | 0 / 5 |
| H.265/45 | -0.66% | -1.72% | 0 / 2 |
| H.265/50 | -1.41% | -1.97% | 2 / 4 |

The two original static-texture sources provide negative evidence: new actions
did not beat the existing bank's best mild option at any codec/QP. Most new
changes saved less than1%. Only H.264 texture1/QP45 and texture2/QP50 had new
actions saving>=1% (two each). H.265/QP50 texture1 remains header dominated,
with no>=1% saving from either bank.

This supports a narrow engineering conclusion: changing source low-frequency
brightness adds a rate lever when temporal brightness variation is present;
it offers no broad replacement for detail filtering. Added QP50 gains were
small even on the constructed flicker example. Real encoder-teacher guard
coverage must be measured before claiming AR capacity or policy improvement.
The original12 action definitions were not retuned after these results, and no
TEST evaluation was run. The table describes that original12-action comparison.

The initial12 were kept unchanged when the parent requested four tiny backoff
actions. `outputs/tiny_probe.py` measures only these four on the exact same
synthetic sources and codec/QP grid, verifies source hashes and anchor byte
counts against the original probe, and writes
`outputs/stabilized_probe_extended.json`. Tiny results are engineering evidence
only, not a task-safety claim or DEV feedback used to tune the coefficients.

All72 tiny codec/action trials completed. At QP45/50 **none** of the four tiny
actions saved>=1% on any of the three synthetic sources with either codec.
At low-frequency-flicker H.264/QP40,3/4 tiny actions saved>=1%; the best saved
13.19%, with per-channel changes<=1. Across18 source/codec/QP points, this was
the only group with a tiny action meeting the byte threshold. The near-identity
ladder is therefore implemented for the real-teacher search, but synthetic
QP45/50 results do not provide evidence of increased qualifying byte capacity.
