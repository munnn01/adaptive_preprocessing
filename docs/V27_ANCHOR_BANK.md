# V27 registered anchor reconstruction bank

`adaptive_vcm/anchor_bank.py` appends eight actions after the unchanged V26
34-action prefix. Uniform actions move 25%, 50%, 75% or 100% of each RGB
source-to-anchor reconstruction residual. Core actions multiply that movement
by `1-protection`. Fully protected pixels remain exactly equal to source.
All added actions retain T/H/W, RGB uint8 and separate output storage.

The reference must be the already measured identity roundtrip decoded at the
current codec and QP. The bank rejects absent, wrong-shape or wrong-dtype
references, invalid QP and nonfinite/out-of-range protection. Matching codec/QP
provenance is the caller's responsibility: a decoded array cannot itself prove
which codec settings produced it. Rendering performs no additional encoding.

These are registered pixel directions selected by a learned policy. They do
not add neural pixel weights, change codec parameters or establish task safety.
The unchanged two-teacher guard and actual >=1% coded-byte threshold must be
applied to each measured candidate by the integration pipeline.

`scripts/anchor_probe.py` measures real H264/H265 elementary streams including
headers at QP40/45/50, medium preset, 25fps on three synthetic 16x128x128 clips.
It compares the new directions with V26, reporting source-space MAE and same
geometry separately. Its mild slice is MAE<=10. `best_mild_bytes` can contain
subthreshold changes; `best_guarded_mild_bytes` additionally requires >=1%
actual byte reduction. Neither slice supplies AR/OD teacher evidence or BD-rate.

The focused tests verify literal blend values, color/geometry, protected core,
prefix pixels/order, no mutation, rejection paths, real codec roundtrips and
reporting that excludes subthreshold byte differences from feasible capacity.

The 2026-10-06 local probe measured 756 real roundtrips across 18 codec/QP
points. In the same-geometry MAE<=10 slice, V26 supplied >=1% rate saving at
17/18 points; the new anchor directions supplied 21 feasible actions at 8/18
points. Their best streams totalled 52,617 bytes, compared with 53,266 anchor
bytes and 43,137 best V26 bytes. Adding the directions improved the best V26
stream at 0/18 points and saved zero additional bytes on this synthetic cohort.
The raw-rate improvement hypothesis is unsupported here. Whether the milder
reconstruction directions survive teacher gates more often requires measured
TRAIN and held-out task evaluation. Results and per-stream hashes are saved in
`D:/STUDY/LAB/bao_1/output/adaptive_v27_research/anchor_probe.json`.
