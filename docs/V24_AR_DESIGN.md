# V24 — consistent AR guards and feasible preprocessing profiles

V23 AR learned preprocessing was selected at 0/1,280 operating points;
1,275 learned streams exactly matched the raw anchor. During the last 100
training steps, 99 center streams matched the anchor and 99 SPSA gradients
were zero. At QP50, the complete pipeline saved only 0.376% / 0.043% bytes
against H.264 / H.265 at the same QP. These are the measured problems that
V24 addresses. V24 has no completed task result yet.

## One codec-relative AR guard

`selection.ar_guard` always admits the anchor itself. The relative distortion
remains KL(source || candidate) minus KL(source || anchor), with the existing
0.1 slack. With `ar_require_anchor_decision=true`, a candidate must retain
each teacher's anchor class. A confident source class is protected only when
the codec retained that class. A codec-induced source/anchor disagreement
therefore cannot create two contradictory class requirements.

The V23 audit found 504 identical-anchor learned streams rejected by its
decision guard. Training used a byte-identity shortcut that returned true,
whereas evaluation applied the contradictory guard. AR training now calls
the same guard for every probe. Both train and eval reuse anchor predictions
for identical streams, with no special AR decision override. The OD-specific
unknown-foreground fallback is retained.

## Finite preprocessing actions at high QP

`ProfilePreprocessor` learns a content/QP/codec-conditioned choice among nine
actions: identity, two mild detail filters, two coarse filters, two block/DC
filters, temporal coarse and temporal DC. It uses the V23 semantic protection,
motion gating and cut reset. Fully protected pixels remain source-exact before
coding. Strong spatial actions use absolute blends of 0.60–0.90 rather than
an unconstrained trainable strength that can fall below uint8 rounding.

The fifth spectral expert is block low-pass below QP45 and frame DC at QP45/50.
Temporal actions reuse the previous source low-pass only outside protected,
moving or cut regions. The policy sees the complete source clip; this is an
offline encoder policy, not a claim of causal streaming.

This is learned **profile selection**, not learned filter coefficients. The
classifier can still choose identity for every input; V24 logs both predicted
and oracle profile frequencies so that such a failure remains visible.

## TRAIN objective

For each TRAIN source and sampled codec/QP, encode identity and all eight
nonidentity profiles with the real codec, then apply the frozen encoder
teachers and the same guard used by evaluation. The label is the smallest
actual-byte feasible profile, with at least 1% savings; otherwise identity.
Ties retain the earlier profile. Cross-entropy trains the nine-way policy,
weighted by `1 + min(1, 2 * fractional_byte_savings)` for nonidentity targets
and 0.25 for identity targets. This reflects asymmetric encoder costs: missing
a feasible edit loses savings, while an infeasible predicted edit is rejected
by the existing guard. Identity labels remain in training; the weighting is
preregistered and does not weaken evaluation or force a nonidentity selection.

There is no SPSA penalty gradient in V24, no pixel STE, and no reward for
task-invalid savings. Ground-truth classes and held-out evaluators never
enter profile-label generation. A fresh width24 model trains for 1,000 steps
on 512 TRAIN sources, seed302001, Adam 1e-3, gradient norm cap1. QP sampling
weights remain 1/1/2/3/3 for QP30/35/40/45/50 per codec. Final-LAST is used;
there is no DEV checkpoint selection or early stopping on DEV metrics.

## Paired experiment

The V24 Kaggle recipe first evaluates the exact completed V23 AR checkpoint
with the corrected guard and its unchanged V23 configuration. Checkpoint
SHA256 must equal
`9c1dae49253b1d54d0caec0032e5075e59953d882e83b013c1534a985d1e0839`.
It is mounted from the existing private V23 Kaggle output; a hash mismatch
aborts the run. This `guard_only` result isolates guard correction from the
new policy. The old checkpoint is never used to initialize V24 training.

Then train the new policy and evaluate `configs/v24_screen.json` on the same
128 DEV clips, 16 frames at128x128, stride2, H.264/H.265 medium, QP30/35/40/45/50,
and 2,000 paired video bootstrap draws. Controls, learned_raw and learned_guarded
are reported separately, including rate/Top-1 at QP40/45/50. The profile name
is saved in each learned candidate's selection audit. Both stages use the
corrected guard and identical fixed analytic controls.

Report three comparisons: original V23 versus guard-only; guard-only versus
V24; and within-V24 controls versus learned. Do not attribute a guard-only
gain to the learned policy. Gate values remain strict canonical/PCHIP
BD-rate < -10%, worst same-QP task gap >= -0.5pp, nonincreasing rate, nonnegative
BD-quality, and finite paired uncertainty. r3d_18 is an encoder teacher;
r2plus1d_18 is the held-out evaluator. This reused DEV cohort is not independent
confirmation. OD keeps its V23 preprocessing and is outside this AR experiment.

## Validation and limits

Regression tests cover source/anchor disagreement at high and low confidence,
relative KL rejection, exact protected pixels, identity, scene cuts, infeasible
profile rejection, checkpoint loading and real-codec train/eval integration.
The latter deliberately uses a confident source class different from the
compressed anchor class. Synthetic actual-byte tests cover both codecs at
QP40/45/50. Synthetic savings are not Top-1 or BD-rate evidence.

See [validation record](V24_VALIDATION.md) for executed checks. Real AR gains
remain an empirical question until the paired Kaggle experiment completes.
