# V26 experimental preprocessing and policy

V25 DEV16 exposed two distinct failures: H265/QP50 TRAIN had226/340 trials
saving>=1% but only1 teacher-safe trial and no feasible action; the learned
policy also missed existing opportunities and used4593 more bytes than a
TRAIN codec/QP static top3 on the complete pilot. These findings guide V26.
The independently audited V25 results remain in
[the pilot report](RESULTS_V25_PILOT_2026-10-05.md).

## Fixed intervention before real DEV measurement

The pixel bank retains every V25 action and appends16 luma/DC/exposure actions,
for identity+33 nonidentity actions. None of the new actions resizes or drops
frames. Soft/medium/strong variants cap per-channel edits at4/8/12 atQP50;
four tiny variants cap edits at1. Semantic core pixels remain source-exact
before coding, and motion/cut checks limit temporal exposure correction.
Actual decoded teacher guards remain necessary. See
[bank definitions and bounded synthetic evidence](V26_STABILIZED_BANK.md).

The learned policy is a41-input ridge utility predictor with a codec/QP TRAIN
prior. It predicts **additional actual byte savings beyond the guarded
controls**, using all action measurements. Unsafe or <1%-saving actions have
zero target utility. The known anchor bpp is an input; no evaluator or true
label is an input. Four source-blocked TRAIN folds choose one of24 registered
regularization/prior recipes; the chosen recipe is then fitted once on all
TRAIN records. This is a learned action selector over fixed pixel operators,
not a network learning pixel-filter coefficients. See
[the policy recipe and TRAIN OOF evidence](V26_UTILITY_POLICY.md).

The deployment budget stays atthree proposals. A CV mixture ofzero explicitly
labels winners `trained_prior__`; it is not counted as learned residual
contribution. Evaluation stores compact context/hash, exact proposal order,
prior/residual scores and the mixture for independent replay. The predictor
ranks before any nonidentity measurements are available.

## Measured controls and separation

Each operating point evaluates anchor, adaptive, historical controls,
global-static top3, codec/QP-static top3, learned-guarded, learned-raw,
V25-bank oracle and expanded-bank oracle. The two oracles are audit upper
bounds, never sources for unproposed adaptive candidates. Every scored arm
uses actual decoded streams and the original evaluator roles. Static arms
receive the same three-proposal budget and fixed controls as adaptive.

TRAIN separately measures all historical controls so that the policy's
marginal labels use the actual lowest-byte admissible control. The old and
new action banks are measured on identical source/codec/QP records; reports
count newly feasible **records**, not only more trials from a larger bank.
Cached TRAIN records can be replayed without re-encoding, with source,
bank-code, manifest, context and baseline checks.

The initial V26 pilot is80 TRAIN sources/80 measured source-group records and
DEV16, seed302101. Collection preserves V25's source/group random schedule,
including the first ten codec/QP groups and increased high-QP sampling.
This isolates capacity on the same measured cohort. Full DEV128 V25 remains
a separate running experiment. TEST is untouched and main stays V24.

## Running

```bash
python -m adaptive_vcm.train_utility --root /data/kineticscleaned \
  --count 80 --measurements 80 --seed 302101 --out outputs/v26/train
python -m adaptive_vcm.evaluate --task ar --root /data/kineticscleaned \
  --config configs/v26_screen.json --checkpoint outputs/v26/train/preprocessor_last.pth \
  --count 16 --split dev --bootstrap 0 --ablate-learned --out outputs/v26/eval
```

The ridge fit has no SGD-step count. The private Kaggle runner streams progress
and still clones an immutable full Git SHA, asserts CUDA, and archives partial
or completed output. It receives no credentials inside the notebook.

Local tests and synthetic coded bytes verify implementation, not action
recognition accuracy. The V26 pilot is diagnostic; it cannot establish
BD-rate<-10%, quality retention on a larger cohort, or TEST transfer.
