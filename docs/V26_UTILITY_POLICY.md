# Source-blocked TRAIN utility policy

The fixed hypothesis is that compact, strongly regularized action utility with a
codec/QP prior generalizes better than V25's 1,640-input MLP while preserving the
same three-proposal budget. No DEV or TEST measurement selects its recipe.

The context has41 entries: the16 source statistics,24 scalar teacher features
from V25, and log1p of the actual anchor bpp. Category-indexed probabilities are
omitted. The two registered teachers still provide source/anchor confidence,
margin, entropy, KL, Jensen-Shannon distance and class agreement. Context contains
no true label or held-out evaluator output; anchor bpp uses all coded bytes and
the original source pixel denominator.

Each nonidentity action is supervised directly by real guarded byte utility.
An unsafe action, or one missing the existing1% anchor saving gate, has zero
utility. With `baseline_log_rate` supplied, utility is the additional byte saving
over the already guarded control stream. Legacy replay without that argument
measures saving over the anchor and must retain that narrower attribution.

A TRAIN codec/QP mean shrinks toward the global mean with pseudo-count4 or16.
A standardized linear ridge residual augments that prior with mixture0,.25,.5
or1 and ridge coefficient10,100 or1000. Four folds hold every codec/QP record of
a source together. Each fold fits its own priors, feature scaler and residual
coefficients. The fixed24-recipe grid chooses highest held-out TRAIN coded bytes
saved at exactly three proposals; ties prefer prior-only and stronger
regularization. Full TRAIN records then fit the selected recipe once. The OOF
score used for recipe selection is not an independent confirmation score.

Checkpoint schema `adaptive-vcm-utility-v5` is incompatible with legacy MLP
checkpoints. It saves exact action order, feature normalization, prior means,
residual coefficients, mixture, utility target scope and global static order.
`proposal_details` reports prior and residual scores separately. A zero mixture
reports `prior_only=true`, so those proposals cannot be credited to learned
residual prediction. Global static and raw codec/QP static comparators both use
exactly three actions.

Replay entry point:

```powershell
python -m adaptive_vcm.utility_ranking --records EXISTING_TRAIN_DIRECTORY --out EMPTY_OUTPUT_DIRECTORY
```

The reusable reader verifies the measurements manifest hash, original source
fingerprints, source TRAIN partitions, action order, source geometry/anchor bpp,
legacy context schema and exact bank source hash. It does not re-encode cached
streams. `scripts.audit_utility_cv` additionally reinitializes and fits the fixed
V25 MLP recipe inside each source-blocked TRAIN fold for comparison.

On the existing80-source V25 pilot TRAIN cache, selected ridge1000/mix.5/prior16
retrieves8712 bytes OOF, versus8702 bytes for global static and7254 bytes for raw
codec/QP static. Feasible coverage is12/80,13/80 and14/80 respectively; bank oracle
coverage is30/80. Reinitializing the fixed V25 MLP inside the same TRAIN folds
retrieves7093 bytes with19/80 feasible records. Compact utility retrieves more
bytes than that MLP but covers fewer records. The10-byte gain over global static
is weak evidence, and coverage does not improve. These are anchor-referenced TRAIN OOF measurements, not measured
DEV gains or savings attributable beyond controls. The pilot H.265/QP50 bank has
zero feasible actions in20 measured source/group records, which no ranking change
can repair. A fresh measured cohort and expanded feasible bank remain necessary.
