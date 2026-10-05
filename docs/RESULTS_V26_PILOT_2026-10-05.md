# V26 AR DEV16: preprocessing capacity and conditional attribution

[qktttttttttt/v26-utility-pilot-ar-s302101](https://www.kaggle.com/code/qktttttttttt/v26-utility-pilot-ar-s302101)
completed successfully on Tesla T4, using executable
`3eff6b767436da95fc4b38474266f063af660f09`.
The immutable 80-source TRAIN / 80-measurement schedule and DEV16 cohort
match the previous V25 pilot. There are 160 operating points, nine scored
arms and no bootstrap or TEST. The independent audit checks 1,440 scored
rows and every guarded minimum. Every old-bank TRAIN stream and DEV anchor
is paired with V25; comparisons below use the same cohort.

| Codec/QP | V25 total byte saving | V26 total byte saving | V26 policy increment beyond controls |
|---|---:|---:|---:|
| H.264/40 | 11.768% | 10.435% | 0.0750 pp |
| H.264/45 | 3.775% | 4.584% | 0.8098 pp |
| H.264/50 | 0.970% | 1.155% | 0.1844 pp |
| H.265/40 | 8.641% | 8.555% | 0.7162 pp |
| H.265/45 | 1.479% | 0.794% | 0 pp |
| H.265/50 | 0.171% | 0.128% | 0.1276 pp |

The learned policy wins 27/160 points, versus V25's 18/160. It saves 11,631
additional bytes over controls (1.3858 pp of anchor bytes), versus 4,907
bytes for V25. Adaptive saves 2,670 bytes versus global static and 825 bytes
versus raw codec/QP static. The preregistered counterfactual using the same
fitted shrunk prior without the conditional residual uses 2,148 more bytes;
this separates conditional attribution from prior shrinkage. Counterfactual
accuracy is available only where its stream matched an already scored arm.

The pure-filter contribution is weaker: only 7 selected actions are pure
filters, compared with 15 in V25; 20 actions resample. None of the 16 new
DC/exposure actions wins primary selection. At QP40/45/50 combined the
increment beyond controls is 1,007 bytes (0.3302 pp), versus V25's 1,898 bytes
(0.6224 pp). Therefore neither uniform high-QP improvement nor increased
learned pure filtering is established.

Expanded TRAIN feasibility increases from 30/80 to 33/80 records: one new
H.264/QP50 record and two H.265/QP45 records. These also provide new marginal
capacity beyond controls. H.265/QP50 still has zero feasible records across
20 measurements. On DEV the expanded bank oracle offers 444 additional
bytes beyond the old-bank oracle, but these unproposed candidates cannot
enter adaptive selection. Bank capacity and policy retrieval remain distinct.

Held-out r2plus1d accuracy at H.264/QP35 is still 6.25 pp below anchor, exactly
as in controls. The pilot cannot confirm BD-rate or the final target.

## Superpowers numerical repair and cached replay

Independent review found that float32 action log-rates mixed with float64
control log-rates can award tiny positive credit to equal-byte actions.
New regression tests first failed, then passed after integer-byte supervision:
subtract actual control/action byte counts before normalization and apply the
unchanged 1% anchor gate to actual bytes. Exact one-byte gains remain counted.
The reusable reader now rejects mismatched codec/QP/geometry, incomplete
controls, inconsistent anchor streams and incorrect winner fields. Legacy
MLP comparison rejects compact V26 context before fitting.

`train_utility --reuse-records` refits the repair on unchanged TRAIN archives,
without teacher inference or codec collection. This pilot contains no false
positive TRAIN coverage from the numerical bug; its repaired selected OOF
result is exactly 1,514 bytes over four records. The registered selected
recipe remains ridge 100 / residual mix 1 / prior pseudo-count 16.

All 160 learned top-three orders, all 160 global-static orders and all 160
group-static orders remain identical after the repair. Pixel-bank code and
guards are unchanged. Every scored arm therefore chooses the same measured
bitstream; its existing accuracy remains applicable. This is a cached decision
equivalence check, not a new GPU run. Original execution manifests and
checkpoint hashes remain intact and are not rewritten to claim a new run.

Evidence is under
`D:/STUDY/LAB/bao_1/output/adaptive_v26_results/pilot_audit`: original archive,
`audit_original.json`, `independent_original.json`, repaired TRAIN outputs
and `integer_replay_equivalence.json`. The original executable modules were
checked from a frozen worktree. Two auditor log1p assertions accept only
1e-12 roundoff across Linux/Windows libm; the observed maximum was 6.94e-18.

The complete repaired code suite passed 160/160 tests in 115.61 seconds;
a separate Superpowers reviewer passed 21 targeted tests and closed the two
P2 findings. These engineering checks do not establish scientific efficacy.
Main remains V24 and independent TEST remains unused.
