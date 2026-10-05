# V26 implementation checks,2026-10-05

The complete repository test suite passed **146 tests in126.38s**, with no
skips, using `D:/STUDY/AI/envs/ten_env/python.exe`. Its log is preserved at
`D:/STUDY/LAB/bao_1/output/adaptive_v26_pytest/full.log`.

These cover legacy AR/OD loading and guards, V25 ranking, all30 bounded-bank
tests, source-blocked utility CV, class-identity-free context, prior-only
attribution, marginal-controls labels, cached replay checks, private immutable
Kaggle payloads and actual H264/H265 QP50 selection. The new real-codec selector
check verifies exact three proposals, actual anchor bpp, persisted proposal
context/hash, group-static membership and old/new bank oracle byte ordering.
Teacher fixtures in those engineering tests do not constitute AR accuracy.

An additional four independent-auditor tests validate guarded minimum bytes,
paired old/new feasible-record counts, fold coherence and independent replay
of all24 CV recipes. `scripts/audit_v26.py` verifies actual completed run
artifacts, every scored component arm, checkpoint proposals, source/pixel
separation, TRAIN controls baselines and quality/BD/bootstrap recomputation.

`git diff --check` and Python compilation also passed. A missing pytest parent
directory was created before rerunning; a loader check caught and repaired
the legacy assumption that every checkpoint must have SGD steps. V26 now
requires its genuine TRAIN-CV fit provenance and measurement budget instead.

V25's completed DEV16 remains the only real AR result when these code checks
were recorded. V26 synthetic and TRAIN OOF evidence is explicitly bounded in
the bank/policy documents. No TEST result or confirmed BD-rate target is
asserted by local testing.
