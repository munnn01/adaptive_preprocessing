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

After the user invoked Superpowers, a fresh complete run including the auditor
passed **150/150 tests in130.16s**, no skips. Its log is
`D:/STUDY/LAB/bao_1/output/adaptive_v26_pytest/superpowers.log`.
This applies the plugin's verification-before-completion workflow. A separate
reviewer checks the V25-to-V26 diff against the documented requirements;
review findings and Kaggle results remain independent of a passing test suite.

`git diff --check` and Python compilation also passed. A missing pytest parent
directory was created before rerunning; a loader check caught and repaired
the legacy assumption that every checkpoint must have SGD steps. V26 now
requires its genuine TRAIN-CV fit provenance and measurement budget instead.

After the numerical repair, a fresh full suite passed **160/160 tests in
115.61s**, no skips, with log
`D:/STUDY/LAB/bao_1/output/adaptive_v26_pytest/integer_repair_full.log`.
Nine new regression cases reject equal-byte phantom credit, seven inconsistent
TRAIN metadata cases and invalid compact context in the legacy MLP audit.
Cached TRAIN refitting is verified without teacher/codec collection. A
separate reviewer passed 21 targeted tests and closed both P2 findings.

Completed [V25 DEV128](RESULTS_V25_FULL_2026-10-05.md) and
[V26 DEV16](RESULTS_V26_PILOT_2026-10-05.md) are now audited separately.
Neither reaches the final target; no TEST result or confirmed BD-rate target
is asserted by local testing.
