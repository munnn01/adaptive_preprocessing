# V26 immutable pilot submission

Executable pin: `3eff6b767436da95fc4b38474266f063af660f09`, published on
[v26-utility](https://github.com/munnn01/adaptive_preprocessing/tree/v26-utility).
The requested checkout `D:/STUDY/LAB/hope` holds the same code on `v26-local`.
Main remains V24 pending independently confirmed transfer.

[qktttttttttt/v26-utility-pilot-ar-s302101](https://www.kaggle.com/code/qktttttttttt/v26-utility-pilot-ar-s302101)
was accepted as private kernel version1 at12:38 ICT on2026-10-05. The immediate
API check reported **RUNNING** at submission. The job has since completed
successfully on Tesla T4; its archived output and independent audit are
described in [the completed pilot report](RESULTS_V26_PILOT_2026-10-05.md).

The pilot uses80 TRAIN sources/80 measured source-codec-QP groups, seed302101,
DEV16 and zero bootstrap. It fits the registered source-blocked TRAIN CV
utility policy, with no fabricated SGD-step count. The bank has33 nonidentity
actions; learned/global-static/group-static budgets remain three each.
Teacher/evaluator roles, strict guard, minimum1% byte saving,16x128x128 source
recipe, QP30/35/40/45/50 and codec settings are unchanged.

The new notebook streams subprocess progress, clones the exact full SHA,
asserts CUDA and archives output on exit. Credentials are only read by the
local Kaggle subprocess from external `D:/STUDY/LAB/pool.json`; none is copied
into the notebook, checkout or output.

Payload/submission receipts: `D:/STUDY/LAB/bao_1/output/adaptive_v26_jobs/pilot`.
Result/audit directory: `D:/STUDY/LAB/bao_1/output/adaptive_v26_results/pilot_audit`.
Independent auditing pairs all old-bank TRAIN streams and DEV anchors/controls
with the completed V25 pilot before assigning gains to the extra filters.

Before inspecting V26 DEV output, an additional **rate-only** counterfactual
was registered: retain this fitted model's shrunk TRAIN prior and remove its
conditional residual, propose its top3, then apply the same actual-byte guard
using the complete bank audit. This distinguishes conditional learned gains
from shrinkage and expanded candidate capacity. Its quality is reported only
where the exact stream was already scored; otherwise it remains unscored.
It cannot change adaptive selection or checkpoint fitting.

The [V25 full DEV128 job](RESULTS_V25_FULL_2026-10-05.md) has also completed
and been audited. Neither job evaluates TEST. A small pilot, local tests or
TRAIN OOF improvements cannot establish the BD-rate<-10% target.
