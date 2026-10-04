# V24 AR execution receipt — 2026-10-05

Kaggle acknowledged **version1** for the private notebook
[qktttttttttt/v24-profiles-ar-s302001](https://www.kaggle.com/code/qktttttttttt/v24-profiles-ar-s302001).
A manual status query after submission returned **KernelWorkerStatus.RUNNING**.
This confirms dispatch/running state, not completed metrics or successful
completion of any particular notebook stage. Free NvidiaTeslaT4 was requested;
the notebook asserts CUDA availability at runtime.

Immutable experiment source:
`6e595f0d30648723db6ad29bd13ae7f7097c5fe2`.
Later documentation-only commits do not change this experiment source.

Stages, in execution order:

1. Mount the V23 AR output, resolve its checkpoint by exact SHA256, then
   reevaluate it with the corrected guard and unchanged V23 config. Output:
   `outputs/v24-profiles-ar-s302001/guard_only`.
2. Fresh V24 profile-policy training on 512 TRAIN clips for 1,000 steps,
   seed302001, width24, config `configs/v24_screen.json`. Output: `train`.
3. Paired V24 evaluation on 128 DEV clips, both H.264/H.265, QP30/35/40/45/50,
   2,000 source-video bootstrap draws, and controls/learned component ablations.
   Output: `eval`.

Both evaluations use the same cohort and the corrected guard; a profile name
is recorded for each V24 learned candidate. The baseline checkpoint is used
only for evaluation, never as a V24 training initialization. Task evaluators
and ground-truth labels are excluded from training target generation.

Payload preflight passed notebook schema validation, Bash syntax, full commit
pin, private GPU metadata, V23 dependency and credential checks. Credentials
remain in the external pool and the local CLI subprocess environment. The
notebook contains no credential or pool file.

Local payload/receipt:
`D:\STUDY\LAB\bao_1\output\adaptive_v24_jobs\ar`.

Manual status and collection (PowerShell, one command per line):

```powershell
& D:\STUDY\AI\envs\ten_env\python.exe D:\STUDY\LAB\hope\scripts\kaggle_runner.py status --account qktttttttttt --slug v24-profiles-ar-s302001 --pool D:\STUDY\LAB\pool.json
& D:\STUDY\AI\envs\ten_env\python.exe D:\STUDY\LAB\hope\scripts\kaggle_runner.py download --account qktttttttttt --slug v24-profiles-ar-s302001 --directory D:\STUDY\LAB\bao_1\output\adaptive_v24_results\ar --pool D:\STUDY\LAB\pool.json
```

No V24 real-data improvement or BD-rate target pass is claimed at dispatch.
The [design](V24_AR_DESIGN.md) specifies the comparisons and the
[validation record](V24_VALIDATION.md) reports the completed local checks.
