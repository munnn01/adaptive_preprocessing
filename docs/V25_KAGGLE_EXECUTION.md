# V25 immutable Kaggle submissions

Executable commit: `0bfcead429dcde08d0e67cb0589216aa7a357131`, published on
[v25-ranking](https://github.com/munnn01/adaptive_preprocessing/tree/v25-ranking).
The main branch remains V24; V25 is experimental until measured transfer is
confirmed. Local requested checkout: `D:/STUDY/LAB/hope`, branch `v25-local`.

| Run | TRAIN sources / measured source-group records | Replay steps | DEV | Bootstrap | Handle |
|---|---:|---:|---:|---:|---|
| pilot | 80 /80 | 1500 | 16 | 0 | [qktttttttttt/v25-ranking-pilot-ar-s302101](https://www.kaggle.com/code/qktttttttttt/v25-ranking-pilot-ar-s302101) |
| full | 512 /512 | 1500 | 128 | 2000 | [baoancut/v25-ranking-ar-s302101](https://www.kaggle.com/code/baoancut/v25-ranking-ar-s302101) |

Both submissions were accepted as kernel version1. Last API check on2026-10-05:
both **RUNNING**. This verifies scheduler status, not successful completion,
CUDA runtime, task accuracy or byte savings. Logs/output were not available at
the first download attempt. No V25 task result is asserted here.

Seed302101, batch32, width64, all QP30/35/40/45/50 and both standard codecs,
unchanged source16x128x128 protocol. Private notebooks request free NvidiaT4,
GPU and internet. Each notebook clones the specified repository, checks out the
full commit and asserts runtime CUDA. No pool credentials are copied into the
notebook, repository or downloaded output. Credentials remain in the environment
of the local Kaggle subprocess, from external `D:/STUDY/LAB/pool.json`.

Payloads and submission receipts are preserved outside Git:

- `D:/STUDY/LAB/bao_1/output/adaptive_v25_jobs/pilot`
- `D:/STUDY/LAB/bao_1/output/adaptive_v25_jobs/full`

Results will be downloaded under
`D:/STUDY/LAB/bao_1/output/adaptive_v25_results/pilot` and `full`.
Use `scripts/kaggle_runner.py status/download` with the corresponding account
and slug. After extracting the job archive, run
`python scripts/audit_v25.py --run PATH_TO_outputs/JOB_SLUG --out AUDIT_JSON`.
The audit verifies configuration/checkpoint/tensor fingerprints, TRAIN/DEV
source and pixel separation, actual minimum eligible byte selection in every
arm, and marginal policy contribution against the equal-budget static control.

The pilot and full runs are independent fresh TRAIN fits; the pilot checkpoint
does not initialize the full run. TEST is reserved for a frozen promising
candidate and is not part of either notebook. A future confirmed result must
cite the actual completed output and independent TEST comparison.
