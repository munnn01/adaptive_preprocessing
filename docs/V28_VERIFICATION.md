# V28 engineering verification, 2026-10-06

This is a research implementation report, not a result claiming the requested BD-rate or learned-win target.

## Changes and boundaries

V28 replaces the fixed-bank ranking learner with continuous spatial strength/expert maps. Separate AR/OD checkpoints use a common architecture. Source-only motion and semantic support preserve original geometry; H.264/H.265 and frozen analyzers remain unchanged. Complete TRAIN source × codec × QP grids provide measured feasible targets, with identity when no profile beats guarded controls. Three learned outputs form the primary proposal budget; static/oracle profiles are component audits.

All six local AR papers and the local MoCrop source were audited. MoCrop-informed motion-region protection is implemented without cropping; RGB flow is explicitly a proxy, not compressed-stream motion-vector extraction. Details and source identities are in `LITERATURE_V28.md`.

## Local evidence

- Before V28: 222 tests passed at V27 commit81eed77; V27 files and jobs were not changed.
- Renderer: focused39/full261 tests passed, covering cuts, camera translation, unreliable flow, protected cores, odd sizes and finite gradients.
- Trainer: focused19/full280 tests passed. Actual FFmpeg AR and OD collections each covered both codecs × five QPs, followed by neural updates. Deterministic fixture teachers allow nine positive targets and one identity target per task; they are not pretrained-model quality evidence.
- First integrated suite: 302 tests passed in449.31s, before review repairs.
- Independent clean-context Superpowers review found two Important issues and no Critical issues. Actual trained-checkpoint loading and complete component selection succeeded for AR24 trials and OD20 trials. Existing controls won these tiny probes.
- Four regression cases were observed failing before repair: absent/legacy checkpoint with V28 config, finite float64 weights overflowing during float32 loading, and finite float32 weights overflowing during neural arithmetic. V28 now validates a trained motion checkpoint before creating outputs or initializing data/analyzers; loaded weights and neural outputs must remain finite before conversion to image bytes.
- Post-repair focused selection/runner suite: 27 passed in10.53s.
- Final complete post-repair suite: **306 passed in227.20s**, exit0, using `D:/STUDY/AI/envs/ten_env/python.exe -m pytest -q`; log `D:/STUDY/LAB/bao_1/output/v28_full_verified.log`.

Synthetic high-QP background-DC edits reduced elementary-stream bytes on a small AR fixture, but decoded protected-core error worsened for H.264. This supports measuring actual codec feasibility and independent task quality; it cannot establish general savings, extra savings over controls, or BD-rate.

## External experiment protocol

Pin one final verified full Git SHA for three private GPU jobs: baoancut joint AR/OD pilot (TRAIN16 and DEV16 per task); trmnguyn111 full AR (TRAIN128/1280 measurements, DEV128/1280 points); baooo25r full OD (TRAIN100/1000 measurements, DEV100/1000 points). All use two codecs, QPs30/35/40/45/50, four epochs, width12, and seed302301. Existing qk and V27 jobs are left running.

Submission and execution receipts live outside the source tree under `D:/STUDY/LAB/bao_1/output/adaptive_v28_jobs`. Requested T4 allocation is not proof of executed GPU work; inspect runtime device, exit status and archive manifests after completion.

Evaluate independent AR Top-1 and OD COCO mAP, each codec's BD-rate/quality gaps and uncertainty, learned-selected counts, incremental learned bytes over controls, per-QP results and same-K static comparison. Training coverage of all1280 AR groups does not mean1280 safe learned wins. The requested <-10% BD-rate and stronger high-QP learned contribution remain unverified until those real runs complete. No TEST gate or promotion to the research best/main branch was performed.
