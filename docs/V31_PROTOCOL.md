# V31 action oracle and adaptive preprocessing

V31 implements the approved action expansion alongside the historical versions. It tests whether task-preserving spatial/temporal actions have useful headroom before spending GPU time on a learned selector. More learned selections alone are not evidence of a better compressor.

The deployment path is source RGB → runtime features and teacher support → anchor encode → selector proposes at most three available actions → RGB preprocessor → H.264/H.265 encode and decode → common teacher guard → smallest admissible packet → independent task evaluation offline. The RGB preprocessor sits before the standard encoder. Resolution reduction, frame subsampling with timestamp restoration, and the existing temporal expert are actions; GOP settings belong to a separate encoder audit.

The oracle is offline, between full-bank measurement/CAL calibration and selector training. It evaluates all teacher-feasible actions on TUNE and gives an empirical upper bound for the frozen bank. It is never called by deployment inference. Backpropagation trains the selector's safety/rate heads on FIT observations; it does not pass through the codecs, COCO evaluation or discrete actions.

| Item | AR | OD |
|---|---|---|
| FIT / CAL / TUNE / DEV | 96 / 32 / 128 / 128 | 75 / 25 / 100 / 100 |
| Canonical analyzer source | 16 centered frames, stride2, 128 square | one 320 letterbox image |
| Primary evaluator | r2plus1d_18 Top1 | ResNet50 COCO mAP@.50:.95 |
| Teachers | r3d_18 and mc3_18 | MobileNet detector |
| Primary rate | sum transmitted bits / sum source duration | sum transmitted bits / sum original image pixels |
| Fixed baseline | always area112; also report area96 | one spatial size frozen on FIT; report every size |

Both tasks use H.264/H.265, medium preset, QP30/35/40/45, 2000 paired source bootstrap draws, and eight selector epochs. Each packet includes a 32-byte recipe. Report elementary and total bytes separately; the guard and comparisons use total bytes. Source IDs and pixel hashes must be disjoint. Unknown or contradictory video timing cannot qualify a rate claim. TEST is not opened; previously observed DEV is exploratory.

Arms A/B/C share a policy-independent B-union measurement grid. A projects the historical spatial bank; B adds the approved temporal/spatial actions under the strict guard; C uses the same bank with CAL-only calibrated teacher feasibility. Their policies, source membership and gates have separate identities. No new filter-profile sweep is introduced.

An empirical oracle gate requires both codecs: PCHIP BD-rate below −15% versus the anchor, improvement over expanded static K3 and fixed spatial, no corresponding-QP primary quality loss exceeding1 percentage point, and complete counts/grids/provenance. Constant quality, insufficient overlap and incomplete measurements do not qualify. Rejected gates stop before optimizer construction. Static K3 and the OD fixed size are frozen on FIT, not TUNE/DEV.

Final learned comparisons use anchor+K3 probes under the same guard. An adaptive-superiority claim additionally needs the upper paired95% CI below zero against expanded static K3 and fixed spatial, at least90% finite draws, and the quality constraint. Anchor comparisons are the third primary CI. Controls, legacy portfolios, fixed-size and unrestricted-union curves are point diagnostics; they do not inherit primary CIs. OD bootstrap recomputes the exact global COCO score ordering, including repeated image occurrences; it does not average per-image AP.

## Local stages

Use `python -m adaptive_vcm.v31.run --stage STAGE --task ar|od --arm all --root DATA --out OUTPUT --device cuda`. OD also requires `--annotations instances_val2017.json`. Stages are `plan`, `startup`, `measure`, `calibrate`, `oracle`, `train`, `dev`, and `all`. The empirical Kaggle oracle notebook executes startup → measure → calibrate → oracle. `all` can proceed to training/DEV only for validated eligible arms.

Startup measures four whole sources at all eight codec/QP cells and projects the measurement plus exact metric cost. It is a cost audit and cannot qualify an oracle. If its conservative projection exceeds10.5 hours, the notebook archives evidence and stops before the full measurement grid. Session-budget projections are estimates, not guaranteed deadlines.

`--plan-file` supports deterministic test fixtures; smaller fixtures cannot satisfy empirical source-count checks. `--shard-manifest` is allowed only for measurement, with complete source IDs and the full parent plan hash. Merge all disjoint complete shards centrally, preserving config/model/code identities, before fitting one CAL policy or oracle gate. No source/QP observations are removed to meet the runtime budget.

Resume requires a verified extracted directory with `resume_manifest.json`, path checks and file hashes. Changed source pixels, code, config, policy, registry, checkpoint or measurement content are rejected. An exact arm subset can continue its original all-arm parent; the parent experiment identity remains intact. Per-execution `runtime.json` stays in the original archive/registry and is excluded from the imported research checkpoint, so the new execution records its own GPU/time receipt. Runtime features exclude labels, primary evaluator output and all candidate outcomes. Frozen DEV replay costs are reconstructed from verified packet observations; live `choose_stream` reports actual encoder/teacher calls and time separately.

## Kaggle orchestration

`python -m scripts.kaggle_v31 prepare --task ar --stage oracle --arm all --commit FULL_SHA --account OWNER --slug SLUG --directory PAYLOAD` creates a private GPU notebook and pinned release manifest. It contains no credentials. `scripts.run_v31_jobs` provides `prepare`, `preflight`, `submit`, `status`, `collect`, `resume`, and complete source-shard `merge` operations.

The account pool is read in insertion order. Fresh verified quota and complete owner inventory are required; active or unknown jobs, pagination ambiguity, low quota and existing slugs prevent use. Exclusive durable submit intents prevent duplicate GPU launches even after stale state or submission failure. Push success is recorded as submitted/unverified until the actual Kaggle status and runtime log prove progress. Existing jobs are never cancelled.

AR oracle is launched first. OD requires a collected, integrity-audited AR oracle from the same immutable commit. A train/DEV notebook additionally requires validated eligible gates, the complete measurement checkpoint and a pinned private dataset reference in the installed CLI's `owner/slug/version` syntax. Only eligible arms train. Interrupted oracle measurements and complete individual shards can also mount verified versioned checkpoints, with every existing cell/packet checked against the full parent. Their audit explicitly sets eligibility false and creates no full completeness/gate; only missing observations are collected on continuation. Archives are downloaded separately, reject traversal/links/special files, and are verified against content checksums and complete measurement membership before promotion.

## Separate GOP capability

`adaptive_vcm.v31.gop` and `configs/v31_gop.json` audit contiguous64-frame AR intervals, GOP8/16/32, scenecut0, closed GOP, B0 and matched presets. Actual ffprobe keyframe positions are verified. Three fixed16-frame stride2 windows starting at0/16/32 are pooled into one source probability vector before Top1 scoring. Compared methods share timestamps and GOP-specific random-access delay; increasing GOP changes that delay and must be labelled separately.

This audit has schema `v31-gop-audit-1`, always declares `primary_oracle_eligible=false`, excludes OD, and cannot qualify primary16-frame preprocessing headroom. Its empirical submission remains deferred until a real-source cost/latency startup audit. Numerical/codec test fixtures establish behavior, not measured VCM gains.

Research rationale and sources are recorded in the approved [design](superpowers/specs/2026-10-07-v31-actions-design.md). The release report distinguishes tested implementation, independently reviewed code, verified live jobs and completed empirical performance evidence.
