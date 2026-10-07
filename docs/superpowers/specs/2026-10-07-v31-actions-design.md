# V31: oracle-gated learned action selection for AR and OD

Status: approved by the user; implementation uses the approved plan and Subagent-driven execution. No empirical V31 improvement is asserted.
Date: 2026-10-07.
Implementation base: V30 commit `59ebb3dc31fc8edd6e1fbb80ed7de508f979edd3`.

## Tóm tắt để duyệt

V31 dùng bốn QP 30/35/40/45, với ba nhánh A/B/C. A học chọn hành động spatial trong bank có các controls và profiles hiện hữu; B bổ sung giảm frame/làm mượt thời gian cho AR, kết hợp resize với xử lý nền hiện hữu cho OD; C giữ bank của B và hiệu chỉnh guard trên CAL. Không bổ sung profile lọc. GOP là nhánh kiểm toán encoder riêng.

Trước khi huấn luyện selector, oracle đo thực trên TUNE phải đạt BD-rate PCHIP dưới -15% ở cả H.264 và H.265, có lợi ích thêm so với static K3 và spatial cố định, đồng thời không giảm quá 1 điểm phần trăm chất lượng tại bất kỳ QP nào. Đây là điều kiện thử nghiệm đề xuất, không phải kết quả đã đạt.

AR: pool TRAIN128 chia FIT96/CAL32; thêm TUNE128 tách biệt và DEV128. OD: pool TRAIN100 chia FIT75/CAL25; thêm TUNE100 tách biệt và DEV100. TUNE lấy từ phần TRAIN khác, không dùng chung nguồn với pool huấn luyện. Các tập DEV đã được quan sát nên kết quả vẫn là exploratory, chưa phải TEST xác nhận.

Selector đề xuất K=3 hành động, so với static K3 ở cùng số lượt encode/teacher. AR so thêm luôn area112; OD chọn spatial cố định trên FIT. Đo cả metadata cần khôi phục sampling. Backprop cập nhật selector qua supervision từ phép đo codec thật; gradient không đi qua H.264/H.265 thật.

Các mặc định mới cần duyệt trong bản này: cách chia FIT/CAL/TUNE; giới hạn giảm chất lượng 1 điểm phần trăm; ba độ phân giải OD 256/224/192; K=3; tám epoch và seed303101; việc tính transport recipe 32 byte cho mọi phương án. Chỉ gửi Kaggle từ commit bất biến sau test và review, theo thứ tự account pool và quota thực tế.

## Intent and scope

The user requested implementation of the illustrated pipeline: expand bit-reducing actions instead of adding filter profiles; measure a feasible oracle before training a learned selector; improve independent AR Top-1 and OD COCO mAP BD-rate against H.264/H.265; compare adaptive selection against strong static alternatives before TEST.

Confirmed constraints: QP `[30,35,40,45]` only, both codecs, existing V29/V30 jobs retained, code ultimately in `D:/STUDY/LAB/hope/v31`, GitHub repository `munnn01/adaptive_preprocessing`, Kaggle credentials remain local in `D:/STUDY/LAB/pool.json`. Use eligible accounts in pool insertion order, after checking active jobs and quota. No cancellation or modification of another job.

This is an architectural change: a learned action selector replaces the V30 continuous filter mixer as the primary learned component. Existing filters are retained as actions and comparators; no extra expert/strength profiles are introduced. V30 remains available for comparison and its modules/configs remain backwards compatible.

## Architecture and execution stages

Offline flow: source plans -> real-codec action measurements -> calibration on CAL -> fit static baselines on FIT -> teacher-feasible oracle on TUNE -> eligibility report -> learned selector training on FIT if eligible -> frozen DEV evaluation.

Deployment simulation: source pixels and frozen teacher predictions -> label-free content/context -> learned top-K action proposal -> action execution -> real encode/decode of proposals and anchor -> common guard -> smallest feasible transmitted packet -> receiver decode/temporal reconstruction -> analyzer.

Selection is encoder-side. Ground-truth labels and independent evaluator outputs are not runtime context, action inputs, ranking inputs or guard inputs. TUNE/DEV outcomes never supervise selector weights or static portfolios. A failed oracle gate is a recorded experiment outcome, not permission to weaken the threshold automatically.

Three arms isolate contributions:

| Arm | AR | OD | Guard |
|---|---|---|---|
| A: spatial | Existing controls/profiles plus explicit spatial actions | Existing controls/profiles plus resize actions | Existing anchor-relative teacher rule |
| B: composed actions | A plus frame reduction and spatial/temporal compositions | A plus resize/background compositions | Same as A |
| C: calibrated | Same action registry as B | Same action registry as B | CAL-fitted continuous teacher-regret policy |

GOP is a separate encoder audit, described below; it does not enter A/B/C primary claims.

## Action contract and registry

Each immutable action descriptor identifies task, spatial size, temporal recipe, existing filter/profile identifier, and parameter values. The registry has a canonical hash and deterministic tie order. An execution result records transmitted RGB, original source geometry and temporal recipe, receiver reconstruction recipe, content hash, and any required packet metadata.

AR starts from the historical 16-frame, stride-2, 128-square recipe. A contains the existing nine AR controls and twelve V29-A reference profiles, with identical-pixel duplicates encoded once but registry aliases preserved. This already includes area112 and area96; their presence is not claimed as new headroom. Explicit spatial actions execute at 112 or 96 before encoding, rather than reducing the analyzer input after decoding.

B adds six AR actions: `drop2` at 128/112/96, and the existing V30-C causal temporal treatment at 128/112/96 with the existing full-strength reference setting. There is exactly one temporal smoothing setting, not a new strength sweep. In compositions, apply temporal treatment at source resolution, then spatial resizing, then frame reduction where applicable. The initial registry does not combine temporal smoothing with drop2, which limits interaction cost and keeps ablations interpretable.

OD starts from the historical 320-square letterbox recipe. A contains the existing five OD controls and twelve V29-A reference profiles, plus direct area resize to 256, 224 and 192. B adds those same three sizes after the existing background8 control. OD accepts one frame only; temporal actions are rejected. Each detection is mapped back to the canonical source coordinate system before guard comparison and to original image coordinates for COCO scoring. Record both resize/letterbox transforms, including rounding.

Existing protected filters retain their pre-codec protected-core behavior. Whole-image resizing and frame reduction do not promise pixel-exact protected cores; their acceptability is established by the guard and independent evaluation.

## Temporal correctness and byte accounting

For AR, record original source FPS, sampled source indices, padding status, and the nominal duration of the frozen analyzer window. Do not reinterpret a sparse 16-frame sample as a new contiguous source clip. All candidates represent the same window and reconstructed analyzer positions as the anchor.

`drop2` sends source sample indices `[0,2,4,6,8,10,12,14]`. The receiver repeats each decoded sample twice to reconstruct 16 analyzer positions. Its encoded frame rate is half the anchor sampled frame rate, preserving nominal duration. Source FPS is read from the source, with rational values preserved; missing/nonfinite FPS disables temporal reduction and is reported rather than silently replaced by 25. Short-clip padding is recorded; frame-reduction actions are disabled for padded clips in this first version.

Spatial AR output stays at its coded size; the existing analyzer performs its fixed resize to 112. Temporal reconstruction is deterministic and has no learned receiver network. All teacher and evaluator calls see the same reconstructed sequence for an action.

A fixed 32-byte transport recipe accompanies every V31 candidate, including anchors and fixed baselines, carrying version/codec, coded geometry/frame count, nominal duration and temporal reconstruction mode. Its complete binary layout is locked in the implementation plan and tested by round trips. This is transport sampling metadata, not a semantic ROI/label/feature channel. Report elementary-stream bytes and total transmitted bytes separately. Total bytes, including codec headers and the recipe, determine selection, utility and primary rates. Do not omit metadata when reporting savings.

AR primary rate is transmitted bits / original nominal window duration. Report original-pixel bpp and aggregate bytes additionally. OD primary rate uses total transmitted bits / original source pixels. Fixed geometry/duration definitions are shared by all compared methods. V31 anchors must be measured afresh: duration-aware settings and packet overhead mean historical V30 streams are not interchangeable references.

## Data partitions and locked budget

Use the existing label-independent source ordering and top-level TRAIN/DEV/TEST partition function, then deterministic source-level allocation inside TRAIN:

- AR: first 128 TRAIN sources form the training pool, split into FIT96 and CAL32; next 128 TRAIN sources form TUNE128; DEV128 comes from the DEV partition.
- OD: first 100 TRAIN sources form the training pool, split into FIT75 and CAL25; next 100 TRAIN sources form TUNE100; DEV100 comes from the DEV partition.
- TEST is not read or submitted in this release.

The FIT/CAL split uses a seed-specific source hash, not labels, predictions or candidate outcomes. Source IDs and source-pixel fingerprints must be disjoint among FIT/CAL/TUNE/DEV/TEST; detect duplicate content and fail with evidence rather than silently changing the frozen plan. Existing DEV has been studied and remains exploratory.

Each measured source has all eight codec/QP conditions. AR training-pool conditions = 1024, TUNE = 1024, DEV = 1024. OD training-pool conditions = 800, TUNE = 800, DEV = 800. These are source/codec/QP conditions, not candidate encodes; record candidate slots and distinct measured encodes separately. Measure the B action union once, reuse immutable observations for A/B/C where action, transport, source and code hashes are identical. Guard policy changes may rescore observations but may not fabricate bytes or predictions.

Start with executable synthetic smoke checks and a bounded four-source real-data startup check; neither can qualify oracle eligibility. Then run AR oracle stage for all arms, followed by OD oracle stage. Each stage writes a resumable registry. Resume measurements by validated content/provenance keys and verify grid completeness before computing a gate.

## Guards and calibration

A/B retain the current AR r3d_18 + mc3_18 source/anchor-relative KL and anchor top-1 protection, minimum 1% anchor saving, and OD MobileNet relative detection-distance rule. This gives a strict-policy comparator under the new actions.

C AR fits a separate temperature for each teacher on CAL labels and source/anchor predictions. Fit positive temperature by bounded scalar NLL minimization over `[0.25,8]`. Freeze temperatures and report calibration before/after; a fitting failure uses temperature 1 and records the fallback. This step does not update teacher weights.

C runtime uses (a) the equal-weight calibrated ensemble's source-relative soft cross-entropy regret versus anchor and (b) each teacher's pseudo-class log-loss regret versus anchor when source/anchor argmax agree; otherwise that teacher contributes source-relative soft cross-entropy regret. The pseudo-class is the agreed source/anchor argmax. For each teacher, protect a hard decision only when its calibrated source and anchor agree and both maximum probabilities are at least the preregistered confidence 0.6. A teacher with disagreement contributes distribution regret rather than a contradictory source/anchor class constraint. An anchor-identical decoded sequence is feasible without approximation. All new guard quantities are continuous, relative to the codec anchor, and finite-checked.

Freeze C AR thresholds from CAL using a finite grid `[0,0.01,0.03,0.05,0.10]` for ensemble regret and per-teacher pseudo-class log-loss regret. Evaluate complete CAL selections for each grid pair. Require non-worsening calibrated ensemble ground-truth Top-1 and mean NLL separately for each codec/QP; maximize total transmitted-byte saving among valid policies; break ties toward smaller thresholds. If no non-identity policy is valid, C uses identity for that group. CAL labels are used only in this offline policy fit, never in deployed guards. Primary independent r2plus1d_18 is not used to fit the guard.

C OD uses the same label-free detection-distance features at inference. Fit its relative-distance threshold on CAL over `[0,0.01,0.02,0.03]`, maximizing transmitted-byte saving subject to non-worsening MobileNet COCO mAP per codec/QP on CAL. Freeze the chosen threshold per group; disable non-identity actions when no policy qualifies. ResNet50 remains outside calibration. No reliable source detection forces anchor fallback. No score temperature is invented for Faster R-CNN detections.

Report C relative to B with identical measured bank observations, showing rejected action counts, permitted class changes, CAL quality and independent TUNE/DEV effects. Calibration is a hypothesis about transfer, not a guarantee of independent task accuracy. Do not prioritize learned over a smaller feasible control; deterministic learned preference is allowed only on an exact byte/feasibility tie and is reported.

## Oracle and eligibility

The deployability-related oracle tries every registered action and selects the smallest total-byte candidate satisfying the same frozen guard used by that arm. Compute independent AR Top-1 or complete-set OD COCO mAP curves from these selected decoded results. This is teacher-feasible full-bank selection, not a primary deployable learned result.

AR may additionally report a ground-truth hindsight upper bound, explicitly diagnostic and kept outside training targets. OD does not label per-image AP winners a global mAP oracle. No DEV label can enter either runtime selection or selector supervision.

For each task/arm, release `oracle_gate.json` on TUNE with curves, overlap, PCHIP/cubic, paired uncertainty, hashes and reasons. Eligible requires:

1. Finite PCHIP BD-rate strictly below -15% versus each of H.264 and H.265 anchors, with measured common quality overlap; no extrapolation.
2. Negative PCHIP BD-rate versus the arm's TRAIN-frozen static K3 and the declared fixed spatial baseline for both codecs. This is a headroom check; final adaptive superiority needs the paired CI criterion below.
3. No more than 1 percentage point Top-1 loss for AR, or 1 mAP point loss for OD, at any corresponding QP relative to the codec anchor. Report all gaps; never discard a bad QP or source to pass.
4. Complete source/codec/QP/action observations, disjoint partitions, immutable provenance, and byte accounting including transport metadata.

Both codecs must pass for a task/arm before its selector is empirically trained. AR success does not qualify OD, or vice versa. A failure report identifies bank headroom, guard rejection or insufficient overlap separately. The new selector infrastructure may be unit-tested on synthetic data before the gate; real FIT training is disabled until a verified eligibility artifact exists.

## Learned selector and baselines

Reuse the existing ranking design where appropriate, with a V31 task-aware context and registry-specific output dimension. The context contains codec/QP, source appearance/motion/semantic support statistics, source/anchor teacher summaries and measured anchor rate. It excludes labels, independent evaluator predictions, DEV/TUNE statistics and unmeasured candidate outcomes.

Use a small MLP with safety and byte-ratio heads. FIT targets come exclusively from freshly measured action feasibility and total byte ratios under the frozen arm guard. Train with admission BCE and feasible-action log-rate regression, with a utility term rewarding retrieval of feasible savings. Keep identity/unsafe rows as supervision; do not require every condition to have a winning non-identity action. Record losses and measured oracle retrieval, not only selected-action counts.

Backpropagation updates selector parameters through this supervised loss. There is no gradient through real H.264/H.265, COCO mAP or discrete action execution. Defaults: width64, eight epochs, Adam, seed303101, final-LAST checkpoint; optimizer details and loss coefficients are frozen in the implementation plan before any TUNE/DEV result is read. C thresholds/temperatures are frozen before FIT labels are materialized.

Primary learned simulation probes anchor plus three distinct learned-proposed non-identity actions (K=3), then uses the common guard. It does not also probe all controls: that would confound learned retrieval with the full bank. An additional union-with-controls result is diagnostic and reports its larger runtime budget. If fewer than three actions exist, probe those available; collapse identical pixels/packets with alias audit.

Static comparators:

- Legacy static K3 fitted on FIT over the existing twelve reference profiles.
- Expanded static K3 fitted on FIT over the new arm bank, excluding identity and using the same greedy marginal-byte coverage objective, guard and runtime budget as learned K3.
- AR always-area112 and always-area96 curves without adaptive fallback, plus their guarded variants; area112 is the declared fixed spatial comparator.
- OD fixed spatial action chosen once on FIT among 320/256/224/192 under the same frozen objective; report all fixed-size curves. Lock this choice before TUNE.
- Unprocessed codec anchors and controls-only selection, with explicit measured probe costs.

Static portfolios are frozen from FIT for each codec/QP group. They cannot choose a different size based on a DEV outcome. Report selected learned actions, savings beyond static/controls, encoder/teacher calls, timing, and packet bytes.

## Independent evaluation and claims

AR primary: frozen r2plus1d_18 Top-1, with true-class probability, NLL and Brier score as secondary evaluation metrics. OD primary: frozen Faster R-CNN ResNet50 COCO mAP@.50:.95 on original-coordinate annotations. Teacher metrics are diagnostics and are identified as overlapping selection models.

Compute all four task/codec outcomes independently. Primary BD interpolation is PCHIP; cubic is a sensitivity result. Record unique quality points, common measured overlap, plateaus and undefined values. A missing overlap is inconclusive, not zero gain. Do not reuse five-QP curves or CIs.

Use 2,000 paired source-level bootstrap resamples, preserving every sampled source's complete eight-condition/action block. Resampling duplicate OD images must remap IDs consistently and recompute complete COCO mAP for each replicate. Compute paired learned-versus-static BD directly per replicate; do not subtract two unrelated confidence intervals. Report valid/invalid draw counts and both interpolation results. These CIs are conditional on frozen trained weights/policies.

An adaptive-superiority claim requires negative upper 95% paired CI against expanded static K3 and the declared fixed spatial comparator for the corresponding task/codec, together with the quality-gap constraint. The application target remains BD-rate strictly below -10% against each codec; the -15% oracle target is an investment gate, not an asserted learned result. Learned selection frequency alone is not success.

No automatic held-out TEST submission in this release. Freeze architecture, registry, calibration, checkpoint and protocol after exploratory DEV; subsequent TEST must have its own immutable manifest and execution record.

## Separate GOP audit

Primary A/B/C use a single matched codec configuration per source and action. Changing GOP is not attributed to the RGB preprocessor.

Implement explicit GOP settings and validation for a separate AR audit. Compare GOP lengths 8/16/32 on contiguous 64-frame windows, under identical preset, QP, scene-cut rule and random-access latency constraints for compared methods. Retain fixed analyzer windows and restore all transmitted-frame actions to the same timestamps. First demonstrate actual encoded keyframe placement and rate accounting in a real-codec test. Report GOP tuning versus an equally tuned unprocessed static codec. An experiment requiring a latency relaxation is labelled as such.

GOP audit cannot qualify the 16-frame A/B/C oracle gate or a preprocessor-only claim. OD images are excluded. Longer-window data and GPU runtime are checked before submitting this optional audit; the first empirical stage is the primary action oracle.

## Integrity, failure handling and release

V31 modules/checkpoints/manifests use their own schema, and reject V30 checkpoints, missing QP cells, unknown action recipes, altered source plans, policy/bank/config/commit mismatch and uncounted packet metadata. Old five-QP modules remain compatible with their own historical artifacts. LF-normalized file hashes support Windows/Linux portability without weakening commit integrity.

Every measured observation stores source/transport/action/config/code hashes, encoded stream hash, byte counts, reconstructed geometry/frame count and frozen teacher/evaluator predictions. Candidate failures fail closed to the anchor but are counted and surfaced; systematic codec/data failures stop the stage. Partial artifacts cannot qualify the oracle.

Local tests cover packet/temporal round trips with real FFmpeg; spatial box mapping; independent rate accounting; decision-policy calibration and no label use at runtime; source disjointness; four-QP grid validation; oracle eligibility and train blocking; learned/static equal budgets; COCO paired bootstrap with duplicated image IDs; notebook execution ordering and immutable SHA verification. Run the complete inherited suite as well as V31 tests. Use fresh-context whole-branch review and resolve material findings before release.

Develop in an isolated V31 checkout from the immutable V30 SHA, then mirror verified tracked files to `D:/STUDY/LAB/hope/v31`, commit to a V31 feature branch and push to the authorized GitHub repository. No Kaggle run starts from uncommitted code. Account preflight, submitted version, actual kernel status, archived outputs and resume stage are recorded in a V31-only registry. Submission success is not proof of execution or completion.

## Research links and limits

- Local `GOP-Based_Deep_Preprocessing_for_Video_Coding.txt`: GOP-matched differentiable proxy research motivates temporal dependency analysis, not guaranteed VCM task gains. DOI https://doi.org/10.1109/PCS60826.2024.10566387.
- Guo et al., calibration: https://proceedings.mlr.press/v70/guo17a.html. Temperature scaling changes confidence, not argmax by itself.
- Bjontegaard measurement: https://arxiv.org/abs/2304.12852. Sparse/nonstandard quality curves require overlap and interpolation sensitivity checks.
- Local SSVC and MoCrop readings remain architectural inspiration; their semantic side channels, cropping protocols and task models are not interchangeable with these standard RGB/decode measurements.

## Self-review outcome

The design distinguishes oracle and deployable selector, measured actions and new filter profiles, TRAIN-derived policies and TUNE/DEV evaluation, preprocessor actions and GOP encoder settings, decoded quality and teacher feasibility, source counts and candidate encode counts. It specifies small-sample limits, metadata overhead, no-overlap outcomes and cases where training must stop. User review of this written design is the next Superpowers architectural gate; an implementation plan follows that review.
