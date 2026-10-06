# V28 literature and MoCrop source audit

Read-only audit dated 2026-10-06. Only this research directory was written. No MoCrop scripts, training, codec jobs, or V27 code were executed or changed. Page references below are one-based PDF pages, followed by printed page numbers where useful.

## Implementation decision

Use the MoCrop idea of motion-density-guided spatial importance as an additional **encoder-side protection mask**, with ordinary full-size RGB output. Do not transplant its crop-and-resize pipeline into OD. Keep the source coordinate system and combine the motion prior with the existing teacher/object protection using a union or pointwise maximum. Apply a spatially smooth, clip-stable background filter outside the protection mask, then run unmodified H.264/H.265 and ordinary RGB decoding. This recommendation is an engineering adaptation, not a result established by these papers.

The papers establish useful motion/appearance cues and action-inference efficiency. They do not establish rate-accuracy gains, H.264/H.265 BD-rate, COCO object-detection improvements, or transfer to the current Kinetics 16-frame 128-pixel frozen 3D analyzers. Every such gain must be measured in V28.

## Papers, local identities, and publication status

| Local file | Identity / status | Relevant local pages | SHA-256 |
|---|---|---|---|
| `MOCROP.pdf` | Huang, Yao, Chen, G. Wang, Q. Wang, Dev. *MoCrop: Training Free Motion Guided Cropping for Efficient Video Action Recognition*. arXiv:2509.18473v1, 22 September 2025; preprint. Local metadata explicitly identifies v1, 5 PDF pages, 3,901,492 bytes. | p1 scope/assumptions; p2 complete algorithm; p3 Table 1/setup; p4 Table 2, limitations, future work. | `b2546fdea604ecc5f4f001052c51934a6a4093ba9cf8b407fdcdb4a89e073013` |
| `Compressed Video Action Recognition.pdf` | Wu, Zaheer, Hu, Manmatha, Smola, Krahenbuhl. CVPR 2018, pp6026-6035; peer-reviewed conference, not journal. 10 PDF pages. | pp2-4 compressed representations and accumulation; p4 printed6029 architecture; pp5-7 evaluation. | `985d78e058d63b2fd5c0a4c54052ec653a13d38cf01b230b0a4cc272be7944ce` |
| `Compressed Video Action Recognition with Refined Motion Vector.pdf` | Cao, Yu, Feng. arXiv:1910.02533v1, submitted6 October2019; preprint. Local title page says Hanyuan Cao, whereas arXiv metadata says Haoyuan Cao. 9 PDF pages. | pp1-3 MV versus optical flow; p4 confidence/refinement pipeline; p6 evaluation. | `eeb1ac25aa40125cc927cece7e17bd2ab7880fa4b11b7a36e8001349cd5d1a35` |
| `MM-ViT Multi-Modal Video Transformer for Compressed Video Action Recognition.pdf` | Jiawei Chen and Chiu Man Ho. WACV2022, pp1910-1921; peer-reviewed conference, not journal. 12 PDF pages. | p3 printed1912 modality representation; pp5-6 printed1914-1915 train/inference and modality ablations; p8 conclusions. | `2c73304cd613e2abfd7157e22bfab8b62293a53bdfb857bb03cf12899aa57e18` |
| `Action recognition A comprehensive survey of tasks, methods, and challenges.pdf` | Heo, Moon, Jung. *ICT Express* 12(1), 32-49, February 2026. DOI 10.1016/j.icte.2025.11.015; peer-reviewed journal review. Online 29 November 2025. 18 PDF pages. | pp1-3 taxonomy and task distinctions; p14 trade-offs/limits, p15 conclusion. | `63a33a138ecdfa31c1ecdbcce965f8e939af4824bb4242a314d8bfacdd2373c4` |
| `Action Recognition and Detection Based on Deep Learning.pdf` | Li, Liang, Gan, Cui. *Computers, Materials & Continua*77(1),1-23, published31 October2023. DOI10.32604/cmc.2023.042494; peer-reviewed journal review. 23 PDF pages. | pp1-2 action recognition versus temporal/spatiotemporal action detection; later architecture/dataset review. | `180940e9eff7a45e0d59e89111ab02cad8f326a6828dd044ec69a25db49693db` |

Complete metadata, byte counts, and extraction paths are in `manifest.json`; local page text is in `paper_1_pages.txt` through `paper_6_pages.txt`. MoCrop v1 algorithm/table pages2 and4 were rendered and visually inspected to verify equations and tables.

Primary URLs:

- MoCrop v1: https://arxiv.org/abs/2509.18473v1
- MoCrop current v2: https://arxiv.org/abs/2509.18473v2 ; https://arxiv.org/html/2509.18473v2 ; https://arxiv.org/pdf/2509.18473v2
- MoCrop author code: https://github.com/microa/MoCrop
- CoViAR: https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Compressed_Video_Action_CVPR_2018_paper.html
- MM-ViT: https://openaccess.thecvf.com/content/WACV2022/html/Chen_MM-ViT_Multi-Modal_Video_Transformer_for_Compressed_Video_Action_Recognition_WACV_2022_paper.html
- Refined MV: https://arxiv.org/abs/1910.02533v1
- 2026 survey: https://doi.org/10.1016/j.icte.2025.11.015 ; https://www.sciencedirect.com/science/article/pii/S2405959525001869
- 2023 survey: https://www.techscience.com/cmc/v77n1/54495

## Local MoCrop source coverage and provenance

Read all source modules (`src/mocrop_dataset.py`, `src/models.py`, `src/transforms.py`), all scripts (`scripts/train.py`, `scripts/test.py`, `scripts/calculate_flops.py`), setup, requirements, environment, license, README, and every notebook source cell plus result summaries. No applicable AGENTS.md was found in the AR/code tree or checked ancestor directories. `code_manifest.json` hashes all12 non-git files.

The clone is clean at commit **38c60e618da211db3144173b4c322bfdfa959427**, committed27 September2025 (`Update README.md`), with origin `https://github.com/microa/MoCrop.git`. Five local commits are present. This proves the inspected local snapshot identity; live upstream HEAD could not be separately verified because direct networking was restricted and the web tool could not fetch source/API pages. The browsed GitHub README identifies the same repository, layout, external maps, and results. Do not describe the local snapshot as a verified February2026 source release.

License is MIT, copyright(c)2025 MoCrop. Preserve its copyright and permission notice if copying substantial source. Its SHA-256 is `e4303242b5cbf2db26089b38a521f656dbc3cafc8bfb86abf1fbb7cbe69397b1`.

Core algorithm file SHA-256: `48335cdd96b1c9c62efeef49e08ae8f58dde920e61d7ee81d97fd0e070f3af88`.

### What the code actually implements

- `src/mocrop_dataset.py:24` `mv2yolo` takes an already-computed **2-D scalar density grid**, not RGB clips or raw motion vectors. It exhaustively enumerates rectangles with grid area within `area_ratio*rows*cols` +/-10%, maximizing plain `np.sum`. Ties retain the first enumerated rectangle. No aspect-ratio/actor-coverage constraint is present.
- Lines57-90 convert the selected normalized rectangle to RGB coordinates, clip bounds, slice the image, and return the original image for invalid coordinates. The grid conversion defaults to320x240 and uses integer cell sizes; grids not dividing these dimensions exactly can introduce coordinate rounding/truncation bias. Use exact normalized grid fractions in V28.
- Lines222-227 locate external maps at hard-coded Linux path `/home/mbin/data/ucf101/extract_mvs/<category>/<video>/motion_vectors_denoised_all_mc_<h>_<w>.npy`. Lines249-255 load the same map for each selected frame and apply its crop. Missing maps cause a warning and original-frame fallback.
- **Absent from the code:** codec MV extraction; denoise/merge implementation; Monte Carlo map generation; global-camera-motion model; outlier rejection beyond whatever the external map already encodes; map smoothing; multi-box preservation; rate optimization; encode/decode; OD evaluation. The environment lists `motion-vector-extractor`, but the source does not invoke it.
- Frames are read by arbitrary OpenCV frame indices, not by explicitly selecting codec I-frames. The `representation='iframe'` and `accumulate` constructor values do not change frame loading. Hence the published I-frame protocol is not fully expressed in this code.
- Models are 2-D ImageNet backbones with replaced UCF101 heads; segment logits are averaged. They are not the present Kinetics pretrained frozen3-D video teachers/evaluator.

### Reproducibility caveats from static inspection

- Notebook calls absent historical files `test_unified.py` and `calculate_flops_clean.py`. Its baseline commands load `val-normal` checkpoints; MoCrop commands generally load different `val-mocrop` checkpoints. This does not establish the paper's claim that the same frozen weights produce the baseline and MoCrop result.
- Current `scripts/test.py:70` accepts `--input-size`, but lines152 and166 pass `transform=None` to the dataset. No resizing uses that argument; samples can retain variable crop/source dimensions and fail collation. This conflicts with224/192-pixel comparisons in the notebook/paper.
- Test lines130-140 strip `base_model.` or `backbone.` checkpoint prefixes, then call `model.load_state_dict(...,strict=False)` on a model whose expected keys still begin with `backbone.` (line32). Keys can all be mismatched without a fatal load error. The script discards incompatible-key diagnostics. Do not use this loading protocol in V28.
- Train lines253/267 add transform instances using `+`, although these classes define no `__add__`; this is a static runtime-error finding. `adjust_learning_rate` line117 references `args`, but it is local to `main`. The unexecuted source therefore needs fixes before claiming reproducibility.
- `src/transforms.py` includes unused helper problems (default scale875 rather than.875, width/height naming and slicing swaps). Do not transplant these helpers.
- FLOPs script lines62/69 feeds `(1,3,S,S)` into only the2-D backbone. It counts per-frame inference and omits the8-frame multiplier, map extraction, preprocessing, decode, and encode. Thus4.11 versus3.02GFLOPs is not end-to-end video-compression cost.
- README describes old minimum versions while `environment.yml` records torch2.4.1/torchvision0.19.1 and modern packages. No old environment installation is needed for a concept adaptation.

## MoCrop v1 versus current v2

Current arXiv v2 is dated1 February2026 and has6 pages. It changes near-static filtering to top-k magnitude retention (example1%), uniform sampling to magnitude^beta importance sampling (beta4, gamma.1), and plain rectangle sum to a.6sum+.4mean objective on an example16x9 grid (SecIII-C, PDFp3). It adds complexity/overhead analysis and explicit failure cases (pp3-5). The inspected2025 code only supplies the v1-style sum search. The headline UCF101 accuracy/GFLOPs figures remain. Neither version reports encoded bitrate/BD-rate or COCO OD results. v2p5 discusses camera shake, perspective, minimally moving actors, and scene-wide motion; proposed mitigations are global-motion suppression, temporal smoothing, and residual cues, not supplied algorithms. Sources: https://arxiv.org/abs/2509.18473v2 and https://arxiv.org/pdf/2509.18473v2 .

## Actionable mapping to encoder-side VCM

| Evidence | Safe V28 use | Boundary / pitfall |
|---|---|---|
| MoCrop v1p2 forms a motion-density grid and one box per clip. | Form one clip-level spatial protection prior; reuse it across the16 frames. Use an integral image or small deterministic grid for bounded cost. | Per-frame crop/zoom or rapidly varying masks can create artificial motion and prediction residuals. One small box can miss interacting people/objects. |
| MoCrop v1p4 and v2p5 identify camera/scene-wide motion as failure conditions. | Estimate and remove robust global translation before motion-density aggregation; reject unreliable estimates; fall back to the appearance/object mask or original source. | Median flow only handles translation approximately. Foreground occupying most pixels can bias it; pan/zoom/rotation need a richer fit or conservative fallback. Global suppression is a proposed mitigation, not a validated MoCrop implementation. |
| Refined MVpp3-4 propagates source edge confidence, median-filters, pools16x16 blocks, suppresses low-confidence MVs. | Confidence-weight a motion proxy using local texture/edge support; use robust clipping of extreme displacement and finite/consistency checks. | Source threshold.0075 and16x16 blocks are tied to its normalization/GOP/representation. Do not transplant them blindly to128-pixel RGB flow. The paper also alternates Scharr/Sobel names; this is not a precise universal recipe. |
| CoViARpp3-4 accumulates codec MVs/residuals to the I-frame. | Preserve temporal context and combine motion with appearance/change evidence. | Its accumulated fields follow codec reference chains; unwarped RGB frame differences do not implement this. CoViAR modifies/trains analyzer streams, which is outside the current pipeline. |
| MM-ViTp3 andp6 show complementary I-frame/MV/residual/audio modalities. | Use RGB appearance, motion, and change evidence as encoder-side priors, with reliability-aware fusion. | MM-ViT is a trained multimodal analyzer, not encoder-only preprocessing. Audio/modality gains cannot be claimed for frozen RGB analyzers. |
| Surveys distinguish classification from action localization/detection. | Keep Kinetics clip Top-1/Top-5 and COCO object mAP as distinct evaluation tracks. | Their term 'action detection' means locating actions over time/space, not COCO category detection on still images. |

### Recommended deterministic prototype

1. For T>1 only, compute low-resolution motion from adjacent source RGB frames. If using optical flow or block matching, label it **pre-encode RGB motion proxy**; true codec MVs exist only after a codec motion search. A probe encode is possible but its CPU/time cost must be reported. MVs are prediction choices optimized for compression, not exact optical flow, and depend on codec, CRF, references, block partitions, and GOP.
2. Estimate background/global translation from reliable flow vectors, preferably outside already protected foreground. Subtract it, cap residual magnitudes robustly (e.g. percentile/MAD clipping), and suppress nonfinite, tiny, and poorly supported vectors. Do not merely keep the largest1%: isolated high displacements and camera shake can dominate. Reject ambiguous global fits; retain the original/appearance protection as fallback.
3. Aggregate robust residual evidence over the clip into a small grid. Normalize only when evidence is nontrivial. Use deterministic pooling or weighted counts rather than random Monte Carlo sampling at128px, where sampling overhead/variance has little value. If a rectangle is desired, search conservative area ratios and retain it as a **mask proposal**, not a coordinate-changing crop. Extend to multiple connected regions or union with appearance protection.
4. Set protection to the pointwise maximum of teacher appearance importance, motion-region importance, and all valid detector box masks. Add margins and feather outside the hard protected interior. Compute temporal union/fixed mask for the clip; appearance context remains protected even for stationary actions.
5. For OD T=1, explicitly bypass temporal motion. Use all source detector boxes, retaining small/low-confidence objects according to the existing protection policy, and preserve original image coordinates. A zero density map must not select an arbitrary top-left crop. If no trustworthy boxes/prior exist, use identity or conservative full-frame filtering.
6. Produce `y = protection*x + (1-protection)*smooth(x)` in the same shape, value range, and coordinate system, using temporally consistent filter parameters. A guard or accepted candidate may leave the source unchanged. Mask feathering must not degrade object interiors; decoded task guards remain necessary.
7. Check actual encoded bytes and decoded RGB results. Added mask edges, movement, sharpening, or textured preservation can increase codec cost, so an expected proxy reduction is not proof of a bitrate gain.

Mask computation does not require a transmitted mask when the final decoder simply decodes a standard full-frame RGB bitstream. If a design transmits crop coordinates, packed ROIs, inverse transforms, residual side streams, or model metadata needed at the receiver, count all of those bits and disclose the changed decoder protocol. Cropping/resizing invalidates direct COCO box geometry; inverse coordinate transforms would add protocol complexity and cannot recover discarded objects.

## Required evidence for rate-accuracy claims

- Evaluate identity H.264 and H.265 at identical source clips/images, codec settings, resolutions, frame counts, chroma format, and frozen analyzer transforms. Keep AR guard teachers separate from the independent r2plus1d_18 evaluator; keep OD detector evaluation and source box generation explicit. Do not use test labels/GT boxes to select per-sample preprocessing.
- Report encoded-file/payload byte definition, rate per clip orbits/pixel/frame, raw baselines, decoded Top-1/Top-5 and COCO mAP. Fixed-CRF equality is not matched bitrate equality. Evaluate multiple operating points and interpolate only within supported overlaps.
- BD-rate requires a common accuracy interval and enough usable operating points. Discrete/noisy task curves, plateaus, crossings, or empty overlaps can make it undefined; report this rather than extrapolating or inventing a benefit.
- Separate teacher consistency, bitrate savings, task accuracy, and timing. Source-confidence preservation can preserve wrong predictions and is not measured Top-1/mAP. A mask may save bits and merely preserve accuracy; call that preservation, not improvement. Joint AR-and-OD improvement requires evidence in both tracks for each codec or explicit scope limits.
- Since the pipeline retains128px and the frozen3-D analyzer, its inference FLOPs will ordinarily remain constant. Any compression gains differ from MoCrop's smaller-input2-D inference GFLOP gains. Include preprocessing/probe-encode/guard overhead in timing.

## Confidence and remaining limits

High confidence: local file hashes/version, local complete source behavior, original-coordinate/timing/bitrate distinctions, missing stages, and per-frame FLOPs measurement. Current v2 date/formula/failure-mode differences are confirmed from primary arXiv text. Live GitHub HEAD/source retrieval remained unavailable; inspected code is the exact clean local author-repository snapshot. No empirical V28 improvement is claimed by this audit.
