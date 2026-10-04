# Local validation

Latest V23 code: [64-test full suite and measured-codec smoke](V23_VALIDATION.md).
The historical V22 validation below is retained for provenance.

2026-10-04, Windows CPU, `D:/STUDY/AI/envs/ten_env`, Python3.11,
PyTorch2.5.1, TorchVision0.20.1, FFmpeg7.1 with libx264/libx265.

`python -m pytest tests -q --basetemp=<writable workspace>/pytest`: **46 passed**.
No codec integration test is skipped. Coverage includes:

- Full real H.264/H.265 encode/decode, strict frame counts and byte-based
  original-pixel normalization across resolution changes.
- Semantic-core equality, motion protection, scene-cut resets, deterministic
  QP-dependent transforms and causal learned-gate/temporal behavior.
- Bounded multiscale learned forward pass, live QP/codec conditioning,
  finite gradients, two-step optimizer execution and final checkpoint loading.
- AR and OD paired runner integration, actual bitstreams, COCO aggregation,
  handling of undefined flat-quality curves and no false target pass.
- Multi-teacher selection, invalid signal fallback, unique detection matches,
  paired bootstrap multiplicity, invalid source failure and rounded geometry.
- Private Kaggle metadata, immutable commit pinning and isolated pool auth.

The AR/OD integration tests use tiny deterministic task fixtures, so they
verify software execution, pairing and reporting only. They do not measure
pretrained-model task accuracy. No local result proves the BD-rate target.
Actual Kinetics/COCO experiments must run separately on Kaggle.

After the first GPU runs, the focused Kaggle runner suite reports **4 passed**.
The added regression test verifies explicit UTF-8 subprocess configuration,
Unicode output handling and credential redaction for Windows log collection.
The full 46-test baseline above records validation of the original experiment
commit; the unchanged suite was not rerun solely for a download encoding fix.
See [actual V22 development results](RESULTS_v22_2026-10-04.md) for task metrics.
