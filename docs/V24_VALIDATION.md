# V24 local validation — 2026-10-05

Executed with `D:\STUDY\AI\envs\ten_env\python.exe` on the staged source.

- Full suite: **80 passed**, no skips, 72.27 seconds.
- After strengthening the integration fixture to make confident source and
  anchor decisions disagree: **15 profile tests passed**, 17.91 seconds.
- After adding the registered 0.25 identity-label weight: the affected
  real-codec profile training/component evaluation test passed again,
  15.87 seconds. No other executable code changed after that check.
- 20,000 randomized probability-vector cases: the new AR guard accepted
  the anchor itself with exact relative KL=0, with strict mode both on/off.
- Actual H.264/H.265 byte tests passed for AR profiles at QP40/45/50. These
  use a synthetic static textured background and exact protected foreground;
  they are compressibility fixtures, not real action-recognition evidence.
- The completed V23 AR checkpoint resolver located the expected artifact and
  verified its SHA256 before allowing the guard-only baseline to use it.
- Kaggle payload test verifies the V23 input dependency, hash resolver,
  separate `guard_only` output, V24 config/trainer, private GPU metadata,
  component evaluation and immutable code pin.

The first suite invocation failed to create pytest temporary directories
because their parent directory did not exist (67 passed, 13 setup errors).
The parent directory was created and the complete suite passed as above.

Validation artifacts are under
`D:\STUDY\LAB\bao_1\output\adaptive_v24_validation` (`pytest.xml`,
`guard_integration.xml`, `weighted.xml`). Source was staged in
`D:\STUDY\LAB\bao_1\adaptive_preprocessing_v24_build` before publication
to the requested `D:\STUDY\LAB\hope` repository.

No completed V24 real-data Top-1, BD-rate, or high-QP rate gain is claimed.
The paired Kaggle experiment in [the design](V24_AR_DESIGN.md) is required
to measure those outcomes and to separate guard correction from policy gains.
