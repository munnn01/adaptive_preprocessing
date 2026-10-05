# V25 local verification

These checks verify code and actual bit accounting. They do not measure Kinetics
AR accuracy or establish the BD-rate target.

Python: `D:/STUDY/AI/envs/ten_env/python.exe`. FFmpeg is the same environment's
`Library/bin/ffmpeg.exe`, discovered by the existing codec adapter.

- Task bank:19 tests passed, including actual x264/x265 at QP40/45/50.
- Policy executor:5 tests passed, including all ten codec/QP measurement groups,
  strict decision rejection of a cheaper class-flipped stream and nonzero replay
  gradients. Conditional safety/rate supervision learns different useful actions
  on a controlled context fixture.
- Integrated ranking evaluator:3 tests passed. A deliberately cheaper unproposed
  action stays out of the primary pool. The equal-budget static arm and full-bank
  oracle remain separate. A complete actual H.264/H.265 evaluation loads a ranking
  checkpoint with `weights_only=True`, renders variable geometry, records all five
  component arms, and verifies oracle bytes <= adaptive bytes <= control bytes.

The first coordinator test invocation used a missing parent temporary directory;
that harness error was repaired before the completed rerun. An accounting fixture
also initially gave duplicate pixel outputs inconsistent mock byte costs; the
fixture was corrected. Production selector behavior did not require a change.

The synthetic capacity probe reports pixel-space diagnostics only. Some sources
have no HEVC QP50 opportunity because the stream is already near its overhead
floor. No headers, frame count or encoder settings were altered to hide this.

Full regression: **107 passed in335.89s**, no skips; the subsequently added
integrated evaluator test also passes (3/3 in its module). Total108 distinct
passing tests. `git diff --check` passed. Kaggle receipts are recorded separately
after submission. TEST has not been inspected during development.
