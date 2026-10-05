# Primary literature informing the preprocessing investigation

The implementation and its measured evidence remain distinct from published
results. These papers support design questions and evaluation choices; their
reported bitrate gains are not gains of this repository.

1. Guo Lu et al., **Preprocessing Enhanced Image Compression for Machine
   Vision**, IEEE Transactions on Circuits and Systems for Video Technology
   34(12),13556-13568,2024,
   [DOI10.1109/TCSVT.2024.3441049](https://ieeexplore.ieee.org/document/10632166/).
   [Author preprint](https://arxiv.org/abs/2206.05650) and
   [author implementation](https://github.com/XingtongGe/PreprocessingICM).
   It learns a quantization-adaptive pixel preprocessor before conventional
   image codecs, with proxy networks for backpropagation. It motivates QP
   conditioning and semantic preservation. V26 instead learns to rank fixed
   pixel operators using actual byte measurements; it does not reproduce that
   paper's neural pixel coefficients or claim that image-task gains transfer
   to temporal AR.

2. **Task-Switchable Pre-Processor for Image Compression
   for Multiple Machine Vision Tasks**, IEEE Transactions on Circuits and
   Systems for Video Technology 34(7),6416-6429,2024,
   [DOI10.1109/TCSVT.2023.3348995](https://ieeexplore.ieee.org/document/10379180/).
   Task-specific semantic features modulate a pixel preprocessor before an
   off-the-shelf image codec. This supports investigating task-dependent
   preprocessing for OD and AR rather than assuming a single filter preserves
   both tasks. Its image experiments do not establish AR safety of luma/DC
   stabilization; the unchanged two-teacher guard is still measured here.

3. Yuan Tian et al., **A Coding Framework and Benchmark towards Low-Bitrate
   Video Understanding**, IEEE Transactions on Pattern Analysis and Machine
   Intelligence 46(8),5852-5872,2024.
   [Author preprint](https://arxiv.org/abs/2202.02813),
   [author repository and journal citation](https://github.com/tianyuan168326/VCS-Pytorch).
   Its video-understanding benchmark and label-free semantic modeling motivate
   independent temporal-task evaluation at low rates. Its collaborative
   traditional/neural coding design transports semantic information in an
   additional stream. V26 does not implement that decoder or side stream;
   comparison rates here count every byte of the conventional codec stream.

4. Yuan Tian et al., **Non-Semantics Suppressed Mask Learning for
   Unsupervised Video Semantic Compression**, ICCV2023,13610-13622,
   [official paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Tian_Non-Semantics_Suppressed_Mask_Learning_for_Unsupervised_Video_Semantic_Compression_ICCV_2023_paper.pdf).
   It studies video semantics and downstream action recognition, with semantic
   mining/compensation and masked learning. It reinforces the distinction
   between reducing texture entropy and preserving machine-relevant semantics.
   Applying that distinction to the V25 observation of byte-saving but
   teacher-unsafe QP50 actions is an engineering inference, not a published
   guarantee about this bank.

The V26 bounded DC/exposure coefficients, motion/cut thresholds and strength
ladder are registered engineering hypotheses. Their narrow synthetic effects,
TRAIN feasibility and held-out DEV task quality are reported separately.
The two IEEE TCSVT papers above are journal publications, not mislabeled
conference proceedings or preprints.
