# SGC + SSDG post-stage design

Date: 2026-05-17

## Goal

Run SGC and SSDG post-stage experiments in parallel. The two groups are independent:

- SGC is trained as a standalone residual-correction module from the N04 Fishr checkpoint.
- SSDG is trained from scratch and does not load the N04 checkpoint.
- No SGC checkpoint is fed into SSDG in this launcher.

The SGC teacher checkpoint is:

`/home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth`

GPU allocation:

- GPU 0-2: SGC standalone residual correction experiments.
- GPU 3-5: SSDG two-stage semi-supervised experiments.

## SSDG Protocol

SSDG starts from randomly initialized model weights and uses a source split stratified by transmitter, receiver, day, and equalization:

- labeled: 0.1
- unlabeled: 0.6
- validation: 0.3

Training uses two explicit stages:

1. Label stage: train for 150 epochs on the 0.1 labeled split. This stage uses the N04-style settings: Lite-D, ID no-DAC, domain no-stats, RCN domain enhancer, same-tx cross-domain MixStyle, mixed-orbit satellite classification, and Fishr. It supervises both transmitter classification and domain-backbone classification, because the later domain gate depends on a trained domain head.
2. Pseudo stage: after epoch 150, continue training with labeled CE plus unlabeled pseudo-label CE.

Unlabeled samples receive pseudo-labels only when all enabled gates pass:

- confidence gate: weak-view transmitter confidence is above threshold.
- domain gate: domain-backbone prediction matches the known unlabeled domain label.
- consistency gate: weak and strong views predict the same transmitter label.
- temporal gate: neighboring samples from the same receiver/day/equalization sequence predict the same transmitter label.

The temporal gate is conservative: a sample is trusted only when at least one contiguous neighbor in the same rx/day/eq stream has the same weak-view pseudo-label, sufficient confidence, and adjacent dataset order. The dataset-order check avoids comparing unrelated transmitters that happen to share similar `sig_i` values.

## Audit Fixes

- SSDG unlabeled loader is deterministic instead of shuffled, so contiguous samples can be checked.
- Domain label mapping is built from the full source pool, not just the 0.1 labeled subset, so valid source domains are not accidentally mapped to unknown during pseudo-label gating.
- N04-style SSDG label training explicitly includes domain CE, adversarial domain CE, orthogonal loss, hard-domain CE, Fishr, mixed-orbit satellite CE, Lite-D/no-DAC/no-stats/RCN, MixStyle, and batch size 256.

## Experiments

SGC:

- S1: default RSGC, clean validation selection.
- S2: stronger feature/budget regularization with satellite worst-case selection.
- S3: no-residual negative control.

SSDG:

- U0: label-only low-label baseline.
- U1: two-stage domain + temporal pseudo-label SSDG.
- U2: U1 plus EMA teacher pseudo-label generation.

## Launcher

Add `code/scripts/run_sgc_ssdg_6gpu.sh`. It launches fixed one-job-per-GPU runs, writes logs to `logs/sgc_ssdg_n04`, and writes checkpoints to `runs/sgc_ssdg_n04`.
