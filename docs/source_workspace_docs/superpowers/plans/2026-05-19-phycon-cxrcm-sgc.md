# PhyCon-CxRCM-SGC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a physics-constrained IQ-level residual diffusion and consistency reconstruction frontend for SGC.

**Architecture:** The new frontend lives under `code/SGC/recon` and is independent from `SGC/v3`. It predicts bounded IQ residuals, conditions on satellite channel metadata or proxy statistics, preserves Frozen Base identity features, and optionally uses a differentiable satellite channel consistency loss.

**Tech Stack:** PyTorch, existing WiSig data loaders, existing `sat_channel.py`, existing Frozen Base checkpoint loading, YAML configs, unittest/pytest.

---

### Task 1: Recon Package Contract Tests

**Files:**
- Create: `E:/type10-7/tests/test_phycon_recon.py`

- [ ] **Step 1: Write failing tests**

Tests must cover:
- `CxResUNet1D` maps `[B,2,T]` plus condition and timestep to `[B,2,T]`.
- Base model parameter count is between 0.18M and 0.25M.
- `ResidualSafetyGate` returns `[B,1,1]` values in `[0,1]`.
- `apply_bounded_residual` honors `rho`.
- `PhyConditionEncoder` accepts real meta and proxy fallback without tx labels.
- `stft_mag_phase_loss`, `identity_preservation_loss`, and `channel_consistency_loss` are finite.
- `DifferentiableSatChannel` propagates gradients to `x_hat`.
- `CxResDiff` and `CxConsistency` expose deployment-friendly `correct(...)`.

- [ ] **Step 2: Run tests and verify they fail**

Run: `conda activate ssr-gpu; python -m pytest E:/type10-7/tests/test_phycon_recon.py -q`

Expected: import failures for missing `SGC.recon` modules.

### Task 2: Core Recon Modules

**Files:**
- Create: `E:/type10-7/code/SGC/recon/__init__.py`
- Create: `E:/type10-7/code/SGC/recon/complex_ops.py`
- Create: `E:/type10-7/code/SGC/recon/time_embedding.py`
- Create: `E:/type10-7/code/SGC/recon/cx_unet_1d.py`
- Create: `E:/type10-7/code/SGC/recon/residual_gate.py`
- Create: `E:/type10-7/code/SGC/recon/condition_encoder.py`

- [ ] **Step 1: Implement complex helpers**

Provide IQ/complex conversion, RMS normalization, residual ratio, and bounded residual application.

- [ ] **Step 2: Implement condition encoder**

Encode `orbit`, `state`, `weather`, and continuous normalized physics fields into `condition_dim=24`; provide `normalize_sat_meta(...)` and `estimate_phy_proxy(...)`.

- [ ] **Step 3: Implement CxResUNet-1D-020M**

Use depthwise residual blocks, three-level encoder/decoder, timestep embedding, condition broadcast, and final tanh output. Keep default params in the 0.18M-0.25M range.

- [ ] **Step 4: Run Task 1 subset**

Expected: shape, parameter, condition, and residual tests pass.

### Task 3: Losses And Differentiable Channel

**Files:**
- Create: `E:/type10-7/code/SGC/recon/stft_losses.py`
- Create: `E:/type10-7/code/SGC/recon/identity_losses.py`
- Create: `E:/type10-7/code/SGC/recon/channel_losses.py`
- Create: `E:/type10-7/code/SGC/recon/diff_sat_channel.py`

- [ ] **Step 1: Implement STFT losses**

Add complex IQ STFT, log magnitude loss, phase loss, and STFT L1 helper.

- [ ] **Step 2: Implement identity loss**

Freeze-compatible feature extraction and classifier helpers, cosine identity loss, CE identity loss, and combined identity loss.

- [ ] **Step 3: Implement differentiable channel**

Keep fixed sampled phi/meta, apply path gain, fading, frequency rotation, phase noise, multipath, mild AGC, and IQ imbalance using differentiable tensor operations.

- [ ] **Step 4: Implement channel consistency loss**

Compute time-domain L1 plus `0.3 * STFT` consistency and residual safety loss.

### Task 4: Diffusion And Consistency Wrappers

**Files:**
- Create: `E:/type10-7/code/SGC/recon/cx_resdiff.py`
- Create: `E:/type10-7/code/SGC/recon/cx_consistency.py`

- [ ] **Step 1: Implement cosine diffusion schedule and v-prediction helpers**

Expose `sample_timesteps`, `q_sample`, `target_v`, and `v_prediction_loss`.

- [ ] **Step 2: Implement `CxResDiff`**

Wrap UNet, condition encoder, residual gate, `correct(...)`, and training loss composition.

- [ ] **Step 3: Implement `CxConsistency`**

Support 1/2/4-step correction, pseudo-Huber consistency loss, EMA-friendly state loading, and same bounded residual contract.

### Task 5: Scripts And Configs

**Files:**
- Create: `E:/type10-7/code/SGC/train_recon_diffusion.py`
- Create: `E:/type10-7/code/SGC/distill_recon_consistency.py`
- Create: `E:/type10-7/code/SGC/eval_recon_frontend.py`
- Create: `E:/type10-7/code/SGC/train_recon_sgc_joint.py`
- Create: `E:/type10-7/code/SGC/configs/recon_cxresdiff_020m.yaml`
- Create: `E:/type10-7/code/SGC/configs/recon_cxconsistency_020m.yaml`
- Create: `E:/type10-7/code/SGC/configs/recon_sgc_joint.yaml`

- [ ] **Step 1: Implement dry-run capable scripts**

Each script must build data/model/checkpoint wiring and support `--dry_run` for fast CI-style validation.

- [ ] **Step 2: Implement training loops**

Diffusion script uses pair, identity, residual, TF, and optional channel losses. Distillation script uses teacher, consistency, residual, identity, TF, and optional channel losses. Joint script trains only recon head/gate and SGC adapter/gate.

- [ ] **Step 3: Implement frontend evaluation**

Report raw base, recon base, residual ratio mean/p95, identity cosine, clean drop, per-scenario sat metrics, params, and latency.

### Task 6: Verification

**Files:**
- Read: `E:/type10-7/AGENTS.md`

- [ ] **Step 1: Run focused tests**

Run: `conda activate ssr-gpu; python -m pytest E:/type10-7/tests/test_phycon_recon.py -q`

- [ ] **Step 2: Run existing SGC tests**

Run: `conda activate ssr-gpu; python -m pytest E:/type10-7/tests/test_sgc_v3.py -q`

- [ ] **Step 3: Run script dry-runs**

Run each new script with `--dry_run` and minimal loader limits when a valid teacher checkpoint is available. If no checkpoint/data exists locally, verify parser/import contracts and report the blocker.
