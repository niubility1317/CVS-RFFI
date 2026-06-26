# Split-BEX02 Alternatives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full four-phase Split-BEX02 alternative stack for federated CVS-RFFI, with honest approximation labels, compression accounting, diagnostics, and an 8-GPU validation matrix.

**Architecture:** Keep ordinary `fedavg` and `fedprox` behavior unchanged. Treat `fedcvs_vmb` as the main VMB approximation route and add `split_bex02` as an explicit compressed-feature approximation route that reuses VMB stage logic while exposing activation-token payloads. Server-side shared signals are kept as compact packets: prototypes, style codes, gradient-stat/conflict summaries, logit anchors, and activation-token summaries.

**Tech Stack:** Python, PyTorch, current single-process federated simulator, `unittest`/`pytest`, PowerShell local verification, N607 sync/report workflow.

---

### Task 1: Tracker And Guard Tests

**Files:**
- Create: `E:/type10-7/code/analysis/split_bex02_alternatives_traceability.md`
- Create: `E:/type10-7/code/tests/test_split_bex02_alternatives.py`
- Modify: `E:/type10-7/code/tests/test_federated_train_integration.py`

- [ ] **Step 1: Write failing tests for new public APIs**

Add tests that import:

```python
from federated.gradient_stats import conflict_aware_aggregate_gradients
from federated.distill_anchors import LogitAnchorBank, build_logit_anchor_stats, logit_anchor_kd_loss
from federated.activation_tokens import ActivationTokenCodec
from federated.style_packet import style_code_from_stats
```

The tests must assert: opposing gradients are reported and corrected, unreliable KD samples are gated, activation tokens reduce payload under quantization, and style codes are bounded fixed-size vectors.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
conda activate ssr-gpu
cd E:/type10-7/code
python -m pytest tests/test_split_bex02_alternatives.py -q
```

Expected: import failures for the new modules/functions.

- [ ] **Step 3: Create traceability tracker**

Create a Markdown table with SBX-01 through SBX-11, status, files, knobs, tests, log evidence, N607 evidence, and notes. Initial code status can become `implemented`; N607 fields remain `pending` until sync and launch evidence exist.

### Task 2: Utility Modules

**Files:**
- Create: `E:/type10-7/code/federated/gradient_stats.py`
- Create: `E:/type10-7/code/federated/distill_anchors.py`
- Create: `E:/type10-7/code/federated/activation_tokens.py`
- Modify: `E:/type10-7/code/federated/style_packet.py`

- [ ] **Step 1: Implement conflict-aware aggregation**

Expose:

```python
conflict_aware_aggregate_gradients(client_gradients, weights, mode="none")
```

Return `(aggregated, metrics)`. Modes: `none`, `cosine_clip`, `pcgrad`. Metrics include `conflict_mode`, `conflicts_detected`, `conflicts_resolved`, `grad_cos_mean_before`, `grad_cos_mean_after`, `grad_norm_before`, and `grad_norm_after`.

- [ ] **Step 2: Implement KD logit anchors**

Expose `LogitAnchorBank`, `build_logit_anchor_stats`, `merge_logit_anchor_stats`, `logit_anchor_kd_loss`, and `logit_anchor_stats_payload_size_bytes`. Use confidence/margin/correctness gating before uploading anchors.

- [ ] **Step 3: Implement activation-token codec**

Expose `ActivationTokenCodec.encode(features)` returning token metadata with shape, payload bytes, compression ratio, route, bits, quantization error, and a decode path for quantized tokens. Routes: `none`, `quantized`, `sketch`, `lowrank`.

- [ ] **Step 4: Implement style-code helper**

Expose `style_code_from_stats(stats, dim)` and allow `StylePacket` to carry optional `style_code` in `to_dict`, `from_dict`, and `vector`.

- [ ] **Step 5: Run utility tests and verify GREEN**

Run the same `pytest` command from Task 1.

### Task 3: Trainer And CLI Integration

**Files:**
- Modify: `E:/type10-7/code/train.py`
- Modify: `E:/type10-7/code/federated/fed_trainer.py`
- Modify: `E:/type10-7/code/configs/fedcvs_rffi_vmb.yaml`
- Modify: `E:/type10-7/code/tests/test_federated_trainer_smoke.py`
- Modify: `E:/type10-7/code/tests/test_federated_train_integration.py`

- [ ] **Step 1: Write failing trainer/config tests**

Tests must assert that `train_mode=split_bex02`, `--fl_conflict_agg`, `--use_logit_anchors`, `--activation_token_route`, `--fl_style_code_dim`, `--fl_probe_every`, and `--feature_probe_export` are exposed. A tiny VMB round must write config/log/metrics fields for KD anchors, activation-token payload, conflict aggregation, and approximation label.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
conda activate ssr-gpu
cd E:/type10-7/code
python -m pytest tests/test_federated_train_integration.py tests/test_federated_trainer_smoke.py -q
```

- [ ] **Step 3: Add CLI knobs without changing fedavg/fedprox defaults**

Add choices and flags for local virtual BEX02, conflict aggregation, KD anchors, activation tokens, style code, feature probes, and Stage1 LR multiplier. `split_bex02` must be explicit and should not silently claim strict centralized equivalence.

- [ ] **Step 4: Wire trainer state**

Create logit anchor bank, activation-token codec, style-code enrichment, conflict-aware aggregation, and `split_bex02` VMB reuse. Add config snapshot sections: `distillation`, `compression`, `conflict_aggregation`, and `feature_probe`.

- [ ] **Step 5: Add logging and metrics**

Append metrics for `train_loss_logit_kd`, `kd_active`, `anchor_count`, `anchor_payload_bytes`, `activation_token_payload_bytes`, `activation_token_compression_ratio`, `activation_token_quant_error`, `vmb_conflicts_detected`, and `vmb_conflicts_resolved`.

- [ ] **Step 6: Run trainer tests and verify GREEN**

Run the same focused tests from Step 2.

### Task 4: Experiment Matrix And Review Gate

**Files:**
- Create: `E:/type10-7/code/scripts/launch_split_bex02_alternatives_8gpu.sh`
- Create: `E:/type10-7/automation_reports/CV-SincNet/20260528_split_bex02_alternatives/report.md`
- Modify: `E:/type10-7/code/SYNC_MANIFEST.txt`

- [ ] **Step 1: Add 8-GPU launcher**

The launcher defines one formal run per GPU: `SBX02_LVMB_r010`, `SBX02_PROTO_r010`, `SBX02_FISHR_r010`, `SBX02_STYLE_r010`, `SBX02_KDLOGIT_r010`, `SBX02_QTOKEN_r010`, `SBX02_SATCE_r010`, and `SBX02_COMBO_r010`. Every formal command must include `--wisig_train_ratio 0.1 --epochs 200 --fl_rounds 200 --fl_client_key receiver`.

- [ ] **Step 2: Add report template**

The report must include objective, hypothesis, files changed, verification commands, exact future N607 command template, GPU allocation, metrics to watch, risks, and pass/fail criteria. Mark server launch as `not_started` until local verification and sync happen.

- [ ] **Step 3: Run local verification**

Run:

```powershell
conda activate ssr-gpu
cd E:/type10-7/code
python -m py_compile train.py federated/fed_trainer.py federated/fedcvs_vmb.py federated/gradient_stats.py federated/distill_anchors.py federated/activation_tokens.py model_dual_cvsincnet.py
python -m pytest tests/test_split_bex02_alternatives.py tests/test_fedcvs_vmb.py tests/test_federated_trainer_smoke.py tests/test_federated_train_integration.py tests/test_federated_aggregation.py tests/test_fed_pvs_style_bank.py tests/test_federated_d_style_plumbing.py -q
```

- [ ] **Step 4: Dispatch completion supervisor and code-review subagents**

Completion supervisor checks SBX-01 through SBX-11 against `split_bex02_alternatives_traceability.md`. Code reviewer checks behavior preservation, approximation honesty, privacy/payload accounting, and tests.

