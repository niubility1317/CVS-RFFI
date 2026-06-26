# FedCVS-RFFI-VMB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the FedCVS-RFFI-VMB design report as an opt-in federated training route with traceable prototypes, domain-balanced synchronized gradient aggregation, diagnostics, and evaluation helpers while preserving existing defaults.

**Architecture:** Add a focused `federated.fedcvs_vmb` utility module for prototype banks, balanced sampling, gradient aggregation, and server updates. Wire it into `FederatedTrainer` only when `train_mode=fedcvs_vmb`; keep old FedAvg/FedProx state aggregation untouched. Add an optional `tx_adv_head` to the dual model only when explicitly enabled.

**Tech Stack:** Python, PyTorch, unittest/pytest, existing `train.py`, `model_dual_cvsincnet.py`, `federated/`, and `tests/`.

---

### Task 1: Tests For VMB Helpers And Opt-In CLI

**Files:**
- Create: `E:/type10-7/code/tests/test_fedcvs_vmb.py`
- Modify: `E:/type10-7/code/tests/test_federated_aggregation.py`
- Modify: `E:/type10-7/code/tests/test_federated_train_integration.py`

- [ ] Write failing tests that import `federated.fedcvs_vmb`.
- [ ] Assert TX/RX prototype banks update normalized EMA prototypes and counts.
- [ ] Assert domain-balanced aggregation gives each receiver/domain equal total weight.
- [ ] Assert `train.py` exposes `fedcvs_vmb`, VMB CLI flags, and keeps `receiver`/0.1/200 defaults.
- [ ] Run:
  `conda activate ssr-gpu; python -m pytest tests/test_fedcvs_vmb.py tests/test_federated_aggregation.py tests/test_federated_train_integration.py -q`
  Expected now: fail because the new module and CLI are not implemented.

### Task 2: Core VMB Utility Module

**Files:**
- Create: `E:/type10-7/code/federated/fedcvs_vmb.py`
- Modify: `E:/type10-7/code/federated/fed_aggregate.py`
- Modify: `E:/type10-7/code/federated/__init__.py`

- [ ] Implement `FedCVSVMBPrototypeBank` with TX class prototypes, RX client prototypes, EMA, count tracking, clipping, and JSON summaries.
- [ ] Implement `prototype_contrastive_loss` for TX and RX prototype CE losses.
- [ ] Implement transmitter-balanced index selection from client sample metadata.
- [ ] Implement domain-balanced client sampling and domain-balanced gradient/state weighting helpers.
- [ ] Implement aggregate gradient and server SGD-momentum update helpers.
- [ ] Run the Task 1 helper tests until green.

### Task 3: Model And Trainer Wiring

**Files:**
- Modify: `E:/type10-7/code/model_dual_cvsincnet.py`
- Modify: `E:/type10-7/code/federated/fed_trainer.py`
- Modify: `E:/type10-7/code/train.py`

- [ ] Add optional `tx_adv_head` on `z_dom` with GRL, controlled by `use_tx_adv_on_zdom`.
- [ ] Add `train_mode=fedcvs_vmb` to CLI and `FederatedTrainer`.
- [ ] Add VMB stage flags: `fl_vmb_stage`, `fl_vmb_pretrain_rounds`, `fl_vmb_batches_per_client`, `fl_vmb_server_lr`, `fl_vmb_server_momentum`, `fl_vmb_freeze_rx_stage2`, `fl_vmb_domain_balanced_sampling`, `fl_vmb_domain_balanced_aggregation`, prototype temperatures and loss weights.
- [ ] In VMB mode, collect one/few batches of gradients from the same global state, aggregate gradients domain-balanced, then apply a server update.
- [ ] Add VMB prototype losses and prototype-stat collection to local objective.
- [ ] Keep FedAvg/FedProx path exactly on `aggregate_state_dicts`.
- [ ] Run smoke tests until green.

### Task 4: Config, Evaluation Helpers, And Traceability Updates

**Files:**
- Create: `E:/type10-7/code/configs/fedcvs_rffi_vmb.yaml`
- Create: `E:/type10-7/code/evaluation/fedcvs_vmb_probe.py`
- Create: `E:/type10-7/code/evaluation/fedcvs_vmb_analysis.py`
- Modify: `E:/type10-7/code/analysis/fedcvs_rffi_vmb_traceability.md`

- [ ] Add config template matching the report's MVP-1, Stage 1, Stage 2, and ablation route.
- [ ] Add probe helpers for `Acc(y|z_t)`, `Acc(d|z_t)`, `Acc(d|z_r)`, `Acc(y|z_r)`.
- [ ] Add prototype/gradient analysis helpers for compactness, drift, and cosine summaries.
- [ ] Update traceability statuses and verification evidence.

### Task 5: Snapshot, Verify, And Review

**Files:**
- Create: `E:/type10-7/code/snapshots/<timestamp>_fedcvs_vmb/`
- Modify: `E:/type10-7/code/SYNC_MANIFEST.txt` only if syncing is requested.

- [ ] Create a local snapshot of changed code/config/script files before any remote use.
- [ ] Run:
  `conda activate ssr-gpu; python -m py_compile train.py model_dual_cvsincnet.py federated/fed_trainer.py federated/fedcvs_vmb.py federated/fed_aggregate.py evaluation/fedcvs_vmb_probe.py evaluation/fedcvs_vmb_analysis.py`
- [ ] Run targeted pytest suite.
- [ ] Spawn sub-agent review for design-report correctness.
- [ ] Spawn sub-agent analysis for cross-domain rationale and likely performance impact.
