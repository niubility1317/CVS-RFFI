# StyleBank Heterogeneous GRL Collaborative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add StyleBank virtual heterogeneous receiver training plus paper-inspired virtual collaborative inference to federated CVS-RFFI.

**Architecture:** Keep pure fusion math in `federated/reliability_fusion.py`; keep StyleBank-aware eval/training orchestration in `federated/fed_trainer.py`; expose controls through `train.py`, launcher, docs, and tests. Since `E:\type10-7\code` is not guaranteed to be a git repository, replace commit steps with a timestamped local snapshot.

**Tech Stack:** Python, PyTorch, unittest/pytest, local `ssr-gpu` conda env, existing FederatedTrainer/StyleBank modules.

---

### Task 1: Traceability And Test Targets

**Files:**
- Create: `E:/type10-7/code/analysis/stylebank_hetero_grl_collab_traceability.md`
- Modify: `E:/type10-7/code/tests/test_fed_pvs_proto_fusion.py`
- Modify: `E:/type10-7/code/tests/test_federated_d_style_plumbing.py`
- Modify: `E:/type10-7/code/tests/test_federated_train_integration.py`

- [ ] Write traceability rows R1-R10 from the design spec.
- [ ] Add failing fusion tests for `collaborative_probability_fusion`.
- [ ] Add failing trainer test for `global_style_collab_fusion`.
- [ ] Add failing integration assertions for CLI/config/log/launcher/docs reachability.
- [ ] Run focused pytest and confirm expected failures are missing-symbol or missing-token failures.

### Task 2: Pure Collaborative Fusion Utilities

**Files:**
- Modify: `E:/type10-7/code/federated/reliability_fusion.py`
- Modify: `E:/type10-7/code/federated/__init__.py`

- [ ] Implement `collaborative_reliability_from_probabilities(p)` using normalized confidence `1 - entropy/log(C)`.
- [ ] Implement `collaborative_probability_fusion(p_base, aux_probabilities, mode, aux_reliabilities, base_weight, max_aux_weight)`.
- [ ] Validate shapes and normalize all probability tensors.
- [ ] Export both helpers from `federated/__init__.py`.
- [ ] Run the fusion tests and confirm they pass.

### Task 3: Trainer Style Collaborative Evaluation

**Files:**
- Modify: `E:/type10-7/code/federated/fed_trainer.py`

- [ ] Import the new fusion helpers.
- [ ] Add `_style_collab_enabled`, `_style_packet_reliability`, `_style_collab_view_packets`, and `_evaluate_style_collab_fusion`.
- [ ] Call `_evaluate_style_collab_fusion()` inside `_evaluate()`.
- [ ] Add `global_style_collab_fusion` to per-round rows, metrics CSV, config snapshot, and summary.
- [ ] Run trainer-focused tests and fix reachable-path failures.

### Task 4: Training Semantics Tightening

**Files:**
- Modify: `E:/type10-7/code/federated/fed_trainer.py`
- Modify: `E:/type10-7/code/tests/test_federated_d_style_plumbing.py`

- [ ] Ensure default StyleBank training batches use sequential constructed `d_style` labels `0..K`.
- [ ] Preserve raw target receiver labels in `d_raw` and metadata for diagnostics.
- [ ] Ensure GRL/Fishr receive `d_style` only after StyleBank DG maturity gates.
- [ ] Run `test_federated_d_style_plumbing.py`.

### Task 5: CLI, Launcher, And Docs

**Files:**
- Modify: `E:/type10-7/code/train.py`
- Modify: `E:/type10-7/code/scripts/run_fed_fl82_validation_4gpu.sh`
- Modify: `E:/type10-7/code/docs/federated_style_transfer_settings.md`
- Modify: `E:/type10-7/code/tests/test_federated_train_integration.py`

- [ ] Add CLI flags: `use_style_collab_eval`, `style_collab_views`, `style_collab_fusion`, `style_collab_base_weight`, `style_collab_max_aux_weight`.
- [ ] Add FL82 launcher variant with conservative StyleBank plus collaborative eval.
- [ ] Document style-transfer settings and paper-inspired collaborative inference.
- [ ] Run `train.py --help` and launcher dry-run.

### Task 6: Verification And Snapshot

**Files:**
- Create: `E:/type10-7/code/snapshots/<timestamp>_stylebank_hetero_grl_collab/`
- Update: `E:/type10-7/code/analysis/stylebank_hetero_grl_collab_traceability.md`

- [ ] Run focused pytest under `ssr-gpu`.
- [ ] Run `py_compile` for modified Python files.
- [ ] Run `bash -n` and dry-run for the FL82 launcher.
- [ ] Compute hashes for changed files.
- [ ] Create a timestamped local snapshot because the code tree may be `NOT_GIT`.
- [ ] Mark traceability rows verified/deferred/rejected/blocked with evidence.
