# Baseline Pseudo Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in pseudo-label self-training module to all CVS baseline comparison trainers.

**Architecture:** Implement pseudo-label filtering and unlabeled CE in `baselines.common.pseudo_labels`, then expose the same CLI switches in every baseline CVS trainer. Baseline behavior remains unchanged unless `--use_pseudo_labels` is passed.

**Tech Stack:** Python, PyTorch, unittest, existing `baselines.common.cvs_trainer` and CVS dataloaders.

---

### Task 1: Shared Pseudo-Label Module

**Files:**
- Create: `E:/type10-7/baselines/common/pseudo_labels.py`
- Test: `E:/type10-7/tests/test_baseline_pseudo_labels.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert the parser exposes pseudo-label controls, the epoch gate is disabled before `pseudo_start_epoch`, and confident samples produce CE loss plus selection metrics.

- [ ] **Step 2: Verify tests fail**

Run: `conda activate ssr-gpu; python -m unittest tests.test_baseline_pseudo_labels -v`
Expected: FAIL because `baselines.common.pseudo_labels` does not exist yet.

- [ ] **Step 3: Implement minimal shared helpers**

Create `PseudoLabelConfig`, `add_pseudo_label_args`, `build_pseudo_label_config`, `PseudoLabelBatchResult`, and `compute_pseudo_label_loss`.

- [ ] **Step 4: Verify tests pass**

Run the same unittest command. Expected: PASS.

### Task 2: Trainer Integration Surface

**Files:**
- Modify: `E:/type10-7/baselines/common/cvs_trainer.py`
- Test: `E:/type10-7/tests/test_baseline_pseudo_labels.py`

- [ ] **Step 1: Write failing integration test**

Assert `run_validation_gated_training` can accept a pseudo-label callback, merge pseudo metrics into epoch train loss, and skip the callback before the start epoch.

- [ ] **Step 2: Implement callback hook**

Add optional `pseudo_step_fn` to `run_validation_gated_training`; after each supervised `train_step_fn`, call it with the same model, device, epoch, and step. Include returned `loss` in `train_loss` so logs reflect total training objective.

- [ ] **Step 3: Verify tests pass**

Run: `conda activate ssr-gpu; python -m unittest tests.test_baseline_pseudo_labels -v`

### Task 3: Wire CVS Baseline Entrypoints

**Files:**
- Modify: `E:/type10-7/baselines/cvcnn/train_cvs.py`
- Modify: `E:/type10-7/baselines/drift/train_cvs.py`
- Modify: `E:/type10-7/baselines/riei/train_cvs.py`
- Modify: `E:/type10-7/baselines/receiver_agnostic_rffi/train_cvs.py`
- Modify: `E:/type10-7/baselines/tifs2025_channel_receiver_rffi/train_cvs.py`
- Test: `E:/type10-7/tests/test_baseline_pseudo_labels.py`

- [ ] **Step 1: Add parser tests**

Assert every CVS baseline accepts `--use_pseudo_labels --pseudo_start_epoch 2 --pseudo_threshold 0.9 --lambda_pseudo 0.5`.

- [ ] **Step 2: Add parser and callback wiring**

Each trainer calls `add_pseudo_label_args`, creates `pseudo_cfg`, builds a pseudo loader from the training dataset when enabled, and passes a method-specific pseudo callback to `run_validation_gated_training`.

- [ ] **Step 3: Verify targeted tests pass**

Run: `conda activate ssr-gpu; python -m unittest tests.test_baseline_pseudo_labels tests.test_baseline_training_behaviors -v`

### Task 4: Final Verification

**Files:**
- Read-only verification across changed files.

- [ ] **Step 1: Run baseline behavior tests**

Run: `conda activate ssr-gpu; python -m unittest tests.test_baseline_pseudo_labels tests.test_baseline_training_behaviors -v`

- [ ] **Step 2: Run help smoke for all CVS trainers**

Run each trainer with `--help` to confirm CLI registration.

Note: The workspace root is not a git repository, so commit steps are intentionally omitted.
