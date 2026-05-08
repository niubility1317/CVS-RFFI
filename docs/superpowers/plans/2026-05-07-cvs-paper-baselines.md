# CVS Paper Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build fair CVS-RFFI comparison training entries for the four paper baselines and a basic CVCNN baseline.

**Architecture:** Reuse the root WiSig/CVS-RFFI split and named test loaders through shared baseline utilities. Keep paper-specific models and losses isolated in their existing packages, with one CVS training entry per method and validation-gated test evaluation.

**Tech Stack:** Python, PyTorch, `unittest`, existing `dataset_wisig.py` split utilities.

---

### Task 1: Regression Tests

**Files:**
- Create: `tests/test_cvs_paper_baselines.py`

- [x] **Step 1: Write failing tests**

```python
import unittest
import torch

class TestPaperBaselineParity(unittest.TestCase):
    def test_riei_mi_uses_signed_cosine_from_paper(self): ...
    def test_drift_style_transfer_center_module_clusters_same_receiver(self): ...
    def test_best_val_gate_runs_tests_only_on_improvement(self): ...
    def test_cvcnn_baseline_forward_outputs_logits(self): ...
```

- [x] **Step 2: Verify tests fail**

Run: `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m unittest tests.test_cvs_paper_baselines -v`

Expected: missing modules/functions and RIEI MI assertion failure.

### Task 2: Shared CVS Data And Trainer

**Files:**
- Create: `baselines/common/cvs_data.py`
- Create: `baselines/common/cvs_trainer.py`

- [ ] **Step 1: Add data args and loader builders**

Expose WiSig/CVS arguments matching root `train.py` defaults and collate samples into dictionaries with `iq`, `label`, `domain`, `receiver`, and `meta`.

- [ ] **Step 2: Add validation-gated trainer**

Evaluate validation every epoch. Run named test loaders only when validation accuracy improves. Save `best_by_val.pt` and `metrics.json`.

### Task 3: Paper-Parity Loss Fixes

**Files:**
- Modify: `baselines/riei/losses.py`
- Modify: `baselines/drift/losses.py`

- [ ] **Step 1: Change RIEI MI to signed cosine**

Remove `abs()` so the loss follows the paper equation.

- [ ] **Step 2: Add DRIFT receiver style/center loss**

Add `receiver_style_transfer_center_loss` as the receiver-specific style regularizer and use it in `compute_drift_loss`.

### Task 4: Baseline Training Entrypoints

**Files:**
- Create: `baselines/cvcnn/model.py`
- Create: `baselines/cvcnn/train_cvs.py`
- Create: `baselines/riei/train_cvs.py`
- Create: `baselines/drift/train_cvs.py`
- Create: `baselines/receiver_agnostic_rffi/train_cvs.py`
- Create: `baselines/receiver_agnostic_rffi/finetune_cvs.py`
- Create: `baselines/tifs2025_channel_receiver_rffi/train_cvs.py`

- [ ] **Step 1: Add CVCNN**

Basic widely-linear complex convolution network with only cross-entropy training.

- [ ] **Step 2: Add CVS entrypoints**

Use shared loaders and paper-specific losses. For TIFS 2025, run NT-Xent pretraining, Siamese fine-tuning, and single-branch validation/test.

### Task 5: Documentation And Verification

**Files:**
- Modify: `baselines/README.md`

- [ ] **Step 1: Document commands**

List CVS training commands and note that only data split differs from paper datasets.

- [ ] **Step 2: Run verification**

Run focused `unittest` and compile/import checks with the PyTorch environment.
