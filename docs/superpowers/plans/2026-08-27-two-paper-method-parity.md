# Two-Paper Method-Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the two paper-reproduction pipelines except for external data assets, while recording every non-disclosed choice as an explicit reproducible default.

**Architecture:** Each paper receives independent configuration, train/evaluate entrypoints and result metadata. Tweak builds a shared-encoder metric-learning pipeline around its existing core. Hu replaces the unverifiable encoder interpretation with a Figure-6-first model and adds pure-torch preprocessing, augmentation, training and evaluation components. Neither path imports CVS-aligned code.

**Tech Stack:** Python 3, PyTorch, pytest, JSON configuration.

**Spec:** `docs/superpowers/specs/2026-08-27-two-paper-strict-method-parity-design.md`

## Global Constraints

- Work only in the existing isolated worktree and branch.
- Keep `paper_reproduction` independent of `cvs_aligned` and `feature_separation_crossrx`.
- Do not download datasets, launch training, or contact N607.
- All unpublished choices are serialized as `UNPUBLISHED_DEFAULT` metadata.

---

### Task 1: Shared method-configuration metadata

**Files:**
- Create: `paper_reproduction/gaskin_tweak_2023/strict_method.json`
- Create: `paper_reproduction/gaskin_tweak_2023/method_config.py`
- Create: `paper_reproduction/hu_feature_separation_2024/strict_method.json`
- Create: `paper_reproduction/hu_feature_separation_2024/method_config.py`
- Test: `tests/test_paper_method_configs.py`

- [ ] Write tests that reject malformed unpublished-default records and assert result metadata contains each configured default.
- [ ] Run the new tests and observe failure because the modules do not exist.
- [ ] Implement immutable config loading and `method_metadata()` using UTF-8 JSON.
- [ ] Run `pytest -p no:cacheprovider tests/test_paper_method_configs.py -q` and require pass.
- [ ] Commit the configuration slice.

### Task 2: Tweak training, domain calibration and grouped decisions

**Files:**
- Modify: `paper_reproduction/gaskin_tweak_2023/calibration.py`
- Create: `paper_reproduction/gaskin_tweak_2023/training.py`
- Create: `paper_reproduction/gaskin_tweak_2023/evaluation.py`
- Modify: `paper_reproduction/gaskin_tweak_2023/metrics.py`
- Test: `tests/test_gaskin_tweak_pipeline.py`

- [ ] Write failing tests for shared-encoder triplet gradients, fixed-N-per-class calibration, domain-separated calibration states, M=10 aggregation, checkpoint selection and five balanced open-set trials.
- [ ] Run the targeted test file and observe the intended missing-import or assertion failure.
- [ ] Implement the minimal training, calibration, group-decision and metric functions; use batch-hard and the frozen five-value LR grid from `strict_method.json`.
- [ ] Run both Tweak test files and require pass.
- [ ] Commit the Tweak slice.

### Task 3: Hu representation, Figure-6-first network and loss metadata

**Files:**
- Create: `paper_reproduction/hu_feature_separation_2024/preprocess.py`
- Modify: `paper_reproduction/hu_feature_separation_2024/representation.py`
- Modify: `paper_reproduction/hu_feature_separation_2024/model.py`
- Modify: `paper_reproduction/hu_feature_separation_2024/losses.py`
- Test: `tests/test_hu_feature_separation_method.py`

- [ ] Write failing tests for deterministic IQ preprocessing, Welch defaults, exactly five residual stages, one shared-feature attention point and positive loss composition.
- [ ] Run the target test file and observe the intended failure.
- [ ] Implement the Figure-6-first encoder and documented preprocessing/PSD defaults without importing project-specific datasets.
- [ ] Run Hu representation/model/loss tests and require pass.
- [ ] Commit the Hu core slice.

### Task 4: Hu augmentation, training, fine-tuning and evaluation matrix

**Files:**
- Create: `paper_reproduction/hu_feature_separation_2024/augmentation.py`
- Create: `paper_reproduction/hu_feature_separation_2024/training.py`
- Create: `paper_reproduction/hu_feature_separation_2024/evaluation.py`
- Modify: `paper_reproduction/hu_feature_separation_2024/finetune.py`
- Test: `tests/test_hu_feature_separation_pipeline.py`

- [ ] Write failing tests for seeded channel augmentation, one training epoch with validation checkpointing, frozen RX branch/BN during TX-only fine-tune and all required evaluation matrix labels.
- [ ] Run the target test file and observe the intended failure.
- [ ] Implement minimal pure-torch augmentation, trainer, fine-tune freezing and metric matrix construction.
- [ ] Run both existing and new Hu tests and require pass.
- [ ] Commit the Hu pipeline slice.

### Task 5: Reverse audit and delivery

**Files:**
- Modify: `docs/reproduction/two-paper-strict-method-parity/traceability.md`
- Modify: `docs/reproduction/two-paper-strict-method-parity/progress.md`
- Test: `tests/test_gaskin_tweak_2023.py`, `tests/test_gaskin_tweak_pipeline.py`, `tests/test_hu_feature_separation_2024.py`, `tests/test_hu_feature_separation_method.py`, `tests/test_hu_feature_separation_pipeline.py`

- [ ] Run the focused suite with pytest cache disabled and record the exact result.
- [ ] Review every traceability row; set only tested requirements to `verified`, and document each unpublished default.
- [ ] Run `git diff --check`, commit the traceability completion and push.
- [ ] Independently compare `git rev-parse HEAD` with the remote branch OID.
