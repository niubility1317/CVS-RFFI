# DADDA Cross-Receiver Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-faithful reproduction surface for Feng et al., "Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment."

**Architecture:** Keep this reproduction separate from CVS deployment extensions. Add `paper_reproduction/dadda_cross_receiver/` for DADDA model, losses, protocol, data tasking, and a gated training entrypoint; keep CVS metrics in `paper_reproduction/cvs_aligned/` only if a later extension is explicitly requested.

**Tech Stack:** Python, PyTorch, pytest, WiSig compact-pkl loader from `paper_reproduction.common.wisig_runtime`.

---

## File Structure

- Create `paper_reproduction/dadda_cross_receiver/model.py`: ResNet18-style IQ feature extractor `G_f`, multiscale feature extractor `G_m`, two-layer classifier `G_l`, and composed `DADDANet`.
- Create `paper_reproduction/dadda_cross_receiver/losses.py`: MMD, LMMD, dynamic adaptive factor alpha, and total DADDA objective.
- Create `paper_reproduction/dadda_cross_receiver/data.py`: ManySig task builder for the 12 Table II receiver-transfer tasks and the day-domain control task.
- Create `paper_reproduction/dadda_cross_receiver/train.py`: dry-run payload, source-only/proposed smoke runner, formal-run gate, and JSON output writer.
- Create `paper_reproduction/dadda_cross_receiver/paper_checklist.md`: paper-to-code evidence matrix.
- Create `paper_reproduction/configs/dadda_cross_receiver_manysig_paper_faithful.json`: paper-faithful configuration with unresolved data path left explicit.
- Modify `paper_reproduction/README.md`: register DADDA as a paper-original baseline and state that it is closed-set UDA, not CVS Stage2 evidence.
- Modify `paper_reproduction/paper_original_matrix.md`: add DADDA row family and current status.
- Create `tests/test_dadda_cross_receiver.py`: unit and smoke tests for model shapes, loss formulas, protocol guards, and smoke training.

### Task 1: Core Model

**Files:**
- Create: `paper_reproduction/dadda_cross_receiver/__init__.py`
- Create: `paper_reproduction/dadda_cross_receiver/model.py`
- Test: `tests/test_dadda_cross_receiver.py`

- [ ] **Step 1: Write model shape tests**

Run target: `python -m pytest tests/test_dadda_cross_receiver.py::test_dadda_model_outputs_paper_named_modules -q`

Expected first failure: `ModuleNotFoundError` or missing `DADDANet`.

- [ ] **Step 2: Implement model modules**

Implement `DADDANet(num_classes=6, feature_dim=128, multiscale_dim=128)` returning a dict with `global_features`, `local_features`, and `logits`.

- [ ] **Step 3: Verify model tests**

Run: `python -m pytest tests/test_dadda_cross_receiver.py::test_dadda_model_outputs_paper_named_modules -q`

Expected: PASS.

### Task 2: MMD, LMMD, and Dynamic Objective

**Files:**
- Create: `paper_reproduction/dadda_cross_receiver/losses.py`
- Test: `tests/test_dadda_cross_receiver.py`

- [ ] **Step 1: Write formula tests**

Cover Eq. (2) MMD symmetry/non-negativity, Eq. (3)-(4) LMMD source labels plus target soft labels, Eq. (5) alpha in `[0,1]`, and Eq. (9) total loss `CE + lambda*((1-alpha)*MMD + alpha*LMMD)`.

- [ ] **Step 2: Implement losses**

Use differentiable PyTorch tensors and an RBF kernel with deterministic bandwidth fallback.

- [ ] **Step 3: Verify loss tests**

Run: `python -m pytest tests/test_dadda_cross_receiver.py::test_dadda_dynamic_objective_combines_ce_mmd_lmmd -q`

Expected: PASS.

### Task 3: Paper Protocol and Data Tasking

**Files:**
- Create: `paper_reproduction/dadda_cross_receiver/data.py`
- Create: `paper_reproduction/dadda_cross_receiver/train.py`
- Create: `paper_reproduction/configs/dadda_cross_receiver_manysig_paper_faithful.json`
- Test: `tests/test_dadda_cross_receiver.py`

- [ ] **Step 1: Write dry-run and task-builder tests**

Assert dry-run declares paper title, method id `dadda_cross_receiver`, Table II tasks, source labels are training labels, target labels are evaluation-only, and CVS extension is false.

- [ ] **Step 2: Implement task builder and dry-run payload**

Support `1-1->8-8` receiver tasks and `d01->d23` day-domain control using WiSig compact metadata.

- [ ] **Step 3: Gate formal CLI**

Non-dry-run CLI must fail before writing output unless a real WiSig pkl path is supplied and `--formal` is explicit.

### Task 4: Smoke Runner

**Files:**
- Modify: `paper_reproduction/dadda_cross_receiver/train.py`
- Test: `tests/test_dadda_cross_receiver.py`

- [ ] **Step 1: Write smoke training test**

Use a synthetic ManySig compact fixture, one receiver-transfer task, methods `source_only` and `proposed`, two epochs, CPU, and one batch per epoch.

- [ ] **Step 2: Implement smoke runner**

Train source-only with CE. Train proposed with DADDA objective using unlabeled target inputs and target labels only for final accuracy audit.

- [ ] **Step 3: Verify smoke**

Run: `python -m pytest tests/test_dadda_cross_receiver.py -q`

Expected: PASS.

### Task 5: Documentation and Paper-to-Code Audit

**Files:**
- Create: `paper_reproduction/dadda_cross_receiver/paper_checklist.md`
- Modify: `paper_reproduction/README.md`
- Modify: `paper_reproduction/paper_original_matrix.md`

- [ ] **Step 1: Add paper checklist**

Use columns: paper item, paper evidence, code evidence, test/result evidence, current status.

- [ ] **Step 2: Update reproduction docs**

Register DADDA as paper-faithful closed-set single-source UDA. State explicitly that same-label-space target receiver adaptation is not CVS Stage2-C new-class enrollment.

- [ ] **Step 3: Run verification**

Run: `python -m pytest tests/test_dadda_cross_receiver.py -q` and `git diff --check`.

Expected: PASS and no whitespace errors.

## Self-Review

- Spec coverage: model scaffold, losses, data protocol, training smoke, documentation, and paper-to-code audit are covered. Full paper reproduction remains pending for Table II baselines and real metrics, Table III/IV/V/VI, and Fig.5-8.
- Placeholder scan: no task relies on `TBD`; the only intentionally unresolved field is the real WiSig pkl path in the config.
- Type consistency: module names and output keys are fixed across tasks.
