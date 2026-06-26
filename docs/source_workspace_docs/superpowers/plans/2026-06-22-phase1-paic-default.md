# Phase1 PAIC Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make future Phase1 optimizer-generated experiments default to the new CVS-SAT-PAIC star-ground channel enhancement, while preserving explicit CEN51 refresh controls as controls.

**Architecture:** Add validator-level requirements for launchable Phase1 Safe-SSDG rows, update test helpers to emit the new default fields and command flags, and push the same rule into prompt, contract, manifest, and state so the standing automation cannot drift.

**Tech Stack:** Python validator/tests, Markdown control files, JSON state, local Conda env `ssr-gpu`.

---

### Task 1: Red Tests

**Files:**
- Modify: `code/tests/test_optimizer_workflow_tools.py`

- [x] **Step 1: Add failing validator test**

The test strips PAIC star-ground fields from eight Phase1 rows and expects `phase1_star_ground_aug_default_required`, `phase1_star_ground_aug_requires_paic_route_family`, `phase1_star_ground_aug_requires_concat_ce_only_mode`, and `phase1_star_ground_aug_requires_schedule`.

- [x] **Step 2: Add failing control-surface test**

The test requires active prompt, contract, manifest, and state to expose `phase1_star_ground_aug_default_enabled`, `CVS-SAT-PAIC`, and `concat_sat_ce_only`.

- [x] **Step 3: Verify RED**

Run:

```powershell
conda activate ssr-gpu; $env:PYTHONPATH='E:\type10-7\code;E:\type10-7\tools;E:\type10-7'; python -m pytest -p no:cacheprovider -q code\tests\test_optimizer_workflow_tools.py::test_validate_matrix_rejects_phase1_without_default_paic_star_ground_aug code\tests\test_optimizer_workflow_tools.py::test_phase1_control_surfaces_require_default_paic_star_ground_aug
```

Expected: 2 failed because validator/control surfaces do not yet enforce the rule.

### Task 2: Validator And Test Helper

**Files:**
- Modify: `tools/optimizer_validate_matrix.py`
- Modify: `code/tests/test_optimizer_workflow_tools.py`

- [x] **Step 1: Add PAIC Phase1 constants**

Add the canonical PAIC schedule and required command/field tokens to the validator.

- [x] **Step 2: Add Phase1 star-ground default validation**

For current Phase1 Safe-SSDG rows, require default PAIC fields unless the row is an explicit CEN51 refresh control exemption.

- [x] **Step 3: Update Phase1 helper**

Make `_phase1_training_item()` emit PAIC default fields and command flags.

- [x] **Step 4: Verify GREEN for focused tests**

Run the two red tests again and expect pass.

### Task 3: Control Surfaces

**Files:**
- Modify: `automation_reports/CV-SincNet/automation_prompt_backups/20260615_001820_stage2_closed_loop_v4/stage2_prompt.md`
- Modify: `tools/optimizer_workflow_contract.md`
- Modify: `tools/optimizer_control_manifest.md`
- Modify: `automation_reports/CV-SincNet/stage2_optimizer_state.json`

- [x] **Step 1: Prompt update**

Add Phase1 optimizer instructions that PAIC star-ground augmentation is default-on and must be explored in idea cards.

- [x] **Step 2: Contract update**

Add required Phase1 candidate fields and command flags, plus the CEN51 refresh control exemption boundary.

- [x] **Step 3: Manifest update**

Record ownership of the PAIC default across prompt/contract/validator/state.

- [x] **Step 4: State update**

Add machine-readable `phase1_star_ground_aug_default` under `phase1_ground_dg_direction`.

### Task 4: Verification

**Files:**
- Validate all modified files.

- [x] **Step 1: Run focused optimizer tests**

```powershell
conda activate ssr-gpu; $env:PYTHONPATH='E:\type10-7\code;E:\type10-7\tools;E:\type10-7'; python -m pytest -p no:cacheprovider -q code\tests\test_optimizer_workflow_tools.py
```

- [x] **Step 2: Run syntax and JSON checks**

```powershell
conda activate ssr-gpu; $env:PYTHONPATH='E:\type10-7\code;E:\type10-7\tools;E:\type10-7'; python -m py_compile tools\optimizer_validate_matrix.py tools\optimizer_state_current_view.py
python -m json.tool automation_reports\CV-SincNet\stage2_optimizer_state.json > $null
```

- [x] **Step 3: Confirm no N607 action**

Record that this was local-only: no SSH, SCP, launch, kill, or remote cleanup.
