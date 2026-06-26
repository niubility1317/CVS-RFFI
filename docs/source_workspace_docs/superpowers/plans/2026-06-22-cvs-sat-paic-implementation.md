# CVS-SAT-PAIC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the CVS-SAT-PAIC design as local code, matrix/report generation, protocol validation, traceability, and tests without launching N607.

**Architecture:** Add one small source module for PAIC route specs, gates, Stage2 protocol rows, and satellite metadata summaries. Add one tool script to emit JSON/Markdown artifacts from that module. Extend the optimizer validator only for PAIC-labelled rows so existing Stage2/OA-MSE validation remains stable.

**Tech Stack:** Python standard library, existing `tools/optimizer_validate_matrix.py`, pytest/unittest, local Conda env `ssr-gpu`.

---

### Task 1: PAIC Route Spec Tests

**Files:**
- Create: `tests/test_cvs_sat_paic_matrix.py`
- Create: `code/tests/test_paic_star_ground.py`

- [ ] **Step 1: Write failing tests**

Tests must import `cvsrffi.paic_star_ground` and assert:

```python
from cvsrffi.paic_star_ground import (
    PAIC_CURRICULUM_SCHEDULE,
    build_paic_matrix,
    summarize_satellite_meta,
)

def test_paic_schedule_matches_design():
    assert PAIC_CURRICULUM_SCHEDULE == (
        "1@0.30:mixed_orbit;"
        "41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;"
        "91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp"
    )

def test_paic_matrix_contains_central_federated_and_stage2_rows():
    payload = build_paic_matrix()
    ids = {row["candidate_id"] for row in payload["candidates"]}
    assert {"C2_PAIC_CURRICULUM_CE_ONLY", "C3_PAIC_LATE_WEAK_ALIGN", "F2_FL_PAIC_CURRICULUM", "S2C_PAIC_PROTOCOL_CHECK"} <= ids
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
conda activate ssr-gpu; python -m pytest -q tests/test_cvs_sat_paic_matrix.py code/tests/test_paic_star_ground.py
```

Expected: import failure for missing `cvsrffi.paic_star_ground`.

### Task 2: Implement PAIC Route Module

**Files:**
- Create: `code/cvsrffi/paic_star_ground.py`

- [ ] **Step 1: Implement dataclasses and builders**

Implement:

```python
PAIC_CURRICULUM_SCHEDULE = "1@0.30:mixed_orbit;41@0.60:mixed_orbit*2,low_elev_leo,rain_leo;91@0.80:mixed_orbit,low_elev_leo,rain_leo,storm_mp"
PAIC_SCENARIOS = ("clear_leo", "low_elev_leo", "rain_leo", "storm_mp", "mixed_orbit")

def build_paic_matrix() -> dict:
    return {"schema": "cvs_sat_paic_matrix_v1", "candidates": [...]}
```

Rows must cover C0-C5, F0-F4, S2-A/B/C and include route metadata, CLI args, gates, and protocol fields.

- [ ] **Step 2: Implement satellite metadata summary**

Implement:

```python
def summarize_satellite_meta(meta: Mapping[str, Any], scenario: str | None = None) -> dict:
    ...
```

It must return count, quantiles for numeric fields, and ratios for `orbit` and `state`.

- [ ] **Step 3: Verify GREEN**

Run the tests from Task 1.

### Task 3: PAIC Artifact Generator

**Files:**
- Create: `tools/cvs_sat_paic_matrix.py`
- Modify: `tests/test_cvs_sat_paic_matrix.py`

- [ ] **Step 1: Write failing CLI tests**

Assert the tool renders JSON and Markdown containing `CVS-SAT-PAIC`, PAIC schedule, C/F/S2 rows, and evidence-boundary text.

- [ ] **Step 2: Implement CLI**

The tool must support:

```powershell
python tools/cvs_sat_paic_matrix.py --output-root tmp_paic
```

Outputs:

```text
tmp_paic/cvs_sat_paic_matrix.json
tmp_paic/cvs_sat_paic_report.md
```

- [ ] **Step 3: Verify GREEN**

Run:

```powershell
conda activate ssr-gpu; python -m pytest -q tests/test_cvs_sat_paic_matrix.py
```

### Task 4: PAIC Validator Checks

**Files:**
- Modify: `tools/optimizer_validate_matrix.py`
- Modify: `code/tests/test_optimizer_workflow_tools.py` or create top-level focused test if lower-risk.

- [ ] **Step 1: Write failing validator tests**

Tests must show a PAIC row is rejected when it lacks:

```text
route_family=CVS-SAT-PAIC
clean_view_role=control_only
target_channel_view=satellite/LEO for Stage2 rows
unknown_query_eval_only=true for Stage2 rows
fl_baseline_view_ce_only=true for FL PAIC rows
```

- [ ] **Step 2: Implement validator helper**

Add `paic_required_field_issues(item)` and call it from the existing validate path.

- [ ] **Step 3: Verify GREEN**

Run focused validator tests plus the PAIC matrix tests.

### Task 5: Traceability and Verification

**Files:**
- Modify: `analysis/cvs_sat_paic_traceability_20260622.md`

- [ ] **Step 1: Update statuses**

Mark implemented and verified rows with exact files and commands.

- [ ] **Step 2: Final verification**

Run:

```powershell
conda activate ssr-gpu; python -m py_compile code/cvsrffi/paic_star_ground.py tools/cvs_sat_paic_matrix.py tools/optimizer_validate_matrix.py
conda activate ssr-gpu; python -m pytest -q tests/test_cvs_sat_paic_matrix.py code/tests/test_paic_star_ground.py
```

Optional if time permits:

```powershell
conda activate ssr-gpu; python -m pytest -q tests/test_spaceborne_fewshot_da_matrix.py code/tests/test_optimizer_workflow_tools.py
```
