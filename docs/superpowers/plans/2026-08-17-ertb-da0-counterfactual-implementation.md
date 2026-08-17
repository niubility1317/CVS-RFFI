# ERTB-IDR DA0 Counterfactual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict no-target-old-preadaptation`DA0_REG1`counterfactual to the frozen ERTB-IDR Target125 experiment.

**Architecture:** A new single-arm matrix reuses immutable Target125 identities but invokes a predictor that leaves the old-class Phase1 state untouched and registers only new-class support. An independent analyzer joins the resulting`DA0`states to the existing`DA1`states without cross-row metric splicing.

**Tech Stack:** Python, PyTorch/NumPy artifacts, pytest, existing D92 E0 Target125 runner/closure utilities.

## Global Constraints

- The matrix is exactly125 outer×3 LEO scenes=375 result rows.
- The predictor must not open target-old support for fit, update, selection, center estimation, or covariance estimation.
- New support is the existing fixed LEO weak K-shot observation and query remains test-only.
- `DA0_REG0`new-class accuracy and harmonic metrics are`N/A`; they are never encoded as zero.
- Old/new predictions compete in one all-registered-class head after registration.
- All implementation follows TDD and is committed before N607 sync.

---

### Task 1: DA0 predictor and fail-closed audit

**Files:**
- Create:`code/scripts/predict_d92_e0_da0_reg_only.py`
- Create:`tests/test_predict_d92_e0_da0_reg_only.py`

**Interfaces:**
- Consumes: immutable Phase1 old-class aggregate, sealed target-new support, sealed query package.
- Produces: immutable`DA0_REG0`and`DA0_REG1`prediction artifacts plus fit/resource audit.

- [ ] **Step 1: Write failing tests for unopened target-old support and immutable old state**
- [ ] **Step 2: Verify RED, implement the minimal new-only registration path, and serialize explicit opened-role receipts**
- [ ] **Step 3: Add negative tests for old-support access, old-state mutation, query access before model lock, and non-all-class scoring**
- [ ] **Step 4: Run focused tests and commit**

Commit message:`feat: add ERTB DA0 new-only registration predictor`

### Task 2: Frozen Target125 matrix and runner

**Files:**
- Create:`code/cvsrffi/stage2_d92_e0_da0_target125.py`
- Create:`code/scripts/run_d92_e0_da0_target125.py`
- Create:`configs/stage2_d92_e0_da0_target125_v1.json`
- Create:`tests/test_stage2_d92_e0_da0_target125.py`
- Create:`tests/test_run_d92_e0_da0_target125.py`

**Interfaces:**
- Consumes: Task1 predictor and existing Target125 outer identities.
- Produces: one immutable arm, one real-checkpoint smoke, eight deterministic shards, and systemic failure receipts.

- [ ] **Step 1: Write failing exact-identity/count/path/stop-rule tests**
- [ ] **Step 2: Verify RED, implement the matrix/runner by reusing established closure utilities without changing existing E0_FULL_ONLY**
- [ ] **Step 3: Add smoke-before-full and non-overwrite tests**
- [ ] **Step 4: Run focused tests and commit**

Commit message:`feat: freeze ERTB DA0 Target125 matrix`

### Task 3: Four-state analysis and report preregistration

**Files:**
- Create:`code/cvsrffi/stage2_d92_e0_da0_target125_analysis.py`
- Create:`code/scripts/analyze_d92_e0_da0_target125.py`
- Create:`tests/test_stage2_d92_e0_da0_target125_analysis.py`
- Create:`automation_reports/CV-SincNet/d92_e0_da0_target125_20260817_v1/report.md`

**Interfaces:**
- Consumes: complete DA0 artifacts and existing exact-row DA1 artifacts.
- Produces: same-row four-state table, four required causal differences, resources, gates, and runner handoff.

- [ ] **Step 1: Write failing tests for exact outer/scene join,`N/A`REG0 new metrics, and all four differences**
- [ ] **Step 2: Verify RED, implement analysis, and reject partial/mismatched matrices**
- [ ] **Step 3: Run focused/adjacent suites, compile checks, and`git diff --check`; update report and commit**

Commit message:`release: prepare ERTB DA0 Target125 comparison`
