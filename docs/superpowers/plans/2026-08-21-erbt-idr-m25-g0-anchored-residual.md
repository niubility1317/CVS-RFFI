# ERBT-IDR M2.5 G0锚定交叉拟合残差实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不替换D92 E0主判别几何的前提下，实现support-only、margin门控、幅度封顶并由support留一证据选择强度的局部残差，在B0–B3每臂完整125矩阵上与去RF32基线同row比较。

**Architecture:** B0复用已证明注册后等价的R1 256维编译头。B1–B3在B0分数上叠加逐query独立的局部support残差；高margin样本保持B0原分数，K1/K2固定残差强度为0，K5/K10只允许从预冻结强度集合中按support留一证据选择。B2把常数不确定性bias改为query依赖的收缩类半径，B3只在最小簇大小、SSE下降和jackknife稳定性同时通过时使用按簇大小加权的双原型。

**Tech Stack:** Python 3、NumPy、pytest、现有CVS-RFFI Stage2 row executor、truth-last scorer和N607 Git Bash工作流。

**Spec:** `docs/ERBT_IDR_M25_G0_ANCHORED_TRACE_20260821.md`

## Global Constraints

- `protocol_schema=p2_min_v1`，只读取冻结Phase1 bundle、`VALIDATED_ONCE`固定received IQ和当前row合法support。
- query不参与拟合、强度选择、阈值选择、原型选择或状态更新；每条query独立面对全部注册类。
- B0–B3各运行5 receiver×5 seed×5 K/new条件，共500个方法行和1500个场景单元。
- K1/K2残差强度固定为0，预测必须与B0逐query一致。
- 主基线固定为去RF32的D92 E0/R1，不覆盖既有`erbt_idr_m24_invariance_break_full125_20260820_v1`。
- prediction全部闭合后才能由独立scorer连接truth；低性能不停止实验。

---

### Task 1: 锚定残差状态与测试

**Files:**
- Create: `code/cvsrffi/stage2_m25_anchored_residual.py`
- Create: `tests/test_stage2_m25_anchored_residual.py`

**Interfaces:**
- Consumes: `M24InferenceState`、`physical_if256()`、R1冻结log-diag和合法support。
- Produces: `fit_m25_anchored_residual(...) -> (M25AnchoredResidualState, audit)`；状态提供`score()`、`predict()`和`metric_features()`。

- [ ] **Step 1: Write failing tests** for K1/K2 exact fallback, high-margin exact preservation, bounded logit perturbation, class-permutation symmetry, support-only lambda selection, shrinkage radius, and stable dual-prototype admission.
- [ ] **Step 2: Run focused tests and verify RED** because the production module does not exist.
- [ ] **Step 3: Implement the minimal state and fitting functions** with frozen `lambda_grid=(0.0,0.02,0.04,0.08)`, `margin_gate=0.10`, residual normalization to `[-1,1]`, role-balanced support loss, per-role non-degradation, and p10 true-margin protection.
- [ ] **Step 4: Run focused tests and verify GREEN** with no query labels or cross-query state.

### Task 2: Row executor and complete matrix lifecycle

**Files:**
- Modify: `code/cvsrffi/stage2_m24_row_executor.py`
- Create: `code/scripts/run_m25_anchored_residual_full125.py`
- Create: `code/scripts/score_m25_anchored_residual_full125.py`
- Create: `code/scripts/summarize_m25_anchored_residual_full125.py`
- Create: `tests/test_stage2_m25_integration.py`

**Interfaces:**
- Consumes: B0–B3 arm constants and existing base feature caches.
- Produces: immutable prediction matrix, row receipts, truth-last score matrix and complete machine summary.

- [ ] **Step 1: Write failing integration tests** for four-state columns, B0 parity, 500-row matrix geometry, per-arm125 closure, B3 MAC accounting and scorer/summarizer wiring.
- [ ] **Step 2: Run integration tests and verify RED** for missing runner integration.
- [ ] **Step 3: Implement executor and scripts** while preserving existing M2.4 arms and outputs.
- [ ] **Step 4: Run integration and adjacent regression tests and verify GREEN**.

### Task 3: Release, N607 and full125 evidence

**Files:**
- Create: `automation_reports/CV-SincNet/erbt_idr_m25_g0_anchored_residual_full125_20260821_v1/report.md`
- Update: `docs/ERBT_IDR_M25_G0_ANCHORED_TRACE_20260821.md`
- Update after scoring: `docs/D92_E0_ALL_ABLATION_EXPERIMENTS_REPORT_20260819.md`

**Interfaces:**
- Consumes: locally verified Git commit and B0–B3 runner.
- Produces: N607 prediction/score evidence, Chinese formal report, machine summary and remotely verified Git publication.

- [ ] **Step 1: Run the focused protocol negatives, real-checkpoint no-query smoke, compile check and one independent P0/P1 review.**
- [ ] **Step 2: Commit and push only the implementation, tests, plan, trace and preregistration report; verify remote OID equals HEAD.**
- [ ] **Step 3: Publish one release archive to a new N607 output root, compare the archive SHA once and compile remotely.**
- [ ] **Step 4: Launch B0–B3 full125 once and verify PID/CWD/cmdline/log growth.**
- [ ] **Step 5: Confirm `PREDICTIONS_COMPLETE_TRUTH_UNOPENED`, `row_count=500`, `paired_input_identity_count=125` and B0–B3 each125 before scoring.**
- [ ] **Step 6: Run the preregistered truth-last scorer and full analyzer.**
- [ ] **Step 7: Report overall, K/new, receiver, seed, scene, four-state, old/new, class, margin, center angle, help/harm, `F_within/F_std` and resources; commit, push and independently verify remote OID.**

