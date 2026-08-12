# D92 E0 FULL BIDIRECTIONAL NEWGUARD MAXMIN Hard11 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or the assigned TDD worker brief. Every production behavior starts with a witnessed RED test.

**Goal:** Build one support-only, single-FULL-fit candidate whose Hard10 paired performance is strictly Pareto-better than E0_FULL_ONLY and whose query compute/state are unchanged.

**Architecture:** Fit the existing E0_FULL_ONLY head once. Build six class-permutation-equivariant old-class directions in the orthogonal complement of the augmented registered-new support span, then solve one deterministic small max-min program for six direction strengths and one non-positive shared old-envelope shift. Publish the result as the same single FP32 affine head, or byte-exactly fall back to E0 on numeric/infeasible closure.

**Tech Stack:** Python3.10, NumPy, SciPy, pytest, `p2_min_v1`, existing D92 E0D prediction/scoring closure, 8-shard N607 RTX3090.

## Global Constraints

- Candidate ID: `d92_e0_full_bidirectional_newguard_maxmin`; arm: `E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN`; registered mode: `newguard_maxmin`.
- Exactly one E0 FULL component fit for active K>2; no BLOCK, OCF, LOO, Fisher, Pareto, encoder, feature or covariance change.
- New-class affine rows are byte-exact. Query uses one affine head with exact E0 query MAC and persistent-state bytes.
- Bottom tail fraction is exactly `0.20`, NumPy `method="lower"`; the fixed tail is selected once from baseline E0 support margins.
- Nullspace rank tolerance is the deterministic LAPACK-style rule `eps * max(X_new.shape) * largest_singular_value`.
- Internal residual obeys `X_new @ delta_internal=0` and old-group zero sum; shared old intercept shift obeys `tau<=0`.
- Only one deterministic support-side max-min solve; no parameter/arm/query/checkpoint scan.
- K<=2 is the exact existing D92 FULL alias; numeric/infeasible K>2 closure is byte-exact E0 fallback and cannot count as success.
- Reuse the frozen Hard10 performance rows and K1 liveness row from `stage2_d92_floorboost_hard11.py`; 11jobs, 33scene-arm, 8shards.
- Historical baseline path and SHA are exactly those in the approved specification; D92/E0 are not rerun.
- Data capsule remains `VALIDATED_ONCE`; do not repeat data validation or build new P2 governance.

---

### Task 1: Scientific affine state and receipts

**Files:**
- Modify: `code/scripts/probe_d92_registration_balanced_covariance.py`
- Modify: `code/cvsrffi/stage2_d92_e0d_slim.py`
- Modify: `code/cvsrffi/stage2_d92_e0d_query_evaluation.py`
- Modify: `tests/test_probe_d92_registration_balanced_covariance.py`
- Modify: `tests/test_stage2_d92_e0d_slim.py`
- Modify: `tests/test_stage2_d92_e0d_query_evaluation.py`

**Interfaces:**
- Add registered mode `newguard_maxmin` and arm `E0_FULL_BIDIRECTIONAL_NEWGUARD_MAXMIN`.
- Add `_build_newguard_affine_state(...) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]` consuming the single FULL support rows/labels/head.
- Emit stable `d92_newguard_*` at probe level and `d92_e0d_newguard_*` at slim/query level for all fields listed in traceability NG-20.

- [ ] Write RED tests for exact nullspace protection, new-row bytes, `tau<=0`, all-old-class fixed-tail nondecrease, class permutation, one FULL fit, K1/K2 alias, numeric fallback and receipt drift.
- [ ] Run the three focused files under `conda activate ssr-gpu`; confirm failure is caused by the missing arm/helper/receipts.
- [ ] Implement the minimum deterministic direction construction, small max-min solve, closure checks and E0 fallback.
- [ ] Run the focused tests to GREEN, then relevant E0D/FloorBoost regression tests and `py_compile`.
- [ ] Commit only Task1-owned files with a concise scientific-method commit.

### Task 2: Frozen Hard11 single-arm matrix, runner and analyzer

**Files:**
- Create: `configs/stage2_d92_full_bidirectional_newguard_hard11_v1.json`
- Create: `code/cvsrffi/stage2_d92_newguard_hard11.py`
- Create: `code/cvsrffi/stage2_d92_newguard_hard11_analysis.py`
- Create: `code/scripts/run_d92_newguard_hard11.py`
- Create: `code/scripts/analyze_d92_newguard_hard11.py`
- Create: `tests/test_stage2_d92_newguard_hard11.py`
- Create: `tests/test_run_d92_newguard_hard11.py`
- Create: `tests/test_stage2_d92_newguard_hard11_analysis.py`

**Interfaces:**
- Reuse the exact frozen row keys and K1 smoke key from FloorBoost Hard11; create a new schema/candidate/run identity without importing FloorBoost performance.
- Reuse the existing E0OCF/FloorBoost prediction closure, truth-free smoke, shared distinct-outer stop ledger and post-prediction scorer.
- Analyzer consumes 11 candidate scores plus the frozen historical `paired_rows.csv` and produces `summary.json`, `gates.json`, `paired_rows.csv`, `analysis.md`.

- [ ] Write RED tests for 11/33/8 matrix identity, smoke-before-shard, exact receipt closure, eight strict Pareto metrics, stability/resource gates and the three verdicts.
- [ ] Run the three new test files under `ssr-gpu`; confirm missing-module/behavior RED.
- [ ] Implement only the narrow single-arm wrappers and analyzer; do not clone new generic infrastructure.
- [ ] Run new tests to GREEN, existing runner/analyzer regressions, JSON parse, both CLIs `--help` and `py_compile`.
- [ ] Commit only Task2-owned files.

### Task 3: Integration, independent release review and preregistration

**Files:**
- Update: `analysis/d92_newguard_hard11_traceability_20260812.md`
- Create: `automation_reports/CV-SincNet/d92_e0_full_bidirectional_newguard_hard11_20260812_v1/report.md`
- Create: `automation_reports/CV-SincNet/d92_e0_full_bidirectional_newguard_hard11_20260812_v1/launch.sh`
- Mirror report to: `E:\type10-7\automation_reports\CV-SincNet\d92_e0_full_bidirectional_newguard_hard11_20260812_v1\report.md`

- [ ] Run integrated focused tests, `py_compile`, JSON parse, CLI help and `git diff --check` under `ssr-gpu`.
- [ ] Verify actual code path, one-FULL inventory, query zero-access and real-checkpoint truth-free smoke.
- [ ] Obtain independent P0=0/P1=0 review; fix only real P0/P1 and rerun covering tests.
- [ ] Write the minimal report with immutable v1 paths, exact command/env/GPU mapping, hashes, expected artifacts and health stop rule.
- [ ] Commit the integrated release and freeze runtime archive/config/launch hashes.

### Task 4: Sole N607 Hard11 run and artifact retrieval

- [ ] Use the ordinary-account direct preflight and verify all four v1 paths are absent plus GPU/process state.
- [ ] SCP only the frozen archive/config/launch; verify remote hashes, Python/CUDA, archive entries and `bash -n`.
- [ ] Launch the exact detached command once; verify PID/CWD/cmdline/GPU, prepared 11-job manifest and real-checkpoint K1 smoke before shards.
- [ ] Monitor with short connections for health only; never read performance to stop.
- [ ] Retrieve source/logs/smoke/output to a new local artifact root and prove 11 receipts, 22 state artifacts and 8 PASS summaries.

### Task 5: Independent analysis, reverse audit and verdict

- [ ] Run the frozen analyzer only after complete artifact retrieval and baseline SHA verification.
- [ ] Produce the full 10-performance-row paired table, D92/E0/candidate means, receiver/scene/K/new-count/class/confusion/resource breakdowns and fallback counts.
- [ ] Apply strict verdict: any one of the eight mean directions wrong is `REJECT_ROUTE`; all eight positive but below magnitude is `REVISE_ONCE`; all magnitude/stability/resource gates pass is `ADVANCE_TO_TARGET125_CANDIDATE`.
- [ ] Update both report copies and every traceability row with direct verification evidence; run a requirement-by-requirement reverse audit.
- [ ] Commit the result report. Do not launch Target125 automatically.
