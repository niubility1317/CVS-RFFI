# Phase1 CLIC Source Metrics Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Every implementation task starts with a failing behavior test.

**Goal:** Seal the missing same-checkpoint source-known clean, three-scene source-LEO-weak and source-proxy evidence for all six C/G fold pairs without changing training, checkpoints, source-L tail calibration or any target artifact.

**Scientific boundary:** `source_clean_proxy.npz` v4 already contains the held source-V clean rows and PAIR v3 already contains the source-proxy score-only diagnostic. Existing source-LEO v4 contains only source-L rows used to fit geometry and tail thresholds; it is not source-V performance evidence. The only new data asset is one immutable source-V single-observation LEO cache per fold, shared byte-for-byte by C/G. Because its exact scene assignment is being completed after the target run was sealed, all results are labelled `POST_TARGET_COMPLETION_AUDIT_NON_SELECTION`; they may complete the rejected candidates' evidence rows but may not select, tune, retry, revive or promote a candidate.

**Tech stack:** Python 3.10, PyTorch, NumPy, JSON, Bash, pytest, existing `ssr-gpu` environment.

## Frozen contracts

- Inputs remain training v5, clean v4, source-L cache/export v3/v4 and PAIR v3. No trainer, checkpoint, source-L cache, target package, prediction or target scorer is modified.
- The formal evaluation role is `source_validation_known_leo_weak`; it is the 16,800-row held V slice in each fold, not the 3,920-row fitted L slice and not the fixed-400 proxy.
- Reconstruct V from the same sealed source split used by clean v4. Require exact validation index/order hashes, globally unique physical IDs and pairwise-disjoint L/V/proxy physical sets.
- Within each `(tx_id,rx_id)` cell, sort opaque physical IDs and assign the frozen scene tuple round-robin. Every physical ID receives exactly one scene, one seed stream and one received-IQ row; scene physical-ID sets are pairwise disjoint. C/G in a fold consume the same cache path and raw SHA.
- Channel parameters and seeds are frozen before execution: reuse the existing source-LEO channel configuration and `checkpoint.seed + 991 + scene_index * 1,000,003`. The assignment and seed policy are independent of all target values.
- Source-V never fits geometry, thresholds, tails, checkpoints, model state, selection or stopping. All `fit_rows` and `threshold_fit_rows` fields remain zero.
- Clean-V correctness is unique `argmax(tx_logits)==truth`. LEO-V correctness requires `decision=registered` and the unique predicted local class equals truth; `unknown` and `defer` are known-class errors.
- Each clean and `fold×scene` receipt stores positive raw numerators/denominators for overall, class, RX and day, plus macro accuracy and minimum class/RX/day floors. Missing or zero denominator fails closed.
- Per fold, proxy evidence is the existing PAIR diagnostic: `AUROC_unknown` and `u_gap=mean(e_proxy)-mean(e_V)`, with L-only geometry and V/proxy zero-fit.
- Non-compensating gates: every fold clean and every one of 18 fold-scene cells requires each G-C delta in overall/min-class/min-RX/min-day to be at least `-2pp`; each fold's three-scene equal-weight overall and the global 18-cell equal-weight overall also require `>=-2pp`; every fold requires both proxy `delta_AUROC>0` and `delta_u_gap>0`.
- These source results cannot compensate for the already failed target-real-unknown 70% gate.

---

### Task 1: Source-V Single-Observation LEO Cache

**Files:**
- Create `code/build_phase1_clic_source_v_leo_iq.py`.
- Create `code/tests/test_build_phase1_clic_source_v_leo_iq.py`.
- Create/update `automation_reports/CV-SincNet/phase1_clic_source_metrics_20260813_v1/report.md`.

- [ ] Write RED tests for exact V reconstruction, L/V/proxy disjointness, stable permutation-invariant scene assignment, one physical ID/one scene, exact C/G byte sharing, finite received IQ, positive scene/class/RX/day coverage, immutable outputs and input TOCTOU rejection.
- [ ] Run the focused test under `ssr-gpu` and confirm failure due to the missing builder API.
- [ ] Implement the smallest V-only builder by reusing current split/channel/safe-bridge helpers; do not generalize or alter the L builder.
- [ ] Seal schema `cvs.phase1.clic_source_v_leo_received_iq.v1` with validation index/order hashes, physical/order/assignment/channel/cache SHA, scene seeds, all coverage counts, C/G checkpoint+terminal bindings and all zero-access/zero-fit fields.
- [ ] Verify focused GREEN, `py_compile`, CLI `--help`, immutable-output negative and `git diff --check`.

---

### Task 2: Source-V Forward and Same-Checkpoint Metrics

**Files:**
- Create `code/export_phase1_clic_source_v_leo_features.py`.
- Create `code/evaluate_phase1_clic_source_metrics.py`.
- Create `code/tests/test_phase1_clic_source_metrics.py`.
- Modify only the source-gate helper in `code/evaluate_phase1_clic_postfreeze_pair.py` if the new scorer cannot contain the missing equal-scene/global aggregation checks without duplication.
- Update the source metrics report.

- [ ] Write RED tests for strict checkpoint/cache/class-order/SHA reopening, one forward per V row, C/G shared-cache binding, no L/proxy/target access, no fit/update/selection, and no legacy Torch/NumPy bridge.
- [ ] Write RED scoring tests for clean and each scene's overall/macro/class/RX/day raw cells and floors; known `unknown/defer` as errors; zero denominator, nonfinite, role overlap, scene reuse and post-open mutation rejection.
- [ ] Write RED gate tests covering every fold/cell, per-fold scene-equal overall, global 18-cell equal overall and strict dual-positive proxy improvements.
- [ ] Implement file-only forward and metrics modes. Reuse the already sealed source-L policy state and existing proxy diagnostic; do not refit or reinterpret it.
- [ ] Verify focused GREEN, affected postfreeze tests, `py_compile`, CLI file invocation and `git diff --check`.

---

### Task 3: Twelve-Arm Source Metrics Release

**Files:**
- Create `code/scripts/launch_phase1_clic_source_metrics12_v1_20260813.sh`.
- Create `code/tests/test_phase1_clic_source_metrics_launcher.py`.
- Update `automation_reports/CV-SincNet/phase1_clic_source_metrics_20260813_v1/report.md`.
- Update only P1 source rows in `analysis/phase1_phase3_goal_traceability_20260813.md` after artifacts exist.

- [ ] Write launcher RED tests: six shared V-cache builds, 12 forwards, 12 metrics receipts, exact fold/arm/path mapping, CPU/GPU bounds, source-only flags, fresh roots and no target inputs.
- [ ] Implement one immutable run with cache build first, then C/G forward and metrics; retry is `NO` and performance never controls dispatch or stopping.
- [ ] Run complete local verification and independent `P0=0/P1=0` review. The reviewer must explicitly audit post-target timing labelling and ensure results cannot feed selection.
- [ ] Commit, build a clean Git archive, preregister the report, and hand the exact release to one N607 runner. Require one F1 cache+consumer structural smoke before the unique formal launch.
- [ ] After 12/12 receipts return, report each checkpoint's four source groups on the same row and update the seven-gate audit. Preserve failed target candidates as failed; do not promote or redesign from target observations.

## Stop and evidence rules

- Stop only for protocol/hash/overwrite/access violations or at least two distinct rows with the same deterministic pre-receipt exception. Never stop or retry because source accuracy, proxy AUROC or any target metric is poor.
- Preserve all run/log/cache/receipt artifacts and record PID/GPU/SSH cleanup. Technical completion is not gate success.
- `P0`: L/V/proxy overlap, multi-scene physical reuse, C/G cache mismatch, V fit/threshold/selection access, target input, nonpositive denominator, or checkpoint/class-order/SHA drift.
- `P1`: missing `POST_TARGET_COMPLETION_AUDIT_NON_SELECTION` boundary, incomplete equal-scene/global aggregation, ambiguous closed-set formula, or a launcher path capable of feeding results back into training/selection.
