# ADV3B02-PairBiCAD-CV2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, publish, and launch the 24-row PairBiCAD-CV2 Phase1 source-only matrix with convergence control, staged bidirectional adversarial training, optional pair hinge, TailGuard, and two experiments per N607 GPU.

**Architecture:** Keep the existing PairBiCAD-P1 backbone and strict single-forward Clean/LEO pairing. Add isolated convergence/SWAD, adversarial-game, and TailGuard modules with candidate-specific switches, then integrate them through the existing trainer and entrypoint. Freeze all 12 candidate definitions before launch so the matrix never changes in response to live results.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing CV-SincNet/SSDG runner, N607 8×RTX3090.

**Spec:** `docs/superpowers/specs/2026-08-31-pairbicad-cv2-design.md`

## Global Constraints

- Phase1 is source-only; target receiver, Phase2, support, query and truth are forbidden before method freeze.
- Use ManySig source receivers `[1,3,4,6,8]`, day1/day2/day3, and `L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`.
- Preserve strict Clean/LEO single-forward pairing and separate Clean, `leo_clear_weak`, `leo_low_elev_weak`, and `leo_rain_weak` artifacts.
- Use test-first RED/GREEN for every production behavior.
- Edit locally first; N607 receives only committed, verified release content.
- One launch owner; no more than two matrix training processes per GPU.

---

### Task 1: Convergence and SWAD primitives

**Files:**
- Create: `code/cvsrffi/phase1_bicad_xr/convergence.py`
- Create: `code/cvsrffi/phase1_bicad_xr/swad.py`
- Create: `code/tests/phase1_bicad_xr/test_convergence.py`
- Create: `code/tests/phase1_bicad_xr/test_swad.py`

**Interfaces:**
- Produces `CoverageLedger`, `DGObservation`, `ConvergenceDecision`, `ConvergenceController.observe(...)`, `SWADAccumulator.consider(...)`, and `SWADAccumulator.averaged_state_dict()`.
- Must not import the training entrypoint or access filesystem paths outside caller-provided checkpoint data.

- [ ] Write failing tests for U/L coverage, candidate-specific activation age, plateau slope, LR reduction count, safety stop, and nonfinite rejection.
- [ ] Run both test files and record the expected RED failures caused by missing modules.
- [ ] Implement the smallest finite, deterministic convergence controller that satisfies the spec.
- [ ] Write failing SWAD tests for score/floor admission, tensor averaging, integer-buffer preservation, and empty-window rejection.
- [ ] Implement SWAD and rerun both files GREEN.
- [ ] Commit only Task 1 files.

### Task 2: Adversarial-game and TailGuard primitives

**Files:**
- Create: `code/cvsrffi/phase1_bicad_xr/adversarial_game.py`
- Create: `code/cvsrffi/phase1_bicad_xr/tailguard.py`
- Modify: `code/cvsrffi/phase1_bicad_xr/gradients.py`
- Create: `code/tests/phase1_bicad_xr/test_adversarial_game.py`
- Create: `code/tests/phase1_bicad_xr/test_tailguard_cv2.py`
- Modify: `code/tests/phase1_bicad_xr/test_gradients.py`

**Interfaces:**
- Produces `AdversarialGamePlan`, `build_adversarial_optimizers(...)`, `DualRatioController`, `margin_group_risks(...)`, `margin_rex_cvar_loss(...)`, and `bounded_hard_group_weights(...)`.
- The plan must use one backbone forward and explicit detached-discriminator/encoder phases.

- [ ] Write focused failing tests for parameter disjointness, LR ratio, detached features, local projection allowlist, two independent gradient ratios, CVaR tail selection, REx variance, and 30% cap.
- [ ] Run focused tests RED.
- [ ] Implement minimal production functions and run GREEN.
- [ ] Run existing gradient/loss/trainer tests to detect regressions.
- [ ] Commit only Task 2 files.

### Task 3: Candidate registry, metrics, and matrix analysis

**Files:**
- Modify: `code/cvsrffi/phase1_bicad_xr/config.py`
- Modify: `code/cvsrffi/phase1_bicad_xr/metrics.py`
- Create: `code/scripts/analyze_phase1_pairbicad_cv2_matrix.py`
- Modify: `code/tests/phase1_bicad_xr/test_config.py`
- Modify: `code/tests/phase1_bicad_xr/test_metrics.py`
- Create: `code/tests/phase1_bicad_xr/test_cv2_matrix_analysis.py`

**Interfaces:**
- Produces frozen candidate IDs `B0-B3,D0-D3,T0-T3`, source-only method-lock payloads, `S_DG` aggregation, same-row mainline/TailGuard gates, and complete four-scenario closure validation.

- [ ] Write failing registry tests proving candidate diffs are static and deferred mechanisms remain disabled.
- [ ] Run config tests RED, implement candidate registry, and run GREEN.
- [ ] Write failing synthetic-artifact tests for `S_DG`, same-row gates, missing scenario rejection, and negative scientific results.
- [ ] Implement analyzer/metrics changes and run GREEN.
- [ ] Commit only Task 3 files.

### Task 4: Main trainer and entrypoint integration

**Files:**
- Modify: `code/cvsrffi/phase1_bicad_xr/trainer.py`
- Modify: `code/SSDG/train_ssdg.py`
- Modify: `code/cvsrffi/phase1_bicad_xr/__init__.py`
- Modify: `code/tests/phase1_bicad_xr/test_trainer.py`
- Modify: `code/tests/phase1_bicad_xr/test_ssdg_entry.py`
- Modify: `code/tests/phase1_bicad_xr/test_real_checkpoint_smoke.py`

**Interfaces:**
- Consumes Tasks 1—3 modules and method locks.
- Produces structured training events, candidate-specific stopping states, final/EMA/SWAD checkpoint candidates, one-forward adversarial update routing, TailGuard loss telemetry, and runtime reconstruction fields.

- [ ] Add failing tests for new CLI/method-lock fields, no-query fail-closed behavior, one-forward backward plan, convergence stop states, and strict runtime reconstruction.
- [ ] Run focused tests RED.
- [ ] Integrate primitives with minimal trainer/entrypoint changes and run GREEN.
- [ ] Run all `code/tests/phase1_bicad_xr` tests.
- [ ] Commit only Task 4 files.

### Task 5: N607 launcher and preregistration

**Files:**
- Create: `code/scripts/launch_phase1_pairbicad_cv2_n607_20260831.py`
- Create: `code/scripts/launch_phase1_pairbicad_cv2_n607_20260831.sh`
- Create: `code/tests/phase1_bicad_xr/test_cv2_launcher.py`
- Update: `analysis/phase1_pairbicad_cv2_traceability.md`
- Create/mirror: formal run report for the immutable run ID.

**Interfaces:**
- Produces exactly 24 rows for 12 candidates×fold1/fold8×seed392002, eight GPUs, two concurrent slots per GPU, immutable row roots, and exact expected artifacts.

- [ ] Write failing launcher tests for the exact 24-row set, two-slot GPU packing, collision refusal, source-only CLI, safety status, and expected scenario artifacts.
- [ ] Run launcher tests RED.
- [ ] Implement launcher and run GREEN plus dry-run plan generation.
- [ ] Update traceability statuses and create the minimal preregistration report.
- [ ] Commit and push exact intended files; verify remote OID equals local `HEAD`.

### Task 6: Independent review, real-checkpoint smoke, release, and launch

**Files:**
- Update: `analysis/phase1_pairbicad_cv2_traceability.md`
- Update: formal run report and Git mirror.

**Interfaces:**
- Consumes the committed release and launcher.
- Produces local verification evidence, one independent P0/P1 review, real-checkpoint no-query smoke, one release archive SHA comparison, remote compile, detached launch, and post-launch binding evidence.

- [ ] Run focused protocol tests, full PairBiCAD-XR suite, Python compilation, launcher dry-run, and analyzer tests.
- [ ] Run one independent P0/P1 correctness review; if necessary, perform at most one scoped fix/re-review.
- [ ] Run a real historical ADV3B02 checkpoint no-query smoke with one optimizer step and strict four-scenario reconstruction.
- [ ] Commit/push final implementation and independently verify remote OID.
- [ ] Run the ordinary-account N607 preflight, verify paths/resources and unrelated GPU occupancy.
- [ ] Create one release archive, compare local/remote SHA once, compile remotely once, and launch the immutable 24-row run.
- [ ] Verify dispatcher/worker PID, CWD, cmdline, run root, GPU mapping and log growth once; report state as `RUNNING` until artifacts close.
