# Stage2-C MRIOR-SDA Preadaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a frozen MRIOR-SDA target-old preadaptation stage before the existing CSIL and MoPC-HR Stage2-C enrollment paths.

**Architecture:** A plan builder deduplicates MRIOR work by`receiver/seed/K/scenario`, a preadaptation module produces hash-bound frozen backbone artifacts without opening query, and the existing truth-free CI predictor loads those artifacts for two new method IDs. Historical v7 no-preadaptation rows remain immutable paired references.

**Tech Stack:** Python3.10/3.13-compatible code, PyTorch, NumPy, pytest, JSON/NPZ/PT artifacts, existing ADV3B02 predictor packages.

## Global Constraints

- Use`protocol_schema=p2_min_v1`for scientific naming, while comparison methods retain their explicit source-access exception.
- New-class support and query are the existing fixed LEO weak IQ; query has zero fit, update, selection, truth, role, quota, and global-reassignment access.
- MRIOR uses exactly 200 adaptation steps, Adam lr`0.0006`, estimate steps`7`, target CE weight`1.0`, DV-KL weight`0.005`, and`mu=0.5`.
- Preadaptation artifacts are keyed only by`receiver/seed/K/scenario`and are shared across new-counts and CSIL/MoPC-HR.
- Original CSIL/MoPC-HR code paths and parameter locks remain unchanged.
- All implementation follows TDD: each production behavior is preceded by a focused failing test and an observed expected failure.
- Local tests use the verified`ssr-gpu`interpreter; N607 receives only committed, locally verified files.

---

### Task 1: Frozen MRIOR preadaptation artifact

**Files:**
- Create:`paper_reproduction/cvs_aligned/adv3b02_mrior_preadapt_ci.py`
- Test:`tests/test_adv3b02_mrior_preadapt_ci.py`

**Interfaces:**
- Consumes: ADV3B02 identity backbone, verified source loader, target-old support tensor/labels, frozen method lock.
- Produces:`MRIORPreadaptResult`, `fit_mrior_preadapted_backbone`, `write_mrior_preadapt_artifact`, and`load_verified_mrior_preadapt_artifact`.

- [ ] **Step 1: Write failing tests for deterministic preadapt keys and artifact identity**

```python
def test_preadapt_key_excludes_new_count_and_downstream_method():
    assert preadapt_key("20-1", 713101, 5, "leo_rain_weak") == (
        "rx_20_1__seed_713101__k_5__scene_leo_rain_weak"
    )
```

- [ ] **Step 2: Run the focused test and verify it fails because the module/API is absent**

Run:`python -m pytest -q tests/test_adv3b02_mrior_preadapt_ci.py`

- [ ] **Step 3: Implement MRIOR adaptation by reusing the existing minimax batch step**

```python
def fit_mrior_preadapted_backbone(
    backbone, source_loader, target_old_x, target_old_y, *, seed: int,
    adapt_steps: int = 200, learning_rate: float = 6.0e-4,
    estimate_steps: int = 7, target_ce_weight: float = 1.0,
    dvkl_weight: float = 0.005, mu: float = 0.5,
) -> MRIORPreadaptResult:
    config = {
        "method_id": "mrior_sda",
        "seed": int(seed),
        "adapt_steps": int(adapt_steps),
        "mrior_adapt_learning_rate": float(learning_rate),
        "mrior_estimate_steps": int(estimate_steps),
        "target_ce_weight": float(target_ce_weight),
        "dvkl_weight": float(dvkl_weight),
        "mrior_mu": float(mu),
    }
    model = ADV3B02MethodModel(
        copy.deepcopy(backbone), method="mrior_sda",
        feature_dim=int(backbone.emb_dim),
    ).to(target_old_x.device)
    trace, resource = _adapt(
        config, model, source_loader, target_old_x, target_old_y,
        scenario="sealed_by_caller", device=target_old_x.device,
    )
    return MRIORPreadaptResult.from_model(model, trace=trace, resource=resource)
```

The result stores only model state, loss/resource trace, input digests, and the query-unopened receipt; it stores no query rows or truth.

- [ ] **Step 4: Add rejection tests for wrong checkpoint SHA, source-cache SHA, support-token SHA, receiver, seed, K, scene, and method lock**

- [ ] **Step 5: Run the focused tests and commit**

Commit message:`feat: add frozen MRIOR CI preadapt artifacts`

### Task 2: Deduplicated 300-job/800-cell plan

**Files:**
- Create:`paper_reproduction/scripts/build_adv3b02_mrior_preadapt_ci_plan.py`
- Test:`tests/test_build_adv3b02_mrior_preadapt_ci_plan.py`

**Interfaces:**
- Consumes: authorized v7 CI plan, v7 package seals, source cache-set manifest path/SHA.
- Produces:`cvs.phase2.adv3b02_mrior_preadapt_ci_plan.v1`with300 preadapt jobs and800 CI cells.

- [ ] **Step 1: Write a failing miniature-plan test that expects deduplication across two methods and two new-counts**

```python
assert len(plan["preadapt_jobs"]) == receiver_count * seed_count * k_count * 3
assert len(plan["cells"]) == receiver_count * seed_count * k_count * new_count_count * 2
```

- [ ] **Step 2: Verify RED, then implement canonical ordering, exact counts, immutable paths, SHA binding, and smoke IDs**

- [ ] **Step 3: Add negative tests for non-v7 source plan, altered support identities across new-count packages, duplicate job keys, wrong source-cache scope, and missing LEO scenes**

- [ ] **Step 4: Run focused tests and commit**

Commit message:`feat: build matched MRIOR preadapt CI matrix`

### Task 3: Truth-free CI predictor integration

**Files:**
- Modify:`paper_reproduction/scripts/run_adv3b02_paper_full_ci_truth_free_predictor.py`
- Modify:`paper_reproduction/cvs_aligned/adv3b02_paper_full_ci.py`
- Modify:`tests/test_adv3b02_paper_full_ci_plan.py`
- Test:`tests/test_adv3b02_mrior_preadapt_ci.py`

**Interfaces:**
- Consumes: verified scenario-specific MRIOR artifact from Task1.
- Produces: existing`PaperFullState`for`mrior_sda_then_csil_paper_full`or`mrior_sda_then_mopc_hr_paper_full`.

- [ ] **Step 1: Write failing integration tests proving the adapted backbone is loaded before new support is read**

- [ ] **Step 2: Verify RED, then add the two method IDs and strict CLI artifact binding**

- [ ] **Step 3: Preserve original method receipts while adding`DA1_REG0`lineage, preadapt artifact SHA, source-access declaration, and`query_opened_after_model_lock=true`**

- [ ] **Step 4: Add mutation tests showing original CSIL/MoPC IDs never accept a preadapt artifact and preadapt IDs fail without one**

- [ ] **Step 5: Run focused tests and commit**

Commit message:`feat: enroll CI methods from MRIOR adapted backbones`

### Task 4: Matrix runner, analysis, and release report

**Files:**
- Create:`paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py`
- Create:`paper_reproduction/scripts/analyze_adv3b02_mrior_preadapt_ci.py`
- Test:`tests/test_run_adv3b02_mrior_preadapt_ci_plan.py`
- Test:`tests/test_analyze_adv3b02_mrior_preadapt_ci.py`
- Create:`automation_reports/CV-SincNet/adv3b02_mrior_preadapt_ci_20260817_v1/report.md`

**Interfaces:**
- Consumes: complete preadapt plan/artifacts, v7 immutable score table, independent truth sidecars.
- Produces: smoke receipt, per-cell prediction/score receipts, full integrity summary, paired rows, stratified tables, and runner handoff.

- [ ] **Step 1: Write failing tests for smoke-before-full authorization, immutable output paths, systemic two-row exception stop, and exact300/800/2400 closure**

- [ ] **Step 2: Implement the runner and verify RED→GREEN**

- [ ] **Step 3: Write failing analysis tests requiring exact same-row joins and rejecting unmatched v7 reference rows**

- [ ] **Step 4: Implement paired analysis without cross-method best-value splicing**

- [ ] **Step 5: Run focused and adjacent suites, compile checks, and`git diff --check`; update traceability/report and commit**

Commit message:`release: prepare MRIOR preadapt CI comparison`
