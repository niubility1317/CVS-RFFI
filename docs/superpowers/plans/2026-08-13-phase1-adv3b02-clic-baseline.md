# Phase1 ADV3B02 CLIC-Equivalent Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and release six source-only ADV3B02 baselines whose training and known-target test configurations are equivalent to the six CLIC folds, then produce six immutable rich known-target references consumable by the existing strict CLIC combined scorer.

**Architecture:** A frozen six-row training launcher reuses the current production `train_ssdg.py` and the historical ADV3B02 loss profile while replacing only the obsolete data split with each CLIC fold's `0.07/0.63/0.30` roles. A separate file-only target evaluator seals role-blind predictions from the already validated IQ-only package, then opens truth in a second mode to compute rich registered-known metrics and calls the existing strict reference ingester. ADV3B02 remains a known-DG comparison component; it does not acquire or claim a real-unknown decision rule.

**Tech Stack:** Bash; Python 3.10; PyTorch; NumPy; JSON; pytest; existing `ssr-gpu` Conda environment.

## Global Constraints

- Follow root `AGENTS.md`, root `项目.md`, and `E:\codex\home\attachments\c75febfd-60b9-42bb-9825-a0b3b9eda0bb\goal-objective.md`.
- Training, checkpoint selection, model state and all ADV hyperparameters remain source-only; target prediction is zero-training, zero-adaptation, zero-update, zero-selection and zero-retry.
- Do not reuse or relabel the historical `tx_rx_day_1_7_2`, `0.10/0.70/0.20` checkpoint. The new method identity is `ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL` so its `0.07/0.63/0.30` data configuration is not mislabeled as `RHO10`.
- Freeze `split_mode=tx_rx_day_1_6_3`, `labeled_ratio=0.07`, `unlabeled_ratio=0.63`, `source_val_ratio=0.30`, seed `392002`, 200 epochs, 130 label epochs, 70 pseudo epochs, `from_scratch=true`, and final-only checkpoint selection.
- Preserve the historical ADV3B02 loss/mechanism flags exactly from `code/scripts/launch_phase1_adv3_mechanism32_queue_20260701.sh` for `ADV3B02_CORE90_SOFT_E200`; do not tune from current target results.
- Six folds use the exact CLIC TX partitions already sealed by training v5. One fold-level ADV reference is shared by its C and G candidates; references may not cross folds.
- Target inputs come from the existing validated confirmation cache/IQ-only package and use all 3120 rows for blind prediction; truth and roles are inaccessible until the separate metrics mode.
- Rich known metrics contain exactly three formal scenes and positive-denominator `overall`, `by_class`, `by_receiver`, `by_day`, `by_class_receiver`, `by_class_day`, macro/min floors, known false-reject, known defer and accepted-known.
- ADV real-unknown metrics are `N/A—no independently frozen endpoint decision rule`; never substitute proxy/vaccept, never write zero, never claim the 70% unknown gate passed.
- The original combined CLIC scorer remains strict. Missing/incomplete/config-inequivalent ADV references yield `FAIL/CANNOT_ESTABLISH`, never a pass.
- All project tests run serially under `ssr-gpu`; preserve the unrelated untracked `conversation_index/`; local changes are committed before N607 release.

---

### Task 1: Six-Fold Source-Only ADV3B02 Training Entry

**Files:**
- Create: `code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`
- Create: `code/tests/test_phase1_adv3b02_clic6_baseline.py`
- Create: `automation_reports/CV-SincNet/phase1_adv3b02_clic6_20260813_v1/report.md`
- Modify: `analysis/phase1_phase3_goal_traceability_20260813.md`

**Interfaces:**
- Consumes: `code/SSDG/train_ssdg.py`, ManySig, and the exact historical ADV3B02 parameter profile.
- Produces: six immutable run directories `F1_ADV3B02_CLIC` through `F6_ADV3B02_CLIC`, each with final checkpoint, terminal/config receipts and complete log.

- [ ] **Step 1: Write launcher behavior tests before the launcher exists**

  Execute the launcher under `--dry-run` and assert exactly six rows; literal fold TX partitions; the frozen ratios/split/seed/epochs/profile; final-only selection; target/package/truth/query paths absent; run/log roots rejected if pre-existing; and any wrong fold, legacy split, non-final selection or loss-profile drift rejected by a production preflight validator rather than a source-text grep.

- [ ] **Step 2: Run the focused tests and confirm RED**

  Run serially in `ssr-gpu`:

  ```text
  python -m pytest code/tests/test_phase1_adv3b02_clic6_baseline.py -q
  ```

  Expected: failure because the new launcher/profile validator is absent.

- [ ] **Step 3: Implement the minimal six-row launcher and frozen profile validation**

  Reuse the historical launcher's command flags without changing their values. Add only fold role flags, new run/candidate IDs, the current split ratios, final-only checkpoint selection, fresh-root protection, deterministic GPU mapping and a bounded `--dry-run` path. Do not add target inputs or a generalized experiment framework.

- [ ] **Step 4: Verify GREEN and syntax**

  Run serially:

  ```text
  python -m pytest code/tests/test_phase1_adv3b02_clic6_baseline.py -q
  bash -n code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh
  bash code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh --dry-run
  python -m py_compile code/SSDG/train_ssdg.py
  ```

  Expected: all tests and syntax pass; dry-run emits exactly six source-only commands.

- [ ] **Step 5: Preregister report and commit Task 1**

  The report records objective, exact six-fold matrix, historical-profile identity, local verification, run/log/output paths, GPU mapping, expected artifacts, technical stop rule, retry=`NO`, and the explicit unknown-metric N/A boundary. Commit only Task 1 files.

---

### Task 2: ADV Blind Target Prediction Artifact

**Files:**
- Create: `code/evaluate_phase1_adv3b02_target_leo.py`
- Modify: `code/tests/test_phase1_adv3b02_clic6_baseline.py`
- Modify: `automation_reports/CV-SincNet/phase1_adv3b02_clic6_20260813_v1/report.md`

**Interfaces:**
- Consumes: one completed ADV checkpoint/terminal, its generated immutable train-data config, and the existing CLIC IQ-only target package.
- Produces: `cvs.phase1.adv3b02_target_prediction.v1` with 3120 opaque rows and only prediction-time fields; no truth, role, TX/RX/day identity or scorer result.

- [ ] **Step 1: Write failing prediction tests**

  Cover real checkpoint strict reconstruction, all 3120 package rows forwarded exactly once, package/predictor/train-config SHA binding, truth/role/config path invisibility, zero fit/update/retry/selection, immutable output and prediction-before-truth ordering. Include checkpoint/package/train-config mutation negatives and a no-Tensor.numpy/no-torch.from_numpy real-forward smoke.

- [ ] **Step 2: Run the prediction selection and confirm RED**

  Expected: import/API absence, not fixture or syntax failure.

- [ ] **Step 3: Implement file-only prediction API and CLI**

  Provide one mode whose only inputs are checkpoint/terminal, sealed train-config and IQ-only package. Reuse current strict model reconstruction and safe Torch/NumPy bridge patterns. Seal source local class order and SHA, opaque-token lineage, forward count and input artifact SHAs. Do not open the truth sidecar or known-test config.

- [ ] **Step 4: Verify focused GREEN and real F1 fixture**

  Run the focused tests, `py_compile`, CLI `--help`, and a real-file invocation against a synthetic or local real-checkpoint fixture with no target truth access.

- [ ] **Step 5: Commit Task 2**

  Update the report with the API/CLI contract and verification evidence; commit only Task 2 files.

---

### Task 3: Truth-Side Rich Known Metrics and Strict Reference Ingest

**Files:**
- Modify: `code/evaluate_phase1_adv3b02_target_leo.py`
- Modify: `code/evaluate_phase1_clic_target_leo.py`
- Modify: `code/tests/test_phase1_adv3b02_clic6_baseline.py`
- Modify: `code/tests/test_phase1_clic_postfreeze.py`
- Create: `code/scripts/launch_phase1_adv3b02_target_reference6_v1_20260813.sh`
- Modify: `automation_reports/CV-SincNet/phase1_adv3b02_clic6_20260813_v1/report.md`

**Interfaces:**
- Consumes: one sealed ADV prediction, the sealer-only truth sidecar, candidate-equivalent known-test config and raw train-config.
- Produces: `cvs.phase1.adv3b02_target_known_metrics.v1` plus one immutable verified ADV reference per fold accepted by `ingest_adv3b02_target_known_reference`.

- [ ] **Step 1: Write failing truth/reference tests**

  Assert prediction and all source/config/package bindings are verified before the first truth open; inactive union-known rows are audited and excluded from the fold local-four; each scene's rich crossed cells reproduce both class and RX/day marginals; denominators are positive; missing scene/cell, zero denominator, marginal drift, post-open mutation, fold mismatch and cross-fold reference reuse fail closed; different capsule bytes with semantically equal configs pass.

- [ ] **Step 2: Run the focused selection and confirm RED**

  Expected: missing truth-metrics/reference API or missing launcher, not a fixture error.

- [ ] **Step 3: Implement minimal metrics, ingest CLI and six-fold launcher**

  Reuse the existing CLIC truth join and rich-known arithmetic rather than duplicating formulas. Expose the existing reference ingester through a file-only CLI if needed. The target-reference launcher runs six predictions first, then six truth metrics/reference ingests; it contains no candidate C/G scoring and no target feedback into training.

- [ ] **Step 4: Verify GREEN and full strict scorer compatibility**

  Run focused ADV tests, the CLIC combined-scorer compatibility selection, `py_compile`, both CLI `--help` paths, `bash -n`, and launcher dry-run. Confirm each fold reference is accepted for both F#C and F#G only and incomplete evidence remains non-promotable.

- [ ] **Step 5: Commit Task 3**

  Update report and traceability statuses for implemented interfaces; commit only Task 3 files.

---

### Task 4: Local Completion Gate and N607 Release Handoff

**Files:**
- Modify: `automation_reports/CV-SincNet/phase1_adv3b02_clic6_20260813_v1/report.md`
- Modify: `analysis/phase1_phase3_goal_traceability_20260813.md`

**Interfaces:**
- Consumes: Tasks 1–3 commits and tests.
- Produces: one reviewed Git commit/release handoff for training, followed after training by a separate reviewed prediction/reference release and finally the existing 12-row combined scorer.

- [ ] **Step 1: Run complete local verification**

  Run the dedicated ADV tests, affected CLIC postfreeze tests, `py_compile`, all CLI help, both launcher syntax/dry-runs and `git diff --check` serially under `ssr-gpu`.

- [ ] **Step 2: Independent P0/P1 review**

  Review exact fold/profile/config equivalence, source-only training, target truth ordering, rich-cell closure, immutable artifact/TOCTOU behavior, unknown N/A boundary and strict combined-scorer failure semantics. Require `P0=0/P1=0` before N607.

- [ ] **Step 3: Real-checkpoint technical smokes**

  Before formal training, run one F1 source-only three-batch smoke with no target input. After the six checkpoints complete, run one F1 blind single-IQ forward and one complete F1 target prediction/reference ingest smoke; neither may use performance to select or retry.

- [ ] **Step 4: Version, report and hand off to one N607 runner**

  Commit all intended files, build a clean Git archive, preregister immutable run IDs, exact commands/environment/CWD, GPU/log/output paths, artifacts and systemic technical stop rules. The runner performs one formal launch per run ID with retry=`NO`.

- [ ] **Step 5: After artifacts complete, run strict comparison without tuning**

  Seal six references, run the existing combined scorer for all 12 CLIC predictions, and report same-row known noninferiority alongside the already sealed target unknown/DG result. Any target gate or noninferiority failure remains a valid failed Phase1 result and cannot cause threshold tuning, candidate selection or selective rerun.
