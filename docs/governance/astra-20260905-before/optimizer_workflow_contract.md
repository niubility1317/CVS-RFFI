# CV-SincNet Optimizer Workflow Contract

This contract defines durable gates for the standing CV-SincNet/CVS-RFFI N607
automation. `AGENTS.md` remains the highest project rule. If this contract
conflicts with `AGENTS.md`, follow `AGENTS.md` and record the conflict in the
run report.

## 2026-07-15 Current qKNNv42 Stage2 Contract

This section is the current Stage2-B/C scalar policy. It supersedes older
OPGAC/JREF/OA-MSE/unknown-rejection/old80-first route defaults elsewhere in
this contract; those clauses are retained only as historical comparator and
audit context.

- Base: the sealed `ADV3B02_CORE90_SOFT_E200` checkpoint with SHA256
  `2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`.
- Data: every Phase2 sample and every adaptation, calibration, reference,
  prototype, threshold, rollback, and TTA signal is from a sealed
  `leo_*_weak` artifact. Raw/clean data and clean-derived signals are
  physically unreachable after Phase2 starts.
- Decision: each query sample faces all registered classes independently.
  Query role, true batch class count, class quota, label/order hints, global
  assignment, query fitting, dense query-query graphs, and scorer feedback are
  forbidden. Prediction is sealed before an independent scorer reads truth.
- Development: only K=10 development evidence may select the adapter, head,
  thresholds, TTA policy, epoch, or hyperparameters. K=1/5/20 are locked
  confirmation slices.
- Matrix: five confirmed target receivers × at least five independent
  confirmation seeds × three fixed `leo_*_weak` scenarios, using real nested
  5/10/20 seen-new TX sets. One prediction cell emits the three scenario
  predictions before scoring, so 300 cells produce 900 joint scenario rows.
  launchable Phase2 rows must expose target-old and target-new sample coverage;
  the target receiver domain may contain one or more receivers, must be
  disjoint from CEN51 train receivers, and validators must not require exactly
  one r_sat.
- Performance: at K=10, old accuracy ≥92%, minimum old-class accuracy ≥88%,
  and seen-new accuracy ≥92%/90%/86% for 5/10/20 new classes. K=5 matched drop
  is ≤3pp. K=1 is non-negative versus identity-only and ≥+2pp versus strict
  direct ADV3B02 overall and per receiver with paired 95% CI lower bound >0.
  Forgetting at K=1/5/10/20 is no worse than matched identity-only.
- Resources: preferred caps are ≤50,000 trainable parameters, ≤20 adaptation
  epochs, ≤256KB persistent state, no dense query graph, and adaptive
  per-sample 1→3→5-view inference with one view by default.
- Formal boundary: field declarations alone never grant launch authority.
  Candidate provenance, immutable input snapshot/TOCTOU closure, real Linux
  isolation, post-run access ledger, target TX coverage, and independent
  sealed prediction/scoring must all pass. Until then the outcome is
  `LOCAL_PROTOCOL_REPAIR_REQUIRED` with `formal_launch_authority=false`.
- Capacity: the current `AGENTS.md` total limit is at most two concurrent
  training experiments per GPU. Any older four-per-GPU or three-Phase2-per-GPU
  scalar is historical and must not drive Runner.

## Source-Of-Truth Map

- `AGENTS.md`: safety, environment, SSH/SCP, local-first edits, reporting, and
  version-management constraints.
- `项目.md`: CVS scientific scenario, source/target data protocol, single
  satellite receiver deployment semantics, Stage2-A/B/C boundaries, and metric
  claim limits.
- `tools/optimizer_control_manifest.md`: control-plane ownership, priority, and
  duplication rules.
- Active `stage2_prompt.md`: execution order for the recurring automation.
- This contract: hard gates, schemas, Phase2 sample protocol, and validation.
- `automation_reports/CV-SincNet/stage2_optimizer_state.json`: mutable evidence
  state, current indicators, route ledgers, queue history, and next-turn handoff.

The prompt may summarize this contract, but the contract owns the full rule.
The state file may store the current value of a rule or evidence item, but it
does not weaken safety, protocol, or runner gates.

## Current State Read Rule

Current monitor, runner, capacity, matrix-count, sample-protocol, and completion
decisions must read only top-level current state fields, especially:

- `latest_two_lane_monitor_result`
- `latest_optimizer_runner_result`
- `lane_monitor_policy`
- `lane_capacity_policy`
- `idle_lane_execution_policy`
- top-level `stage2_sample_protocol`

Use `tools/optimizer_state_current_view.py` as the preferred read-only helper
for this current-decision surface. `objective_changelog`, `target_changelog`,
lane subtrees, and older run summaries are audit-only unless a live top-level
field explicitly points to them.

Use `tools/optimizer_preflight_decision.py` as the read-only local preflight
bundle once a current matrix and launcher exist. It checks control-file
readability, the compact state view, matrix validation, launcher identity, and
duplicate registry/command hashes, then reports `PASS`, `BLOCKED`,
`PENDING_LOCAL_ARTIFACTS`, or `PENDING_REMOTE_MONITOR`. `PENDING_REMOTE_MONITOR`
does not authorize remote action; the helper must not run SSH/SCP or launch,
and remote work remains gated by `AGENTS.md` N607 preflight plus live
process/CWD/cmdline/GPU monitoring.

Lane-local mirrors, `active_focus`, `objective_changelog`,
`target_changelog`, historical run summaries, older command text, and previous
batch reports are audit evidence only. They cannot drive the current gate.

If this contract and top-level current state disagree on a scalar policy, stop
before remote actions with `CONTRACT_STATE_DRIFT`, unless `AGENTS.md` or the live
user instruction explicitly resolves the conflict.

## Gate Classes

Hard blockers stop remote action for the affected lane:

- Required control file unreadable.
- AGENTS safety conflict.
- `项目.md` protocol conflict.
- N607 route, identity, or SSH target ambiguity.
- Local verification failure.
- Unrepairable runner identity failure after repair-first preflight, or a real
  run/log path collision or registry duplicate that remains after regeneration.
- Remote hash/syntax/dry-run/path verification failure.
- Run/log/checkpoint/report path collision.
- Registry evidence of duplicate launch.
- Unsafe ambiguous CVS training process that cannot be assigned to a lane.
- GPU capacity above lane policy.
- User explicit pause or stop.

Repairable gates do not stop the whole loop:

- Missing current-run matrix, current validator output, local dry-run, launcher
  identity preflight, or local launch artifact.
- Missing candidate fields.
- Candidate count drift.
- Retired or invalid route rows.
- Missing evidence slices.
- Missing subagent tools.
- Literature or telemetry gaps.
- Metrics below target.
- Capacity shortage for a subset of rows.

Repairable gates must be resolved by filling fields, replacing rows, marking
rows as `NON_LAUNCH_DIAGNOSTIC`, or deferring rows with exact next commands.

## Idle Launchable Lane Obligation

When a current idle lane has `LANE_HAS_LAUNCHABLE_ROWS` in validator
`launchability_summary.by_lane`, and the launchable rows satisfy protocol and
schema validation, the controller must continue to the local-first Runner gates
for that lane. It must not return `MONITOR_ONLY_CONTINUE`, report-only,
state-only, dry-run-only, or protocol-only as that lane's outcome unless a hard
blocker from the `Gate Classes` section is present.

The following are not launch blockers for that lane:

- opposite lane active or awaiting completion audit, as long as the current lane
  has lane-specific capacity under `AGENTS.md` and this contract;
- older `latest_phase2_defer_result`, lane-local historical mirrors, or stale
  all-lane local-patch state that conflicts with current launchability summary;
- subagent disagreement without a cited hard blocker from `AGENTS.md`,
  `项目.md`, this contract, validator output, or current state;
- metrics below target, missing telemetry, literature gaps, or missing post-run
  completion audit from another lane;
- report-only, state-only, dry-run-only, or protocol-only progress after local
  validator has exposed launchable rows.

If the controller blocks an idle launchable lane, hard blocker must be one of `Gate Classes`, and the report/state must record the exact blocked outcome and artifact. Otherwise the lane must proceed through local verification, remote verification, launch, startup health, registry/state/report update, and SSH/SCP cleanup.

## Idle Lane Must Execute / Repair-Until-Launch Policy

When a server lane is process-idle (`phase*_monitor_state=1`), the controller's
standing objective is: idle lane must execute an experiment. This is a
repair-until-launch obligation bounded only by `AGENTS.md`, `项目.md`, SSH/N607
safety, current lane capacity, and the `Gate Classes` hard blockers.

missing current-run matrix is repair work, not a terminal outcome. The same is
true for missing current validator output, stale matrix identity, runner
identity drift, duplicate/retired rows, shallow idea pools, missing candidate
fields, or metrics below target. The controller must mutate the approach within
the same automation turn: compress evidence, generate or repair the current-run
64-row matrix, validate, run repair-first launcher identity preflight, local
dry-run, remote hash/syntax/path/capacity/dry-run gates, and launch at least one
safe row for the idle lane when no hard blocker remains.

For an idle lane, these are disallowed terminal outcomes:

- `analysis_only`
- `report_only`
- `state_update_only`
- `local_dry_run_only`
- `protocol_pass_only`
- `NO_CURRENT_MATRIX_VALIDATION`
- `NOT_RUN_NO_CURRENT_REPAIRED_MATRIX`
- `DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION`

`DEFERRED_RETRY_LOCAL_VERIFY` may be used only as a row-level status, or as a
lane outcome after at least one other safe row in that idle lane has been
executed, unless a hard blocker from `Gate Classes` prevents every row from
launching. If every candidate is blocked by `AGENTS.md`, `项目.md`, N607 access,
capacity, path/registry collision after regeneration, or local/remote
verification failure after bounded repair, the terminal outcome must name that
hard blocker, usually `USER_REQUIRED_SAFETY_STOP` or a concrete
`LOCAL_VERIFY_FAILURE_*` / `REMOTE_VERIFY_FAILURE_*` code, not a generic route
repair defer.

## Evidence-First Current-Run Matrix Policy

The standing optimizer must operate as an evidence-first current-run matrix
controller. For every automation wakeup, it must read the current completed
relevant experiment evidence before matrix selection, including completed
scheduler lifecycle, metrics, score tables, manifests, local reports, state
handoff, and registry context that are allowed by this contract.

After that evidence review, the optimizer must generate or repair the 64-row
mixed matrix for the same automation run, validate it under the current control
plane, and continue to local-first Runner gates for launchable or explicitly
deferred rows. A next-run matrix handoff is audit-only: older matrices,
handoff JSON, dry-runs, or report-only candidate proposals may seed the evidence
review, but they are not current launch authority. They cannot satisfy the
optimizer objective unless the controller regenerates or repairs them under the
current run ID, revalidates them after the evidence sweep, and executes the
result in the same automation run.

If the controller stops after producing a matrix for a future turn while the
current lane is idle and has no hard blocker, that is a contract violation, not
a valid `DEFERRED_RETRY_*` outcome. If the evidence sweep shows that a previous
matrix is still the correct mechanism set, the controller may reuse it only by
forking it into a current-run matrix artifact with fresh paths, registry keys,
command hashes, validation output, and report/state references.
If top-level state, the registry, or the current run report already records the
same matrix/run ID as completed, diagnostic-negative, or otherwise analyzed, the
controller must not relaunch that matrix in place. It may only run bounded
diagnostics over the completed artifacts, or fork the mechanism set into a fresh
current-run matrix with new identity, paths, registry keys, validation, and
report/state authority.

## Monitor Boundary

Monitor mode is only a process/CWD/cmdline/GPU classifier.

- Emit `phase1_monitor_state` and `phase2_monitor_state` on every run.
- A Phase1 process blocks only Phase1 optimizer/runner.
- A Phase2 process blocks only Phase2 optimizer/runner.
- Stop both lanes only if both lanes are active, or if a CVS training process is
  unsafe ambiguous and cannot be assigned to Phase1 or Phase2.
- Do not read logs, metrics, configs, checkpoints, final markers, handoff JSON,
  registry rows, or historical reports to decide optimizer entry.
- Active-lane logs and metrics may not be used for effect conclusions in an
  idle lane.
- Exclude monitor helpers and literal-string false positives.

Phase1 process signals include `Phase1-GroundDG`, `Safe-SSDG-CVS-R01`,
`safe_ssdg`, `source-only DG`, `CEN51_REFRESH_CONTROL`, or `code/train.py`
commands with formal source-only CEN51/Safe-SSDG ground-DG semantics. Legacy
`Meta-SSL-CVS-R04`, `meta_ssl`, and `phase1_meta_ssl` remain monitor-only
classification signals for historical or already-running Phase1 jobs; they are
not the current Phase1 optimizer direction unless the live user explicitly
reopens them.

Phase2 process signals include `stage2_spaceborne`, `next64*`,
`eval_spaceborne_fewshot.py`, `train_target_adapt.py`,
`export_spaceborne_features.py`, satellite/LEO target view, score-table,
manifest, rollback, or spaceborne few-shot/adaptation semantics.

## Lane Runner Contract

When a lane has monitor state `1` and at least one safe launchable or explicitly
deferred candidate, the automation must drive that lane to a server-landed
runner outcome:

- `LAUNCHED`
- `DEFERRED_RETRY_CAPACITY`
- `DEFERRED_RETRY_RUNTIME_BUDGET`
- `DEFERRED_RETRY_LOCAL_VERIFY`
- `DEFERRED_RETRY_REMOTE_VERIFY`
- `MONITOR_ONLY_CONTINUE`
- `USER_REQUIRED_SAFETY_STOP`

Report-only, state-only, dry-run-only, and protocol-only work do not satisfy an
idle-lane runner objective.

The required local-first order is:

1. Local report.
2. Local verification under `ssr-gpu`, including repair-first runner identity
   preflight for the current matrix and launcher.
3. Snapshot/manifest for changed code/config/script files.
4. SCP of locally verified changes only.
5. Remote hash/syntax/dry-run/path/capacity checks.
6. Launch.
7. Startup health after about 4-5 minutes.
8. Registry/state/report update.
9. SSH/SCP cleanup verification.

## Runner Identity Preflight

Before SCP, remote dry-run, or launch, the rendered launcher and current matrix
must pass repair-first runner identity preflight. The purpose is to remove
avoidable local blockers, not to add a terminal gate. Run the validator in
repair mode with the matrix and launcher, for example:

`tools/optimizer_validate_matrix.py <stage2_candidate_matrix.json> --expected-count 64 --launcher <local-launcher.sh> --repair-launcher-identity`

This preflight must check that:

- matrix top-level `n607_run_id` is present for current-run launch authority;
- each row `estimated_run_path` is under `/runs/<n607_run_id>/`;
- each row `estimated_log_path` is under `/logs/<n607_run_id>/`;
- each row `registry_key` starts with `<n607_run_id>:` when present;
- the launcher default RUN_ID in `RUN_ID="${RUN_ID:-...}"` equals the same
  `n607_run_id`;
- launcher `RUNS_ROOT` and `LOG_ROOT` derive from `${RUN_ID}`;
- if Phase2 has launchable rows, the launcher must not default
  `PHASE2_LOCAL_PATCH_REQUIRED=1`;
- generated launchers that source `stage2_queue_runner_template.sh` must not
  call `stage2_acquire_launcher_lock` directly, because the template owns lock
  acquisition.

Deterministic launcher mismatches are auto-repairable and must be repaired
locally, then validator and local dry-run must be rerun. They must not be used
as a reason to stop an idle launchable lane. Remote action is blocked only if
repair fails, the matrix itself still points at colliding run/log roots,
registry duplicate evidence remains, or `AGENTS.md` / `项目.md` safety and
protocol gates forbid launch.

## Candidate Matrix Contract

Default current queue: exactly 64 mixed candidates.

- 8 Phase1 / Safe-SSDG-CVS-R01 or matched CEN51_R04 source-only DG
  non-regression rows. Legacy Meta-SSL/meta-learning DG rows are diagnostic or
  negative evidence unless explicitly reopened by the live user.
- 56 Phase2 Stage2-A/B/C rows.
- Eight rows per GPU across GPU0-GPU7.
- Normally one Phase1 row and seven Phase2 rows per GPU.
- Initial active capacity for this standing automation: one Phase1 row per GPU
  plus up to three Phase2 rows per GPU only when those Phase2 rows are verified
  lightweight route-switch/evaluation jobs, or when the live user explicitly
  overrides `AGENTS.md` training concurrency for the run.
- Any row that starts a training process, heavy target-adaptation process,
  centralized job, federated/FL job, or unclear compute job falls back to
  `AGENTS.md` safety, including the default two-training-experiments-per-GPU
  limit, and the relevant lane cap unless the live user explicitly overrides
  the scope.
- Candidate size changes require live user override or explicit evidence-backed
  contract update.

Each candidate must include:

- `candidate_id` or `experiment_id`
- `lane`
- `stage2_mode` for Phase2 rows
- `parent_run`
- `lineage`
- `route_signature`
- `route_family` for mechanism families such as `OPGAC_NET` or `OA_MSE_HEAD`
- `hypothesis`
- `control`
- `key_changes`
- `parameters`
- `gpu`
- `runtime_class`
- `estimated_runtime_min`
- run path, log path, report path, and registry key
- exact command or exact dry-run command
- `cross_domain_target_metric`
- `satellite_channel_target_metric`
- `allowed_tradeoff`
- `must_not_regress_floor`
- `comparability_status`
- `expected_failure_signals`
- `fallback_or_alternative`
- `launchability_status`

Phase1 Safe-SSDG rows must additionally include:

- `route_family=SAFE_SSDG_CVS_R01` or `CEN51_REFRESH_CONTROL`
- `ground_dg_claim_scope=source_only`
- `source_ssl_split=0.1L/0.7U/0.2Val`
- `no_target_receiver_in_training=true`
- `cen51_base_checkpoint_or_config`
- `cen51_parent_run_or_control`
- `phase1_non_regression_target=matched_CEN51_R04`
- `CEN51_COMPARABLE=true` only when the source receiver/day split, WiSig train
  ratio or `rho_label<=0.1`, old/new/unknown TX semantics, no-target-leakage
  rule, and the declared satellite/LEO evaluation view are comparable. Legacy
  CEN51 five-scenario views may be used only for historical comparability
  controls; future star-ground enhancement and sample overlay rows must use
  the simplified LEO residual channel defined by `项目.md`.
- `must_not_regress_floor` containing at least `overall>=88.57`,
  `strict_udu>=84.87`, `receiver_floor>=79.53`, `sat_mean_5>=46.564`, and
  `sat_floor_5>=41.52`
- `optimization_target=exceed_matched_CEN51_R04`
- `target_lift_over_cen51`, with positive lift intent over CEN51_R04 rather
  than equality-as-success
- `satellite_channel_primary_metric=true`
- `satellite_channel_lift_target`, naming the deployment-primary simplified
  LEO residual channel metrics. Legacy five-scenario metrics such as
  `sat_mean_5_delta_pp>0` and `sat_floor_5_delta_pp>0` are historical
  comparator/control metrics, not the default future optimization target.
- `star_ground_channel_impl=simplified_leo_residual` for future star-ground
  channel enhancement, sample overlay, satellite stress, or PAIC rows.
- `target_channel_view` or `sat_train_scenarios` must use the simplified
  deployment-primary views from `项目.md` such as `leo_clear_weak`,
  `leo_low_elev_weak`, and `leo_rain_weak`, unless the row is explicitly
  labeled as a legacy five-scenario control/diagnostic.
- `phase1_star_ground_aug_default_enabled=true` for every future launchable
  Phase1 Safe-SSDG training row, unless the row is an explicit
  `CEN51_REFRESH_CONTROL` comparator/control and is marked with a
  `phase1_star_ground_aug_policy` control exemption.
- `phase1_star_ground_aug_route_family=CVS-SAT-PAIC`.
- `phase1_star_ground_aug_mode=concat_sat_ce_only_paic_curriculum`.
- `use_concat_sat_channel_aug=true`, `concat_sat_ce_only=true`,
  `use_sat_consistency=true`, and a simplified-channel PAIC schedule using
  `leo_clear_weak`, `leo_low_elev_weak`, and `leo_rain_weak`. The old
  `mixed_orbit/low_elev_leo/rain_leo/storm_mp` schedule is permitted only for
  explicitly marked legacy controls.
- `star_ground_aug_exploration_axis`, describing which PAIC dimension is being
  explored, such as curriculum probability, scenario mix, CE-only satellite
  branch weight, late weak `z_id` consistency, DSQ/domain audit, or explicit
  CEN51 refresh control.
- `pseudo_precision_audit_target>=0.95`
- `pseudo_coverage_is_risk_metric=true`
- `forbid_meta_learning_dg_mainline=true`

Phase1 ground prototype/mask/feature-distribution rows additionally require:

- `phase1_ground_prototype_mask_openworld_enabled=true`
- `phase1_ground_feature_distribution_objective=true`
- `source_domain_prototype_outputs_required=true`
- `phase1_enable_ground_prototype_stats=true`
- `phase1_enable_feature_distribution_audit=true`
- `phase1_enable_feature_masks_aux=true`
- `phase1_enable_txrx_geometry_audit=true`
- `prototype_mask_modules` containing at least `phase2_prototypes`,
  `feature_masks`, and `tx_rx_geometry`
- `target_receiver_usage=forbidden_in_phase1`
- `unknown_query_role` remaining eval-only or not available in Phase1 training
- CEN51 must be declared as non-regression/comparison experience and not as a
  narrowed route family.

If any required comparability or non-regression field is missing, the Phase1
row must be `NOT_CEN51_COMPARABLE`, `LOCAL_PATCH_REQUIRED`, or
`NON_LAUNCH_DIAGNOSTIC`, not launchable. Startup PASS, protocol PASS, high
pseudo-label coverage, and a single partial seed are not Phase1 DG completion
or CEN51 no-worse evidence. CEN51_R04 is a hard floor. The main Safe-SSDG
optimization target is to exceed CEN51_R04, with star-ground satellite/LEO
performance treated as deployment-primary. A row that improves only clean or
non-satellite metrics while failing to lift `sat_mean_5` or `sat_floor_5`
cannot be promoted as the new Phase1 base.

The default Phase1 optimizer policy is now `CVS-SAT-PAIC` star-ground channel
enhancement on, using the simplified LEO residual channel implementation. The
generated training command, dry-run command, or structured parameters must
expose the CE-only PAIC flags:
`--use_concat_sat_channel_aug`, `--concat_sat_ce_only`,
`--sat_view_schedule`, and `--use_sat_consistency`, plus structured metadata
`star_ground_channel_impl=simplified_leo_residual`. This is a source-domain
derived satellite view only; it must not use target receiver samples, target
BN/statistics, Stage2 support/query labels, or target validation metrics.
Explicit CEN51 refresh rows may omit PAIC augmentation only as comparator
controls, and they must not be promoted as the new Phase1 base by matching
CEN51 alone.

Phase1 Safe-SSDG rows default executable. A future optimizer-generated Phase1
Safe-SSDG row must not use row-scoped
`DEFERRED_RETRY_LOCAL_VERIFY_PHASE1_SAFE_SSDG_CVS_R01_LAUNCHER_SCHEMA_REQUIRED`
as its normal result. The generated launcher must expose
`run_phase1_safe_ssdg_candidate`, and row-level commands must point to that
launcher entrypoint or the executable direct entrypoint
`python ${ROOT}/code/SSDG/train_ssdg.py`. The nonexistent `code/train.py
--use_safe_ssdg_cvs` path is not a valid launch command. Phase1 Safe-SSDG may
defer only for real capacity/runtime budget, active same-lane process,
repair-failed local verification after an actual executable branch exists,
`AGENTS.md`/`项目.md` safety or protocol conflict, SSH/N607 gate failure, or
explicit user pause.

OPGAC rows must additionally include:

- `route_family=OPGAC_NET`
- `stage2_base_model_id=JREF_C9_MULTICOMP_M2_E220`
- `stage2_base_model_role=receiver_floor_diagnostic_not_deployment_success`
- `opgac_stage` (`old_calibration`, `old_new_enrollment`,
  `strict_eval`, or `confirm_new_sensitivity`)
- `opgac_memory_policy=support_only`
- `opgac_local_code_hook=code/cvsrffi/opgac_net.py`
- `opgac_eval_tool=tools/evaluate_opgac_stage2.py`
- `opgac_query_update_forbidden=true`
- `unknown_query_eval_only=true` only when unknown query is present as Phase3
  backup metadata
- `target_new_query_not_threshold_fit=true`
- `model_output_semantics`, explicitly distinguishing old label, seen-new
  label when Stage2-C is legal, ambiguous, and defer; reject is required only
  when Phase3 backup is enabled
- `opgac_overlap_policy`, requiring provisional or ambiguous handling for old
  versus seen-new overlap rather than forced registration
- `opgac_rollback_policy`, requiring rollback to ground-old memory or the
  previous support-only snapshot on drift/harm
- `opgac_metric_bundle`, containing at least `old_acc`, `old80_gap`,
  `seen_new_acc` when Stage2-C is active, `H_old_new` when Stage2-C is active,
  coverage, `old_FRR`, rollback rate, defer rate, old/new confusion counts,
  and `same_row_rank`; unknown/FAR metrics are Phase3 backup only
- `opgac_primary_selection_metric`, such as constrained old/new same-row score,
  `H_old_new`, or OLD80-first deficit before seen-new optimization
- `opgac_same_row_ranking_required=true`
- `opgac_score_table_required_columns`, containing at least candidate label,
  best old score, best seen-new score, top-2 margin, threshold delta,
  `opgac_old_score`, and `opgac_new_score`; best reject score is Phase3 backup
  only
- `stage2_priority_phase=PHASE2_ADAPT_NEWCLASS_FIRST`
- `old_acc_target>=0.80`, only as an intermediate old-class recovery gate
- `deployment_success_claim_allowed=false` while using the OLD80 intermediate
  gate

Current OPGAC rows use `JREF_C9_MULTICOMP_M2_E220` as the Stage2 base because
recent JREF evidence shows it is the strongest local-mode receiver-floor
diagnostic. This does not promote JREF as Phase1 mainline success, does not
replace the project protocol, and does not create deployment evidence by
itself.

OA-MSE rows must additionally include:

- `oa_mse_stage` (`mse_lite`, `mse_subspace`, or `oa_mse_head`)
- `source_target_fusion_policy`
- `fusion_inputs`
- `threshold_selection_label_scope`
- `unknown_query_role=eval_only` only when unknown query is present as Phase3
  backup metadata
- `unknown_query_eval_only=true` only when unknown query is present as Phase3
  backup metadata
- `target_new_query_not_threshold_fit=true`
- `model_output_semantics`, explicitly distinguishing old label, seen-new
  label, uncertain, and defer; reject is required only when Phase3 backup is
  enabled
- `unknown_FAR_target<=0.05` only for Phase3 backup rows
- `uncertain_policy`
- `onboard_low_compute_training=true`
- `compute_budget_profile`, naming feature-level / low-rank / no-full-backbone
  update constraints
- `adapter_trainable_params_cap`
- `max_adapt_steps`
- `old_acc_target>=0.90`, except rows explicitly marked
  `stage2_priority_phase=PHASE2_ADAPT_NEWCLASS_FIRST`, where
  `old_acc_target>=0.80` is allowed
  only as an intermediate old-class recovery gate and never as deployment
  success
- `seen_new_acc_target>=0.75`
- `target_adapter_required=true`
- `seen_new_evidence_gate_required=true`
- `seen_new_anchor_gate_required=true`
- `accepted_only_online_update_required=true`
- `oa_mse_onboard_adaptation_bundle=target_adapter+seen_new_evidence_gate+seen_new_anchor_gate+accepted_only_online_update+stage2_receiver_domain`, with Weibull EVT, pseudo-unknown energy, and Siamese verifier allowed only as Phase3 backup components

The Phase2 onboard adaptation bundle is indivisible for launchable OA-MSE rows.
A row that proposes only a subset of Target Adapter, seen-new evidence gate,
seen-new anchor gate, accepted-only online update,
and Stage2 receiver-domain separation must be classified as `LOCAL_PATCH_REQUIRED` or
`NON_LAUNCH_DIAGNOSTIC`, not launchable.

Validate Stage2 matrices with:

`tools/optimizer_validate_matrix.py --expected-count 64`

or a stricter equivalent before runner launch.

Validator `verdict=PASS` means schema/protocol validation passed. Runner
readiness must additionally inspect `launchability_summary.by_lane`:

- A lane with `LANE_HAS_LAUNCHABLE_ROWS` may proceed to local-first runner gates.
- A lane with `LANE_LOCAL_PATCH_REQUIRED_NO_LAUNCHABLE_ROWS` must stay local
  repair/deferred and must not generate remote dry-run, SCP, or launch attempts.
- Route-duplication repair counts are launch blockers for the affected rows,
  even when the overall matrix schema is valid.

`PHASE2_LOCAL_PATCH_REQUIRED` MUST NOT be set to `1` for the whole Phase2 lane when `launchability_summary.by_lane.phase2_spaceborne_fsl.runner_readiness=LANE_HAS_LAUNCHABLE_ROWS`.
In mixed Phase2 matrices, local-patch deferral must be row-scoped: only rows
whose `launchability_status` or route flags contain `LOCAL_PATCH_REQUIRED`,
`NON_LAUNCH_DIAGNOSTIC`, or route-duplication repair markers may be deferred
for local verification. If Phase2 has at least one launchable row and the row
passes local validator, protocol, path, hash, dry-run, capacity, and registry
gates, launcher generation must omit or set `PHASE2_LOCAL_PATCH_REQUIRED=0`
and let the scheduler launch the verified Phase2 rows while preserving exact
deferred retry records for non-launchable rows.

For Phase2 sample selection, target receiver domain may contain one or more receivers. The target receiver domain must be disjoint from CEN51 train receivers, and launchable Phase2 mainline rows must expose target-old and target-new sample coverage under the simplified LEO target view. Open-set / unknown rejection is Phase3 backup, not a Phase2 mainline blocker. The controller must repair or replace only invalid rows; do not require exactly one r_sat, and do not set all-lane PHASE2_LOCAL_PATCH_REQUIRED when any Phase2 row is launchable.

## Phase2 Sample Protocol

Phase2 launchable rows must obey the corrected 2026-07-07 sample boundary: target-old adaptation plus target-new learning on the same target receiver domain and simplified LEO target view is the Phase2 mainline; open-set rejection and unknown FAR are Phase3 backup.

### LEO_weak-only Data Reachability

- Phase2 receives only samples after one of `leo_clear_weak`,
  `leo_low_elev_weak`, or `leo_rain_weak` has actually been applied. This covers
  every Stage2-A/B/C target-old/target-new and optional Phase3-backup unknown
  support/query sample, adaptation/calibration/enrollment input, model-selection
  signal, rollback/ranking signal, and formal-evaluation input.
- Phase2 must not read, cache, reconstruct, or derive features, logits,
  prototypes, thresholds, TTA decisions, or any other decision signal from clean
  samples. Clean controls may run only in Phase1 or in a fully isolated offline
  reference workflow whose outputs do not enter Phase2.
- Every launchable Phase2 row must declare
  `phase2_sample_view_policy=leo_weak_only_no_clean_access`,
  `clean_sample_access=false`, `target_channel_view=leo_weak_only`, explicit
  `target_channel_scenarios` drawn only from the three allowed `leo_*_weak`
  scenarios, and a satellite seed or equivalent per-sample overlay provenance.
- Missing fields, a clean token in the target view/scenario, a scenario outside
  the allowed `leo_*_weak` family, or inability to prove that overlay was
  applied is `LOCAL_PROTOCOL_REPAIR_REQUIRED`. Such a row must not enter matrix
  launchability, runner execution, promotion, or formal claims. Historical
  clean-access artifacts may only be sealed as
  `PROTOCOL_INVALID_FOR_PHASE2`; do not generate new clean-access diagnostics.

### Query Role And Class-Quota Oracle Ban

- Every Phase2 query must be decided independently against the same registered
  class set plus the allowed reject/uncertain/defer outputs. The decision path
  must not receive the query's true old/new/unknown role, the true class count
  of a query batch, any per-class query quota, query ordering/block membership,
  or a batch-level Hungarian, optimal-transport, quota-matching, or reassignment
  result.
- Every launchable Phase2 row must declare
  `phase2_query_decision_policy=per_sample_all_registered_classes`,
  `phase2_query_role_oracle_access=false`,
  `phase2_query_true_batch_class_count_access=false`,
  `phase2_query_class_quota_access=false`, and
  `phase2_query_batch_global_assignment=false`.
- `phase2_query_class_count_access` is deprecated because it ambiguously mixes
  the legal registered-class count with the forbidden truth-derived query-batch
  class composition. It does not satisfy the current launchable schema.
- Missing fields, a true-valued guard, a semantic alias that enables the same
  information, or an exact command/config that enables role Oracle, class-count
  Oracle, class quota, Hungarian/optimal-transport assignment, or global batch
  reassignment is `LOCAL_PROTOCOL_REPAIR_REQUIRED`. Such a row must not launch,
  pass validation, enter promotion/ranking, or support a formal claim.
- This ban does not prohibit Stage2-B/C support labels, enrollment identities,
  or a pre-registered positive integer K-shot support count per class. It also
  does not change Phase1 source-side pseudo-label quota audits. Ground-truth
  query labels may be read only after predictions are frozen, solely to compute
  evaluation metrics.
- Historical role/quota Oracle artifacts may only be sealed as
  `PROTOCOL_INVALID_FOR_DEPLOYMENT` with subtype
  `ROLE_OR_CLASS_QUOTA_ORACLE`; do not generate new Oracle diagnostics.

### Base Model

- Current Phase2 OPGAC rows use `JREF_C9_MULTICOMP_M2_E220` for on-orbit
  inference/adaptation. Selection is explicit user direction plus recent JREF
  evidence that this row best preserves local receiver modes and receiver-floor
  behavior among the JREF diagnostics.
- Treat `JREF_C9_MULTICOMP_M2_E220` as a Stage2 base-model choice, not as a
  Phase1 mainline replacement, paper success, or deployment evidence.
- Do not select a checkpoint only because it is newest.
- Candidate fields: `stage2_base_model_id`,
  `stage2_base_model_role`, and `cen51_base_checkpoint_or_config` or an
  equivalent checkpoint/config pointer.

### Receiver Split

- Working assumption until audit: `cen51_train_rxs=rx0,rx1,rx2,rx3,rx4,rx5,rx6`.
- Each launchable Phase2 run must declare a non-empty target receiver domain,
  using `target_receiver_ids` or an equivalent target receiver field.
- The target receiver domain may contain one or more receivers, for example
  `rx7` or `rx7,rx8,rx9,rx10,rx11`.
- The target receiver domain must be disjoint from CEN51 train receivers.
- This applies equally to target-old support/query and target-new
  support/query. Old-class target support is still target-domain data and must
  not be sampled from CEN51/train receivers.
- Before completion claims, verify the actual CEN51 train/test receiver IDs from
  the CEN51 config, manifest, or logs.
- Candidate fields: `cen51_train_rxs`, `target_receiver_ids`; the latter may
  contain one or more target receivers for launchable Phase2 rows.
- WiSig compact-subset rows should also expose `source_receiver_labels` and
  `target_receiver_labels` when the real `rx_list` entries are labels such as
  `20-1` or `3-19`. Numeric receiver IDs may remain as compatibility fields,
  but sample selection and cross-pkl alignment must use the receiver labels.

### Transmitter Split

- Old classes are the six ManySig transmitters.
- Working TX assumption until audit: `target_old_tx_ids=0,1,2,3,4,5`.
- `target_new_tx_ids` must be outside the six old TX IDs.
- New TX samples must be received by the target receiver domain used for
  old-class target samples.
- `unknown_tx_ids` are required only for Phase3 open-set backup rows or
  optional evaluation-only metadata. When present, they must be outside the six
  old TX IDs and disjoint from `target_new_tx_ids`.
- WiSig `ManyTx` rows must expose resolved `target_new_tx_labels` and
  `unknown_tx_labels` copied from the actual `ManyTx.pkl tx_list`. Synthetic
  numeric placeholders, subset-local ranks, or prose such as "resolve labels
  later" are not launchable IDs. If compatibility `*_tx_ids` fields are kept,
  they must contain the same resolved labels or be treated as descriptive
  metadata only by the runner.
- Before launch, each resolved target-new TX label must be checked for all of
  the following under the row's target receiver label: not in the ManySig
  old-label set, resolvable in `ManyTx.tx_list`, and enough receiver-specific
  samples for the row's support/query request. If a Phase3 backup row carries
  unknown TX labels, those labels must pass the same checks and remain disjoint
  from `Y_new`. Aggregate non-old counts or aggregate sample totals do not
  prove per-TX launchability.
- Launchable Phase2 rows must expose target-old and target-new sample coverage;
  if local data lacks one side, mark the row `LOCAL_DATASET_EXTENSION_REQUIRED`,
  `LOCAL_PROTOCOL_REPAIR_REQUIRED`, or `NON_LAUNCH_DIAGNOSTIC` instead of
  claiming full Stage2-C.
- The legacy split `source_tx_ids=0,1,2,3` with `new_tx_ids=4,5` is invalid for
  launchable Phase2 rows unless a fresh TX audit proves a different ManySig map.

### Evidence Grain

The sample grain is `target receiver x transmitter`.

- `target_old`: ManySig old TX received by target/satellite receiver.
- `target_new`: non-ManySig TX received by target/satellite receiver.
- `unknown`: Phase3 backup non-old TX held out for rejection, received by
  target/satellite receiver. It may be present as evaluation-only metadata in
  Phase2 rows, but it is not a Phase2 mainline success gate.
- Source receiver samples may anchor prototypes, controls, replay, or base
  CEN51 references. They are not evidence that target-receiver old classes
  improved.

### Stage2 Modes

`Stage2-A_zero_label_deploy`

- Support: empty target-label support.
- Query: target-old and target-new reference query on target/satellite
  receivers. Unknown query is optional Phase3-backup evaluation metadata.
- Output space: old classes, plus optional Phase3-backup rejection.
- Allowed claims: old target recognition and target-new non-enrolled reference.
- Forbidden claims: new identity recognition and target-label threshold fitting.
- OPGAC mapping: build memory from ground/source old prototypes only; evaluate
  target-old query and reject target-new/unknown query. No target support,
  target threshold fitting, memory update, or query-driven overlap repair.
- OA-MSE mapping: source-only MSE-lite or source-calibrated open gate only.
  `U_orbit`, target-old fusion, and target-new registration are unavailable.

`Stage2-B_old_label_calibration`

- Support: small labeled target-old support only.
- Query: separate target-old query plus target-new non-enrolled reference query
  on the same target/satellite receiver set. Unknown query is optional
  Phase3-backup evaluation metadata.
- Output space: old classes, with optional uncertain/defer behavior.
- Allowed claims: old target lift, old retention, rollback, and deployment
  cost.
- Forbidden claims: seen-new identity accuracy.
- OPGAC mapping: target-old support may update support-only old Gaussian
  calibration, radii, and rollback/defer thresholds under the same target
  receiver domain. Target-new support is forbidden; target-new/unknown query is
  rejection evaluation only.
- OA-MSE mapping: target-old support may update old prototype/mask/radius,
  estimate shared `U_orbit`, and calibrate old energy/OpenMax/Mahalanobis
  gates. Target-new support is forbidden.

`Stage2-C_old_new_enrollment`

- Only applies if labeled target-new support is explicitly allowed.
- Support: target-old plus target-new seen support.
- Query: target-old and seen-new query. Unknown query is optional
  Phase3-backup evaluation metadata.
- Output space: old classes, seen-new classes, uncertain, and defer; rejection
  is optional when Phase3 backup is enabled.
- If new labels are unavailable, this mode is not a valid Phase2 mainline
  Stage2-C row and must not report seen-new identity accuracy.
- OA-MSE mapping: full old calibration plus seen-new registration is allowed
  only when target-new support is explicit. Unknown query remains evaluation
  only and cannot fit thresholds.
- OPGAC mapping: target-old support calibrates old Gaussian memory and
  target-new support registers seen-new Gaussian states. Unknown query remains
  evaluation only and cannot fit thresholds, overlap policies, rollback
  triggers, or memory updates.

### Required Phase2 Fields

Launchable Phase2 rows and reports must expose:

- `stage2_base_model_id`
- `stage2_base_model_role`
- `cen51_base_checkpoint_or_config`
- `cen51_train_rxs`
- `target_receiver_ids`
- `target_old_tx_ids`
- `target_new_tx_ids`
- `unknown_tx_ids` only for Phase3 backup rows or optional evaluation-only
  metadata.
- `target_new_tx_labels` for WiSig `ManyTx` rows; these must be exact
  `tx_list` labels, not synthetic numeric ranks or unresolved explanatory text.
- `unknown_tx_labels` when Phase3 backup is enabled; these must also be exact
  `tx_list` labels.
- `target_old_leo_support` for Stage2-B/C, or explicit empty/NA for Stage2-A
- `target_old_leo_query`
- `target_new_leo_support` only for Stage2-C seen-new enrollment; it must be
  empty/NA for Stage2-A/B
- `target_new_leo_query`
- `unknown_leo_query` only for Phase3 backup rows or optional evaluation-only
  metadata.
- `k_shot` or explicit `target_old_support_per_tx` / `target_new_support_per_tx`
  for Stage2-B/C. K must be a positive integer. `{1,2,5,10,15,20,50}` are
  recommended anchor values for comparable curves, not the only launchable
  values. K values above 20 must be labeled higher-shot, medium-shot, or
  saturation, not strict few-shot.
- `old_support_query_split`
- `new_support_query_split`, with empty target-new support for Stage2-A/B
- `phase2_sample_view_policy=leo_weak_only_no_clean_access`
- `clean_sample_access=false`
- `target_channel_view=leo_weak_only`
- `target_channel_scenarios` containing only `leo_clear_weak`,
  `leo_low_elev_weak`, and/or `leo_rain_weak`
- `satellite_seed` or an equivalent sample-level overlay provenance field
- `threshold_selection_label_scope`
- `unknown_query_eval_only=true` when unknown query is present
- `target_new_query_not_threshold_fit=true`
- `unknown_FAR_target<=0.05` only for Phase3 backup rows
- `FPR95_target` only for Phase3 backup rows
- `uncertain_policy`
- `old_acc_delta_pp`
- `new_acc_drop_pp` when Stage2-C is active
- `H_old_new` when Stage2-C is active
- `unknown_FAR` only for Phase3 backup rows
- AUROC/FPR95 only for Phase3 backup rows
- old->new and new->old confusion; unknown confusion only for Phase3 backup
- score-table diagnostics: candidate label/group, best old score, best
  seen-new score, seen-new-minus-old contrast, threshold deltas, and
  seen-new anchor similarity/delta
- OPGAC score-table diagnostics when `route_family=OPGAC_NET`: candidate
  label, best old score, best seen-new score, best reject score, top-2 margin,
  threshold delta, `opgac_old_score`, and `opgac_new_score`
- OPGAC optimizer metrics when `route_family=OPGAC_NET`: `old80_gap`,
  `old_new_hmean` or `H_old_new`, same-row rank, metric deficit vector,
  rollback rate, defer rate, and confusion counts; unknown/FAR metrics are
  Phase3 backup only
- `rescue`, `harm`, `net_gain`, `changed_pred_rate`
- rollback and deployed FAR status

If local code cannot expose these split/manifest/score-table fields, mark the
affected row `LOCAL_PATCH_REQUIRED` or `NON_LAUNCH_DIAGNOSTIC` and do not
launch it as a Stage2 job.

## Metric Definitions

- `old_acc = P[h(x)=y | y in Y_o]`.
- `seen_new_acc = P[h(x)=y | y in Y_n_seen]`; only Stage2-C may use this.
- `unknown_FAR = P[h(x) != reject | y in Y_n_unseen]`; Phase3 backup only.
- `unknown_rejection = 1 - unknown_FAR`; Phase3 backup only.
- `old_FRR = P[h(x)=reject | y in Y_o]`.
- `H_old_new = 2*old_acc*seen_new_acc/(old_acc+seen_new_acc)`.
- Optional `H_open` may include unknown rejection, but it must be labeled
  separately and not confused with `H_old_new`.
- `old80_gap = max(0, 0.80 - old_acc)` for current
  PHASE2_ADAPT_NEWCLASS_FIRST route
  selection. It is a repair deficit, not a deployment-success score.
- `old_unknown_hmean = 2*old_acc*(1-unknown_FAR)/(old_acc+1-unknown_FAR)`.
  Use it only as a Phase3 backup selector; it must not replace Phase2
  Stage2-C `H_old_new`.
- `opgac_same_row_rank` ranks complete candidate rows using metrics from the
  same run/candidate. Do not combine a best old accuracy from one row with a
  best FAR, AUROC, or coverage from another row.
- `opgac_metric_deficit_vector` records the active repair deficits, such as
  old80, FAR, coverage, old_FRR, rollback/defer, AUROC/FPR95, and scenario
  harm. Optimizer actions should target the deficit vector rather than a
  single decorative aggregate.

Stage2-C Phase2 mainline success requires constrained old/new improvement, not
raw acceptance:

- maximize `H_old_new` or constrained `seen_new_acc`
- satisfy `old_acc >= old_floor`
- satisfy `new_acc_drop_pp <= 2`
- satisfy `rollback = False`
- satisfy deployment cost bound `Cost <= C_max`

Exploratory FAR stress and `unknown_FAR <= 0.05` belong to Phase3 backup. They
must be labeled separately and cannot be reported as Phase2 mainline success.

Current H06/Phase2 optimization order is `PHASE2_ADAPT_NEWCLASS_FIRST`:

- First recover target-old performance to `old_acc>=0.80` under the applicable
  Stage2-B or Stage2-C protocol.
- Only after the OLD80 gate is reached may the optimizer treat `seen_new_acc`
  and `H_old_new` as the next primary objectives.
- Stage2-B rows may carry the OLD80 gate and optional Phase3-backup unknown
  diagnostics, but they must not claim `seen_new_acc`; seen-new optimization
  requires Stage2-C with target-new support/query legality.
- For current OPGAC rows, PHASE2_ADAPT_NEWCLASS_FIRST means first reduce
  `old80_gap` using
  support-only old Gaussian calibration under `JREF_C9_MULTICOMP_M2_E220`.
  A low `unknown_FAR`, high AUROC, or high coverage row is Phase3-backup
  diagnostic evidence only and cannot promote a Phase2 route that keeps
  `old_acc<0.80` or lacks target-new learning evidence.
- OLD80 is an intermediate route-selection gate. It does not weaken
  deployment success requirements, Stage2-C old/new success requirements,
  target split rules, unknown-query calibration bans, or clean-view claim
  boundaries.

## Deployment View

Satellite/LEO target view is deployment-primary. For future star-ground
enhancement, sample overlay, satellite stress, and PAIC rows, the
deployment-primary implementation is the simplified LEO residual channel
defined by `项目.md`.

- Phase2 is stricter than a deployment-primary preference: it is
  `LEO_weak-only` and has no clean-sample access. Clean view is a
  control/reference only for Phase1 or a fully isolated offline workflow whose
  outputs cannot affect Phase2.
- Report simplified-channel per-view results for `leo_clear_weak`,
  `leo_low_elev_weak`, and `leo_rain_weak` when those views exist.
- Report legacy results for `clear_leo`, `low_elev_leo`, `rain_leo`,
  `storm_mp`, and `mixed_orbit` only when the row is explicitly marked as a
  legacy control/diagnostic.
- Do not let Phase2 read or receive any signal derived from clean samples, and
  do not promote clean-view success into deployment claims.
- Do not let Phase2 receive query true-role, query class-count, per-class query
  quota, or batch-global assignment signals. Vectorized batching must preserve
  the same per-sample decision permission.

## Evidence Sweep

Before designing a next matrix for a completed lane, collect completed-lane
evidence and label gaps. Required labels include:

- `FULL_ARTIFACT_SWEEP_REQUIRED`
- `LOCAL_ARCHIVE_ROOTS`
- `REMOTE_CURRENT_ROOTS`
- `REMOTE_ARCHIVE_GAP_CHECK`
- `NO_TAIL_ONLY_ANALYSIS`
- `NO_LAUNCHER_ONLY_CONCLUSION`
- `LOGS_JSONL_METRICS_CSV_REQUIRED_WHEN_PRESENT`
- `STDOUT_FULL_SCAN_REQUIRED`
- `SCHEDULER_EVENTS_LIFECYCLE_REQUIRED`
- `SFE_ALL_GATES_NOT_BEST_ONLY`
- `SFE_SCORE_TABLE_REQUIRED`
- `MANIFEST_PROTOCOL_INTEGRITY_REQUIRED`
- `FTRC_ALL_EPOCHS_REQUIRED`
- `LOSS_CURVE_REQUIRED`
- `FULL_TRAINING_LOG_ANALYSIS_REQUIRED`
- `LOSS_LOG_OBSERVABILITY_REQUIRED`
- `FULL_LOSS_TELEMETRY_REQUIRED`
- `PER_EPOCH_METRICS_REQUIRED`
- `ADAPTER_LOSS_TRACE_REQUIRED`
- `LOSS_NORMAL_CLAIM_REQUIRES_CURVE`
- `CONFIG_LOSS_ALIGNMENT_REQUIRED`
- `ROLLBACK_DEPLOYED_VS_RAW_REQUIRED`
- `FINAL_VS_BEST_CHECKPOINT_REQUIRED`
- `PARTIAL_EVIDENCE_BOUNDARY`
- `REMOTE_CURRENT_REQUIRED`
- `DEFERRED_RETRY_CAPACITY_IS_NOT_FAILURE`

Unlaunched or deferred rows must not enter completed metrics.

## Training Log Telemetry Contract

For any row that performs training, fine-tuning, low-compute adaptation, or
federated local updates, the completed-lane evidence sweep must include a full
training-log analysis before it can make an optimization-effect claim.

Required evidence by row type:

- Phase1 / Safe-SSDG training rows must emit a config snapshot, full stdout,
  per-epoch loss telemetry, and either `metrics_epoch.csv` or
  `metrics_epoch.jsonl`. Every active loss term must be represented with raw
  and weighted values, including classification, domain, adversarial,
  consistency, group CE, Fishr, satellite classification/consistency,
  unlabeled/pseudo losses, gradient norms, pseudo-label stats, val/test
  metrics, and final-vs-best checkpoint state.
- Federated rows must emit per-round and per-client loss/component telemetry
  when training occurs. Non-finite JSON values must be normalized or reported
  as a telemetry defect; non-standard `NaN` JSON is a report-quality issue.
- Stage2 rows that train a low-compute target adapter must emit a per-step
  `loss_trace` containing at least total loss, CE loss, source-anchor loss,
  learning rate, gradient norm, and support accuracy. Pure evaluation rows must
  be labeled `EVAL_ONLY_NO_TRAINING_LOSS` instead of being judged on loss
  descent.

The optimizer must compare actual command/config values against the matrix row,
state policy, scheduler events, and emitted telemetry. A loss term that is
enabled by config but absent from telemetry, disabled but non-zero without an
explanation, or monitored under the wrong grain is
`CONFIG_LOSS_ALIGNMENT_REQUIRED` / `LOSS_LOG_OBSERVABILITY_REQUIRED`.

Allowed loss conclusions:

- `TRAINING_LOG_ANALYSIS_PASS`: full logs and structured telemetry are present,
  active loss terms are aligned to config, no fatal runtime marker exists, and
  the trend analysis supports the stated claim.
- `LOSS_TREND_PARTIAL_EVIDENCE`: some evidence exists but is incomplete; the
  report may discuss observed values, but must not claim "loss normally
  decreased" or "optimization was correct".
- `MISSING_LOSS_TELEMETRY`: no durable curve or step trace exists for a
  training/adaptation row. Artifact completion, startup PASS, 56/56 metrics, or
  absence of NaN/OOM is not sufficient to claim normal loss behavior.
- `LOSS_ANOMALY` or `RUNTIME_FAILURE`: non-finite losses/gradients, skipped
  steps, exploding/flat loss, final-vs-best regression, or loss/metric conflict
  must be reported as a defect or diagnostic limitation.

Post-run evidence integrity belongs in a dedicated report/evidence validation
surface, not in the pre-launch matrix validator. The matrix validator checks
launchability; completed logs, loss curves, optimizer parameters, monitor
parameters, and report artifacts are post-run evidence.

## Anchor And Diagnostic Floors

Centralized integrated anchor minimums:

- `strict_udu >= 84.0`
- `sat_floor >= 41.0`
- `receiver_floor >= 73.0`

Centralized diagnostic-only rule:

- If `sat_floor >= 41.0` but `strict_udu < 82.0` or
  `receiver_floor < 65.0`, label diagnostic-only.

Federated/VMB integrated anchor minimums:

- `strict_udu >= 77.5`
- `rx8_udu` or `receiver_floor >= 58.0`
- `sat_floor >= 37.0`

Federated/VMB diagnostic-only rule:

- If `sat_floor >= 37.5` but `rx8_udu` or `receiver_floor < 50.0`, label
  diagnostic-only.

Diagnostic candidates may inform mechanism hypotheses, but they must not be
reported as proven integrated anchors.

## Route Retirement And Invalidity

Candidate-level retired-route gates must replace invalid rows instead of
stopping the whole loop or shrinking the matrix.

Retire a `route_signature` when at least three independent completed evidence
points directly contradict the route's deployment claim. Record retired
signatures in `stage2_optimizer_state.json` under the Phase2 route retirement
policy.

Maintain `route_invalidity_ledger` for routes rejected by both principle and
experiment. Each entry should include:

- `route_signature`
- `status=PRINCIPLE_AND_EXPERIMENT_REJECTED`
- `exploration_count`
- `principle_rejection`
- `experimental_rejection`
- `do_not_retry_as`
- `evidence_refs`
- `replacement_policy`
- `reopen_policy`
- `last_reviewed_local`

Do not retry retired or double-rejected routes by rename, reseed, GPU move, or
knob-only stress. Reopen only with new mechanism evidence, protocol evidence, or
score-table evidence, labeled `REOPEN_REQUIRES_NEW_MECHANISM_EVIDENCE`.

## Subagent Review Requirements

Each optimizer turn should include independent review roles:

- Protocol review: Phase2 sample protocol and Stage2-A/B/C boundary.
- Evidence review: completed evidence, retired route, invalidity ledger, and
  metric interpretation.
- Runner review: local-first verification, path/registry uniqueness, capacity,
  remote verification, startup health, and SSH cleanup.
- Validation review after matrix generation.

Disagreement must be reported and resolved by source priority. Lack of subagent
tools is not a launch blocker by itself.

## Report Requirements

For every N607 design or runner attempt, create or update a local report before
ending the turn. The report must include:

- run ID, timestamp, operator, objective, hypothesis, and comparison target
- loaded control files and conflicts
- lane monitor states and process snapshots
- evidence scope and partial-evidence labels
- training log analysis artifact, loss trend verdict, loss telemetry gaps,
  optimizer/config alignment, monitor parameter snapshot, and any
  post-run evidence integrity result
- subagent review summaries
- local files changed and verification commands
- snapshot/manifest/hash details when files changed
- local-to-remote sync mapping
- exact remote command, Conda/Python environment, cwd, PID, GPU, log path, run
  path, and expected outputs
- dataset split, receiver split, TX split, seed, and metrics to watch
- startup health
- detailed final result tables, anomalies, interpretation, and recommended next
  inspection

Finished-run reports must include tabular result sections:

- a per-candidate or per-experiment table where each row contains metrics from
  the same candidate/run, including candidate ID, mechanism/category, lane,
  receiver/TX split, K-shot, seed, primary metrics, coverage/defer/rollback
  fields, loss/adapter summary, and verdict
- a comparison table against prior relevant runs, with claim boundaries for
  diagnostic-negative, startup PASS, runner completion, and deployment evidence
- a failure-mode table covering missing artifacts, runtime failures, protocol
  downgrades, metric conflicts, and next inspection

Do not report standalone maxima or minima as the main result. If marginal
max/min values are included, attach the candidate/run ID and the full same-row
metric context, or label them as distribution-only statistics. Interpretations
must use joint candidate rows or an explicitly named joint-ranking criterion,
not unrelated extrema reached by different candidates.

Preserve datasets, checkpoints, logs, metrics, reports, and run outputs. Do not
delete or overwrite them unless the user explicitly requests it.
