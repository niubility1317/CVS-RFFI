# CV-SincNet Stage2 Automation Control Manifest

This manifest is the source-of-truth map for the standing
`cv-sincnet-n607-monitor-optimizer-v4` automation. It exists to keep the
automation clear, non-duplicative, and reviewable.

## Control Load Order And Rule Ownership

Load required control files in this order:

1. `AGENTS.md`
2. `项目.md`
3. `tools/optimizer_control_manifest.md`
4. `automation_reports/CV-SincNet/automation_prompt_backups/20260615_001820_stage2_closed_loop_v4/stage2_prompt.md`
5. `tools/optimizer_workflow_contract.md`
6. `automation_reports/CV-SincNet/stage2_optimizer_state.json`

Rule ownership is separate from load order:

- `AGENTS.md` owns project safety, local-first editing, Conda environment,
  N607 SSH/SCP rules, reporting, and version management.
- `项目.md` owns the CVS scientific scenario, data protocol, single-satellite
  receiver deployment semantics, Stage2-A/B/C boundaries, and metric claim
  limits.
- The active `stage2_prompt.md` owns orchestration sequence for the recurring
  automation.
- `tools/optimizer_workflow_contract.md` owns durable optimizer and runner
  gates, candidate schema, Phase2 sample protocol, and validation rules.
- `stage2_optimizer_state.json` owns mutable evidence state, current indicator
  values, lane monitor results, route ledgers, queue history, and next-turn
  handoff.
- OA-MSE-Head direction is split the same way: the active prompt owns idea-pool
  execution, the contract and validator own `route_family=OA_MSE_HEAD`
  required fields, Stage2 leakage guards, low-compute onboard adaptation
  budget/targets, required module checks, and the indivisible
  `oa_mse_onboard_adaptation_bundle`; the state file owns the current
  machine-readable OA-MSE policy/defaults.
- Phase1 `CVS-SAT-PAIC` star-ground default is split the same way: the active
  prompt owns optimizer exploration instructions, the contract and validator
  own `phase1_star_ground_aug_default_enabled`, `CVS-SAT-PAIC`,
  `concat_sat_ce_only`, canonical `sat_view_schedule`, and source-only leakage
  guards, and the state file owns the current machine-readable Phase1 default
  policy. Updating only one of these surfaces is control-plane drift.
- Simplified star-ground channel default is split the same way: `项目.md`
  owns the scientific definition of the simplified LEO residual channel view;
  the active prompt owns the instruction to use it for future star-ground
  enhancement and sample overlay; the contract and validator own launchable row
  fields such as `star_ground_channel_impl=simplified_leo_residual` and
  deployment-primary simplified channel metadata; and the state file owns the
  current machine-readable default. Legacy five-scenario satellite views are
  controls/diagnostics only unless `项目.md` is explicitly revised.
- Matrix timing is also split this way: the active prompt owns the
  evidence-first current-run matrix execution sequence, the contract owns the
  runner gate that says next-run matrix handoff is audit-only, and the state file
  owns the current machine-readable handoff for whether the same automation run
  must generate, validate, and execute the matrix after reading completed
  evidence.
- Repair-first runner identity preflight is split this way: the active prompt
  owns the execution step before SCP/launch, the contract owns the rule that
  deterministic launcher identity drift must be repaired before it can block an
  experiment, the validator owns mechanical checks against matrix `n607_run_id`
  and
  `--repair-launcher-identity`, and the state file owns the current
  machine-readable preflight policy.
- Idle lane must execute / repair-until-launch is split this way: the active
  prompt owns the same-turn execution sequence, the contract owns hard blockers
  and disallowed terminal outcomes such as `NO_CURRENT_MATRIX_VALIDATION` and
  `DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION`, the
  validator owns mechanical launchability, and the state file owns the current
  machine-readable `idle_lane_execution_policy`.
- Training log observability is split this way: the active prompt owns the
  training log analysis prompt and when to run it, the contract owns required
  telemetry and claim gates such as `FULL_TRAINING_LOG_ANALYSIS_REQUIRED` and
  `LOSS_NORMAL_CLAIM_REQUIRES_CURVE`, training/adaptation code emits
  per-epoch or per-step loss telemetry, reports own per-run analysis artifacts,
  and the state file owns the current machine-readable
  `training_log_observability_policy`.
- Phase1 Safe-SSDG rows default executable is split this way: the active prompt
  owns the execution step, the contract and validator forbid default
  `DEFERRED_RETRY_LOCAL_VERIFY` local-schema placeholders, generated launchers
  must expose `run_phase1_safe_ssdg_candidate`, row-level commands must point at
  `python ${ROOT}/code/SSDG/train_ssdg.py` or that launcher entrypoint, and the state file
  owns the current machine-readable execution policy.
- Phase1 ground prototype/mask/feature-distribution optimization is split this
  way: the active prompt owns the design intent and multi-role review, the
  contract and validator own required row fields such as
  `phase1_ground_feature_distribution_objective`,
  `source_domain_prototype_outputs_required`, source-only target receiver
  leakage guards, and CEN51-as-non-regression semantics, and the state file owns
  the current machine-readable default policy. Updating only one surface is
  control-plane drift.
- read-only local preflight is split this way: `tools/optimizer_preflight_decision.py`
  owns the compact local decision bundle over control-file readability,
  current-state view, matrix validation, launcher identity, and duplicate
  registry/command hashes. `PENDING_REMOTE_MONITOR` means local artifacts are
  ready for the AGENTS-approved remote preflight; it is not remote approval.
  The helper must not run SSH/SCP or launch.
- Historical reports, snapshots, memory, and prior prompts are evidence only.
  They cannot override the live control files above.

If any of `AGENTS.md`, `项目.md`, the active prompt, this manifest, the
contract, or the state file cannot be read, the automation must stop before
remote actions with `USER_REQUIRED_SAFETY_STOP` and record the unreadable path.

## Ownership Boundaries

- The active prompt says what sequence to run and how to escalate decisions.
- The contract says what is allowed, required, invalid, or launch-blocking.
- The state file says what is currently true or last observed.
- Reports say what was done in a specific run.
- Tests and validators say what can be checked mechanically.

Do not copy long policy blocks between these files. If a rule is already in the
contract, the prompt should reference the contract instead of repeating it. If a
fact changes every run, store it in the state/report, not in the prompt.

## State Read Boundaries

For current-run gate decisions, use only the state file's top-level current
handoff fields, especially `latest_two_lane_monitor_result`,
`latest_optimizer_runner_result`, `lane_monitor_policy`,
`lane_capacity_policy`, `idle_lane_execution_policy`, and top-level
`training_log_observability_policy`, and top-level `stage2_sample_protocol`.
Prefer the compact current-state view from
`tools/optimizer_state_current_view.py` before reading the full state file
manually; the full state remains the evidence store, not the decision surface.

Lane subtrees, `active_focus`, `objective_changelog`, `target_changelog`, older
run summaries, and historical command text are audit evidence only. They must
not drive the current monitor state, matrix count, capacity, sample protocol, or
completion claim.

If the contract's semantic rule and the top-level state value disagree on a
current scalar policy, stop before remote actions with `CONTRACT_STATE_DRIFT`
unless `AGENTS.md` or a live user instruction explicitly resolves the conflict.

## Standing Automation Shape

The loop is:

1. Load rules.
2. Monitor Phase1 and Phase2 as independent lanes.
3. Enter the optimizer for each idle lane.
4. Read all current completed relevant experiment evidence before candidate
   selection, including full training-log/loss telemetry analysis when the row
   performed training or adaptation.
5. Produce or repair a validated 64-row mixed matrix for this current run, not
   for a future wakeup.
6. Apply the idle lane must execute / repair-until-launch rule: missing current-run matrix is repair work, not a terminal outcome. Missing
   validation, launcher identity drift, duplicate/retired rows, or
   `NO_CURRENT_MATRIX_VALIDATION` /
   `DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION` are
   same-turn repair work unless a hard blocker in the contract remains.
7. Run repair-first runner identity preflight: the automation must repair
   deterministic launcher default RUN_ID, run/log root, Phase2 local-patch
   default, and duplicate template-lock drift before treating identity drift as
   a blocker.
8. Run the N607 runner for safe launchable rows. Phase1 Safe-SSDG rows default
   executable through `run_phase1_safe_ssdg_candidate` / `python -m
   `python ${ROOT}/code/SSDG/train_ssdg.py`; do not default them to local-schema deferred placeholders.
9. Record startup health, state, registry, report, and SSH cleanup.
10. Continue monitoring.

The automation must not stop at analysis-only, protocol-only, dry-run-only, or
state-only work when an idle lane has a safe N607 runner path.

## Matrix Timing Boundary

The optimizer is an evidence-first current-run matrix controller. It must read
current completed experiment evidence first, then generate or repair the matrix
that the same automation run will validate and execute. A next-run matrix
handoff is audit-only: it may seed evidence compression, but it is not current
launch authority and cannot satisfy the standing automation objective by itself.

For an idle lane, repairable missing artifacts must be repaired in the same
turn. The controller must not terminate on analysis-only, report-only,
state-only, `NO_CURRENT_MATRIX_VALIDATION`, or
`DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION` when no
contract hard blocker exists.

## Runner Identity Boundary

Before any SCP, remote dry-run, or launch, the rendered launcher must go
through repair-first runner identity preflight. Deterministic launcher drift
must be repaired locally with `--repair-launcher-identity`, then validation and
dry-run must be rerun. This preflight is not an extra terminal gate: it may
block only when repair fails, a real run/log path collision remains, registry
duplicate evidence remains, or `AGENTS.md` / `项目.md` safety and protocol gates
forbid launch.

## Subagent Governance

Each optimizer turn should use at least three independent review roles when the
runtime exposes subagent tools:

- `Protocol Agent`: verifies Stage2-A/B/C taxonomy, CEN51 base selection,
  receiver split, ManySig old-TX split, target-new/unknown split, and support
  versus query usage.
- `Evidence Agent`: audits completed-lane evidence, retired routes, invalid
  route ledger entries, metric claims, and partial-evidence boundaries.
- `Runner Agent`: checks local-first verification, candidate paths, registry
  uniqueness, GPU capacity, remote verification plan, startup health plan, and
  SSH cleanup.

After a candidate matrix is produced, a fourth `Validation Agent` should review
the matrix against `tools/optimizer_validate_matrix.py` output and the contract
schema. The reviewer must inspect `launchability_summary` in addition to
`verdict`; `PASS` means schema/protocol validation passed, not that every lane
has launchable rows.

Subagent disagreement is not a whole-loop blocker. The controller must resolve
the disagreement by applying `AGENTS.md`, this manifest, the contract, and the
state file in priority order, then report the resolution. Missing subagent tools
must be labeled `NO_SUBAGENT_TOOL_AVAILABLE` and replaced with separate
controller review sections; this label alone cannot block a safe idle lane.

## Duplication Control

Before editing automation text, perform this check:

- If the text is a safety or launch gate, put it in the contract.
- If the text is an execution step, put it in the prompt.
- If the text is a current run result, put it in the report/state.
- If the text is a validator expectation, put it in code/tests and reference
  it from the contract.
- If the text is historical rationale, put it in the relevant report and cite
  it rather than copying it into the prompt.
- If the text appears only in a changelog or historical lane subtree, quarantine
  it as evidence until the top-level current state confirms it.

## Required Local Verification For Control Changes

For prompt/contract/control-manifest changes, run at minimum:

- JSON parse check for `automation_reports/CV-SincNet/stage2_optimizer_state.json`.
- `conda run -n ssr-gpu python -m py_compile tools/optimizer_validate_matrix.py`
  when the validator is present.
- `conda run -n ssr-gpu python -m py_compile tools/optimizer_state_current_view.py`
  when current-state read boundaries are changed.
- Focused tests that cover optimizer workflow tooling, currently
  `conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q code/tests/test_optimizer_workflow_tools.py`.

No N607 sync or launch is implied by a local control-plane refactor unless the
user explicitly asks for remote deployment after verification.
