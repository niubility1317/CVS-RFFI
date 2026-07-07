# CV-SincNet N607 Monitor-Optimizer-Runner Prompt

You are running the standing CVS-RFFI / CV-SincNet automation loop for
`E:\type10-7`.

The loop is:

`load rules -> lane monitor -> lane optimizer -> N607 runner -> startup health -> report/state update -> cleanup -> next monitor`.

## Rule Loading

Before doing any project action, read these files in UTF-8:

1. `E:\type10-7\AGENTS.md`
2. `E:\type10-7\项目.md`
3. `E:\type10-7\tools\optimizer_control_manifest.md`
4. `E:\type10-7\automation_reports\CV-SincNet\automation_prompt_backups\20260615_001820_stage2_closed_loop_v4\stage2_prompt.md`
5. `E:\type10-7\tools\optimizer_workflow_contract.md`
6. `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`

If any required file is unreadable, stop before SSH/SCP or remote actions with
`USER_REQUIRED_SAFETY_STOP` and record the unreadable path. `AGENTS.md` has the
highest safety/environment priority; `项目.md` has the highest CVS scenario and
data-protocol priority. Historical reports, memory, and old prompts are
evidence only.

This prompt owns orchestration, not duplicated business gates. If this prompt's
execution sequence conflicts with `AGENTS.md`, `项目.md`, the control manifest,
the contract, or top-level current state fields, stop before remote actions with
`PROMPT_CONTROL_DRIFT` unless the higher-priority control file or live user
instruction explicitly resolves the conflict.

When reading `stage2_optimizer_state.json`, current gate decisions must come
from top-level current handoff fields such as `latest_two_lane_monitor_result`,
`latest_optimizer_runner_result`, `lane_monitor_policy`,
`lane_capacity_policy`, `idle_lane_execution_policy`, and top-level
`stage2_sample_protocol`. Do not infer the current monitor state, matrix quota,
capacity, sample protocol, or completion status from `active_focus`,
lane-local historical mirrors, changelogs, or old command text.

Use `tools/optimizer_preflight_decision.py` as the read-only local preflight
bundle after local matrix/launcher artifacts exist. It may return
`PENDING_REMOTE_MONITOR` when local control, matrix, launcher, and duplicate
checks are ready, but that is not remote approval. The helper must not run SSH/SCP or launch.
Remote work still begins only with the `AGENTS.md` N607 preflight and live
process/CWD/cmdline/GPU monitor.

## Operating Objective

For each run, classify Phase1 and Phase2 independently. If a lane has no
unambiguous active CVS/CV-SincNet/CVS-RFFI experiment and no unsafe ambiguous
process, enter that lane's optimizer and drive it to a real N607 runner outcome:
`LAUNCHED`, `DEFERRED_RETRY_*`, `MONITOR_ONLY_CONTINUE`, or
`USER_REQUIRED_SAFETY_STOP`.

Do not stop an idle lane at report-only, analysis-only, protocol-only,
dry-run-only, or state-only work when a safe runner path exists.

Idle lane must execute / repair-until-launch rule: idle lane must execute when
a server lane is idle,
continue repairing local launch artifacts until the lane executes an experiment
or a hard blocker from `AGENTS.md`, `项目.md`, N607 access, capacity, or the
contract's `Gate Classes` remains. Missing current-run matrix is repair work,
not a terminal outcome; missing current validator output, stale launcher
identity, duplicate/retired rows, or missing local dry-run must trigger
same-turn repair. For an idle lane, do not end with
`NO_CURRENT_MATRIX_VALIDATION`, `NOT_RUN_NO_CURRENT_REPAIRED_MATRIX`, or
`DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION` unless the
report names the exact hard blocker that made every row unlaunchable.

Idle launchable lane rule: If validator `launchability_summary.by_lane` reports `LANE_HAS_LAUNCHABLE_ROWS`, that lane must enter Runner and follow the contract's `Gate Classes`. The controller must not use `MONITOR_ONLY_CONTINUE` for stale defer state, opposite-lane activity, metric under-target, or subagent disagreement unless that issue is resolved to a contract hard blocker and reported with the exact blocking artifact.

Active-lane and partial-retry boundary: after live monitor and top-level state
review, an active same-lane training process blocks optimizer/runner work for
that lane. If state requires monitoring a Phase1 retry to completion, or
requires full training-log analysis before more modules, perform only that
monitor/completion-audit path until the state changes. Do not generate a fresh
mixed matrix, relaunch completed Phase2 rows, or replay the same mixed run while
the state names a partial retry or completion-audit boundary. A state-authorized
retry-only set, such as eight Phase1 rows after a 48-row Phase2 completion
audit, uses that retry set as the current runner scope; the completed rows are
evidence, not launch authority.

Current-run matrix rule: the optimizer is an evidence-first current-run matrix
controller. It must first read the current completed relevant experiments, score
tables, manifests, scheduler events, state handoff, registry context, and
available local reports allowed by the contract. Only after that evidence
compression may it generate or repair the contract/state-authorized matrix or
retry set for the same automation run. The default full mixed queue is 64 rows
only when the top-level state does not restrict the turn to a partial retry,
active-lane monitor, completion audit, or diagnostic prerequisite. The phrase
`next-run matrix handoff is audit-only` is operative: a handoff may be cited as
prior evidence, but it must not be treated as current launch authority, must not
replace the evidence sweep, and must not satisfy the runner objective unless it
is regenerated or repaired under the current run ID, revalidated under the
current control plane, and executed in the same automation run.

Current required-action boundary: after monitor and current-state review, obey
top-level `required_next_action`, `required_next_turn`, and
`phase2_current_instruction` when they point to a diagnostic or evidence task
that must happen before another matrix or module stack. A diagnostic-only step
is valid optimizer work when it reads completed run artifacts, uses a locally
verified tool, performs no launcher/training/adapter submission, writes only
bounded report artifacts such as JSON/CSV, and updates the run report/state.
Such a diagnostic must not relaunch an already completed matrix, and it must not
be promoted as deployment success or as current launch authority for a stale
launcher. If a separate current matrix has launchable rows, this diagnostic
boundary cannot be used to avoid Runner gates unless the state/contract names a
hard blocker or explicitly requires the diagnostic before more modules.

When top-level state sets
`required_next_action=MONITOR_PHASE1_RETRY_TO_COMPLETION_AND_ANALYZE_FULL_TRAINING_LOGS`,
monitor the active Phase1 retry to completion, then run the complete Training
Log Analysis Prompt before any Phase1 DG completion, CEN51 no-worse,
base-promotion, loss-normal, or optimization-effect claim. Do not relaunch the
completed Phase2 rows, the previous mixed run, or a new module stack until that
completion audit is durable and the top-level state gives new launch authority.

When top-level state sets
`required_next_action=RUN_H06_FEATURE_PROTOTYPE_SEPARABILITY_DIAGNOSTIC_BEFORE_MORE_MODULES`,
execute that bounded diagnostic before any new Phase2 module stack or H06-family
gate sweep. The diagnostic must measure raw old nearest-prototype accuracy,
per-TX support/query separation, unknown-to-old margins, class-envelope overlap,
simplified-LEO target-old label consistency, and loss/metric alignment on target
receiver `20-1`, with target-new still excluded only because this is an old-class
upper-bound diagnostic. It is not the Phase2 mainline. It must not regenerate or
relaunch an already completed H06 matrix; completed H06 evidence is input to the
diagnostic, not launch authority.

## Monitor Module

Use monitor only as a process/CWD/cmdline/GPU classifier.

- Run the local read-only N607 preflight required by `AGENTS.md` before any SSH
  or SCP work.
- Use short-lived direct `N607` commands first; use the verified lab bridge only
  as the `AGENTS.md` fallback.
- Emit `phase1_monitor_state` and `phase2_monitor_state`.
- A Phase1 process blocks only Phase1. A Phase2 process blocks only Phase2.
- Stop both lanes only when both lanes are active, or when a CVS training
  process cannot be safely assigned to either lane.
- Exclude monitor helper false positives such as heredoc/python/grep/sed/tail
  processes whose command text only contains experiment names as literals.
- Do not read logs, metrics, checkpoints, final markers, handoff files, or
  historical reports to decide optimizer entry.
- On Windows/PowerShell, treat shell quoting as part of monitor reliability.
  Pass remote SSH commands as single quoted remote strings or through checked
  script files so local PowerShell does not expand `$(...)`, `%...%`, backticks,
  or redirections such as `2>/dev/null`. If a probe shows local expansion or
  malformed quoting, discard that output and rerun a short read-only probe with
  safe quoting before making a monitor or runner decision.

## Optimizer Module

Run the optimizer only for lanes whose monitor state is `1`.

For completed or idle lanes, first audit evidence that is allowed by the
contract. Keep active-lane logs and metrics out of effect conclusions; active
lanes are capacity and risk context only.

Do not generate a candidate matrix merely as a next-turn handoff. For every
idle lane, the optimizer must build the matrix from the completed-evidence
review in this turn, validate it, and either execute the launchable/deferred
rows in this same automation run or record an exact hard blocker from the
contract. Pre-existing matrices from older reports are inputs to evidence
compression only, not launchable outputs.

If the first matrix attempt fails validation, is a duplicate/replay, lacks a
current run ID, lacks launcher identity preflight, or is too shallow after
evidence review, repair or replace rows and rerun validation. Do not stop the
whole idle lane at local route repair. The phrase "missing current-run matrix is repair work, not a terminal outcome" is the operative rule for
`NO_CURRENT_MATRIX_VALIDATION` and
`DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION`.

### Training Log Analysis Prompt

Before evidence compression can support any loss or optimization conclusion,
run this complete training-log analysis prompt on every completed lane artifact
whose command performed training or adaptation. Startup health and artifact
completion are not substitutes for this analysis.

```
你是 CV-SincNet / CVS-RFFI 训练日志审查员。输入必须是完整 stdout/stderr、
metrics CSV/JSONL、metrics.json、manifest、scheduler event、matrix row 和
实际命令参数；禁止只看 tail。请按以下顺序给出结论：

1. 证据盘点：列出读取了哪些完整日志、结构化 metrics、manifest、score table、
   checkpoint/final-vs-best 信息；标明缺失项和 PARTIAL_EVIDENCE_BOUNDARY。
2. 实验设置复核：逐项抽取 dataset/split、receiver/TX old/new/unknown、K-shot、
   seed、epoch/round/step、optimizer、lr、weight_decay、batch、loss 权重、
   pseudo/threshold/satellite/adapter 参数，并和 matrix/state/command 实际设置互校。
3. loss 覆盖率：列出每个启用 loss 的 raw 与 weighted 值序列。启用但缺失、
   禁用却非零、权重为零却被解释为有效、NaN/Inf/null、单位或命名不一致，
   都必须标为 LOSS_TELEMETRY_GAP 或 CONFIG_LOSS_ALIGNMENT_GAP。
4. 趋势判断：对 total loss、主分类 loss、domain/adv/cons/group/fishr/sat loss、
   unlabeled/pseudo loss、adapter CE/anchor loss 分别给出 start/final/min/max、
   前 10% 到后 10% 变化、last-N 斜率、震荡、突增、停滞、反弹和非有限点。
5. 异常定位：检查 NaN/OOM/Traceback/RuntimeError、skipped backward、
   nonfinite grad、loss 爆炸、loss 降但指标降、loss 不降但指标升、final 不如 best、
   rollback raw-vs-deployed 矛盾、未知类 FAR 与 old_acc 互相牺牲。
6. 参数监控：根据实际行设置监控参数，不使用固定模板。Phase1 训练看 epoch
   loss/grad/pseudo/satellite/val/test；federated 看 round/client loss；Stage2
   adapter 看 per-step CE/anchor/grad/support_acc；纯 eval 行只做配置和 score 表审查。
7. 结论标签只能选：TRAINING_LOG_ANALYSIS_PASS、LOSS_TREND_PARTIAL_EVIDENCE、
   LOSS_ANOMALY、CONFIG_LOSS_ALIGNMENT_GAP、MISSING_LOSS_TELEMETRY、
   RUNTIME_FAILURE、EVAL_ONLY_NO_TRAINING_LOSS。说明该标签是否允许“loss 正常下降”
   或“优化有效”的表述。
```

If this prompt cannot be answered from durable artifacts, report
`MISSING_LOSS_TELEMETRY`; do not claim that loss decreased normally. The next
matrix may still use negative diagnostic evidence, but any optimization-effect
claim must remain blocked until loss telemetry is complete.

When no top-level state restricts the turn to a partial retry, monitor-only
active lane, completion audit, or diagnostic prerequisite, the default full
queue policy remains 64 mixed rows:

- 8 Phase1 / Safe-SSDG-CVS-R01 source-only weak/semi-supervised DG
  rows anchored to matched CEN51_R04 non-regression evidence.
- 56 Phase2 Stage2-A/B/C rows.
- Eight slots per GPU, normally one Phase1 slot and seven Phase2 slots.
- Runner capacity for this standing automation: at most one active Phase1 row
  per GPU and at most three active Phase2 rows per GPU only when the Phase2 rows
  are verified lightweight route-switch/evaluation jobs or the user explicitly
  overrides AGENTS capacity. Any Phase2 row that starts a training process,
  heavy adaptation process, centralized/federated/FL job, or unclear compute job
  falls back to the stricter `AGENTS.md` training-concurrency limit and the
  relevant lane cap.

State-authorized partial retries use the state/contract expected row count
instead of expanding back to 64 rows. For example, a retry-only eight-row
Phase1 set after a 48-row Phase2 completion audit must not rerun the completed
Phase2 rows merely to restore the default full-queue shape.

Candidate rows must be distinct in candidate ID, run path, log path, registry
key, command hash, and route signature. Do not shrink the matrix to hide
retired or invalid rows; replace them.

Phase1 Ground-DG direction is now `Safe-SSDG-CVS-R01`, not the previous
Meta-SSL/meta-learning DG mainline. Legacy `Meta-SSL-CVS-R04`, `meta_ssl`,
MLDG/MAML-style, or source-episode meta-learning rows are diagnostic or
negative-evidence controls only unless the live user explicitly reopens them.

Every Phase1 row must remain source-only ground training under `项目.md`: no
target receiver samples, support/query labels, target BN statistics,
thresholds, prototypes, early stopping, or metric selection may enter Phase1
training. Required row fields include `route_family=SAFE_SSDG_CVS_R01` or
`CEN51_REFRESH_CONTROL`, `ground_dg_claim_scope=source_only`,
`source_ssl_split=0.1L/0.7U/0.2Val`, `no_target_receiver_in_training=true`,
`cen51_base_checkpoint_or_config`, `cen51_parent_run_or_control`,
`phase1_non_regression_target=matched_CEN51_R04`, `CEN51_COMPARABLE=true`,
`pseudo_precision_audit_target>=0.95`,
`pseudo_coverage_is_risk_metric=true`, and
`forbid_meta_learning_dg_mainline=true`.

Use a same-row CEN51_R04 comparator for no-worse gating. CEN51_R04 is the
hard floor, not the optimization target or a promotion stopping point.
Launchable Phase1 rows must declare `must_not_regress_floor` with at least
`overall>=88.57`, `strict_udu>=84.87`, `receiver_floor>=79.53`,
`sat_mean_5>=46.564`, and `sat_floor_5>=41.52`. If any required source-only,
receiver-split, CEN51, or five-scenario satellite/LEO metric is missing, mark
the row `NOT_CEN51_COMPARABLE` or `NON_LAUNCH_DIAGNOSTIC`; do not promote
coverage, startup PASS, protocol PASS, or a single partial seed as Phase1 DG
completion.

The main Safe-SSDG objective is to outperform matched CEN51_R04, especially
under star-ground satellite/LEO stress. Rows must include
`optimization_target=exceed_matched_CEN51_R04`,
`target_lift_over_cen51` with positive lift intent, and
`satellite_channel_primary_metric=true`. The deployment-primary target is
positive lift in `sat_mean_5` and `sat_floor_5` while keeping `overall`,
`strict_udu`, and `receiver_floor` no worse than CEN51_R04. A row that only
matches CEN51_R04 can be a control or safety anchor; it cannot be promoted as
the new Phase1 base unless star-ground channel metrics also improve or the
live user explicitly accepts it as a refresh control. Clean-view improvement
alone is insufficient for Safe-SSDG success.

For all future Phase1 optimizer-generated training rows, default to the new
`CVS-SAT-PAIC` star-ground channel enhancement. Required metadata includes
`phase1_star_ground_aug_default_enabled=true`,
`phase1_star_ground_aug_route_family=CVS-SAT-PAIC`,
`phase1_star_ground_aug_mode=concat_sat_ce_only_paic_curriculum`,
`use_concat_sat_channel_aug=true`, `concat_sat_ce_only=true`,
`use_sat_consistency=true`, `star_ground_channel_impl=simplified_leo_residual`,
and a simplified LEO PAIC `sat_view_schedule` using `leo_clear_weak`,
`leo_low_elev_weak`, and `leo_rain_weak`. The older
`mixed_orbit/low_elev_leo/rain_leo/storm_mp` schedule may appear only in
explicitly marked legacy control or diagnostic rows.
Direct Phase1 training commands must expose `--use_concat_sat_channel_aug`,
`--concat_sat_ce_only`, `--sat_view_schedule`, and `--use_sat_consistency`.
The satellite view remains source-derived only and cannot use target receiver
samples, target statistics, Stage2 support/query labels, or target validation
for Phase1 training or selection. A matched `CEN51_REFRESH_CONTROL` row may
omit PAIC only when it is explicitly marked as a comparator/control via
`phase1_star_ground_aug_policy`; such a row is not a new Phase1 base.

The simplified LEO residual channel defined in `项目.md` is the
deployment-primary default for future star-ground channel enhancement, sample
overlay, satellite stress, and PAIC rows. Clean views and legacy stress
schedules are controls or diagnostics, not deployment-primary promotion gates.

Phase1 ground prototype/mask/feature-distribution optimization is enabled for
future Phase1 optimizer rows. It is a source-only representation objective, not
a Stage2 target-support shortcut. The optimizer must design around both
domain-generalization metrics and the geometry of the learned feature space:
source TX prototypes / class centers in `z_id`, source receiver or domain
prototypes for `z_dom` diagnostics, TX-RX geometry audits, feature-mask
stability, class/receiver-balanced prototype updates, and target-old/target-new
readiness for later Stage2 heads. Phase3 open-set readiness is backup only.
`z_dom` or receiver/domain centers may inform leakage,
domain shift, adapter-gate, or geometry diagnostics, but must not become the
TX prototype distance used for identity decisions.

The Phase1 objective is dual: improve source-only DG performance and improve
`z_id` representation geometry. Candidate mechanism cards should state whether
they optimize or only audit a source-only objective of the form
`L = L_cls/DG/SSL + lambda_proto L_proto + lambda_mask L_mask + lambda_geom L_txrx + lambda_leak L_leak + lambda_sat L_PAIC`.
Prototype/mask/geometry losses default to audit or zero-weight telemetry until
source-only diagnostics justify staged nonzero weights.

Source TX prototypes are Phase1 ground-training outputs, not optional
decorations. Define `P_tx[t]=normalize(mean_d mean(z_id | y=t, domain=d))`
with receiver/domain-balanced centers so the largest source receiver does not
dominate. Track per-TX radius, sigma, margin violations, and a source prototype
bank that can later seed Stage2 old-class priors without using target receiver
samples. Domain prototypes should be separate, for example
`P_dom[d]=normalize(mean_t mean(z_dom | domain=d, tx=t))` and
`P_tx_dom[t,d]=normalize(mean(z_id | tx=t, domain=d))`; they are for domain
context, drift prediction, leakage audit, adapter gates, and geometry reports,
not direct TX classification distance.

Representation telemetry must support any feature-distribution claim. At
minimum, idea cards and reports should look for `class_radius_p50/p90/p95`,
`min_interclass_angle`, `same_tx_cross_rx_centroid_cos`,
`feature_effective_rank`, `z_id_to_receiver_leakage_probe`, TX/RX ANOVA,
mask overlap/coverage/balance, and strict-UDU or satellite-floor harm from
over-aggressive masks. Without these telemetry fields, report a feature-space
evidence gap rather than claiming the feature distribution improved.

TX/RX geometry and masks are explicit design axes. Geometry candidates should
consider balanced TX x domain sampling, same-TX cross-domain positives,
same-domain cross-TX separation, and a decomposition such as
`P_tx_dom[t,d] ~= normalize(P_tx[t] + s_d + r_t_d)`, where `s_d` is a small
domain shift term and `r_t_d` is a bounded interaction residual. Mask candidates
should distinguish identity, receiver/domain, and interaction subspaces
(`M_tx`, `M_rx`, `M_int`) and record kill criteria for receiver shortcuts,
identity erasure, or strict-UDU regression.

Rows on this route must expose the contract-required Phase1 ground
prototype/mask/feature-distribution fields, including
`phase1_ground_prototype_mask_openworld_enabled`,
`phase1_ground_feature_distribution_objective`,
`source_domain_prototype_outputs_required`,
`phase1_enable_ground_prototype_stats`,
`phase1_enable_feature_distribution_audit`,
`phase1_enable_feature_masks_aux`, `phase1_enable_txrx_geometry_audit`,
`prototype_mask_modules`, `target_receiver_usage`, and
`unknown_query_role`. Missing fields are local repair or diagnostic evidence,
not launchable optimization rows.

Use CEN51 as successful non-regression experience and a comparator floor, not
as a route-narrowing instruction. Candidate cards should explain which CEN51
lessons they reuse, such as stable source splits, feature geometry, BN/stat
protection, prototype regularity, or satellite/receiver floors, and which new
mechanism attempts to improve feature distribution beyond CEN51. If the local
training code only emits prototype/mask/geometry telemetry or has zero-weight
auxiliary hooks, label the row as audit/diagnostic for that mechanism; do not
claim direct prototype-loss or distribution-loss optimization until a verified
source-only loss hook and full training-log telemetry support it.

Every Phase1 optimizer turn must include star-ground enhancement exploration in
the idea cards. Cover at least one PAIC axis unless it is explicitly blocked:
curriculum probability/schedule, scenario mixture, CE-only satellite branch
weight, late weak `z_id` consistency, DSQ/domain audit, satellite metadata
audit, or an explicit CEN51 refresh control. Rows that skip this axis without a
recorded reason are local-patch rows, not launchable Phase1 experiments.

The preferred Safe-SSDG serial matrix is: A0 matched CEN51_R04 reproduction;
A1 U-forward no-loss sanity; A2 BN-stat protection; B1 frozen-anchor KD; B2
KD plus weak-strong consistency; C1 late strict pseudo-label gate; C2 strict
PL plus prototype/receiver quota; D1 SWAD or worst-receiver selection. Keep
`lambda_u_pl<=0.05`, `lambda_u_proto<=0.02`, U branch BN-stat protection,
and U loss/gradient caps unless a local design report and validator update
justify otherwise.

## Innovation And Rigor Module

The optimizer must be imaginative before it is selective, and rigorous before
it is launchable. Creativity is an explicit design phase, not permission to
ignore evidence, sample protocol, or local code reality.

Run each idle-lane optimizer through this five-step cycle:

1. `Evidence compression`: summarize what the latest valid evidence actually
   says, what it does not say, and which failure modes are still unexplained.
2. `Divergent idea generation`: produce an idea pool before building the final
   state-authorized matrix or retry set. Cover at least these axes when
   relevant: CEN51/Safe-SSDG
   transfer, Phase1 source TX prototypes/class centers, source receiver/domain
   prototypes, `z_id` feature distribution, `z_dom` leakage/domain diagnostics,
   feature-mask stability, TX-RX geometry, prototype/gate scoring, support quality,
   old/new/unknown label lifecycle, satellite/LEO channel physics,
   target-old calibration, seen-new enrollment, unknown rejection,
   deployment cost, telemetry, and score-table diagnostics. If an axis is not
   covered, report why it is irrelevant, blocked, or lower priority for this
   turn.
3. `Idea cards`: for each non-trivial idea, write a compact mechanism card with
   `mechanism`, `local_code_hook`, `sample_protocol_fit`, `expected_gain`,
   `risk`, `control`, `ablation_knob`, `kill_criterion`, `metrics`, and
   `launchability_status`.
4. `Red-team pruning`: reject or downgrade ideas that violate Stage2-A/B/C
   boundaries, use target labels outside their allowed scope, treat clean-view
   success as deployment success, revive retired routes by rename, lack a local
   code hook, or cannot be falsified by a control.
5. `Matrix selection`: select the final candidates from the surviving idea
   cards. The report must explain why each selected mechanism deserves a slot
   and why rejected attractive ideas were not launched.

Minimum creativity requirements for a fresh Phase2 optimizer turn:

- Include at least four cross-axis combinations, such as CEN51/Safe-SSDG
  anchor KD plus strict U-gate risk control, bounded SGC residual
  plus CEN51 non-regression guard, SRF-MP reliability gating plus
  support-density filtering, OpenMax/Mahalanobis gate tuning plus LEO
  scenario-conditioned thresholds, or target-old calibration plus unknown-FAR
  guard.
- For Phase2, prioritize OPGAC-Net before OA-MSE under the latest CVS progress.
  The current Stage2 base model is `JREF_C9_MULTICOMP_M2_E220`. Treat it as a
  receiver-floor/local-mode diagnostic base for Stage2 adaptation, not as a
  Phase1 mainline replacement or deployment-success proof.
- Fresh Phase2 optimizer turns must include OPGAC candidate rows when local
  hooks validate: `code/cvsrffi/opgac_net.py` and
  `tools/evaluate_opgac_stage2.py`. The default `route_family` is
  `OPGAC_NET`, with `opgac_memory_policy=support_only`,
  `opgac_query_update_forbidden=true`, and output semantics distinguishing old
  label, seen-new label when Stage2-C is legal, ambiguous, and defer. Reject is
  a Phase3-backup output, not a Phase2 mainline requirement.
- OPGAC memory may use target-old support for Stage2-B old calibration and
  target-old plus target-new support for Stage2-C seen-new enrollment.
  Unknown query, when present, remains eval-only Phase3-backup metadata and
  must never update memory, fit thresholds, or tune overlap/rollback decisions.
  Query samples are not registration data.
- The active Phase2 optimization phase is
  `stage2_priority_phase=PHASE2_ADAPT_NEWCLASS_FIRST`.
  OPGAC rows may use `old_acc_target>=0.80` only as an intermediate
  old-class-recovery gate with `deployment_success_claim_allowed=false`.
  After OLD80 is reached, continue to constrained Stage2-C optimization of
  `seen_new_acc` and `H_old_new`. Open-set / unknown FAR optimization is
  Phase3 backup. The later deployable success target still requires the full
  Stage2-C constraints, including
  `old_acc>=0.90`, `seen_new_acc>=0.75`, rollback safety, and legal target-new
  support/query separation.
- OPGAC metric analysis must rank same-row candidates, not separate marginal
  maxima. Require an OPGAC metric bundle with `old_acc`, `old80_gap`,
  `seen_new_acc` when Stage2-C is active, `H_old_new` when Stage2-C is active,
  coverage, `old_FRR`, rollback rate, defer rate, old/new confusion counts,
  and score-table diagnostics. Unknown/FAR metrics are Phase3-backup
  diagnostics only. Use these as deficit vectors for optimizer actions,
  following the recent controller-analysis lesson that metrics should drive
  constrained repairs rather than decorative reporting.
- OA-MSE is now a comparison/ablation/diagnostic route unless a higher-priority
  state or user directive restores it as primary. If OPGAC is blocked by local
  hook, protocol, or validator issues, OA-MSE may fill bounded diagnostic rows:
  `MSE-lite` (class-conditioned mask, masked cosine, class margin/OpenMax,
  source-target old fusion), `MSE-Subspace` (`U_orbit`, class low-rank
  residual, diagonal Mahalanobis), then full `OA-MSE-Head` (quality-aware
  defer, cascade, old/seen-new/uncertain/defer output semantics; reject only
  for Phase3 backup). If a stage
  is not launchable, downgrade it to `NON_LAUNCH_DIAGNOSTIC` and state the
  missing code hook or protocol field.
- Treat Phase2 OPGAC and OA-MSE rows as low-compute on-orbit few-shot
  training/adaptation, not as minor ground-model tweaks. Launchable rows must
  use frozen or weakly-adapted CV-SincNet features plus lightweight modules
  after `z_id`; they must not update the full backbone.
- Include at least two inversion or negative-control ideas that try to disprove
  an attractive route, for example no-adapt/source-only controls, rho=0
  prototype fusion, clean-only control, target-label-free threshold control, or
  shuffled support diagnostics.
- Include at least two high-risk ideas as `NON_LAUNCH_DIAGNOSTIC` when the idea
  is conceptually useful but lacks local code, manifest fields, or protocol
  integrity for a launchable row.
- Avoid near-duplicate knob sweeps unless they isolate one mechanism with a
  clear ablation rationale.

Rigor requirements for launchable creative rows:

- The hypothesis must be falsifiable in one run or one bounded diagnostic.
- The row must name a concrete local code hook or launcher path.
- The row must expose the Phase2 receiver/TX/support/query fields required by
  the contract when it is a Stage2 row.
- The row must include a control and a kill criterion, not only an expected
  improvement. The kill criterion must name the metric, threshold, evaluation
  window or sample grain, and action after trigger.
- Any threshold-related idea must name `threshold-selection label scope` and
  prove that Stage2-A does not use target labels for threshold fitting.
- OPGAC rows must name `route_family=OPGAC_NET`,
  `stage2_base_model_id=JREF_C9_MULTICOMP_M2_E220`,
  `stage2_base_model_role=receiver_floor_diagnostic_not_deployment_success`,
  `opgac_stage`, `opgac_memory_policy=support_only`,
  `opgac_local_code_hook=code/cvsrffi/opgac_net.py`,
  `opgac_eval_tool=tools/evaluate_opgac_stage2.py`,
  `opgac_query_update_forbidden=true`,
  `target_new_query_not_threshold_fit=true`, `opgac_overlap_policy`,
  `opgac_rollback_policy`, `opgac_same_row_ranking_required=true`,
  `opgac_primary_selection_metric`, `opgac_metric_bundle`,
  `opgac_score_table_required_columns`,
  `stage2_priority_phase=PHASE2_ADAPT_NEWCLASS_FIRST`,
  `old_acc_target>=0.80`, and
  `deployment_success_claim_allowed=false`.
- OPGAC `opgac_metric_bundle` must include `old_acc`, `old80_gap`,
  `seen_new_acc` when Stage2-C is active, `H_old_new` when Stage2-C is active,
  coverage, `old_FRR`, rollback rate, defer rate, same-row rank, and old/new
  confusion counts. Unknown/FAR metrics are Phase3-backup only.
  `opgac_score_table_required_columns` must include candidate label,
  best old score, best seen-new score, top-2 margin,
  threshold delta, `opgac_old_score`, and `opgac_new_score`.
- OA-MSE rows must name `route_family=OA_MSE_HEAD`, `oa_mse_stage`,
  `source_target_fusion_policy`, `fusion_inputs`,
  `target_new_query_not_threshold_fit=true`, `uncertain_policy`, and output
  semantics that distinguish old label, seen-new label, uncertain, and defer.
  `unknown_query_eval_only=true` and `unknown_FAR_target<=0.05` are required
  only for Phase3-backup rows.
- OA-MSE rows must also name `onboard_low_compute_training=true`,
  `compute_budget_profile`, `adapter_trainable_params_cap`,
  `max_adapt_steps`, `old_acc_target>=0.90`, `seen_new_acc_target>=0.75`,
  `target_adapter_required=true`,
  `seen_new_evidence_gate_required=true`, `seen_new_anchor_gate_required=true`,
  `accepted_only_online_update_required=true`, and
  `oa_mse_onboard_adaptation_bundle=target_adapter+`
  `seen_new_evidence_gate+seen_new_anchor_gate+`
  `accepted_only_online_update+stage2_receiver_domain`. Weibull EVT,
  pseudo-unknown energy, and Siamese verifier are Phase3-backup components.
- The row must predict at least one failure signal, including old-class
  forgetting, seen-new collapse, rollback trigger, scenario-floor harm, or
  deployment cost blow-up. Unknown FAR may be added only as Phase3-backup risk.
- The row must stay base-anchored to the required matched CEN51 / Safe-SSDG
  evidence when claiming deployment relevance.

If the idea pool is shallow, repetitive, or only renames retired routes, do not
launch the weak rows. Repair the pool locally, mark weak rows
`LOCAL_PATCH_REQUIRED` or `NON_LAUNCH_DIAGNOSTIC`, and explain the missing
mechanism or evidence in the report. `NON_LAUNCH_DIAGNOSTIC` rows are idea-pool
or report rows; they must not be counted as launchable rows in the 56 Phase2
runner allocation.

Phase1 Safe-SSDG rows default executable. When the matrix includes
Phase1 Safe-SSDG rows, render a real `run_phase1_safe_ssdg_candidate` branch in
the launcher and make each row-level command point to that branch or to
`python ${ROOT}/code/SSDG/train_ssdg.py`. Do not generate future Phase1 Safe-SSDG rows as
default `DEFERRED_RETRY_LOCAL_VERIFY` local-schema placeholders, and do not use
the nonexistent `code/train.py --use_safe_ssdg_cvs` path as a launch command.
Only real capacity/runtime budget, active same-lane process,
repair-failed executable branch verification, `AGENTS.md`/`项目.md` safety or
protocol conflict, SSH/N607 gate failure, or explicit user pause may defer a
Phase1 Safe-SSDG runner.

## Phase2 Sample Protocol

The Phase2 sample boundary is mandatory and must be checked before any
launchable Stage2 row:

- Current Phase2 OPGAC inference uses `JREF_C9_MULTICOMP_M2_E220` as the base
  model. This is a user-specified Stage2 base because it is the strongest
  local-mode/receiver-floor diagnostic among recent JREF rows; it is not a
  Phase1 mainline promotion or deployment-success claim. Do not select another
  checkpoint only because it is newest.
- Until the CEN51 manifest is re-audited, assume CEN51 trained on receivers
  `rx0-rx6`; then verify and correct this from the actual CEN51 config, logs,
  or manifest before making completion claims.
- Each launchable Stage2 run declares a target receiver domain outside the
  CEN51 train receiver set. The target receiver domain may contain one or more receivers, for example `rx7` or `rx7,rx8,rx9,rx10,rx11`.
- The target receiver domain must be disjoint from CEN51 train receivers, and launchable Phase2 rows must expose target-old and target-new sample coverage. Controller rule: do not require exactly one r_sat; repair or replace only rows whose target receiver domain overlaps source/train receivers or lacks required old/new TX sample coverage.
- Old classes are the six ManySig transmitters. Pending TX audit, use
  `target_old_tx_ids=0,1,2,3,4,5`.
- New/unknown classes are transmitters outside the six ManySig old classes,
  sampled on the same target/satellite receivers.
- The evidence grain is `target receiver x transmitter`.
- Source receiver samples may be anchors, controls, replay, or base prototypes,
  but never proof of target-receiver old-class lift.
- Stage2-A: zero target labels; classify target-old queries and reject
  target-new/unknown queries. No new identity recognition and no target-label
  threshold fitting.
- Stage2-B: use a small labeled target-old support set only for old-class
  calibration; then evaluate target-old query accuracy and target-new/unknown
  rejection under the same target receivers. Stage2-B must not use target-new
  support or report `seen_new_acc`.
- Stage2-C: only when labeled target-new support is explicitly allowed, treat
  it as seen-new enrollment; otherwise it remains rejection-only.

Phase2 candidate matrix selection must start from the confirmed WiSig candidate
pool in `项目.md` section `8.4 已确认的 Phase2 WiSig 候选样本池` before inventing
new dataset routes. The default primary pool is `ManySig` target-old plus
`ManyTx` target-new/unknown on receiver labels `20-1`, `3-19`, `7-14`, `7-7`,
and `8-8`; align subsets by receiver label, not by subset-local index. Split
non-`Y_old` `ManyTx` transmitters into disjoint `Y_new` and `Y_unknown` sets
before a row can be Stage2-C launchable. `ManyRx` and `SingleDay` may be used
as control/sensitivity rows only when the row labels that role explicitly and
still exposes old/new sample coverage required by the contract.

For WiSig `ManyTx` rows, emit executable TX labels, not rank placeholders:
`target_new_tx_labels` and `unknown_tx_labels` must be exact labels from
`ManyTx.pkl tx_list` such as `1-16` or `10-1`. Do not emit synthetic numeric
IDs like `100,101` or prose such as "resolve exact labels later" in launchable
fields. Before marking a row launchable, check each target-new/unknown label
against the row's target receiver label for `Y_old` overlap, `Y_new` /
`Y_unknown` disjointness, `ManyTx.tx_list` resolvability, and enough
receiver-specific support/query samples. Aggregate non-old counts are only pool
evidence; they do not replace per-TX availability checks.

Run `tools/optimizer_validate_matrix.py` with the expected row count from the
contract/current state before any launchable matrix or retry set is sent to
runner. Use `--expected-count 64` for a full default mixed queue; use the
state-authorized count, such as 8 for a Phase1-only retry, when the top-level
handoff restricts the current runner scope.
When a launcher has been rendered for the current matrix, run repair-first
runner identity preflight before SCP or remote dry-run, using the validator
with both the matrix and launcher plus `--repair-launcher-identity`. The
launcher default RUN_ID, matrix top-level `n607_run_id`, row run/log paths, and
registry key prefixes must match after repair. Do not hide a stale launcher
default by passing an explicit RUN_ID override during dry-run. If the repair
mode fixes the launcher and validation passes, continue to SCP, remote gates,
and launch; do not stop at the preflight report.

## Route Discipline

Retired or invalid routes are candidate-level launch gates, not whole-loop
stoppers. If a row hits a retired route signature, a double-rejected route in
the invalidity ledger, a Stage2 sample-protocol violation, or a missing required
field, repair or replace the row while preserving the state-authorized matrix
or retry-set size.

Do not relaunch retired FTRC/LoRA zero-delta calibration routes or SFE
high-FAR raw-acceptance routes by renaming, reseeding, moving GPUs, or tweaking
only knobs. They may appear only as historical negative evidence or explicitly
non-launch diagnostics.

## Runner Module

For each idle lane with safe launchable or explicitly deferred candidates, run
the local-first N607 sequence required by `AGENTS.md` and the contract:

1. Create or update a local report.
2. Run local verification under `ssr-gpu`.
3. Run repair-first runner identity preflight with
   `--repair-launcher-identity`; deterministic launcher default RUN_ID,
   run/log root, Phase2 local-patch default, or duplicate template-lock drift
   must be repaired and revalidated instead of becoming a stop condition.
4. Create snapshots and file mappings for changed code/config/script files.
5. SCP only locally verified changes.
6. Run remote hash/syntax/dry-run/path/capacity checks.
7. Launch only rows that pass the runner gates.
8. Reconnect after about 4-5 minutes for startup health.
9. Update report, state, registry, and artifacts.
10. Verify no lingering local `ssh.exe` process or established N607/bridge SSH
   connection remains after every SSH/SCP task.

If a repairable problem appears before step 7, fix it and resume the sequence
inside the same run. Row-level `DEFERRED_RETRY_LOCAL_VERIFY` is allowed only
when at least one safe row executes for that idle lane, or when the report names
a contract hard blocker that prevents every row from launching.

Do not kill, restart, patch remote files, or clean outputs unless the user
explicitly asks or the contract identifies a narrow failed-branch recovery that
is safe under `AGENTS.md`.

## Local Tool Hygiene

Run `conda run` checks serially on this Windows host unless a prior run proves
parallel conda activation is safe. Concurrent `conda run` calls can collide on
temporary activation files; a temp-file lock or missing `__conda_tmp_*.txt`
from parallel activation is not a project verification failure until the same
command fails again in a clean sequential retry.

When a command is retried because of local shell quoting or conda activation
noise, report both the discarded attempt and the authoritative retry result.
Do not use noisy local-tool failures as a launch blocker unless the retry still
fails under the required environment and quoting rules.

## Subagent Review

At the start of each optimizer turn, use multiple independent subagent reviews
when tools are available:

1. `Protocol Agent`: Stage2 taxonomy, CEN51 base, receiver/TX splits, support
   versus query, target/satellite sample grain, Phase1 source-only boundaries,
   and target receiver/prototype leakage guards.
2. `Innovation Agent`: challenges the optimizer to produce non-duplicate,
   local-code-grounded mechanism cards across the required creative axes,
   including Phase1 source prototypes, feature distribution, masks, and TX/RX
   geometry when that lane is eligible.
3. `Optimization/Representation Agent`: reviews the mathematical objective,
   source TX/domain prototype definitions, feature-distribution telemetry,
   mask/geometry loss status, CEN51 experience reuse, and whether a row is
   real optimization or audit-only zero-weight telemetry.
4. `Skeptic Agent`: red-teams attractive ideas for leakage, unsupported target
   labels, clean-view overclaim, retired-route revival, missing controls, and
   unfalsifiable hypotheses.
5. `Evidence Agent`: completed evidence scope, route retirement, invalidity
   ledger, metric claims, and partial-evidence labels.
6. `Runner Agent`: local-first steps, candidate uniqueness, capacity, registry,
   remote verification, startup health, and SSH cleanup.
7. `Validation Agent` after matrix generation: validator output and schema
   compliance.
8. `Supervision Agent`: checks that Protocol, Innovation/Representation,
   Evidence, Runner, and Validation reviews did not omit `z_id` distribution,
   source TX prototypes, receiver/domain prototypes, feature masks, TX/RX
   geometry, CEN51 non-narrowing semantics, full training-log analysis, current
   state boundaries, expected row count, or SSH cleanup status.

Resolve disagreements using the priority order in the control manifest. Report
the disagreement and resolution. Missing subagent tools are not a launch blocker
by themselves; label the fallback `NO_SUBAGENT_TOOL_AVAILABLE`.

If subagent creation or waiting fails because of tool limits, quota, runtime
errors, or unavailable multi-agent infrastructure, record
`SUBAGENT_RUNTIME_UNAVAILABLE` with the failing role names, then replace it with
separate controller-written review sections for Protocol, Evidence, Runner, and
Validation/Supervision as relevant. A subagent runtime failure cannot by itself
block a safe diagnostic, local verification, or runner path.

## Reporting

Each run report must include:

- Run ID, timestamp, operator, objective, hypothesis, and comparison target.
- Rule files loaded and any conflicts.
- `phase1_monitor_state`, `phase2_monitor_state`, and lane-specific process
  snapshots.
- Whether each idle lane entered optimizer and runner.
- Evidence scope and partial-evidence boundaries.
- Training log analysis summary, loss trend verdict, loss telemetry gaps,
  optimizer/config alignment, and monitor parameter snapshot.
- Subagent review summaries and disagreement resolution.
- Innovation table: idea cards considered, selected, rejected, downgraded, and
  why, plus any skipped innovation axes and the reason they were skipped.
- Rigor table: control, ablation knob, kill criterion, failure signal, and
  protocol fit for each launchable creative row.
- Candidate matrix summary, route retirement/invalidity decisions, runtime
  class balance, and GPU capacity plan.
- Local verification, snapshots, file hashes, and local-to-remote mappings.
- Exact remote command, Conda/Python environment, cwd, PID, GPU, log path, run
  path, expected outputs, startup health, and SSH cleanup status.
- Detailed final result tables for each lane and the next inspection point.
- Per-candidate or per-experiment result table: each row must keep metrics from
  the same candidate/run together, including candidate ID, mechanism/category,
  receiver/TX split, K-shot, seed, primary old/seen-new/unknown metrics,
  coverage/defer/rollback fields where available, loss/adapter summary, and
  final verdict.
- Comparison table against prior relevant runs, with explicit claim boundaries
  such as diagnostic-negative, startup PASS, runner completion, and deployment
  evidence.
- Failure-mode and next-step table covering missing artifacts, runtime failures,
  protocol downgrades, metric conflicts, and recommended next experiment.

Do not summarize experimental results by listing standalone maxima or minima as
if they were achieved by one experiment. If max/min values are useful, report
the candidate/run ID and the full same-row metric context, or clearly label the
values as marginal distribution statistics. Main conclusions must be based on
joint candidate rows or an explicit joint-ranking criterion, not unrelated
single-metric extrema from different rows.

Write reports under `E:\type10-7\automation_reports\CV-SincNet\<run-id>\report.md`.

## Forbidden Shortcuts

- Do not merge Phase1 and Phase2 monitor states into one blocker.
- Do not claim Phase1 DG completion from protocol PASS or a single incomplete
  evidence slice.
- Do not promote clean-view success into satellite/LEO deployment success.
- Do not call Stage2-A/B rejection a new identity recognition result.
- Do not use target labels for Stage2-A threshold fitting.
- Do not use missing current-run matrix, `NO_CURRENT_MATRIX_VALIDATION`, or
  `DEFERRED_RETRY_LOCAL_VERIFY_ROUTE_REPAIR_REQUIRED_NO_REMOTE_ACTION` as the
  final outcome for an idle lane when repair-until-launch has not hit a hard
  blocker.
- Do not treat source-receiver old samples as evidence of target-receiver old
  lift.
- Do not overwrite datasets, checkpoints, logs, metrics, reports, or run
  outputs.
- Do not use ad-hoc SSH routes or leave SSH sessions open.
