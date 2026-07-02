# phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004

## Objective

Run a Phase2 Stage2-C old+seen-new enrollment diagnostic from frozen `ADV3B02_CORE90_SOFT_E200`, using target-domain old-class samples and overlaid simplified LEO satellite-ground channel seen-new samples. Per the live user goal, this run does not consider unknown rejection: unknown TX are excluded from export, evaluation, rollback interpretation, and success metrics.

Success target for this scoped run:

| Metric | Target | Notes |
|---|---:|---|
| `seen_new_acc` | `>=65%` | computed over at least 2 target-new classes; this run uses 3 |
| `old_acc` | `>=80%` | target-old query on the same target receiver domain |
| `H_old_new` | diagnostic | same-row harmonic check for old/new balance |
| unknown FAR | not in scope | not deployment success; open-set validation remains a later separate task |

## Loaded Rules

| File | Status |
|---|---|
| `E:\type10-7\AGENTS.md` | read |
| `E:\type10-7\项目.md` | read |
| `E:\type10-7\tools\optimizer_control_manifest.md` | read |
| `E:\type10-7\automation_reports\CV-SincNet\automation_prompt_backups\20260615_001820_stage2_closed_loop_v4\stage2_prompt.md` | read |
| `E:\type10-7\tools\optimizer_workflow_contract.md` | read |
| `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json` | read |

No `项目.md` revision is made. This is a user-scoped no-unknown Stage2-C diagnostic, not deployable Stage2-C success under the standing open-world contract.

## Protocol

| Field | Value |
|---|---|
| Phase1 base | `ADV3B02_CORE90_SOFT_E200` |
| Checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| Source TX / `Y_old` | `0,1,2,3,4,5` in ManySig |
| Source receivers / `R_s` | `0,1,2,3,4,5,6` |
| Target receiver / `R_t` | `20-1` |
| Target-old support/query | ManySig `Y_old` on `20-1`, K in `{10,20}`, 30 query per TX |
| Target-new / `Y_new` | ManyTx labels `1-16,1-18,1-14`, K in `{10,20}`, 30 query per TX |
| Multi-new-class coverage | 3 seen-new classes, satisfying the user requirement of at least 2 |
| Unknown / `Y_unknown` | excluded by user scope |
| Channel view | simplified LEO residual, `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| Output space | old labels plus seen-new labels; no reject target is optimized |

The split preserves the core support/query boundary for the scoped task: `R_t` is outside source receivers, old and seen-new TX sets are disjoint, and both old and seen-new support/query are drawn from the same target receiver label under LEO overlay.

## Candidate Plan

| Candidate | GPU | K-old | K-new | Mechanism | Purpose |
|---|---:|---:|---:|---|---|
| `ADV3B02_STAGE2C_NOUNK_K10_BALANCED_3NEW` | 0 | 10 | 10 | support-center + soft/multiproto + seen-new override | first few-shot old/new balance check |
| `ADV3B02_STAGE2C_NOUNK_K10_OLDRESCUE_3NEW` | 1 | 10 | 10 | balanced + OLD80 head `support_cv_select/rescue_rejected` | rescue rejected old samples without `replace_all` |
| `ADV3B02_STAGE2C_NOUNK_K20_BALANCED_3NEW` | 2 | 20 | 20 | support-center + soft/multiproto + seen-new override | low-shot saturation check |
| `ADV3B02_STAGE2C_NOUNK_K20_OLDRESCUE_3NEW` | 3 | 20 | 20 | balanced + OLD80 head `support_cv_select/rescue_rejected` | old recovery at K20 while preserving seen-new labels |

`old80_head_apply_policy=replace_all` is intentionally not used because earlier ADV3B02 diagnostics showed it overwrites query decisions and is unsuitable for seen-new classification. The old-rescue variants only apply the OLD80 head to rejected/unknown predictions.

## Local Files

| File | Purpose |
|---|---|
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh` | Four-row Stage2-C no-unknown launcher |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004\report.md` | This report |
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh` | Git-backed mirror |
| `E:\type10-7\github_publish\CVS-RFFI-repo\automation_reports\CV-SincNet\phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004\report.md` | Git-backed mirror |

## Verification Plan

| Check | Status |
|---|---|
| Root version state | `E:\type10-7` is not a Git repository |
| Git-backed release state | publish mirror pending commit; root `automation_reports/` is ignored by default |
| Bash syntax | PASS locally for root launcher and publish mirror |
| Dry-run | PASS locally: 4 candidates, target-new `1-16,1-18,1-14`, no `--unknown_tx_ids`, no `replace_all` |
| N607 preflight | PASS direct-only read-only preflight |
| N607 live occupancy | PASS for launch: no GPU compute or active training processes detected |

## Planned Remote Sync

| Local | Remote |
|---|---|
| `code/scripts/launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh` |
| `automation_reports/CV-SincNet/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/report.md` |

## Claim Boundary

This run can satisfy the live user-scoped goal only if the same candidate row reaches `old_acc>=80%` and `seen_new_acc>=65%` with at least 2 seen-new classes. It cannot be claimed as open-world deployment success because unknown rejection is explicitly excluded.

## Local Verification

Version state before remote access:

| Scope | Evidence |
|---|---|
| Root workspace | `git status -sb` reports `fatal: not a git repository` |
| Git-backed mirror | branch `codex/cvs-rffi-release-20260626`, ahead of origin; this run adds the launcher, force-adds the ignored report mirror, and adds a path-scoped LF rule for the launcher |
| Snapshot | `E:\type10-7\code\snapshots\phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004\` |

| Artifact | SHA256 |
|---|---|
| `code/scripts/launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh` | `8C70A2FC8014DD8C93B93E570669A19620905D2B35C1C5CC11D234C26D91EC9A` |
| `automation_reports/CV-SincNet/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/local_dry_run.out` | `9AE04E78F3D7830D50C5409E86218F01FBF1D0C342E9432B0ECCD8A1840C7F3E` |

| Command | Result |
|---|---|
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh` | PASS |
| `bash -n github_publish/CVS-RFFI-repo/code/scripts/launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh` | PASS |
| `bash code/scripts/launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh --dry-run` | PASS |

Dry-run audit:

| Field | Observed |
|---|---:|
| Candidate rows | 4 |
| Target-new classes | 3 |
| `--unknown_tx_ids` occurrences | 0 |
| `replace_all` occurrences | 0 |

The no-unknown scope is therefore explicit in the generated commands, not only in this report.

The report content itself is fixed by the Git-backed mirror and local snapshot rather than a self-referential hash.

## N607 Preflight And Sync

| Check | Evidence |
|---|---|
| Direct preflight | PASS via `tools\n607_ssh_preflight.ps1`; host `dell-DSS8440`, project root visible, 8 GPUs visible |
| Live training occupancy | `tools\n607_training_inventory.py --direct-only --pretty`: no `gpu_compute`, no active training processes, no launcher context |
| Target path conflict | no existing remote launcher, run root, or log root for this `RUN_ID` |
| Disk | `/home` has 7.7T available, 26% used |
| SSH cleanup | local `ssh_process_count=0`, `n607_or_bridge_established_count=0` after preflight, occupancy, path checks, and sync |

Remote synchronized files:

| Remote file | SHA256 / status |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh` | `8c70a2fc8014dd8c93b93e570669a19620905d2b35c1c5cc11d234c26d91ec9a` |
| `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/report.md` | synced before launch verification |
| remote `bash -n` | PASS |
| remote dry-run | PASS; 4 candidates, `unknown_tx_ids_count=0`, `replace_all_count=0`, target-new line present |

Planned bounded launch command:

```bash
ssh -F E:\type10-7\tools\n607_ssh_config -o BatchMode=yes N607 'cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004 automation_reports/CV-SincNet/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004 && nohup env RUN_ID=phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004 bash code/scripts/launch_phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004.sh > logs/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/scheduler.out 2>&1 < /dev/null & echo scheduler_pid=$!'
```

Expected remote outputs:

| Path | Purpose |
|---|---|
| `logs/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/scheduler.out` | parent launcher log |
| `logs/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/<candidate>.out` | per-candidate export/eval log |
| `runs/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/<candidate>/features.npz` | exported support/query feature bundle |
| `runs/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/<candidate>/metrics.json` | candidate metrics |
| `runs/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/<candidate>/manifest.json` | candidate split/config manifest |
| `runs/phase2_adv3b02_stage2c_multinew_no_unknown_20260703_0004/<candidate>/score_table.csv` | row-level query score table |
