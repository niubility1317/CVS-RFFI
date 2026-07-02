# ADV3B02 Phase2-C receiver sweep, 2-new, no unknown

## Run Identity

| Field | Value |
|---|---|
| Run ID | `phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050` |
| Timestamp | 2026-07-03 00:50 Asia/Hong_Kong |
| Operator | Codex |
| Objective | Test whether another confirmed Stage2-C target receiver can meet `old_acc>=80%` and `seen_new_acc>=65%` with at least two seen-new TX classes |
| Base checkpoint | `runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| Unknown policy | Actual unknown TX are excluded from export, evaluation, thresholds, and success metrics for this live scoped goal |

## Protocol Boundary

This is a Stage2-C diagnostic scoped to the user's no-unknown request. It keeps the CVS rules from `项目.md`:

| Field | Value |
|---|---|
| Source receivers `R_s` | `0,1,2,3,4,5,6` / confirmed ManySig source receiver labels from the current launcher convention |
| Target receivers `R_t` | `3-19`, `7-14`, `7-7`, `8-8`; all are in the confirmed Phase2 target pool and disjoint from `R_s` |
| Old TX | `0,1,2,3,4,5` / `Y_old` |
| New TX | `1-16,1-18` from `ManyTx.pkl`, non-`Y_old`, two seen-new classes |
| K-shot | K-old=10 and K-new=10 |
| Channel view | target-old and target-new both use satellite view with `simplified_leo_residual` and `leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| Success claim | A row passes only if the same candidate reaches old 80% and seen-new 65%; no unknown-rejection claim is made |

## Rationale

The prior two `20-1` runs are diagnostic-negative: simple multi-new Stage2-C peaked at old 50.56% / seen-new 58.89%, while old-retain follow-up peaked at old 61.25% / seen-new 30.00%. Post-hoc support-kNN and old-logit sweeps also failed to produce an 80% / 65% same-row tradeoff on `20-1`. This run tests whether the blocker is the specific target receiver rather than only terminal thresholds.

## Candidate Matrix

| Candidate | GPU | Target receiver | New TX | K-old | K-new | Strategy | Goal |
|---|---:|---|---|---:|---:|---|---|
| `ADV3B02_RXSWEEP_2NEW_RX3_19_BALANCED` | 0 | `3-19` | `1-16,1-18` | 10 | 10 | balanced, no OLD80 overwrite | preserve new-class accuracy |
| `ADV3B02_RXSWEEP_2NEW_RX3_19_OLDRESCUE` | 1 | `3-19` | `1-16,1-18` | 10 | 10 | OLD80 `support_cv_select` + `rescue_rejected` | recover old without replacing accepted new |
| `ADV3B02_RXSWEEP_2NEW_RX7_14_BALANCED` | 2 | `7-14` | `1-16,1-18` | 10 | 10 | balanced, no OLD80 overwrite | preserve new-class accuracy |
| `ADV3B02_RXSWEEP_2NEW_RX7_14_OLDRESCUE` | 3 | `7-14` | `1-16,1-18` | 10 | 10 | OLD80 `support_cv_select` + `rescue_rejected` | recover old without replacing accepted new |
| `ADV3B02_RXSWEEP_2NEW_RX7_7_BALANCED` | 4 | `7-7` | `1-16,1-18` | 10 | 10 | balanced, no OLD80 overwrite | preserve new-class accuracy |
| `ADV3B02_RXSWEEP_2NEW_RX7_7_OLDRESCUE` | 5 | `7-7` | `1-16,1-18` | 10 | 10 | OLD80 `support_cv_select` + `rescue_rejected` | recover old without replacing accepted new |
| `ADV3B02_RXSWEEP_2NEW_RX8_8_BALANCED` | 6 | `8-8` | `1-16,1-18` | 10 | 10 | balanced, no OLD80 overwrite | preserve new-class accuracy |
| `ADV3B02_RXSWEEP_2NEW_RX8_8_OLDRESCUE` | 7 | `8-8` | `1-16,1-18` | 10 | 10 | OLD80 `support_cv_select` + `rescue_rejected` | recover old without replacing accepted new |

## Local Files

| File | Purpose |
|---|---|
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh` | Eight-candidate receiver sweep launcher |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050\report.md` | This report |

## Verification Plan

| Check | Status |
|---|---|
| Git state before edit | root tree is not a Git repo; Git-backed mirror clean on branch `codex/cvs-rffi-release-20260626` before this launcher |
| Bash syntax | PASS via `conda activate ssr-gpu; bash -n code/scripts/launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh` |
| Dry-run audit | PASS: 8 candidates, actual `--unknown_tx_ids` count 0, 4 OLD80 rescue candidates, each receiver has 2 rows |
| Snapshot under `code\snapshots` | PASS: launcher, report, and local dry-run evidence copied under `code\snapshots\phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050` |
| Git-backed mirror commit | PASS: launcher, report, dry-run evidence, and `.gitattributes` entry included in the Git-backed change set before N607 sync |
| N607 preflight/occupancy | PASS before sync/launch |
| Remote syntax/hash/dry-run | pending |
| Launch | pending |

## Local Verification Evidence

| Artifact | SHA256 |
|---|---|
| `code/scripts/launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh` | `D993D7CD8FB431E45688A6ABC0C0FCB71BE971BBB4547BD0A8228AD5724DAF4A` |
| `automation_reports/CV-SincNet/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/local_dry_run.out` | `2F5CA33737D9CFA8A7643DCDDF4F164F17CBD7F2A385C90961BB96CF1912DEC3` |

Dry-run audit:

| Field | Observed |
|---|---:|
| Candidate rows | 8 |
| Actual `--unknown_tx_ids` occurrences | 0 |
| `--oa_mse_old80_head_mode` occurrences | 4 |
| `--old80_head_apply_policy` occurrences | 4 |
| `target_receiver=3-19` candidate lines | 2 |
| `target_receiver=7-14` candidate lines | 2 |
| `target_receiver=7-7` candidate lines | 2 |
| `target_receiver=8-8` candidate lines | 2 |

## N607 Prelaunch Evidence

| Check | Evidence |
|---|---|
| Git-backed version | `github_publish\CVS-RFFI-repo` commit `b6371cc` (`Add ADV3B02 receiver sweep launcher`) |
| Direct preflight | PASS via `tools\n607_ssh_preflight.ps1`; project root visible; 8 GPUs visible |
| Live occupancy | `tools\n607_training_inventory.py --direct-only --pretty`: `gpu_compute=[]`, `active_training_processes=[]`, no launcher context |
| Target path conflict | no existing launcher, run root, log root, or remote report directory for this `RUN_ID` |
| Disk | `/home` has 7.7T available, 26% used |
| SSH cleanup | local `ssh.exe` and established N607/bridge TCP checks clear after preflight, occupancy, and path checks |

Planned remote sync:

| Local | Remote |
|---|---|
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/report.md` |

Planned launch command:

```bash
ssh -F E:\type10-7\tools\n607_ssh_config -o BatchMode=yes N607 'cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050 automation_reports/CV-SincNet/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050 && nohup env RUN_ID=phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050 bash code/scripts/launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh > logs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/scheduler.out 2>&1 < /dev/null & echo scheduler_pid=$!'
```

## Expected Outputs

| Artifact | Remote path |
|---|---|
| Run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050` |
| Log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050` |
| Scheduler log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/scheduler.out` |
| Per-candidate metrics | `runs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/<candidate>/metrics.json` |
| Per-candidate manifest | `runs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/<candidate>/manifest.json` |
| Per-candidate score table | `runs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/<candidate>/score_table.csv` |

## Risks and Inspection Notes

| Risk | Inspection |
|---|---|
| Target receiver has too weak old-domain separation | Compare old acc by receiver and OLD80 rescue mode |
| New TX `1-16` remains unstable across receivers | Inspect per-new-class accuracy in `metrics.json` |
| OLD80 rescue harms new recognition | Compare balanced vs OLDRESCUE within the same receiver |
| No row reaches both thresholds | Treat as diagnostic-negative for ADV3B02 Stage2-C no-unknown with current K=10 two-new setup; next step would need a different feature/update mechanism, not another threshold-only tweak |
