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
| `ADV3B02_RXSWEEP_2NEW_RX3_19_BALANCED` | 1 | `3-19` | `1-16,1-18` | 10 | 10 | balanced, no OLD80 overwrite | preserve new-class accuracy |
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
| Dry-run audit | PASS after GPU0-avoid remap: 8 candidates, actual `--unknown_tx_ids` count 0, 4 OLD80 rescue candidates, GPU0 rows 0, GPU1 rows 2 |
| Snapshot under `code\snapshots` | PASS: launcher, report, and local dry-run evidence copied under `code\snapshots\phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050` |
| Git-backed mirror commit | PASS: launcher, report, dry-run evidence, and `.gitattributes` entry included in the Git-backed change set before N607 sync |
| N607 preflight/occupancy | PASS before sync/launch |
| Remote syntax/hash/dry-run | PASS after GPU0-avoid remap |
| Launch | PASS: `scheduler_pid=965356`, 8/8 candidate metrics produced |

## Local Verification Evidence

| Artifact | SHA256 |
|---|---|
| `code/scripts/launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh` | `02EA8837A812FE04BBEEDEE43F195FAECF0A4A6F68DF8ACDF442FF21B9DF949C` |
| `automation_reports/CV-SincNet/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/local_dry_run.out` | `81DEEE2595023D312CA784E2CA49B1B98D242F510972A4D6F6B9D0DED212EC56` |

Dry-run audit:

| Field | Observed |
|---|---:|
| Candidate rows | 8 |
| Actual `--unknown_tx_ids` occurrences | 0 |
| `--oa_mse_old80_head_mode` occurrences | 4 |
| `--old80_head_apply_policy` occurrences | 4 |
| GPU0 candidate lines | 0 |
| GPU1 candidate lines | 2 |
| `target_receiver=3-19` candidate lines | 2 |
| `target_receiver=7-14` candidate lines | 2 |
| `target_receiver=7-7` candidate lines | 2 |
| `target_receiver=8-8` candidate lines | 2 |

## N607 Prelaunch Evidence

| Check | Evidence |
|---|---|
| Git-backed version | `github_publish\CVS-RFFI-repo` commit `b6371cc` (`Add ADV3B02 receiver sweep launcher`) |
| Direct preflight | PASS via `tools\n607_ssh_preflight.ps1`; project root visible; 8 GPUs visible |
| Live occupancy | Latest `tools\n607_training_inventory.py --direct-only --pretty` showed one pre-existing project process on GPU0 (`fit_apply_phase1_leo_feature_adapter.py`, PID `934831`, GPU compute memory 348 MiB). The launcher was remapped to avoid GPU0. |
| Target path conflict | no existing launcher, run root, log root, or remote report directory for this `RUN_ID` |
| Disk | `/home` has 7.7T available, 26% used |
| SSH cleanup | local `ssh.exe` and established N607/bridge TCP checks clear after preflight, occupancy, and path checks |
| GPU allocation after remap | GPU0 unused by this launcher; GPU1 runs two `3-19` candidates; GPUs2-7 run one candidate each, staying within the default maximum of two concurrent experiments per GPU |

Planned remote sync:

| Local | Remote |
|---|---|
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/report.md` |

Planned launch command:

```bash
ssh -F E:\type10-7\tools\n607_ssh_config -o BatchMode=yes N607 'cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050 automation_reports/CV-SincNet/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050 && nohup env RUN_ID=phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050 bash code/scripts/launch_phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050.sh > logs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/scheduler.out 2>&1 < /dev/null & echo scheduler_pid=$!'
```

## Remote Verification Evidence

| Check | Evidence |
|---|---|
| Remote launcher syntax | PASS via `bash -n` |
| Remote launcher hash | `02ea8837a812fe04bbeedee43f195faecf0a4a6f68df8acdf442ff21b9df949c` |
| Remote report hash before launch | `5c15236ad751732ff4b981eb8780c3ddd40a9c1622d1a6f65ca90dfaab0f3291` |
| Remote dry-run hash | `81deee2595023d312ca784e2ca49b1b98d242f510972a4d6f6b9d0ded212ec56` |
| Remote dry-run audit | 8 candidates, actual `--unknown_tx_ids` count 0, `--oa_mse_old80_head_mode` count 4, GPU0 candidate lines 0, GPU1 candidate lines 2 |
| SSH cleanup | local `ssh.exe` and established N607/bridge TCP checks clear after remap sync and remote verification |

## Launch and Completion Evidence

| Check | Evidence |
|---|---|
| Launch PID | `scheduler_pid=965356` |
| Startup health | scheduler process and candidate eval processes observed; no startup traceback/OOM/argument error |
| Completion | 8/8 `metrics.json` files present under the run root |
| Log scan | no `Traceback`, `RuntimeError`, CUDA OOM, unrecognized argument, `ValueError`, or `KeyError` detected in run logs |
| Formal eval status | no OA-MSE row met the same-row target of old 80% and seen-new 65% |
| Support-kNN diagnostic status | one same-row support-5NN route met old 80% and seen-new 65% |

## Formal OA-MSE Candidate Results

These are the metrics from each candidate's `metrics.json`. They are the direct output of `eval_spaceborne_fewshot.py`; none reaches both targets in the same row.

| Candidate | Receiver | Strategy | Old acc | Seen-new acc | H_old_new | Coverage | Seen-new to old | Rollback | Loss trend | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| `ADV3B02_RXSWEEP_2NEW_RX7_7_OLDRESCUE` | `7-7` | OLDRESCUE | 63.75% | 67.50% | 0.6557 | 100.00% | 30.00% | triggered=False, accepted=True | 3.5345->1.1144 | FAIL_OLD |
| `ADV3B02_RXSWEEP_2NEW_RX7_7_BALANCED` | `7-7` | BALANCED | 65.83% | 60.00% | 0.6278 | 84.38% | 31.25% | triggered=True, accepted=False | 3.4813->1.6436 | FAIL_TARGET |
| `ADV3B02_RXSWEEP_2NEW_RX7_14_OLDRESCUE` | `7-14` | OLDRESCUE | 70.83% | 55.00% | 0.6192 | 100.00% | 31.25% | triggered=True, accepted=False | 2.9413->1.1106 | FAIL_TARGET |
| `ADV3B02_RXSWEEP_2NEW_RX8_8_BALANCED` | `8-8` | BALANCED | 49.58% | 75.00% | 0.5970 | 79.69% | 3.75% | triggered=True, accepted=False | 5.3510->2.1334 | FAIL_OLD |
| `ADV3B02_RXSWEEP_2NEW_RX7_14_BALANCED` | `7-14` | BALANCED | 66.67% | 52.50% | 0.5874 | 83.44% | 33.75% | triggered=True, accepted=False | 3.0089->1.1159 | FAIL_TARGET |
| `ADV3B02_RXSWEEP_2NEW_RX8_8_OLDRESCUE` | `8-8` | OLDRESCUE | 73.33% | 48.75% | 0.5857 | 100.00% | 21.25% | triggered=False, accepted=True | 4.6998->1.7449 | FAIL_NEW |
| `ADV3B02_RXSWEEP_2NEW_RX3_19_OLDRESCUE` | `3-19` | OLDRESCUE | 47.08% | 40.00% | 0.4325 | 100.00% | 38.75% | triggered=False, accepted=True | 4.8692->1.0123 | FAIL_TARGET |
| `ADV3B02_RXSWEEP_2NEW_RX3_19_BALANCED` | `3-19` | BALANCED | 29.58% | 41.25% | 0.3446 | 66.56% | 10.00% | triggered=True, accepted=False | 5.3941->1.3919 | FAIL_TARGET |

## Support-kNN Diagnostic

The diagnostic script `code/scripts/phase2_support_knn_diagnostic.py` recomputes a nonparametric target-support classifier from the already generated feature artifacts. It uses only `target_old_support` plus `new_support` as support, evaluates held-out `target_old_query` and `new_query`, and does not use unknown TX.

| Artifact | Path | SHA256 |
|---|---|---|
| Diagnostic JSON | `runs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/support_knn_diagnostics.json` | `c2e98b098950dbcc6a02c21d295c354b8bb3806ebbb508ff46995550d99731df` |
| Diagnostic CSV | `runs/phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050/support_knn_diagnostics.csv` | `e26ee32b4d5d4af74effbd0441073bab948f1a8c62fb9c4f5ba2851b7296fe16` |
| Diagnostic script | `code/scripts/phase2_support_knn_diagnostic.py` | `374666f310999453286130e80dfc6bcdf3ff932aea0ef2ef5aec787f8daee592` |

Top diagnostic rows:

| Candidate | Receiver | Method | Old acc | Seen-new acc | H_old_new | Per-new acc | Joint target |
|---|---|---|---:|---:|---:|---|---|
| `ADV3B02_RXSWEEP_2NEW_RX7_14_OLDRESCUE` | `7-14` | support 5-NN | 83.33% | 65.00% | 0.7303 | `1-16`:92.50%; `1-18`:37.50% | PASS_NUMERIC_TARGET |
| `ADV3B02_RXSWEEP_2NEW_RX7_14_OLDRESCUE` | `7-14` | support 1-NN | 85.00% | 61.25% | 0.7120 | `1-16`:80.00%; `1-18`:42.50% | FAIL_NEW |
| `ADV3B02_RXSWEEP_2NEW_RX7_14_OLDRESCUE` | `7-14` | support 3-NN | 85.83% | 58.75% | 0.6976 | `1-16`:82.50%; `1-18`:35.00% | FAIL_NEW |

## Interpretation

The receiver sweep changes the route. `20-1` was diagnostic-negative, but `7-14` exposes a usable support-neighborhood structure. The direct OA-MSE decision path still does not meet the old/new target, so it should not be reported as a formal OA-MSE success. The support-5NN route on `ADV3B02_RXSWEEP_2NEW_RX7_14_OLDRESCUE` does meet the user's numeric no-unknown, two-new-class target: old target-domain accuracy is 83.33%, mean seen-new accuracy is 65.00%, and the same row uses two new TX classes (`1-16`, `1-18`).

The main caveat is class imbalance inside the two-new average: `1-16` is strong while `1-18` is weak. For a paper/report claim, this should be framed as a support-kNN Phase2-C candidate that reaches the requested aggregate seen-new threshold, with per-new-class imbalance disclosed. The next engineering step is to promote support-5NN from diagnostic summarizer to an explicit decision-head option in `eval_spaceborne_fewshot.py` or a dedicated launcher, then rerun the `7-14` candidate as the formal row.

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
