# Phase2 OLD80_FIRST ADV3B02 Best Head48 Old-First Rollback Policy

| Item | Value |
|---|---|
| Run ID | `phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035` |
| Timestamp | 2026-07-02 10:35 Asia/Hong_Kong |
| Operator | Codex |
| Objective | Re-run `ADV3B02_CORE90_SOFT_E200` Phase2 with OLD80_FIRST rollback policy so target-old recovery is not reverted by unknown FAR during the old-class-first stage |
| Parent completed run | `phase2_old80first_adv3b02_best_head48_srfix_20260702_1028` |
| Phase1 source run | `phase1_adv3_mechanism32_queue_20260701` |
| Phase1 candidate | `ADV3B02_CORE90_SOFT_E200` |
| Phase1 checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| Matrix | same 48 OLD80_FIRST head candidates |
| K grid | old support per TX: 2,3,5,10 |

## Rationale

The parent run completed 48/48 and showed raw target-old recovery up to `74.17%`, but all rows were deployed as baseline because the default rollback policy treated `unknown_false_accept_rate` as a hard max-rise guard. Under OLD80_FIRST, unknown FAR must be logged but must not be the deployment rollback criterion until old-class accuracy is recovered.

## Policy Change

| File | Purpose | SHA256 |
|---|---|---|
| `E:\type10-7\code\configs\phase2_old80_first_rollback_policy.json` | OLD80_FIRST rollback policy: old/known/coverage guards only; unknown metrics eval-only | `0f93692523908aa9d02732b01d9eb63ce6ac582a01fc24a2c80e0c5595137718` |
| `E:\type10-7\code\scripts\launch_phase2_old80first_head48_oldfirst_policy_20260702_1035.sh` | 48-row launcher with `--rollback_policy_json` inserted in every eval command | `353bb106b0697a41fca725959eaf9286c4ee2efcbe9497f64ab3c2f3975022e7` |

Policy rules:

| Metric | Mode | Threshold | Role |
|---|---|---:|---|
| `old_class_accuracy` | `max_drop` | 0.05 | keep old-class safety guard |
| `known_accuracy` | `max_drop` | 0.05 | keep known-class safety guard |
| `coverage` | `min` | 0.20 | prevent coverage collapse |
| `unknown_false_accept_rate` | eval-only | n/a | logged, not rollback, during OLD80_FIRST |

## Local Verification

```powershell
conda run -n ssr-gpu python -m json.tool code\configs\phase2_old80_first_rollback_policy.json
bash -n code/scripts/launch_phase2_old80first_head48_oldfirst_policy_20260702_1035.sh
```

Launcher audit: 48/48 `eval_spaceborne_fewshot.py` commands contain `--rollback_policy_json`.

## Version And Snapshot

| Item | Value |
|---|---|
| Root Git status | `E:\type10-7` is not a Git repository |
| Local snapshot | `E:\type10-7\code\snapshots\phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035` |
| Git-backed mirror | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| Git-backed status before launch | new policy and launcher mirrored; report evidence pending final commit after launch update |

## Sync Mapping

| Local | Remote |
|---|---|
| `E:\type10-7\code\configs\phase2_old80_first_rollback_policy.json` | `/home/szu2070436088/2510044040/CV-SincNet/code/configs/phase2_old80_first_rollback_policy.json` |
| `E:\type10-7\code\scripts\launch_phase2_old80first_head48_oldfirst_policy_20260702_1035.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_old80first_head48_oldfirst_policy_20260702_1035.sh` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035/report.md` |

## Remote Command

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
mkdir -p logs/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035 runs/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035 automation_reports/CV-SincNet/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035
nohup env RUN_ID=phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035 TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth STAGE2_MAX_ACTIVE_PER_GPU=2 INCLUDE_EXTERNAL_GPU_PROCS=1 SCHEDULER_POLL_SECONDS=5 bash code/scripts/launch_phase2_old80first_head48_oldfirst_policy_20260702_1035.sh > logs/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035/launcher_submit.out 2>&1 &
```

## Expected Outputs

| Path | Expected content |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035/launcher_submit.out` | scheduler output |
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035/OA_MSE_H06_OLD80FIRST*.out` | per-candidate logs |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035/*/metrics.json` | per-candidate metrics |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035/summary_old80first_metrics.csv` | parsed same-row summary |

## Success Criteria

Primary: deployed target-old `old_class_accuracy` reaches at least 72%, preferably 80%. Unknown FAR/AUROC/FPR95 remain recorded as eval-only context and do not override OLD80_FIRST deployment selection in this run.
