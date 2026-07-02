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
| Git-backed policy commit | `fcc0d74` (`Add OLD80_FIRST rollback policy for Phase2`) |

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

## Final Status

| Item | Value |
|---|---|
| Completion | 48/48 candidates completed |
| Failed candidates | 0 |
| Active remote processes after completion | 0 |
| Rollback triggered | 0/48 |
| Summary CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_adv3b02_best_head48_oldpolicy_20260702_1035\summary_old80first_metrics.csv` |
| Summary CSV SHA256 | `4c185d14cc0572092c38ace38071b44eae96c76ab0b4d488958ec988824d2d87` |

## Result Summary

The OLD80_FIRST rollback policy fixed the structural deployment issue: raw and deployed old-class accuracy now match because unknown FAR is recorded but no longer triggers rollback in this stage.

| Metric | Max | Mean | Min |
|---|---:|---:|---:|
| Raw old acc | 77.50 | 66.62 | 56.25 |
| Deployed old acc | 77.50 | 66.62 | 56.25 |
| Baseline old acc | 37.92 | 28.93 | 16.25 |
| Unknown FAR | 100.00 | 100.00 | 100.00 |
| AUROC | 73.28 | 63.71 | 52.19 |

Best deployed OLD80 rows by K:

| K old support | Best candidate | Deployed old acc | Baseline old acc | Deployed full acc | Unknown FAR | AUROC | Verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 2 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_A_MSE_SUBSPACE_KOLD2_KNEW0` | 70.42 | 34.58 | 52.81 | 100.00 | 60.60 | improved, below 72 |
| 3 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU5_B_MSE_SUBSPACE_KOLD3_KNEW0` | 70.00 | 33.33 | 52.50 | 100.00 | 62.77 | improved, below 72 |
| 5 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 72.50 | 33.33 | 54.38 | 100.00 | 66.43 | passes 72 floor |
| 10 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 77.50 | 32.50 | 58.13 | 100.00 | 71.20 | passes 72 floor, near 80 target |

Top same-row candidate:

| Candidate | K | Arm | Deployed old acc | Baseline old acc | Deployed full acc | Unknown FAR | AUROC | FPR95 | Rollback |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 10 | F | 77.50 | 32.50 | 58.13 | 100.00 | 71.20 | 75.83 | false |

## Gate Failure Diagnosis

The current OLD80_FIRST head is not a meaningful open-set gate. It uses `old80_head_apply_policy=replace_all`, so `apply_old80_first_head` applies the old-class head to every query row, forces `accepted=True`, and overwrites the gate reason with `old80_first_*`. On the two representative passing rows:

| Candidate | K | Old rows accepted | Unknown rows accepted | Unknown predicted as old | Gate reason |
|---|---:|---:|---:|---:|---|
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | 240/240 | 80/80 | 80/80 | `old80_first_fused_centroid` |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 10 | 240/240 | 80/80 | 80/80 | `old80_first_support_cv_select` |

This means the run is valid only as an old-class recovery diagnostic. It must not be claimed as open-set rejection, gate restoration, or deployment success evidence.

Post-hoc score sweeps were run on the completed `score_table.csv` artifacts to estimate whether any recorded score can recover unknown rejection while keeping old-class accuracy above the OLD80_FIRST floor. This is a non-deployment oracle diagnostic because thresholds were selected after looking at query labels.

| Scope | Candidate | K | Constraint | Best score gate | Direction | Threshold | Old acc | Unknown FAR | Unknown reject | Full acc | Coverage |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| top-by-K | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | old acc >= 72% | `margin` | low reject | 0.006592 | 72.08 | 95.00 | 5.00 | 55.31 | 97.81 |
| top-by-K | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 10 | old acc >= 72% | `cosine_unknown_score` | high reject | 0.107974 | 72.08 | 65.00 | 35.00 | 62.81 | 83.75 |
| top-by-K | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 10 | old acc >= 75% | `energy` | high reject | 1.919682 | 75.00 | 77.50 | 22.50 | 61.88 | 89.69 |
| all48 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU6_D_MSE_SUBSPACE_KOLD10_KNEW0` | 10 | old acc >= 72% | `best_old_score` | low reject | -2.896687 | 72.08 | 62.50 | 37.50 | 63.44 | 85.62 |

No candidate in this diagnostic produced a feasible old acc >= 80% gate-restoration point. Even with oracle query-label thresholding, the best old acc >= 72% tradeoff still leaves unknown FAR at 62.50%-65.00%. Therefore, the failure is not just rollback policy; the current OLD80_FIRST strategy structurally restores old accuracy by accepting all unknown samples, and the available score margins are only weakly separable.

## Interpretation

This run reaches the OLD80_FIRST old-class recovery intent for K=5 and K=10, and it substantially improves K=2 and K=3, but K=2/K=3 are still below the 72% floor. Unknown FAR remains 100% because the OLD80_FIRST head currently overrides the gate and accepts every query. The next strategy should split the work into two explicit tracks: keep improving K=2/K=3 old-class recovery, and create a separate gate-restoration matrix for K=5/K=10 that applies an unknown veto after the old-class head and is judged by same-row old acc plus unknown FAR. Until that second track passes, this result remains diagnostic-only for open-set rejection.
