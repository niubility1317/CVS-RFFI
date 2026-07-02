# Phase2 OLD80_FIRST ADV3B02 Best Head48 Sample-Rate Fix

| Item | Value |
|---|---|
| Run ID | `phase2_old80first_adv3b02_best_head48_srfix_20260702_1028` |
| Timestamp | 2026-07-02 10:28 Asia/Hong_Kong |
| Operator | Codex |
| Objective | Re-run Phase2 OLD80_FIRST with `ADV3B02_CORE90_SOFT_E200` after fixing checkpoint sample-rate fallback |
| Parent failed run | `phase2_old80first_adv3b02_best_head48_20260702_1018` |
| Phase1 source run | `phase1_adv3_mechanism32_queue_20260701` |
| Phase1 candidate | `ADV3B02_CORE90_SOFT_E200` |
| Phase1 checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| Phase1 prototype export | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/phase2_zid_prototypes.pt` |
| Launcher base | `code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh` |
| Matrix | 48 OLD80_FIRST head candidates |
| K grid | old support per TX: 2,3,5,10 |
| Priority gate | target-old old-class accuracy first: at least 72%, preferably 80% |

## Protocol Boundary

This is a Stage2-B OLD80_FIRST run. It does not optimize unknown query threshold before target-old recovery. Unknown FAR/AUROC/FPR95 are logged as secondary context only.

## Failure Repaired

The parent run reached startup only and produced no metrics. The first candidate failed in `code/export_spaceborne_features.py` while building the model:

```text
ValueError: sample_rate too low or min_band_hz too large.
```

ADV3B02 checkpoint inspection showed `args.sample_rate_hz=0.0` and `baseline_args.sample_rate_hz=25000000.0`. The local repair makes `build_model_from_ckpt` fall back to `25e6` for WiSig when CLI/checkpoint `sample_rate_hz<=0`.

## Local Files And Version State

| File | Purpose | Status |
|---|---|---|
| `E:\type10-7\code\eval_feature_diagnosis.py` | sample-rate fallback used by `export_spaceborne_features.py` | changed locally, snapshotted, synced |
| `E:\type10-7\code\scripts\launch_phase2_old80first_head48_sched_20260702_0055.sh` | existing scheduler-safe 48-row launcher | unchanged |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_adv3b02_best_head48_20260702_1018\report.md` | parent failure report | updated and synced |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_adv3b02_best_head48_srfix_20260702_1028\report.md` | this run report | created before launch |

Version and snapshots:

| Item | Value |
|---|---|
| Root Git status | `E:\type10-7` is not a Git repository |
| Local snapshot | `E:\type10-7\code\snapshots\phase2_old80first_adv3b02_best_head48_20260702_1018_sample_rate_fix\code\eval_feature_diagnosis.py` |
| Git-backed mirror | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| Commit | `f948136` (`Fix Phase2 checkpoint sample rate fallback`) |
| Local code SHA256 | `3eaf80f48cd7b8619444432b3b0e7a21c5d49b050035c56c139c141648fa2fe2` |

Verification:

```powershell
conda run -n ssr-gpu python -m py_compile code\eval_feature_diagnosis.py code\export_spaceborne_features.py
```

```powershell
cd E:\type10-7\code
conda run --no-capture-output -n ssr-gpu python -u -  # build_model_from_ckpt smoke: built DualCVSincNetDisentangle
```

Remote verification:

| Check | Evidence |
|---|---|
| Remote code SHA256 | `3eaf80f48cd7b8619444432b3b0e7a21c5d49b050035c56c139c141648fa2fe2` |
| Remote parent report SHA256 | `dbb96d5cb259812ec5b9af075be2d0b15557a0dd08849ee48a8ff7df19dbe2ab` |
| Remote py_compile | PASS for `code/eval_feature_diagnosis.py` and `code/export_spaceborne_features.py` |
| Pre-launch GPU context | parent launcher PID `3738593` not alive; no active GPU compute processes |

## Sync Mapping

| Local | Remote |
|---|---|
| `E:\type10-7\code\eval_feature_diagnosis.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/eval_feature_diagnosis.py` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_adv3b02_best_head48_20260702_1018\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_old80first_adv3b02_best_head48_20260702_1018/report.md` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_adv3b02_best_head48_srfix_20260702_1028\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028/report.md` |

## Remote Command

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
mkdir -p logs/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028 runs/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028 automation_reports/CV-SincNet/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028
nohup env RUN_ID=phase2_old80first_adv3b02_best_head48_srfix_20260702_1028 TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth STAGE2_MAX_ACTIVE_PER_GPU=2 INCLUDE_EXTERNAL_GPU_PROCS=1 SCHEDULER_POLL_SECONDS=5 bash code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh > logs/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028/launcher_submit.out 2>&1 &
```

## Expected Outputs

| Path | Expected content |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028/launcher_submit.out` | scheduler output |
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028/OA_MSE_H06_OLD80FIRST*.out` | per-candidate logs |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028/*/metrics.json` | per-candidate metrics |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_old80first_adv3b02_best_head48_srfix_20260702_1028/*/score_table.csv` | per-sample score tables |

## Success Criteria

Primary: same-row target-old `old_class_accuracy` reaches at least 72%, preferably 80%. Only after this gate is met should unknown/open-set gate optimization be restored. Unknown metrics are eval-only context for this run.

## Known Risks

Candidate IDs retain the inherited `OA_MSE_H06_OLD80FIRST_HEAD48_*` naming; the run ID and explicit `TEACHER_CKPT` identify the ADV3B02 source. If this checkpoint still underperforms, compare against `latest_safe_ssdg.pth` in a separate run rather than mixing checkpoint variants in the same run.

## Final Status

| Item | Value |
|---|---|
| Completion | 48/48 candidates completed |
| Failed candidates | 0 |
| Summary CSV | `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_adv3b02_best_head48_srfix_20260702_1028\summary_old80first_metrics.csv` |
| Rollback triggered | 48/48 |
| Rollback trigger | all sampled top rows show `unknown_false_accept_rate` max-rise violation |

## Result Summary

The sample-rate repair fixed startup and allowed all candidates to complete. However, the default rollback policy is still structurally misaligned with OLD80_FIRST: raw Phase2 old-class accuracy improved to `74.17%`, but deployment metrics were reverted to baseline because unknown FAR rose during this old-class recovery stage.

| K old support | Best raw candidate | Raw old acc | Deployed old acc | Mean raw old acc | Mean deployed old acc | Unknown FAR at best raw |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU3_A_MSE_SUBSPACE_KOLD2_KNEW0` | 70.00 | 36.25 | 62.86 | 27.55 | 100.00 |
| 3 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU5_B_MSE_SUBSPACE_KOLD3_KNEW0` | 70.00 | 33.33 | 63.70 | 26.82 | 100.00 |
| 5 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU6_C_MSE_SUBSPACE_KOLD5_KNEW0` | 73.33 | 30.42 | 67.79 | 28.05 | 100.00 |
| 10 | `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | 74.17 | 21.67 | 70.39 | 31.56 | 100.00 |

Top same-row raw OLD80 result:

| Candidate | K | Raw old acc | Deployed old acc | Baseline old acc | Deployed full acc | Unknown FAR | Rollback |
|---|---:|---:|---:|---:|---:|---:|---|
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | 10 | 74.17 | 21.67 | 21.67 | 36.56 | 100.00 | true |

## Interpretation

This run confirms that the ADV3B02 checkpoint plus OLD80_FIRST head can recover target-old above the 72% floor in raw Phase2 metrics, but the default rollback gate prevents that recovery from becoming the deployed result. The next run should use an OLD80_FIRST rollback policy that keeps old-class/known/coverage safety guards while treating unknown FAR as eval-only until target-old recovery is stable.
