# Phase2 OLD80_FIRST ADV3B02 Best Head48

| Item | Value |
|---|---|
| Run ID | `phase2_old80first_adv3b02_best_head48_20260702_1018` |
| Timestamp | 2026-07-02 10:18 Asia/Hong_Kong |
| Operator | Codex |
| Objective | Use `ADV3B02_CORE90_SOFT_E200` for Phase2 OLD80_FIRST, recovering target-old accuracy before restoring open-set gate optimization |
| Phase1 source run | `phase1_adv3_mechanism32_queue_20260701` |
| Phase1 candidate | `ADV3B02_CORE90_SOFT_E200` |
| Phase1 checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| Phase1 prototype export | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/phase2_zid_prototypes.pt` |
| Launcher base | `code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh` |
| Phase2 plan | `OA_MSE_H06_OLD80FIRST_HEAD48` |
| Rows | 48 |
| K grid | old support per TX: 2,3,5,10 |
| Priority gate | OLD80_FIRST: target-old `old_acc>=0.80` before restoring open-set optimization |

## Protocol Boundary

This is a Stage2-B OLD80_FIRST run. It uses target-old support from `Y_old` and evaluates unknown query without using unknown query to tune thresholds. Target-new support remains disabled for this old-class recovery stage. Unknown FAR is logged as eval-only context, not as the primary optimization target.

## Hypothesis

`ADV3B02_CORE90_SOFT_E200` has the strongest closed-set DG profile in the ADV3 mechanism32 queue (`strict UDU=84.89`, `receiver_floor=75.55` in local report). Using its best joint-safe checkpoint may improve target-old recovery under the same OLD80_FIRST head-first Phase2 strategy.

## Pre-Launch Evidence

| Check | Evidence |
|---|---|
| `AGENTS.md` | Read before this run |
| `项目.md` | Read before this run; confirms OLD80_FIRST priority and unknown-query threshold boundary |
| N607 preflight | PASS at 2026-07-02 10:15 CST |
| GPU state | 8 RTX 3090 visible, 0 active compute processes at preflight |
| Previous OLD80 scheduler | PID `2896435` no longer alive; previous CEN51/default OLD80 run has 48 logs and 48 metrics |
| B02 checkpoint availability | `best_joint_safe_ssdg.pth` exists, 8,582,116 bytes |
| B02 prototype availability | `phase2_zid_prototypes.pt` exists, 379,654 bytes |
| Missing aliases | no `latest_model.pth` or `best_model.pth`; use explicit `best_joint_safe_ssdg.pth` path |

## Local Files

No code changes are required for this B02 launch. The run uses the already verified scheduler-safe OLD80_FIRST launcher with environment overrides:

| Local file | Purpose |
|---|---|
| `E:\type10-7\code\scripts\launch_phase2_old80first_head48_sched_20260702_0055.sh` | Existing scheduler-safe 48-row OLD80_FIRST launcher |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_adv3b02_best_head48_20260702_1018\report.md` | This run report |

## Remote Command

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
mkdir -p logs/phase2_old80first_adv3b02_best_head48_20260702_1018 automation_reports/CV-SincNet/phase2_old80first_adv3b02_best_head48_20260702_1018
nohup env RUN_ID=phase2_old80first_adv3b02_best_head48_20260702_1018 TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth STAGE2_MAX_ACTIVE_PER_GPU=2 INCLUDE_EXTERNAL_GPU_PROCS=1 SCHEDULER_POLL_SECONDS=5 bash code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh > logs/phase2_old80first_adv3b02_best_head48_20260702_1018/launcher_submit.out 2>&1 &
```

## Expected Outputs

| Path | Expected content |
|---|---|
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_adv3b02_best_head48_20260702_1018/launcher_submit.out` | scheduler output |
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_adv3b02_best_head48_20260702_1018/OA_MSE_H06_OLD80FIRST*.out` | per-candidate logs |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_old80first_adv3b02_best_head48_20260702_1018/*/metrics.json` | per-candidate metrics |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_old80first_adv3b02_best_head48_20260702_1018/*/score_table.csv` | per-sample score tables |

## Success Criteria

Primary: same-row target-old `old_class_accuracy` reaches at least 72%, preferably 80%. Only after this gate is met should unknown/open-set gate optimization be restored. Unknown FAR/AUROC/FPR95 are recorded, but they do not override the OLD80_FIRST decision.

## Risks

The existing launcher candidate IDs do not include the B02 label; the run ID and `TEACHER_CKPT` override are the authority for identifying this run as B02. If B02 best checkpoint differs materially from final, a later `latest_safe_ssdg.pth` comparison can be launched as a separate run.

## Startup Failure

| Item | Evidence |
|---|---|
| Observed status | Launcher started as PID `3738593`; early health found 48 candidate logs, 48 run directories, and 0 metrics |
| First failing log | `logs/phase2_old80first_adv3b02_best_head48_20260702_1018/OA_MSE_H06_OLD80FIRST_HEAD48_GPU0_A_MSE_SUBSPACE_KOLD2_KNEW0.out` |
| Failure point | `code/export_spaceborne_features.py` called `build_model_from_ckpt(...)` before feature export |
| Exception | `ValueError: sample_rate too low or min_band_hz too large.` |
| Root cause | ADV3B02 checkpoint `args.sample_rate_hz` is `0.0`, while `baseline_args.sample_rate_hz` is `25000000.0`; `build_model_from_ckpt` used the nonpositive checkpoint value instead of falling back to the valid WiSig sample rate |
| Protocol impact | No metrics were produced; this run is startup-failed diagnostic evidence only, not Phase2 performance evidence |

## Local Repair

| File | Purpose |
|---|---|
| `E:\type10-7\code\eval_feature_diagnosis.py` | Make `build_model_from_ckpt` fall back to `25e6` for WiSig when checkpoint/CLI `sample_rate_hz<=0` |

Verification:

```powershell
conda run -n ssr-gpu python -m py_compile code\eval_feature_diagnosis.py code\export_spaceborne_features.py
```

```powershell
cd E:\type10-7\code
conda run --no-capture-output -n ssr-gpu python -u -  # build_model_from_ckpt smoke: built DualCVSincNetDisentangle
```

Re-run policy: preserve this failed run and launch a new fixed run ID after local repair is snapshotted, mirrored to the Git-backed release workspace, and synced to N607.
