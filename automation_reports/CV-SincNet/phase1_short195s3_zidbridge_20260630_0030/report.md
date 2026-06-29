# phase1_short195s3_zidbridge_20260630_0030

## Objective

Validate the latest ground-training z_id feature-space bridge on top of `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3`, without changing the CVS source-only ground protocol.

## Baseline Evidence

Read-only N607 check at 2026-06-30 00:21 CST confirmed:

| Field | Value |
|---|---|
| Base candidate | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3` |
| Source run | `phase1_gpu0_jointsafe36_queue_20260629_0930` |
| Entry | `code/SSDG/train_ssdg.py` |
| Final/best epoch | E200 |
| best_score | 84.9483 |
| val_tx | 98.51% |
| test_overall_tx | 90.34% |
| strict_udu | 84.14% |
| receiver_floor | 76.24% |
| sat_mean/sat_floor | 76.62% / 75.39% |
| sat_strict_mean/sat_strict_floor | 70.45% / 69.24% |
| checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3/best_joint_safe_ssdg.pth` |

The base command already imported `phase2_prototypes`、`feature_masks`、`tx_rx_geometry` and `open_world_head` through SSDG audit flags, but active legacy audit weights were all zero. This report's new candidates therefore test active z_id feature-space losses, not just audit presence.

## Local Files Changed

| File | Purpose |
|---|---|
| `code/SSDG/train_ssdg.py` | Adds default-off SSDG bridge for `PrototypeMemoryBank` and `open_world_feature_space_loss`; adds optional Phase2 prototype export; keeps legacy audit weights protected. |
| `code/tests/test_phase2_train_cli.py` | Adds CLI/text guard for SSDG feature-space bridge and export defaults. |
| `code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh` | Four-row variable-epoch ground validation launcher derived from the SHORT195_S3 baseline. |
| `docs/PHASE1_SHORT195S3_ZID_FEATURE_SPACE_VALIDATION.md` | Documents baseline evidence, bridge boundary, matrix, and promotion gates. |
| `automation_reports/CV-SincNet/phase1_short195s3_zidbridge_20260630_0030/report.md` | This report. |

## Local Verification

```powershell
conda run --no-capture-output -n ssr-gpu python -m py_compile code\SSDG\train_ssdg.py code\cvsrffi\losses.py code\cvsrffi\phase2_prototypes.py
conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase2_train_cli.py code\tests\test_open_world_feature_space_loss.py code\tests\test_training_test_eval.py -q
conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --help | Select-String -Pattern "lambda_proto|lambda_open_world_feat|phase2_export|test_eval_final"
conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --dry_run --output_dir runs\dryrun_ssdg_bridge --epochs 3 --label_epochs 2 --pseudo_epochs 1 --lambda_proto 0.004 --lambda_open_world_feat 0.004 --phase2_export_prototypes true --test_eval_policy interval_final --test_eval_start_epoch 1 --test_eval_interval 10 --test_eval_final_window 20 --test_eval_final_interval 2
bash -n code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh
bash code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh --dry-run
```

Result: `py_compile` passed; focused pytest passed (`14 passed`, one `.pytest_cache` permission warning); help shows the new default-off flags; dry-run parsed the combined active bridge, export, and dense-tail schedule; launcher syntax check passed and dry-run emitted four E160 candidates with `--epochs 160 --label_epochs 150 --pseudo_epochs 10`.

Note: one initial parallel `conda run` pytest attempt hit a Windows temp-file lock. It was rerun serially and passed; the lock is not experiment evidence.

## Candidate Matrix

| Candidate | GPU | Seed | Active bridge | Expected evidence |
|---|---:|---:|---|---|
| `PHASE1_SHORT195S3_ZIDBRIDGE_C0_EXPORT_E160` | 1 | 360703 | None; Phase2 export only | Control for dense-tail schedule and prototype export overhead |
| `PHASE1_SHORT195S3_ZIDBRIDGE_C1_PROTO_LOW_E160` | 2 | 360713 | `--use_proto_memory true --lambda_proto 0.004` | Whether EMA source prototypes improve z_id stability without source/sat regression |
| `PHASE1_SHORT195S3_ZIDBRIDGE_C2_OWFEAT_LOW_E160` | 3 | 360723 | `--lambda_open_world_feat 0.004 --ow_feat_domain_align_weight 0.01` | Whether angular compactness/inter-class margin improves without over-regularization |
| `PHASE1_SHORT195S3_ZIDBRIDGE_C3_PROTO_OWFEAT_LOW_E160` | 4 | 360733 | Both low-weight bridges | Whether prototype memory and angular geometry are complementary |

All rows inherit the SHORT195_S3 loss, pseudo-label, satellite, and guard settings, but the first screen is intentionally shorter: `SCREEN_EPOCHS=160`, `SCREEN_LABEL_EPOCHS=150`, `SCREEN_PSEUDO_EPOCHS=10`. These can be overridden through environment variables. E160 rows are mechanism/stability screens; only promoted rows should be expanded to E200/E220 for final comparison against the E200 SHORT195_S3 baseline.

All rows use:

```text
--test_eval_policy interval_final --test_eval_start_epoch 1 --test_eval_interval 10 --test_eval_final_window 20 --test_eval_final_interval 2
--phase2_export_prototypes true --phase2_export_feature_key z_id --phase2_export_split train
```

## Success Gates

| Gate | Required outcome |
|---|---|
| Protocol | source-only ground training; no target receiver, Stage2 support/query, or unknown query in training/model selection/threshold fitting |
| DG metrics | E160 screens must show no obvious collapse; final E200/E220 confirmation must show no >2pp drop versus SHORT195_S3 on overall, strict_udu, or receiver_floor |
| Satellite metrics | no >1pp drop on sat_mean or sat_floor without manual review |
| Active loss telemetry | proto/open-world weighted losses finite and nonzero for active rows |
| Prototype export | `phase2_zid_prototypes.pt` and JSON sidecar written for each candidate |
| Promotion | at most two candidates proceed to Stage2-B old/unknown rejection validation |

## Launch Record

Status: prepared locally, not yet synced or launched at report creation time.
