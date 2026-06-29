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
| `code/cvsrffi/losses.py` | Uses cosine-margin training losses for open-world feature geometry while keeping angle metrics for logging; avoids non-finite `acos` gradients observed in the first all-joint startup. |
| `code/tests/test_phase2_train_cli.py` | Adds CLI/text guard for SSDG feature-space bridge and export defaults. |
| `code/tests/test_open_world_feature_space_loss.py` | Adds near-duplicate-feature finite-gradient regression coverage for the open-world feature-space loss. |
| `code/training_test_eval.py` | Provides dense-tail named-test scheduling through `should_run_training_test(..., final_window, final_interval)`. |
| `code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh` | Seven-row all-joint variable-epoch ground validation launcher derived from the SHORT195_S3 baseline. |
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

Result: initial focused pytest passed (`14 passed`, one `.pytest_cache` permission warning); after the cosine-margin stability fix, focused pytest passed (`15 passed`, same cache warning). `py_compile` passed after serial rerun; help shows the new default-off flags; dry-run parsed the combined active bridge, export, and dense-tail schedule. The launcher was later changed per operator instruction from a four-row ablation/control screen to seven all-joint candidates on GPU1-GPU7.

Note: one initial parallel `conda run` pytest attempt hit a Windows temp-file lock. It was rerun serially and passed; the lock is not experiment evidence.

## Candidate Matrix

| Candidate | GPU | Seed | Active bridge | Expected evidence |
|---|---:|---:|---|---|
| `PHASE1_SHORT195S3_ZIDJOINT_C0_CONSERVE_E160` | 1 | 360703 | `--use_proto_memory true --lambda_proto 0.003 --lambda_open_world_feat 0.003 --proto_domain_align_weight 0.15 --ow_feat_domain_align_weight 0.01` | Conservative all-joint setting; checks whether the combined bridge can train without source/sat regression. |
| `PHASE1_SHORT195S3_ZIDJOINT_C1_LOW_E160` | 2 | 360713 | `--lambda_proto 0.004 --lambda_open_world_feat 0.004 --proto_domain_align_weight 0.25 --ow_feat_domain_align_weight 0.01` | Low balanced all-joint setting. |
| `PHASE1_SHORT195S3_ZIDJOINT_C2_GEOM_E160` | 3 | 360723 | `--lambda_proto 0.004 --lambda_open_world_feat 0.006 --proto_domain_align_weight 0.25 --ow_feat_domain_align_weight 0.02` | Geometry-leaning all-joint setting. |
| `PHASE1_SHORT195S3_ZIDJOINT_C3_PROTO_E160` | 4 | 360733 | `--lambda_proto 0.006 --lambda_open_world_feat 0.004 --proto_domain_align_weight 0.35 --ow_feat_domain_align_weight 0.01` | Prototype-leaning all-joint setting. |
| `PHASE1_SHORT195S3_ZIDJOINT_C4_BALANCED_E160` | 5 | 360743 | `--lambda_proto 0.006 --lambda_open_world_feat 0.006 --proto_domain_align_weight 0.25 --ow_feat_domain_align_weight 0.02` | Mid balanced all-joint setting. |
| `PHASE1_SHORT195S3_ZIDJOINT_C5_DOMAIN_E160` | 6 | 360753 | `--lambda_proto 0.004 --lambda_open_world_feat 0.005 --proto_domain_align_weight 0.50 --ow_feat_domain_align_weight 0.03` | Domain-alignment-leaning all-joint setting. |
| `PHASE1_SHORT195S3_ZIDJOINT_C6_STRONG_E160` | 7 | 360763 | `--lambda_proto 0.008 --lambda_open_world_feat 0.006 --proto_domain_align_weight 0.35 --ow_feat_domain_align_weight 0.03` | Strong all-joint pressure test. |

All rows inherit the SHORT195_S3 loss, pseudo-label, satellite, and guard settings, but the first screen is intentionally shorter: `SCREEN_EPOCHS=160`, `SCREEN_LABEL_EPOCHS=150`, `SCREEN_PSEUDO_EPOCHS=10`. These can be overridden through environment variables. E160 rows are all-joint mechanism/stability screens; only promoted rows should be expanded to E200/E220 for final comparison against the E200 SHORT195_S3 baseline. No new GPU is spent on a no-loss control or single-mechanism ablation; the historical SHORT195_S3 row is the baseline.

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

Pre-launch N607 read-only check at 2026-06-30 00:38 CST:

| Field | Value |
|---|---|
| Existing active training | `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5` on GPU0 |
| Old queue markers | 36 candidates, 36 launched, 32 complete markers, 0 failed markers |
| Pending old candidates | none observed; all 36 candidate launch markers are present |
| Available GPUs for this run | GPU1-GPU7; operator instructed this screen must avoid GPU0 and use GPU5-GPU7 as well |

Planned local-to-remote sync:

| Local file | Remote destination |
|---|---|
| `code/SSDG/train_ssdg.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py` |
| `code/training_test_eval.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/training_test_eval.py` |
| `code/cvsrffi/losses.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py` |
| `code/cvsrffi/phase2_prototypes.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/phase2_prototypes.py` |
| `code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh` |

Remote pre-launch verification:

| Check | Result |
|---|---|
| Initial remote dry-run after syncing `train_ssdg.py` and launcher | Failed with `ImportError: cannot import name 'open_world_feature_space_loss' from 'cvsrffi.losses'`; remote dependency was stale. |
| Corrective sync | Synced `code/cvsrffi/losses.py` and `code/cvsrffi/phase2_prototypes.py` to the matching remote paths. |
| Remote SSDG dry-run | Passed with `--lambda_proto 0.004 --lambda_open_world_feat 0.004 --phase2_export_prototypes true` and dense-tail test-eval flags. |
| Remote launcher syntax | `bash -n code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh` passed. |
| First launch attempt | Submitted four earlier candidates to GPU1-GPU4, but all exited before training with `TypeError: should_run_training_test() got an unexpected keyword argument 'final_window'`; root cause was stale remote `code/training_test_eval.py`. |
| Corrective sync for dense-tail scheduler | Synced `code/training_test_eval.py`; failed attempt is retained as startup audit evidence and is not counted as experiment result. |
| First all-joint launch | `phase1_short195s3_zidbridge_joint7_20260630_0050` submitted seven candidates to GPU1-GPU7 but all showed persistent `train_skipped_nonfinite_grad=1.0` through E009. |
| Stop invalid all-joint run | Precisely terminated only PIDs `1397165 1397172 1397179 1397186 1397256 1397326 1397396`; GPU0 old queue was not touched. |
| Root-cause fix | Replaced open-world feature-space training gradients from `acos(angle)` penalties with cosine-margin hinge losses; retained angle values only as detached metrics. |
| Remote gradient probe | Passed on N607: near-duplicate feature loss finite, `grad_finite=True`, `active=6.0`. |

Corrected launch command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/phase1_short195s3_zidjoint_lossfix7_20260630_0058 && setsid env RUN_ID=phase1_short195s3_zidjoint_lossfix7_20260630_0058 STAGE2_MAX_ACTIVE_PER_GPU=2 SCREEN_EPOCHS=160 SCREEN_LABEL_EPOCHS=150 SCREEN_PSEUDO_EPOCHS=10 bash code/scripts/launch_phase1_short195s3_zidbridge_20260630_0030.sh > logs/phase1_short195s3_zidjoint_lossfix7_20260630_0058/scheduler.out 2>&1 < /dev/null & echo started_setsidshell=$!
```

Corrected launch status at 2026-06-30 00:55 CST:

| Field | Value |
|---|---|
| Run ID | `phase1_short195s3_zidjoint_lossfix7_20260630_0058` |
| Scheduler shell | `1403389` |
| Candidate PIDs | `1403399 1403406 1403413 1403483 1403490 1403560 1403693` |
| GPUs | GPU1-GPU7; GPU0 avoided |
| Log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_short195s3_zidjoint_lossfix7_20260630_0058/` |
| Run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_short195s3_zidjoint_lossfix7_20260630_0058/` |
| Early health | Seven metric files present by E005; `proto_active=6.0` for all candidates; open-world weighted loss nonzero; `stage_test_eval_ran=0.0` before E010, so heavy test is not running every epoch. |
| Remaining watch item | C2/C3/C4/C6 show `train_skipped_nonfinite_grad=0.01538` at E005, i.e. a small batch-level skip rate rather than full-epoch failure. Monitor at E010/E020 before promotion decisions. |
