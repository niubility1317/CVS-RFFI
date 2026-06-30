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

E010 checkpoint at 2026-06-30 00:58 CST confirmed the requested test cadence and stable joint loss:

| Candidate | E010 val_tx | E010 test_tx | skip_grad | proto_active | weighted open-world loss |
|---|---:|---:|---:|---:|---:|
| `PHASE1_SHORT195S3_ZIDJOINT_C0_CONSERVE_E160` | 88.08 | 78.59 | 0.0 | 6.0 | 0.000306 |
| `PHASE1_SHORT195S3_ZIDJOINT_C1_LOW_E160` | 90.57 | 81.57 | 0.0 | 6.0 | 0.000429 |
| `PHASE1_SHORT195S3_ZIDJOINT_C2_GEOM_E160` | 89.61 | 77.65 | 0.0 | 6.0 | 0.000640 |
| `PHASE1_SHORT195S3_ZIDJOINT_C3_PROTO_E160` | 91.39 | 79.11 | 0.0 | 6.0 | 0.000422 |
| `PHASE1_SHORT195S3_ZIDJOINT_C4_BALANCED_E160` | 92.71 | 81.48 | 0.0 | 6.0 | 0.000616 |
| `PHASE1_SHORT195S3_ZIDJOINT_C5_DOMAIN_E160` | 91.61 | 79.73 | 0.0 | 6.0 | 0.000555 |
| `PHASE1_SHORT195S3_ZIDJOINT_C6_STRONG_E160` | 90.52 | 78.90 | 0.0 | 6.0 | 0.000634 |

Metrics rows through E012/E013 show `stage_test_eval_ran` only at E010 so far, matching the requested schedule: every 10 epochs during the main run and every 2 epochs only inside the final 20 epochs. These E010 values are startup/trajectory evidence, not final E160 selection evidence.

## Completion Analysis 2026-06-30 09:51 CST

N607 direct inventory at `2026-06-30T09:51:00+0800` showed no active classified CVS-RFFI training process. A separate GPU-resident process was present, but it was not this experiment: `python main.py --opts dataset celeba train False eval True compute_metrics True solve_inverse_problem False`; it was not touched.

The corrected seven-candidate run `phase1_short195s3_zidjoint_lossfix7_20260630_0058` is complete. Every candidate wrote 160 epoch rows, reached `E160/160`, wrote `best_joint_safe_ssdg.pth`, `latest_ssdg.pth`, `metrics_epoch.csv`, `metrics_epoch.jsonl`, `phase2_zid_prototypes.pt`, and `phase2_zid_prototypes.json`. Full per-candidate stdout logs had 6364 lines each and no `Traceback`, `RuntimeError`, OOM, or killed marker. `stage_test_eval_ran` occurred exactly at epochs `10,20,30,40,50,60,70,80,90,100,110,120,130,140,142,144,146,148,150,152,154,156,158,160`; there were no missing or extra heavy-test epochs.

Baseline comparison target remains `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3` at E200: score 84.9483, test_tx 90.34, strict_udu 84.14, receiver_floor 76.24, sat_mean/sat_floor 76.62/75.39, sat_strict_mean/sat_strict_floor 70.45/69.24.

| Rank | Candidate | Mechanism | GPU | Seed | Best epoch | Score | Δscore | Test TX | Strict UDU | Receiver floor | Sat mean/floor | Sat strict mean/floor | Verdict |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `C5_DOMAIN` | all-joint, domain-align leaning | 6 | 360753 | 160 | 83.5329 | -1.4154 | 88.7873 | 82.8483 | 75.3167 | 73.1415/72.0069 | 67.5050/66.6167 | Best in this screen, but below baseline and satellite gate. |
| 2 | `C3_PROTO` | all-joint, prototype leaning | 4 | 360733 | 148 | 83.5274 | -1.4209 | 88.6951 | 83.7617 | 73.5083 | 74.2753/72.8716 | 67.2161/65.9550 | Near-tie with C5; strict close, receiver/sat still worse. |
| 3 | `C4_BALANCED` | all-joint, mid balanced | 5 | 360743 | 144 | 83.1531 | -1.7952 | 89.0775 | 84.1017 | 74.9833 | 69.7765/68.6083 | 64.3828/63.2650 | Strict nearly baseline, but satellite collapse is too large. |
| 4 | `C6_STRONG` | all-joint, strong pressure | 7 | 360763 | 156 | 82.9151 | -2.0332 | 88.5088 | 83.6683 | 74.3833 | 70.9721/69.9701 | 63.1056/62.4183 | Stronger weights did not help. |
| 5 | `C2_GEOM` | all-joint, geometry leaning | 3 | 360723 | 160 | 82.8088 | -2.1395 | 88.3966 | 82.5117 | 74.6333 | 72.2235/71.2966 | 63.8306/62.8100 | Geometry pressure underperforms. |
| 6 | `C1_LOW` | all-joint, low balanced | 2 | 360713 | 154 | 82.0929 | -2.8554 | 87.6093 | 82.4067 | 68.8167 | 73.6340/72.6324 | 66.6444/65.7133 | Receiver floor regression is unacceptable. |
| 7 | `C0_CONSERVE` | all-joint, conservative | 1 | 360703 | 142 | 81.6933 | -3.2550 | 87.3309 | 81.3600 | 72.2417 | 70.4386/69.4069 | 63.6417/62.7083 | Not promotable. |

Detailed per-candidate Phase1 evidence:

| Candidate | Phase/split/K-shot | Open-world old/new/unknown fields | Generalization splits at selected epoch | Satellite scenario TX | Feature-space telemetry at E160 | Pseudo-label telemetry at E160 | Stability |
|---|---|---|---|---|---|---|---|
| `C5_DOMAIN` | Phase1 source-only ground training; K-shot N/A | N/A; Stage2 rejection not run | unseen-day seen-rx 92.5381; seen-day unseen-rx 89.4750; unseen-day unseen-rx 82.8483 | clear 75.1990; low-elev 72.2186; rain 72.0069 | proto loss/w 0.232230/0.000929; ow loss/w 0.073811/0.000369; min inter 77.6061°; pos angle 34.1374° | 7457 selected, 7455 correct, conf 0.9975, total 8320 | small skipped-grad epochs 10, max 0.0308; logs clean |
| `C3_PROTO` | Phase1 source-only ground training; K-shot N/A | N/A; Stage2 rejection not run | unseen-day seen-rx 92.1667; seen-day unseen-rx 88.7683; unseen-day unseen-rx 83.7617 | clear 76.5657; low-elev 73.3887; rain 72.8716 | proto loss/w 0.224102/0.001345; ow loss/w 0.075682/0.000303; min inter 78.3676°; pos angle 34.5878° | 7472 selected, 7470 correct, conf 0.9983, total 8320 | small skipped-grad epochs 9, max 0.0308; logs clean |
| `C4_BALANCED` | Phase1 source-only ground training; K-shot N/A | N/A; Stage2 rejection not run | unseen-day seen-rx 92.2286; seen-day unseen-rx 89.6417; unseen-day unseen-rx 84.1017 | clear 71.7627; low-elev 68.9583; rain 68.6083 | proto loss/w 0.202595/0.001216; ow loss/w 0.064213/0.000385; min inter 82.0218°; pos angle 33.3341° | 7459 selected, 7458 correct, conf 0.9963, total 8320 | small skipped-grad epochs 8, max 0.0154; logs clean |
| `C6_STRONG` | Phase1 source-only ground training; K-shot N/A | N/A; Stage2 rejection not run | unseen-day seen-rx 92.9440; seen-day unseen-rx 87.1400; unseen-day unseen-rx 83.6683 | clear 72.9637; low-elev 69.9824; rain 69.9701 | proto loss/w 0.222330/0.001779; ow loss/w 0.077282/0.000464; min inter 78.6181°; pos angle 34.6194° | 7427 selected, 7423 correct, conf 0.9972, total 8320 | small skipped-grad epochs 7, max 0.0308; logs clean |
| `C2_GEOM` | Phase1 source-only ground training; K-shot N/A | N/A; Stage2 rejection not run | unseen-day seen-rx 92.8881; seen-day unseen-rx 87.9933; unseen-day unseen-rx 82.5117 | clear 74.0368; low-elev 71.2966; rain 71.3373 | proto loss/w 0.240253/0.000961; ow loss/w 0.083241/0.000499; min inter 76.6712°; pos angle 35.5496° | 7397 selected, 7394 correct, conf 0.9970, total 8320 | small skipped-grad epochs 10, max 0.0308; logs clean |
| `C1_LOW` | Phase1 source-only ground training; K-shot N/A | N/A; Stage2 rejection not run | unseen-day seen-rx 91.3048; seen-day unseen-rx 87.6383; unseen-day unseen-rx 82.4067 | clear 75.3583; low-elev 72.9113; rain 72.6324 | proto loss/w 0.208887/0.000836; ow loss/w 0.068600/0.000274; min inter 79.8909°; pos angle 33.8092° | 7388 selected, 7388 correct, conf 0.9978, total 8320 | small skipped-grad epochs 9, max 0.0154; logs clean |
| `C0_CONSERVE` | Phase1 source-only ground training; K-shot N/A | N/A; Stage2 rejection not run | unseen-day seen-rx 92.1655; seen-day unseen-rx 86.5333; unseen-day unseen-rx 81.3600 | clear 72.2377; low-elev 69.6711; rain 69.4069 | proto loss/w 0.210352/0.000631; ow loss/w 0.067424/0.000202; min inter 80.6250°; pos angle 33.8143° | 7465 selected, 7462 correct, conf 0.9972, total 8320 | small skipped-grad epochs 9, max 0.0154; logs clean |

Interpretation:

1. The latest implementation is technically exercised end-to-end: multi-prototype memory, open-world feature-space loss, pseudo-label continuation, dense-tail test scheduling, and `z_id` prototype export are all active and finite.
2. The feature-space additions did not beat the requested SHORT195_S3 baseline in this E160 screen. The best two rows, C5 and C3, are separated by only 0.0054 score points, so the practical ranking is a tie, but both are still about 1.42 score points below baseline.
3. The main failure is not source validation. Validation TX stays around 98.24-98.65. The regression appears on held-out receiver/day and especially satellite robustness: even the best C5 loses 3.48pp sat_mean and 3.38pp sat_floor versus baseline. C4 almost preserves strict_udu but loses 6.84pp sat_mean, so it is not promotable.
4. The small skipped-gradient rates after the cosine-margin fix are batch-level events rather than the earlier full-epoch `train_skipped_nonfinite_grad=1.0` failure. They do not invalidate the run, but they remain a stability signal to track before any longer run.
5. No candidate should be promoted to Stage2-B old/unknown rejection validation yet. This run validates that the implementation path is executable, but it does not validate the joint mechanism as an improvement over the current Phase1 baseline.

Recommended next action:

Do not spend Stage2/unknown-rejection budget on these checkpoints. The next ground-training experiment should keep the code path but reduce satellite-destructive feature-space pressure: keep C5/C3 as reference settings, lower or warm up `lambda_open_world_feat`/domain-align weights, and add a satellite-preservation gate during selection. If only one confirmation run is allowed, C5 is the better E200/E220 candidate by final score and receiver floor; if the goal is strict_udu preservation, C3 is the better reference. Neither should be claimed as stronger than `SHORT195_S3` without a new run that clears the satellite gate.

## Prototype Geometry Detail 2026-06-30 10:20 CST

Source artifact: N607 run root `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_short195s3_zidjoint_lossfix7_20260630_0058/*/phase2_zid_prototypes.json`. All values below are from exported `z_id` prototype packages and final E160 training telemetry.

Export schema note: each candidate has 6 global `P_tx` transmitter prototypes, plus 84 active `P_tx_dom[t,d]` transmitter-domain prototypes. The JSON tensor keeps 26 possible domain slots, but only 14 domain slots are active for each TX in this source split: `0,1,4,5,8,9,12,13,16,17,20,21,24,25`. Therefore each TX has 15 active prototypes when counting `1 P_tx + 14 P_tx_dom`; each candidate has 90 active prototypes total. These are old/source TX prototypes only; no target receiver support/query or unknown query is used.

| Candidate | Global `P_tx` | Active `P_tx_dom` | Total active prototypes | Export p95 radius mean | Export min interclass angle | Nearest TX pair | Margin violation pairs | E160 open-world loss/weighted | E160 proto loss/weighted |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `C0_CONSERVE` | 6 | 84 | 90 | 9.69° | 88.67° | 1-3 | 0 | 0.067424/0.000202 | 0.210352/0.000631 |
| `C1_LOW` | 6 | 84 | 90 | 8.39° | 88.21° | 4-5 | 0 | 0.068600/0.000274 | 0.208887/0.000836 |
| `C2_GEOM` | 6 | 84 | 90 | 6.81° | 87.46° | 1-5 | 0 | 0.083241/0.000499 | 0.240253/0.000961 |
| `C3_PROTO` | 6 | 84 | 90 | 8.11° | 88.89° | 1-3 | 0 | 0.075682/0.000303 | 0.224102/0.001345 |
| `C4_BALANCED` | 6 | 84 | 90 | 8.13° | 88.67° | 1-3 | 0 | 0.064213/0.000385 | 0.202595/0.001216 |
| `C5_DOMAIN` | 6 | 84 | 90 | 7.21° | 89.03° | 1-3 | 0 | 0.073811/0.000369 | 0.232230/0.000929 |
| `C6_STRONG` | 6 | 84 | 90 | 6.94° | 89.05° | 3-4 | 0 | 0.077282/0.000464 | 0.222330/0.001779 |

Rank-1 candidate `C5_DOMAIN` per-TX prototype detail:

| TX | Samples | `P_tx` | Active `P_tx_dom` | Total prototypes | Active domains | p95 radius | p99 radius | max radius | robust max | sigma | Domain shift mean/max | Domain sample min/max |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1393 | 1 | 14 | 15 | `0,1,4,5,8,9,12,13,16,17,20,21,24,25` | 6.19° | 61.30° | 89.93° | 70.44° | 9.14° | 1.68°/2.94° | 98/100 |
| 1 | 1388 | 1 | 14 | 15 | `0,1,4,5,8,9,12,13,16,17,20,21,24,25` | 15.18° | 89.31° | 89.88° | 89.88° | 11.45° | 2.97°/8.15° | 97/100 |
| 2 | 1389 | 1 | 14 | 15 | `0,1,4,5,8,9,12,13,16,17,20,21,24,25` | 3.78° | 35.38° | 89.82° | 43.98° | 8.60° | 1.69°/2.90° | 97/100 |
| 3 | 1387 | 1 | 14 | 15 | `0,1,4,5,8,9,12,13,16,17,20,21,24,25` | 8.16° | 88.93° | 89.90° | 89.90° | 10.96° | 2.60°/5.92° | 98/100 |
| 4 | 1376 | 1 | 14 | 15 | `0,1,4,5,8,9,12,13,16,17,20,21,24,25` | 4.48° | 38.77° | 89.80° | 46.19° | 7.41° | 1.53°/4.57° | 95/100 |
| 5 | 1387 | 1 | 14 | 15 | `0,1,4,5,8,9,12,13,16,17,20,21,24,25` | 5.49° | 18.10° | 89.84° | 25.02° | 6.93° | 2.13°/3.58° | 97/100 |

Interpretation of these prototype numbers:

1. The prototype package is not empty and not single-prototype-only. It contains one global source TX identity prototype per class and one local source domain prototype for each active TX-domain cell. This is the current local implementation's multi-prototype form: `P_tx` for identity center plus `P_tx_dom` for receiver/day/domain mode diagnostics and later deployment-side evidence.
2. C5 has a good exported global separation surface: nearest global TX pair is 1-3 at 89.03°, and `margin_violation_pairs=0`. This means the exported global centers are not the immediate problem.
3. The weak point is radius tail, not center separation. C5's mean p95 radius is only 7.21°, but TX1 and TX3 have p99/max near 89°, indicating a small but severe tail of source samples far from their class center. That tail explains why open-world/new-class rejection should not be promoted from this checkpoint without a radius/overlap audit.
4. Domain shifts of active `P_tx_dom` against `P_tx` are modest on average, about 1.53°-2.97° for C5, but TX1 reaches 8.15° and TX3 reaches 5.92°. This is compatible with the final result: domain alignment is partly working, but not enough to preserve satellite robustness.

`train_ow_feat_*` interpretation for C5:

| Field | C5 value | Meaning |
|---|---:|---|
| `train_ow_feat_active_classes` | 6.0 | The batch-level geometry loss saw all 6 source TX classes on average; it was active, not skipped due to missing classes. |
| `train_ow_feat_compact` | 0.059466 | Squared hinge penalty for samples outside the desired class angular radius. Nonzero means many batch samples remain wider than the configured 12° radius. |
| `train_ow_feat_inter` | 0.001830 | Squared hinge penalty for class centers closer than the configured 55° margin in some batches. Small but nonzero; not a collapse signal. |
| `train_ow_feat_sample_margin` | 0.008342 | Per-sample own-center versus nearest-negative-center margin penalty. Nonzero means some samples are still too close to another TX center. |
| `train_ow_feat_domain_align` | 0.489642 rad, about 28.05° | Logged domain-center angular misalignment metric. It is not directly the weighted loss term; the actual loss uses a cosine-distance form internally. |
| `train_ow_feat_pos_angle_deg` | 34.14° | Mean sample-to-own-class-center angle across E160 batches. This is much larger than the target radius and explains the compactness penalty. |
| `train_ow_feat_min_inter_deg` | 77.61° | Mean batch-level minimum inter-class center angle. It is comfortably above the 55° margin on average, so class center separation is not the bottleneck. |
| `train_loss_open_world_feat`/`train_w_loss_open_world_feat` | 0.073811/0.000369 | Raw open-world feature-space loss and its weighted contribution with `lambda_open_world_feat=0.005`; finite and active but intentionally small relative to main CE/DG losses. |

Important distinction: exported prototype geometry is computed after extracting the full selected split into one package; `train_ow_feat_*` is averaged over E160 training batches. Therefore C5 can simultaneously show exported `min_interclass_angle=89.03°` and training `train_ow_feat_min_inter_deg=77.61°`; they are different aggregation surfaces.
