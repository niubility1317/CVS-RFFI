# CVS-RFFI staged experiment groups

Date: 2026-05-14

This plan follows `CVS_RFFI_experiment_validation_design.md` for phase 1 and keeps the multi-prototype classifier head and SSDG as separate post-baseline training stages.

## Common baseline settings

All directly runnable phase-1 jobs use:

```bash
python -u train.py \
  --batch_size 256 \
  --eval_batch_size 256 \
  --dataset wisig \
  --wisig_domain rx_day \
  --wisig_train_ratio 0.2 \
  --primary_udu_weight 0.65 \
  --epochs 200 \
  --eval_sat_channel \
  --eval_sat_on test_unseen_day_unseen_rx \
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_eval_max_batches -1
```

Primary reporting should prioritize final/latest checkpoint stability, not a test-selected checkpoint:

```text
final strict_udu >= 86.00
E180-E200 strict_udu mean >= 85.85
E180-E200 strict_udu std <= 0.15
storm_mp >= 38.0
mixed_orbit >= 40.8
```

The current `train.py` still writes best-by-test and best-primary checkpoints, so logs must distinguish analysis checkpoints from formal final-epoch reporting.

## Phase 1: stable baseline

Strictly based on the validation design, the main line is Stable-SAT07:

```bash
--slim_group rxrobust_lite_d_no_dac_refined \
--use_sat_consistency \
--sat_train_scenario mixed_orbit \
--sat_cons_start_epoch 20 \
--lambda_sat_cls 0.08 \
--lambda_sat_cons 0.04
```

| ID | Experiment | Purpose | Extra args |
|---|---|---|---|
| B0 | Stable-SAT07 reproduce | Reproduce SAT07 final-epoch 86 percent plateau | `--slim_group rxrobust_lite_d_no_dac_refined --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04` |
| B1 | Stable-SAT07 + SmoothDRO | Push worst-domain stability without changing backbone family | B0 args + `--group_ce_mode smooth_dro --groupdro_tau 0.45 --groupdro_momentum 0.95` |
| B2 | Stable-SAT07 + Fishr | Compare gradient-stat regularization against SmoothDRO | B0 args + `--lambda_fishr 0.02 --fishr_min_domains 4` |
| B3a | Stable-SAT07 weak consistency | Check whether consistency is too smoothing | B0 args with `--lambda_sat_cons 0.02` |
| B3b | Stable-SAT07 cls-only | Isolate satellite CE contribution | B0 args with `--lambda_sat_cls 0.10 --lambda_sat_cons 0.00` |
| B4 | Stable-SAT07 no MixStyle | Mechanism control for MixStyle x satellite interaction | B0 args + `--no_use_mixstyle` |
| B0s | B0 seed repeat | Use idle GPU for stability replication | B0 args + `--seed 2027` |
| B1s | B1 seed repeat | Use idle GPU for stability replication | B1 args + `--seed 2027` |

B0s and B1s are repeats, not new method groups. They fill all 8 GPUs while preserving the phase-1 design.

Start phase 1:

```bash
bash scripts/run_cvs_rffi_staged_8gpu.sh
```

or explicitly:

```bash
bash scripts/run_cvs_rffi_staged_8gpu.sh --mode phase1 --gpu-ids 0,1,2,3,4,5,6,7
```

Preview without launching training:

```bash
bash scripts/run_cvs_rffi_staged_8gpu.sh --mode phase1 --dry-run
```

## Post stage: multi-prototype and SSDG in one queue

Run these only after choosing the Stable-SAT07 checkpoint. The post queue is designed so multi-prototype and SSDG core experiments run together and automatically fill newly freed GPUs.

Current repository status:

```text
Frozen multi-prototype head exists, but train_fjmp.py is not present.
SSDG design/tools live under `code/SSDG`.
```

Therefore the launcher defaults to not running post-stage jobs until the trainer entrypoints are added or supplied.

Start post-stage queue after trainer entrypoints exist:

```bash
bash scripts/run_cvs_rffi_staged_8gpu.sh \
  --mode post \
  --run-post \
  --base-ckpt runs/cvs_rffi_staged/B1_stable_sat07_smoothdro/latest_model.pth \
  --gpu-ids 0,1,2,3,4,5,6,7
```

If the final selected baseline is B0 or B2, change `BASE_CKPT` accordingly.

## Multi-prototype head groups

| ID | Experiment | Training scope |
|---|---|---|
| P1 | z_id-only K=2 proto head | Freeze backbone, train prototype head only |
| P2 | z_id-only K=3 proto head | Run after or alongside P1 as prototype-count ablation |
| P3 | z_id + stopgrad(z_dom), K=2 | Uses domain residual as head-only conditioning |

Expected command shape:

```bash
python -u train_fjmp.py \
  --baseline_ckpt "$BASE_CKPT" \
  --feature_input z_id \
  --num_prototypes 2 \
  --freeze_backbone true \
  --strict_raw true \
  --output_dir runs/cvs_rffi_staged/P1_proto_zid_k2
```

## SSDG groups

The SSDG core groups run in parallel with SGC and prototype-head experiments:

| ID | Experiment | Purpose |
|---|---|---|
| U0 | 10 percent label-only | Fair low-label baseline |
| U1 | FixMatch global threshold | Basic semi-supervised control |
| U2 | receiver-aware SSDG | Main SSDG line with rx/day thresholds |
| U3 | receiver-aware SSDG + mixed-orbit | Tests satellite consistency synergy |

Expected command shape:

```bash
python -u -m SSDG.train_ssdg \
  --baseline_ckpt "$BASE_CKPT" \
  --split_mode tx_rx_day_1_7_2 \
  --labeled_ratio 0.10 \
  --unlabeled_ratio 0.70 \
  --source_val_ratio 0.20 \
  --pseudo_threshold_mode rx_day_quantile \
  --tau_min 0.80 \
  --tau_max 0.97 \
  --output_dir runs/cvs_rffi_staged/U2_ssdg_receiver_aware
```

Combo groups are intentionally a second post queue because they depend on trained SGC/prototype checkpoints:

| ID | Experiment | Dependency |
|---|---|---|
| U4 | SGC + receiver-aware SSDG | Requires S1 checkpoint |
| U5 | SGC + Proto + receiver-aware SSDG | Requires S1 and P1 checkpoints |

Start combo queue:

```bash
bash scripts/run_cvs_rffi_staged_8gpu.sh \
  --mode combo \
  --run-post \
  --base-ckpt /path/to/stable_baseline.pth
```

## Scheduling policy

The launcher keeps a FIFO queue and launches a new experiment as soon as any GPU job exits. This is different from the older wave scheduler in `run_sat_channel_ablation_8gpu.sh`, which waits for all jobs in a wave.

Recommended execution:

```text
1. Run phase1.
2. Choose B0/B1/B2 based on final strict_udu, E180-E200 stability, and satellite metrics.
3. Implement or supply post-stage trainer entrypoints.
4. Run MODE=post with SGC, prototype, and SSDG core groups in one dynamic queue.
5. Run MODE=combo only after S1/P1 checkpoints exist.
```
