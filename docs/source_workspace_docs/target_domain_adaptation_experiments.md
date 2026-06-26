# Target-Domain Adaptation Experiments

## Goal

Evaluate whether a small amount of target-domain signal improves the strongest baseline on target receiver/date conditions. The target-domain samples are already satellite-channel signals, not clean signals, so the adaptation trainer does not synthesize additional star-ground views.

## Baseline

Use `BEX02_fishr002_mixed_e170` as the teacher checkpoint. The default launcher expects:

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth
```

If needed, override it with `TEACHER_CKPT=/path/to/latest_model.pth`.

## Method

The added trainer is `code/train_target_adapt.py`.

It implements two conservative target-adaptation lines:

- Freeze the trained baseline by default.
- Update a tiny logit calibration layer and normalization affine parameters.
- Select target samples by total sample budget.
- `labeled`: use transmitter labels on the selected satellite-channel target samples for supervised CE adaptation.
- `unlabeled`: do not use transmitter labels; use entropy minimization and confident pseudo-label sharpening.
- Both lines use the provided target samples directly as single-view satellite-channel inputs.
- Both lines exclude the selected adaptation samples from target-domain testing.
- Both lines keep a small teacher-anchor KL term to reduce drift from BEX02.

The unlabeled line is TENT/SHOT-style. The labeled line is supervised few-shot target adaptation with the same parameter-efficient update scope.

## Main Launch Commands

Dry-run:

```bash
bash code/scripts/run_target_adapt_bex02_6gpu.sh --plan SMOKE --dry-run
```

Core run:

```bash
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
bash code/scripts/run_target_adapt_bex02_6gpu.sh --plan CORE --gpu-ids 6,7
```

Full run:

```bash
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
bash code/scripts/run_target_adapt_bex02_6gpu.sh --plan FULL --gpu-ids 6,7
```

## Experiment Matrix

| Plan | Target samples | Label modes | Seeds | Purpose |
|---|---|---|---|---|
| `SMOKE` | `5,10` | `labeled,unlabeled` | `1337` | Validate command and pipeline quickly. |
| `CORE` | `5,10` | `labeled,unlabeled` | `1337` | Main signal-budget comparison. |
| `FULL` | `5,10` | `labeled,unlabeled` | `1337,2027,42` | Robust multi-seed curve. |

All plans adapt on `test_unseen_day_unseen_rx` by default. This can be changed with `--target-loader`.

## Direct Single-Run Command

```bash
cd code
CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PWD" python -u train_target_adapt.py \
  --teacher_ckpt /home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
  --output_dir ../runs/target_adapt_bex02/manual_labeled_n10_seed1337 \
  --target_loader test_unseen_day_unseen_rx \
  --target_channel_view provided_satellite \
  --target_label_mode labeled \
  --target_num_samples 10 \
  --dataset wisig \
  --wisig_pkl /path/to/Dataset_WigSig/ManySig.pkl \
  --wisig_domain rx_day \
  --wisig_train_days 0,1 \
  --wisig_test_days 2,3 \
  --wisig_train_rxs 0,1,2,3,4,5,6 \
  --wisig_test_rxs 7,8,9,10,11 \
  --epochs 20 \
  --lr_adapt 5e-5 \
  --eval_sat_channel false
```

## What To Compare

For each run, compare:

- `[BEFORE-ADAPT] target_tx` vs epoch `[AFTER-ADAPT] target_tx`.
- The log config line `eval_size_after_excluding_adapt`, confirming target testing excludes adaptation samples.
- `labeled` vs `unlabeled` at the same target sample budget.
- Whether source/main OOD aggregate drops sharply. If it does, increase `anchor_weight` or lower `lr_adapt`.
