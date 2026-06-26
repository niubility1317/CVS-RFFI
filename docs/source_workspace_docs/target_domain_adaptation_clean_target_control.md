# Clean-Target Adaptation Control

## Purpose

This control uses 干净的目标域样本 for target-domain adaptation, without adding a satellite channel to the adaptation samples:

```bash
--target_channel_view clean
```

All other sweep settings match `run_target_adapt_bex02_sweep_8gpu.sh`: BEX02 teacher, `test_unseen_day_unseen_rx` target loader, labeled/unlabeled routes, target sample budgets, epoch counts, fine-tuning strength profiles, and per-epoch evaluation.

Satellite-channel evaluation remains enabled during evaluation. This makes the control answer a specific question: does adapting on clean target-domain samples improve target receiver/date robustness, or is adaptation on the provided satellite target samples necessary for 星地信道 generalization?

## Launcher

```bash
code/scripts/run_target_adapt_bex02_clean_target_8gpu.sh
```

The script is a thin wrapper around `run_target_adapt_bex02_sweep_8gpu.sh` with these defaults:

| Setting | Value |
|---|---|
| `TARGET_CHANNEL_VIEW` | `clean` |
| `EXP_PREFIX` | `BEX02_tadapt_clean` |
| `RUN_ROOT` | `runs/target_adapt_bex02_clean_target_8gpu` |
| `LOG_ROOT` | `logs/target_adapt_bex02_clean_target_8gpu` |

## Matrix

`CORE`:

| Dimension | Values |
|---|---|
| Epochs | 20, 50, 100 |
| Target samples per receiver | 5, 10 |
| Label routes | labeled, unlabeled |
| Fine-tuning strength | safe, base, strong |
| Seeds | 1337 |
| Jobs | 36 |

`FULL` repeats the same matrix on seeds `1337,2027,42` for 108 jobs.

## Commands

Dry-run:

```bash
bash code/scripts/run_target_adapt_bex02_clean_target_8gpu.sh --plan CORE --dry-run
```

Run CORE:

```bash
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
bash code/scripts/run_target_adapt_bex02_clean_target_8gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5,6,7
```

Nohup command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/target_adapt_bex02_clean_target_8gpu_${STAMP}"
RUN_DIR="runs/target_adapt_bex02_clean_target_8gpu_${STAMP}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

nohup bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ssr-gpu
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
RUN_ROOT='"$RUN_DIR"' \
LOG_ROOT='"$LOG_DIR"' \
bash code/scripts/run_target_adapt_bex02_clean_target_8gpu.sh \
  --plan CORE \
  --gpu-ids 0,1,2,3,4,5,6,7
' > "$LOG_DIR/nohup_target_adapt_bex02_clean_target_${STAMP}.out" 2>&1 &
```

Monitor:

```bash
tail -f "$LOG_DIR/nohup_target_adapt_bex02_clean_target_${STAMP}.out"
```

## Expected Log Checks

Each job should include:

- `[CONFIG-TARGET] ... view=clean`
- `[EPOCH] ... target_tx=<number>% sat_mean=<number>%`
- `[AFTER-ADAPT] ... target_view=clean ... sat_mean=<number>%`

Compare these against the provided-satellite sweep from `run_target_adapt_bex02_sweep_8gpu.sh` at the same label route, sample budget, epoch count, strength profile, and seed.
