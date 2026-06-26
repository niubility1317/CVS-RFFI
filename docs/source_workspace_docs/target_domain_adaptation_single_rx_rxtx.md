# Single-RX RX-TX Target Adaptation

## Purpose

This experiment gives the adapter only one target receiver domain at a time, instead of mixing all target receivers in the adaptation subset.

The five target receiver domains are run separately:

```text
RX 7, 8, 9, 10, 11
```

For each single RX, the sampler selects 每个发射机 5, 10 target samples. Both labeled and unlabeled routes are run for 30 epochs.

## Matrix

| Dimension | Values |
|---|---|
| Target loader | `test_unseen_day_rx_7`, `test_unseen_day_rx_8`, `test_unseen_day_rx_9`, `test_unseen_day_rx_10`, `test_unseen_day_rx_11` |
| Samples | `rxtx5`, `rxtx10` |
| Label modes | `labeled`, `unlabeled` |
| Epochs | `30` |
| Adapt weight | `base` |
| Jobs | `20` |

For one receiver, assuming all 6 transmitters are present:

- `rxtx5`: `5 x 6 = 30` adaptation samples;
- `rxtx10`: `10 x 6 = 60` adaptation samples.

The unlabeled route uses transmitter labels only for stratified sample selection; labels are not used in the adaptation loss.

## Launcher

```bash
code/scripts/run_target_adapt_bex02_single_rx_rxtx_8gpu.sh
```

This is a wrapper around `run_target_adapt_bex02_sweep_8gpu.sh` with these defaults:

| Setting | Value |
|---|---|
| `TARGET_LOADERS` | `test_unseen_day_rx_7,test_unseen_day_rx_8,test_unseen_day_rx_9,test_unseen_day_rx_10,test_unseen_day_rx_11` |
| `TARGET_SAMPLES_PER_RX_TX` | `5,10` |
| `TARGET_LABEL_MODES` | `labeled,unlabeled` |
| `EPOCHS` | `30` |
| `ADAPT_WEIGHTS` | `base` |
| `EXP_PREFIX` | `BEX02_tadapt_single_rx_rxtx` |

## Commands

Dry-run:

```bash
bash code/scripts/run_target_adapt_bex02_single_rx_rxtx_8gpu.sh --plan CORE --dry-run
```

Run:

```bash
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
bash code/scripts/run_target_adapt_bex02_single_rx_rxtx_8gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5,6,7
```

Nohup:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/target_adapt_bex02_single_rx_rxtx_8gpu_${STAMP}"
RUN_DIR="runs/target_adapt_bex02_single_rx_rxtx_8gpu_${STAMP}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

nohup bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ssr-gpu
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
RUN_ROOT='"$RUN_DIR"' \
LOG_ROOT='"$LOG_DIR"' \
bash code/scripts/run_target_adapt_bex02_single_rx_rxtx_8gpu.sh \
  --plan CORE \
  --gpu-ids 0,1,2,3,4,5,6,7
' > "$LOG_DIR/nohup_target_adapt_bex02_single_rx_rxtx_${STAMP}.out" 2>&1 &
```

Monitor:

```bash
tail -f "$LOG_DIR/nohup_target_adapt_bex02_single_rx_rxtx_${STAMP}.out"
```

## Expected Log Checks

Each job should show:

- `target_loader=test_unseen_day_rx_7` or another single RX loader;
- `samples_per_rx_tx=5` or `samples_per_rx_tx=10`;
- `few_size=30` for `rxtx5`, assuming 6 transmitters are present;
- `few_size=60` for `rxtx10`, assuming 6 transmitters are present;
- `[EPOCH] E030/030 ... target_tx=<number>% sat_mean=<number>%`.
