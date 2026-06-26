# RX-TX Balanced Target Adaptation

## Purpose

This experiment changes the target adaptation sample budget from random samples per receiver to transmitter-balanced samples inside each receiver.

中文描述：每个目标接收机内，对每个发射机分别随机抽取固定数量样本，预算为 2, 3。

For each target receiver, select:

- `2` samples from every transmitter;
- `3` samples from every transmitter.

With the current BEX02/WiSig setup, target receivers are RX `7,8,9,10,11` and the task has 6 transmitters. Therefore the approximate adaptation sizes are:

| Budget | Per receiver | Total over 5 target receivers |
|---|---:|---:|
| `rxtx2` | `2 x 6 = 12` samples | `60` samples |
| `rxtx3` | `3 x 6 = 18` samples | `90` samples |

Both `labeled` and `unlabeled` routes are run. The unlabeled route uses transmitter labels only for stratified sample selection; the adaptation loss still does not use transmitter labels.

## Launcher

```bash
code/scripts/run_target_adapt_bex02_rx_tx_balanced_8gpu.sh
```

It wraps `run_target_adapt_bex02_sweep_8gpu.sh` with:

| Setting | Value |
|---|---|
| `TARGET_SAMPLES_PER_RX_TX` | `2,3` |
| `EXP_PREFIX` | `BEX02_tadapt_rxtx` |
| `RUN_ROOT` | `runs/target_adapt_bex02_rx_tx_balanced_8gpu` |
| `LOG_ROOT` | `logs/target_adapt_bex02_rx_tx_balanced_8gpu` |

All other defaults match the main BEX02 target adaptation sweep: epochs `20,50,100`, weights `safe,base,strong`, label modes `labeled,unlabeled`, target view `provided_satellite`, and per-epoch satellite evaluation.

## Commands

Dry-run:

```bash
bash code/scripts/run_target_adapt_bex02_rx_tx_balanced_8gpu.sh --plan CORE --dry-run
```

Run CORE:

```bash
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
bash code/scripts/run_target_adapt_bex02_rx_tx_balanced_8gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5,6,7
```

Nohup:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/target_adapt_bex02_rx_tx_balanced_8gpu_${STAMP}"
RUN_DIR="runs/target_adapt_bex02_rx_tx_balanced_8gpu_${STAMP}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

nohup bash -lc '
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ssr-gpu
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
RUN_ROOT='"$RUN_DIR"' \
LOG_ROOT='"$LOG_DIR"' \
bash code/scripts/run_target_adapt_bex02_rx_tx_balanced_8gpu.sh \
  --plan CORE \
  --gpu-ids 0,1,2,3,4,5,6,7
' > "$LOG_DIR/nohup_target_adapt_bex02_rx_tx_balanced_${STAMP}.out" 2>&1 &
```

Monitor:

```bash
tail -f "$LOG_DIR/nohup_target_adapt_bex02_rx_tx_balanced_${STAMP}.out"
```

## Expected Log Checks

Each job should show:

- `[WARN] --target_samples_per_rx_tx uses transmitter labels for stratified sample selection`
- `[CONFIG-TARGET] ... samples_per_rx_tx=2` or `samples_per_rx_tx=3`
- `[CONFIG-TARGET] ... few_size=60` for `rxtx2`, assuming all 6 TX are present for all 5 target RX;
- `[CONFIG-TARGET] ... few_size=90` for `rxtx3`, under the same assumption;
- `[EPOCH] ... target_tx=<number>% sat_mean=<number>%`.

Compare this against the random-per-RX `rxn5/rxn10` sweep to separate the effect of more samples from the effect of transmitter-balanced coverage.
