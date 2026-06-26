# Target-Domain Adaptation 8GPU Sweep

## Purpose

`5.20-adapt-logs/target_adapt_bex02_finetune_fix_20260521_001930` confirmed that the BEX02 target-adaptation path can run, but the logged experiments were too narrow:

- only 20 adaptation epochs were tested;
- satellite-channel evaluation was disabled, so `sat_mean=nan%`;
- the adaptation strength was fixed at one setting;
- labeled and unlabeled routes were compared only for rxn5/rxn10 at seed 1337.

The new launcher is:

```bash
code/scripts/run_target_adapt_bex02_sweep_8gpu.sh
```

It keeps the same BEX02 teacher and target-loader setup, uses GPU `0..7` by default, evaluates every epoch, and enables 星地信道 evaluation every epoch.

## Default Matrix

`CORE` is the recommended first full pass:

| Dimension | Values |
|---|---|
| Epochs | 20, 50, 100 |
| Target samples per receiver | 5, 10 |
| Label routes | labeled, unlabeled |
| Fine-tuning strength | safe, base, strong |
| Seeds | 1337 |
| Jobs | 36 |

Fine-tuning strength profiles:

| Profile | `lr_adapt` | `anchor_weight` | Intent |
|---|---:|---:|---|
| `safe` | `5e-5` | `0.10` | Reduce drift for longer 50/100 epoch runs. |
| `base` | `1e-4` | `0.05` | Match the 5.20 run's adaptation scale. |
| `strong` | `2e-4` | `0.02` | Test whether target performance improves with more adaptation freedom. |

`FULL` repeats the same grid on seeds `1337,2027,42` for 108 jobs.

## Commands

Dry-run the default CORE grid:

```bash
bash code/scripts/run_target_adapt_bex02_sweep_8gpu.sh --plan CORE --dry-run
```

Run CORE on all available GPUs:

```bash
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
bash code/scripts/run_target_adapt_bex02_sweep_8gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5,6,7
```

Run the multi-seed FULL grid:

```bash
TEACHER_CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
bash code/scripts/run_target_adapt_bex02_sweep_8gpu.sh --plan FULL --gpu-ids 0,1,2,3,4,5,6,7
```

Run a smaller ablation from the same script:

```bash
bash code/scripts/run_target_adapt_bex02_sweep_8gpu.sh \
  --plan CORE \
  --gpu-ids 0,1,2,3 \
  --target-samples 10 \
  --epochs 20,50,100 \
  --adapt-weights base,strong \
  --target-label-modes labeled,unlabeled \
  --dry-run
```

Custom fine-tuning strength can be passed as `name:lr_adapt:anchor_weight`:

```bash
bash code/scripts/run_target_adapt_bex02_sweep_8gpu.sh \
  --plan CORE \
  --adapt-weights base,very_safe:2.5e-5:0.20 \
  --dry-run
```

## Expected Logs

Each experiment log should show:

- `[BEFORE-ADAPT] ... sat_mean=<number>%`
- `[EPOCH] E001/... target_tx=<number>% sat_mean=<number>%`
- `[AFTER-ADAPT] ... sat_mean=<number>%`
- `[TEST-SPLIT]` and satellite scenario lines every epoch because `--eval_detail_every 1` is the default.

The launcher writes:

- scheduler logs under `logs/target_adapt_bex02_sweep_8gpu/`;
- queue TSVs under the same log root;
- checkpoints under `runs/target_adapt_bex02_sweep_8gpu/<experiment_id>/`.

## Reading Results

Primary comparisons:

- best epoch versus final epoch for 20, 50, and 100 epochs;
- `target_tx` and `sat_mean` together, because the selected checkpoint scores both when satellite evaluation is enabled;
- labeled versus unlabeled at the same sample budget, epoch count, seed, and strength profile;
- safe/base/strong drift behavior, especially whether source/main aggregate drops while target improves.

Initial decision rule:

- prefer runs that improve target accuracy without lowering `sat_mean`;
- for labeled runs, watch whether strong adaptation overfits after early epochs;
- for unlabeled runs, inspect pseudo coverage and whether longer 50/100 epoch schedules collapse after the best epoch.
