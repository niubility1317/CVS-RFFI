# Baseline supervised receiver-curriculum experiments

## Goal

Run supervised baseline receiver-curriculum experiments first, under receiver
hold-out protocols that follow the table in the reference image. Pseudo-label
and SSDG runs are intentionally left off for this first pass.

The fixed data split is:

- Train days: `2021_03_01`, `2021_03_08` (`--wisig_train_days 0,1`)
- Test days: `2021_03_15`, `2021_03_23` (`--wisig_test_days 2,3`)
- Train ratio: `0.2` (`--wisig_train_ratio 0.2`)

## Receiver Mapping

The current WiSig index mapping observed in existing logs is:

| WiSig index | Receiver label |
| --- | --- |
| 0 | `1-1` |
| 1 | `1-19` |
| 2 | `14-7` |
| 3 | `18-2` |
| 4 | `19-2` |
| 5 | `2-1` |
| 6 | `2-19` |
| 7 | `20-1` |
| 8 | `3-19` |
| 9 | `7-14` |
| 10 | `7-7` |
| 11 | `8-8` |

The image-defined receiver anchors are therefore:

- `Rx(1-1)` -> index `0`
- `Rx(1-19)` -> index `1`
- `Rx(14-7)` -> index `2`
- `Rx(7-7)` -> index `10`
- `Rx(8-8)` -> index `11`

## Receiver Sets

### `Rx(1-19)` design block

Image 2-receiver training pairs:

| Pair ID | Training receivers |
| --- | --- |
| `T1_P01` | `1-1,7-7` -> `0,10` |
| `T1_P02` | `1-1,8-8` -> `0,11` |
| `T1_P03` | `1-1,14-7` -> `0,2` |
| `T1_P04` | `7-7,8-8` -> `10,11` |
| `T1_P05` | `7-7,14-7` -> `10,2` |
| `T1_P06` | `8-8,14-7` -> `11,2` |

The curriculum expands each pair to seven training receivers using the ordered
source universe:

`0,10,11,2,5,6,3` = `1-1,7-7,8-8,14-7,2-1,2-19,18-2`.

### `Rx(14-7)` design block

Image 2-receiver training pairs:

| Pair ID | Training receivers |
| --- | --- |
| `T14_P01` | `1-1,1-19` -> `0,1` |
| `T14_P02` | `1-1,7-7` -> `0,10` |
| `T14_P03` | `1-1,8-8` -> `0,11` |
| `T14_P04` | `1-19,7-7` -> `1,10` |
| `T14_P05` | `1-19,8-8` -> `1,11` |
| `T14_P06` | `7-7,8-8` -> `10,11` |

The curriculum expands each pair to seven training receivers using the ordered
source universe:

`0,1,10,11,5,6,3` = `1-1,1-19,7-7,8-8,2-1,2-19,18-2`.

For every image pair, level `K=2` is exactly the image row. Levels `K=3..7`
append the first not-yet-used receivers from the ordered universe. This keeps
the pair identity visible while progressively increasing source receiver
diversity.

## Test Receiver Policy

Testing reuses the existing CVS named-test module. For each training receiver
set, the launcher sets:

`--wisig_test_rxs = all WiSig receiver indices 0..11 not in --wisig_train_rxs`

This means every remaining receiver is evaluated, not only the image block's
nominal held-out receiver. The existing test module will still report the
main splits and per-receiver named splits such as `test_rx_7`, `test_rx_8`,
etc.

## Training Mode

The default launcher mode is supervised only:

- `plain`: original baseline path, no pseudo labels.

The script still accepts `PL_MODES=plain,pseudo` for a later pseudo-label
ablation, but do not set that for the supervised-first run.

## Recommended Run Plan

Start with `CORE` before `FULL`:

- `SMOKE`: one design block/pair, levels 2 and 7, `cvcnn`, supervised only.
- `CORE`: all image pairs for both targets, levels 2 through 7, default method
  `cvcnn`, supervised only.
- `FULL`: all image pairs for both targets, levels 2 through 7, all requested
  methods, supervised only.

## Commands

Dry run:

```bash
conda activate ssr-gpu
bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh --plan CORE --dry-run
```

Run the default CORE queue on GPU 0-5:

```bash
conda activate ssr-gpu
nohup bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh \
  --plan CORE \
  --gpu-ids 0,1,2,3,4,5 \
  > logs/baseline_supervised_rx_curriculum/nohup_core_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

Run all baseline comparison methods:

```bash
conda activate ssr-gpu
METHODS=cvcnn,drift,riei,receiver_agnostic,tifs2025 \
nohup bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh \
  --plan FULL \
  --gpu-ids 0,1,2,3,4,5 \
  > logs/baseline_supervised_rx_curriculum/nohup_full_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

Later pseudo-label ablation, not for the current supervised-first pass:

```bash
PL_MODES=plain,pseudo \
bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh --plan CORE --dry-run
```

If `ManySig.pkl` is not under `./Dataset_WigSig/ManySig.pkl`, pass it explicitly:

```bash
WISIG_PKL=/path/to/Dataset_WigSig/ManySig.pkl \
bash code/scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh --plan CORE
```
