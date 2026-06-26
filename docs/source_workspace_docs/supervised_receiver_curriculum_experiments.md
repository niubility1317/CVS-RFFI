# Supervised Receiver-Curriculum Baseline Experiments

## 1. Experiment Objective

This experiment runs supervised baseline comparison models under a receiver-curriculum protocol. The goal is to evaluate how transmitter-identification generalization changes as the number of training receivers increases from 2 to 7.

This first pass is supervised only:

- No pseudo-label training.
- No SSDG training.
- No semi-supervised target adaptation.
- Training ratio is fixed at `0.2`.
- Testing reuses the existing CVS named-test module.

The experiment follows the receiver-pair design shown in the reference table, then expands each initial 2-receiver pair step by step until 7 receivers are used for training.

## 2. Fixed Dataset Split

The day split is fixed for all experiments:

| Split | Days | CLI |
| --- | --- | --- |
| Train/validation days | `2021_03_01`, `2021_03_08` | `--wisig_train_days 0,1` |
| Test days | `2021_03_15`, `2021_03_23` | `--wisig_test_days 2,3` |

The training fraction is fixed:

```bash
--wisig_train_ratio 0.2
```

The validation data is the remaining contiguous tail after the guard gap:

```bash
--wisig_guard_gap 8
```

## 3. Receiver Index Mapping

The current WiSig receiver index mapping, confirmed from existing logs, is:

| WiSig index | Receiver label |
| --- | --- |
| `0` | `Rx(1-1)` |
| `1` | `Rx(1-19)` |
| `2` | `Rx(14-7)` |
| `3` | `Rx(18-2)` |
| `4` | `Rx(19-2)` |
| `5` | `Rx(2-1)` |
| `6` | `Rx(2-19)` |
| `7` | `Rx(20-1)` |
| `8` | `Rx(3-19)` |
| `9` | `Rx(7-14)` |
| `10` | `Rx(7-7)` |
| `11` | `Rx(8-8)` |

The receivers appearing in the reference table map to:

| Table receiver | WiSig index |
| --- | --- |
| `Rx(1-1)` | `0` |
| `Rx(1-19)` | `1` |
| `Rx(14-7)` | `2` |
| `Rx(7-7)` | `10` |
| `Rx(8-8)` | `11` |

## 4. Receiver-Curriculum Design

Each experiment starts from a 2-receiver pair from the reference table. The pair is then expanded to `K=3`, `K=4`, `K=5`, `K=6`, and `K=7` by appending receivers from a fixed source universe.

This gives a controlled curriculum:

```text
K=2: original table pair
K=3: original pair + one extra receiver
K=4: original pair + two extra receivers
K=5: original pair + three extra receivers
K=6: original pair + four extra receivers
K=7: original pair + five extra receivers
```

### 4.1 Design Block: `Rx(1-19)`

The reference-table block associated with `Rx(1-19)` uses six 2-receiver training pairs:

| Pair ID | Training receivers | WiSig indices |
| --- | --- | --- |
| `T1_P01` | `Rx(1-1), Rx(7-7)` | `0,10` |
| `T1_P02` | `Rx(1-1), Rx(8-8)` | `0,11` |
| `T1_P03` | `Rx(1-1), Rx(14-7)` | `0,2` |
| `T1_P04` | `Rx(7-7), Rx(8-8)` | `10,11` |
| `T1_P05` | `Rx(7-7), Rx(14-7)` | `10,2` |
| `T1_P06` | `Rx(8-8), Rx(14-7)` | `11,2` |

Expansion universe:

```text
0,10,11,2,5,6,3
```

Label form:

```text
Rx(1-1), Rx(7-7), Rx(8-8), Rx(14-7), Rx(2-1), Rx(2-19), Rx(18-2)
```

Example:

| Level | Training indices | Training labels |
| --- | --- | --- |
| `K=2` | `0,10` | `Rx(1-1), Rx(7-7)` |
| `K=3` | `0,10,11` | `Rx(1-1), Rx(7-7), Rx(8-8)` |
| `K=4` | `0,10,11,2` | `Rx(1-1), Rx(7-7), Rx(8-8), Rx(14-7)` |
| `K=5` | `0,10,11,2,5` | add `Rx(2-1)` |
| `K=6` | `0,10,11,2,5,6` | add `Rx(2-19)` |
| `K=7` | `0,10,11,2,5,6,3` | add `Rx(18-2)` |

### 4.2 Design Block: `Rx(14-7)`

The reference-table block associated with `Rx(14-7)` uses six 2-receiver training pairs:

| Pair ID | Training receivers | WiSig indices |
| --- | --- | --- |
| `T14_P01` | `Rx(1-1), Rx(1-19)` | `0,1` |
| `T14_P02` | `Rx(1-1), Rx(7-7)` | `0,10` |
| `T14_P03` | `Rx(1-1), Rx(8-8)` | `0,11` |
| `T14_P04` | `Rx(1-19), Rx(7-7)` | `1,10` |
| `T14_P05` | `Rx(1-19), Rx(8-8)` | `1,11` |
| `T14_P06` | `Rx(7-7), Rx(8-8)` | `10,11` |

Expansion universe:

```text
0,1,10,11,5,6,3
```

Label form:

```text
Rx(1-1), Rx(1-19), Rx(7-7), Rx(8-8), Rx(2-1), Rx(2-19), Rx(18-2)
```

Example:

| Level | Training indices | Training labels |
| --- | --- | --- |
| `K=2` | `0,10` | `Rx(1-1), Rx(7-7)` |
| `K=3` | `0,10,1` | add `Rx(1-19)` |
| `K=4` | `0,10,1,11` | add `Rx(8-8)` |
| `K=5` | `0,10,1,11,5` | add `Rx(2-1)` |
| `K=6` | `0,10,1,11,5,6` | add `Rx(2-19)` |
| `K=7` | `0,10,1,11,5,6,3` | add `Rx(18-2)` |

## 5. Test Receiver Policy

Testing uses all receivers not included in the current training set.

For each experiment:

```text
test_rxs = all WiSig receivers 0..11 minus train_rxs
```

Example 1:

```text
train_rxs = 0,10
test_rxs  = 1,2,3,4,5,6,7,8,9,11
```

Example 2:

```text
train_rxs = 0,10,11,2,5,6,3
test_rxs  = 1,4,7,8,9
```

This means the experiment does not test only one nominal held-out receiver. It tests every receiver left outside the training set.

The existing CVS test module will report:

- `test_unseen_day_seen_rx`
- `test_seen_day_unseen_rx`
- `test_unseen_day_unseen_rx`
- per-receiver splits such as `test_rx_7`, `test_rx_8`, etc.

Satellite-channel evaluation is also enabled for the main test splits:

```bash
--eval_sat_channel
--eval_sat_on main
--eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
```

## 6. Methods

The launcher supports these supervised baseline methods:

| Method key | Python module on server |
| --- | --- |
| `cvcnn` | `cvcnn.train` |
| `drift` | `drift.train` |
| `riei` | `riei.train` |
| `receiver_agnostic` | `receiver_agnostic_rffi.train` |
| `tifs2025` | `tifs2025_channel_receiver_rffi.train` |

Current first-pass command should use:

```bash
PL_MODES=plain
```

Do not use pseudo-label mode for this first supervised run.

## 7. Number of Experiments

For one method:

```text
2 design blocks x 6 pair IDs x 6 K levels = 72 runs
```

For five methods:

```text
72 x 5 = 360 supervised runs
```

With GPU `0..5`, the launcher runs one experiment per GPU at a time. When one GPU finishes its current experiment, it automatically pulls the next queued experiment.

## 8. Output Layout

Training logs should be stored under:

```bash
baseline_log/supervised_rx_curriculum
```

Weights and metrics should be stored under:

```bash
baseline_runs/supervised_rx_curriculum
```

Each experiment writes to:

```bash
baseline_runs/supervised_rx_curriculum/<EXP_ID>/
```

Expected important files:

```text
metrics.json
best_by_val.pt
```

The launcher also writes one scheduler log and one queue TSV:

```text
baseline_log/supervised_rx_curriculum/scheduler_<PLAN>_<timestamp>.log
baseline_log/supervised_rx_curriculum/queue_<PLAN>_<timestamp>.tsv
```

## 9. Recommended Commands

Run from the server `CV-SincNet` root:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
conda activate ssr-gpu
mkdir -p baseline_log baseline_runs
```

### 9.1 Dry Run

Use this first to confirm module names, train receivers, and test receivers:

```bash
METHODS=cvcnn,drift,riei,receiver_agnostic,tifs2025 \
PL_MODES=plain \
RUN_ROOT="$PWD/baseline_runs/supervised_rx_curriculum" \
LOG_ROOT="$PWD/baseline_log/supervised_rx_curriculum" \
WISIG_PKL="$PWD/Dataset_WigSig/ManySig.pkl" \
bash scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh \
  --plan SMOKE \
  --gpu-ids 0,1,2,3,4,5 \
  --dry-run
```

The dry-run command should show modules like:

```bash
python3 -u -m cvcnn.train
python3 -u -m drift.train
```

It should not show:

```bash
python3 -u -m baselines.cvcnn.train
```

### 9.2 Run Full Supervised Experiment Matrix

```bash
METHODS=cvcnn,drift,riei,receiver_agnostic,tifs2025 \
PL_MODES=plain \
GPU_IDS=0,1,2,3,4,5 \
RUN_ROOT="$PWD/baseline_runs/supervised_rx_curriculum" \
LOG_ROOT="$PWD/baseline_log/supervised_rx_curriculum" \
WISIG_PKL="$PWD/Dataset_WigSig/ManySig.pkl" \
nohup bash scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh \
  --plan FULL \
  --gpu-ids 0,1,2,3,4,5 \
  > baseline_log/nohup_supervised_rx_curriculum_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

### 9.3 Check Running Jobs

```bash
tail -f baseline_log/nohup_supervised_rx_curriculum_*.out
```

Check scheduler logs:

```bash
tail -f baseline_log/supervised_rx_curriculum/scheduler_FULL_*.log
```

Check GPU processes:

```bash
nvidia-smi
```

## 10. Restart Behavior

The launcher skips completed experiments by default when:

```text
baseline_runs/supervised_rx_curriculum/<EXP_ID>/metrics.json
```

already exists.

Failed or incomplete experiments can be rerun by launching the same command again. Existing completed runs are skipped, and missing runs continue.

To force rerun everything:

```bash
bash scripts/run_baseline_pseudo_rx_curriculum_6gpu.sh --plan FULL --no-skip-done
```

Use this only when intentionally overwriting or regenerating results.

## 11. Result Interpretation

Primary comparisons:

1. For each method, compare performance as `K` increases from 2 to 7.
2. Compare the six initial table pairs within the same design block.
3. Compare `Rx(1-19)` design block against `Rx(14-7)` design block.
4. Compare methods under identical receiver sets.

Recommended metrics to inspect in `metrics.json`:

- best validation transmitter accuracy or loss, depending on method
- `test_seen_day_unseen_rx`
- `test_unseen_day_unseen_rx`
- per-receiver `test_rx_*`
- satellite scenario aggregate and strict UDU scores

The most important question is whether adding more training receivers improves unseen-receiver and unseen-day-unseen-receiver performance consistently, or whether certain receiver combinations are more useful than simply increasing `K`.
