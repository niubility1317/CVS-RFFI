# Weekend Seed 1337 Experiment Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize baseline-path exploration over one weekend while running only `seed=1337`.

**Architecture:** Use a 48-hour, 8-GPU queue with staged priority. Start with high-information B3b-centered fusions, then expand into satellite CE/Fishr sweeps, MixStyle interaction, satellite scenario choice, start epoch, regularizer alternatives, and small architecture/post-stage probes if time remains.

**Tech Stack:** `train.py`, WiSig `rx_day`, R25 compact `rxrobust_lite_d_no_dac_refined`, 200 epochs for all scheduled runs, satellite evaluation on five scenarios.

---

## Non-Negotiables

```text
Seed: only 1337.
Backbone family: R25 compact unless an experiment explicitly says architecture probe.
Primary objective: find the strongest B3b-centered baseline.
Time budget: Friday night to Monday morning, about 48 hours.
Scheduling model: 8 concurrent GPU workers, dynamic FIFO queue.
```

Do not schedule repeat seeds. The historical seed sensitivity is real, but this weekend is an exploitation/exploration sweep under one fixed seed.

## Runtime Budget

Observed from `B2_stable_sat07_fishr_20260514_111353.log`:

```text
Epoch 200 time: about 81.6s
Full 200 epoch run: about 4.5-5.0 hours plus eval/log overhead
8 GPUs * 48 hours / 5 hours = about 76 theoretical runs
Safe weekend target: 40-48 runs
```

Use this capacity rule:

```text
Must-run core: 16 runs
High-value expansion: 16 runs
Conditional overflow: 16 runs
Total design: 48 runs
Expected completed by weekend: 32-48 depending on environment and GPU health
```

## Common Command Template

Use this template for every W experiment unless the table overrides an argument:

```bash
python3 -u train.py \
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
  --sat_eval_max_batches -1 \
  --run_name <RUN_NAME> \
  --latest_save_path runs/weekend_seed1337/<RUN_NAME>/latest_model.pth \
  --best_save_path runs/weekend_seed1337/<RUN_NAME>/best_val_model.pth \
  --best_primary_save_path runs/weekend_seed1337/<RUN_NAME>/best_primary_ood_model.pth \
  --best_unseen_day_unseen_rx_save_path runs/weekend_seed1337/<RUN_NAME>/best_strict_udu_model.pth \
  --slim_group rxrobust_lite_d_no_dac_refined \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.10 \
  --lambda_sat_cons 0.00 \
  --seed 1337 \
  <EXTRA_ARGS>
```

## Baseline References

Use these as score gates:

```text
B2 global reference:
Primary 88.06, strict_udu 86.66, test_overall 90.66
Satellite: clear 45.05, low 45.35, rain 42.94, storm 37.92, mixed 40.64

B3b satellite reference:
Primary 87.94, strict_udu 86.47, test_overall 90.68
Satellite: clear 45.82, low 45.74, rain 43.37, storm 38.60, mixed 42.05
Satellite mean: 43.116
```

Promotion rules:

```text
Global winner: Primary > 88.06 and strict_udu >= 86.66.
Satellite winner: satellite_mean > 43.116 and storm_mp >= 38.60.
Balanced winner: Primary >= 88.00, strict_udu >= 86.50, satellite_mean >= 43.00, test_overall >= 90.50.
Kill signal: test_overall < 90.20 or strict_udu < 86.00 unless satellite_mean is clearly best.
```

## Wave 1: Must-Run Core, 8 GPUs

These eight runs should start first. They answer whether B3b + Fishr, no-MixStyle, and the main lambda neighborhood can beat both B2 and B3b.

| ID | Run name | Purpose | Extra args |
|---|---|---|---|
| W01 | `W01_fishr002_cls010_cons000` | Main candidate: B3b + Fishr | `--lambda_fishr 0.02 --fishr_min_domains 4` |
| W02 | `W02_fishr001_cls010_cons000` | Weaker Fishr preserves satellite robustness | `--lambda_fishr 0.01 --fishr_min_domains 4` |
| W03 | `W03_fishr003_cls010_cons000` | Stronger Fishr tests strict_udu headroom | `--lambda_fishr 0.03 --fishr_min_domains 4` |
| W04 | `W04_fishr002_cls010_cons000_nomix` | Main no-MixStyle stack | `--lambda_fishr 0.02 --fishr_min_domains 4 --no_use_mixstyle` |
| W05 | `W05_fishr001_cls010_cons000_nomix` | Lower Fishr with no MixStyle | `--lambda_fishr 0.01 --fishr_min_domains 4 --no_use_mixstyle` |
| W06 | `W06_fishr002_cls012_cons000` | Stronger satellite CE | `--lambda_fishr 0.02 --fishr_min_domains 4 --lambda_sat_cls 0.12` |
| W07 | `W07_fishr002_cls008_cons000` | Is B3b gain mostly no-consistency? | `--lambda_fishr 0.02 --fishr_min_domains 4 --lambda_sat_cls 0.08` |
| W08 | `W08_cls010_cons000_nomix` | B3b + no MixStyle without Fishr | `--no_use_mixstyle` |

Expected decision after Wave 1:

```text
If W01 wins global and keeps satellite_mean >= 43.0, center the rest of the weekend on Fishr=0.02.
If W02 beats W01 on satellite_mean, Fishr=0.02 is too strong; prioritize weaker Fishr variants.
If W04/W05 beat W01/W02, MixStyle is the main remaining conflict.
If W06 wins storm/rain but loses overall, use cls=0.12 only for satellite-focused branches.
```

## Wave 2: Fishr x Satellite CE Grid, 8 GPUs

Run after Wave 1 starts or immediately if all 8 GPUs are free in a scheduler queue. This grid fills the most plausible B3b-centered surface.

| ID | Run name | Purpose | Extra args |
|---|---|---|---|
| W09 | `W09_fishr0005_cls010_cons000` | Very light Fishr | `--lambda_fishr 0.005 --fishr_min_domains 4` |
| W10 | `W10_fishr0015_cls010_cons000` | Midpoint between W02 and W01 | `--lambda_fishr 0.015 --fishr_min_domains 4` |
| W11 | `W11_fishr004_cls010_cons000` | Upper Fishr stress before 0.05 | `--lambda_fishr 0.04 --fishr_min_domains 4` |
| W12 | `W12_fishr000_cls010_cons000` | Reproduce B3b in same run root | no extra args |
| W13 | `W13_fishr001_cls012_cons000` | cls=0.12 with weak Fishr | `--lambda_sat_cls 0.12 --lambda_fishr 0.01 --fishr_min_domains 4` |
| W14 | `W14_fishr003_cls012_cons000` | cls=0.12 with stronger Fishr | `--lambda_sat_cls 0.12 --lambda_fishr 0.03 --fishr_min_domains 4` |
| W15 | `W15_fishr001_cls008_cons000` | cls=0.08 with weak Fishr | `--lambda_sat_cls 0.08 --lambda_fishr 0.01 --fishr_min_domains 4` |
| W16 | `W16_fishr003_cls008_cons000` | cls=0.08 with stronger Fishr | `--lambda_sat_cls 0.08 --lambda_fishr 0.03 --fishr_min_domains 4` |

## Wave 3: MixStyle Interaction Grid, 8 GPUs

Use this to test whether satellite degradation was mainly caused by MixStyle rather than Fishr.

| ID | Run name | Purpose | Extra args |
|---|---|---|---|
| W17 | `W17_fishr0005_cls010_cons000_nomix` | Very light Fishr no-MixStyle | `--lambda_fishr 0.005 --fishr_min_domains 4 --no_use_mixstyle` |
| W18 | `W18_fishr0015_cls010_cons000_nomix` | Mid Fishr no-MixStyle | `--lambda_fishr 0.015 --fishr_min_domains 4 --no_use_mixstyle` |
| W19 | `W19_fishr003_cls010_cons000_nomix` | Stronger Fishr no-MixStyle | `--lambda_fishr 0.03 --fishr_min_domains 4 --no_use_mixstyle` |
| W20 | `W20_fishr002_cls012_cons000_nomix` | Strong CE no-MixStyle | `--lambda_sat_cls 0.12 --lambda_fishr 0.02 --fishr_min_domains 4 --no_use_mixstyle` |
| W21 | `W21_fishr002_cls008_cons000_nomix` | Baseline CE no-MixStyle | `--lambda_sat_cls 0.08 --lambda_fishr 0.02 --fishr_min_domains 4 --no_use_mixstyle` |
| W22 | `W22_fishr001_cls012_cons000_nomix` | Weak Fishr + strong CE no-MixStyle | `--lambda_sat_cls 0.12 --lambda_fishr 0.01 --fishr_min_domains 4 --no_use_mixstyle` |
| W23 | `W23_fishr001_cls008_cons000_nomix` | Weak Fishr + low CE no-MixStyle | `--lambda_sat_cls 0.08 --lambda_fishr 0.01 --fishr_min_domains 4 --no_use_mixstyle` |
| W24 | `W24_cls012_cons000_nomix` | Strong CE no-MixStyle without Fishr | `--lambda_sat_cls 0.12 --no_use_mixstyle` |

## Wave 4: Consistency Reintroduction and Start Epoch, 8 GPUs

These check whether `lambda_sat_cons=0` is universally best, or whether tiny consistency becomes useful after adding Fishr/no-MixStyle.

| ID | Run name | Purpose | Extra args |
|---|---|---|---|
| W25 | `W25_fishr002_cls010_cons001` | Tiny consistency | `--lambda_sat_cons 0.01 --lambda_fishr 0.02 --fishr_min_domains 4` |
| W26 | `W26_fishr002_cls010_cons002` | Weak consistency, like B3a but Fishr | `--lambda_sat_cons 0.02 --lambda_fishr 0.02 --fishr_min_domains 4` |
| W27 | `W27_fishr001_cls010_cons001` | Tiny consistency with weak Fishr | `--lambda_sat_cons 0.01 --lambda_fishr 0.01 --fishr_min_domains 4` |
| W28 | `W28_fishr002_cls010_cons001_nomix` | Tiny consistency no-MixStyle | `--lambda_sat_cons 0.01 --lambda_fishr 0.02 --fishr_min_domains 4 --no_use_mixstyle` |
| W29 | `W29_fishr002_cls010_start10` | Earlier sat start | `--lambda_fishr 0.02 --fishr_min_domains 4 --sat_cons_start_epoch 10` |
| W30 | `W30_fishr002_cls010_start30` | Later sat start | `--lambda_fishr 0.02 --fishr_min_domains 4 --sat_cons_start_epoch 30` |
| W31 | `W31_fishr002_cls010_start40` | Conservative sat start | `--lambda_fishr 0.02 --fishr_min_domains 4 --sat_cons_start_epoch 40` |
| W32 | `W32_fishr002_cls010_start60` | Very late sat start, SAT15 style | `--lambda_fishr 0.02 --fishr_min_domains 4 --sat_cons_start_epoch 60` |

## Wave 5: Satellite Scenario Specialization, 8 GPUs

B3b uses `mixed_orbit`; this wave checks whether training on a harder or more correlated scenario gives better strict_udu/satellite transfer.

| ID | Run name | Purpose | Extra args |
|---|---|---|---|
| W33 | `W33_fishr002_cls010_train_rain` | rain_leo has highest correlation with strict_udu | `--lambda_fishr 0.02 --fishr_min_domains 4 --sat_train_scenario rain_leo` |
| W34 | `W34_fishr001_cls010_train_rain` | rain_leo with weaker Fishr | `--lambda_fishr 0.01 --fishr_min_domains 4 --sat_train_scenario rain_leo` |
| W35 | `W35_fishr002_cls010_train_storm` | Direct storm optimization | `--lambda_fishr 0.02 --fishr_min_domains 4 --sat_train_scenario storm_mp` |
| W36 | `W36_fishr001_cls010_train_storm` | storm with weaker Fishr | `--lambda_fishr 0.01 --fishr_min_domains 4 --sat_train_scenario storm_mp` |
| W37 | `W37_cls010_train_rain` | Pure B3b style rain training | `--sat_train_scenario rain_leo` |
| W38 | `W38_cls010_train_storm` | Pure B3b style storm training | `--sat_train_scenario storm_mp` |
| W39 | `W39_fishr002_cls010_train_low` | low_elev specialization | `--lambda_fishr 0.02 --fishr_min_domains 4 --sat_train_scenario low_elev_leo` |
| W40 | `W40_fishr002_cls010_train_clear` | clear specialization control | `--lambda_fishr 0.02 --fishr_min_domains 4 --sat_train_scenario clear_leo` |

## Wave 6: Regularizer and Architecture Overflow, 8 GPUs

Only run these after W01-W40 are launched or if some earlier branch clearly fails. This wave explores alternatives but has lower priority than the B3b/Fishr/MixStyle surface.

| ID | Run name | Purpose | Extra args |
|---|---|---|---|
| W41 | `W41_smoothdro_cls010_cons000` | SmoothDRO combined with cls_only | `--group_ce_mode smooth_dro --groupdro_tau 0.45 --groupdro_momentum 0.95` |
| W42 | `W42_smoothdro_fishr001_cls010_cons000` | Weak Fishr + SmoothDRO | `--lambda_fishr 0.01 --fishr_min_domains 4 --group_ce_mode smooth_dro --groupdro_tau 0.45 --groupdro_momentum 0.95` |
| W43 | `W43_smoothdro_cls010_cons000_nomix` | SmoothDRO no-MixStyle | `--group_ce_mode smooth_dro --groupdro_tau 0.45 --groupdro_momentum 0.95 --no_use_mixstyle` |
| W44 | `W44_groupdro_capped_cls010_cons000` | Capped GroupDRO alternative | `--group_ce_mode smooth_dro_capped --groupdro_tau 0.45 --groupdro_momentum 0.95` |
| W45 | `W45_r19_fishr002_cls010_cons000` | R19 capacity check | `--slim_group rxrobust_lite_b_no_dac_mix015 --lambda_fishr 0.02 --fishr_min_domains 4` |
| W46 | `W46_r19_cls010_cons000` | R19 B3b-style control | `--slim_group rxrobust_lite_b_no_dac_mix015` |
| W47 | `W47_r25_fishr002_cls014_cons000` | Aggressive satellite CE | `--lambda_sat_cls 0.14 --lambda_fishr 0.02 --fishr_min_domains 4` |
| W48 | `W48_r25_fishr005_cls010_cons000` | Strong Fishr stress test | `--lambda_fishr 0.05 --fishr_min_domains 4` |

## Weekend Scheduling

### Friday Night / Saturday Morning

Launch W01-W16.

```text
Goal: answer the main question quickly.
Expected completion: first 16 runs in about 10-12 wall-clock hours on 8 GPUs.
Decision checkpoint: Saturday morning/noon.
```

### Saturday Afternoon / Saturday Night

Launch W17-W32 unless Wave 1 clearly invalidates a branch.

```text
If no-MixStyle is bad in W04/W05/W08, still run W17-W24 only if spare capacity exists.
If Fishr=0.03 is bad in W03, deprioritize W11/W14/W19.
If tiny consistency helps in W25/W26, consider adding more consistency sweeps after the weekend.
```

### Sunday

Launch W33-W48 by priority:

```text
First: W33-W36, because rain/storm training answers satellite generalization directly.
Second: W41-W44, because regularizer alternatives may recover global strict_udu.
Third: W45-W48, because they are useful but more speculative.
```

## Adaptive Pruning Rules

Use these rules to save weekend time:

```text
If a branch has strict_udu < 85.50 by epoch 100 and no satellite scene exceeds B3b by 0.50pp, stop that run.
If cls=0.12 variants consistently lower test_overall below 90.20, stop remaining cls=0.12/0.14 runs.
If no-MixStyle variants lose both strict_udu and satellite_mean across W04/W05/W08, stop W17-W24.
If Fishr=0.03 and 0.04 both regress satellite_mean, do not run W48 Fishr=0.05.
If train_storm improves storm_mp but collapses mixed_orbit by more than 1.0pp, keep it as a diagnostic, not a final baseline.
```

## Results Table To Fill

```text
Run | Primary | strict_udu | test_overall | best_epoch | clear | low | rain | storm | mixed | sat_mean | status
W01 |
W02 |
W03 |
W04 |
W05 |
W06 |
W07 |
W08 |
W09 |
W10 |
W11 |
W12 |
W13 |
W14 |
W15 |
W16 |
W17 |
W18 |
W19 |
W20 |
W21 |
W22 |
W23 |
W24 |
W25 |
W26 |
W27 |
W28 |
W29 |
W30 |
W31 |
W32 |
W33 |
W34 |
W35 |
W36 |
W37 |
W38 |
W39 |
W40 |
W41 |
W42 |
W43 |
W44 |
W45 |
W46 |
W47 |
W48 |
```

## Final Selection

At the end of the weekend, select three named outputs:

```text
1. Best global baseline:
   Highest Primary, strict_udu >= 86.66, test_overall >= 90.50.

2. Best satellite baseline:
   Highest satellite mean, storm_mp >= 38.60, mixed_orbit >= 42.05.

3. Best balanced baseline:
   Primary >= 88.00, strict_udu >= 86.50, satellite_mean >= 43.00.
```

Expected best candidates before running:

```text
Most likely global winner: W01 or W02.
Most likely satellite winner: W04, W05, W20, or W33.
Most likely balanced winner: W01, W02, W06, or W13.
```

## Self-Review

Spec coverage:

```text
Only seed=1337: yes.
Explores many paths: Fishr, cls weight, consistency, MixStyle, start epoch, satellite scenario, GroupDRO, R19 capacity.
Fits weekend: 48 designed runs, 32-48 expected feasible, with pruning.
B3b remains the center: yes.
```

Placeholder scan:

```text
No TBD/TODO entries.
Every run has a concrete name, purpose, and argument delta.
```

Risk:

```text
The exact number completed depends on Python environment, GPU health, and whether eval_sat_channel dominates runtime.
The design is intentionally wider than the minimum so the queue can keep GPUs busy all weekend.
```
