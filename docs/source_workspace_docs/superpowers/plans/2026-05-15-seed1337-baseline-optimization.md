# Seed 1337 Baseline Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explore the strongest CV-SincNet Type10-7 baseline under a single fixed seed, `seed=1337`, with B3b satellite classification-only training as the main hypothesis.

**Architecture:** Keep the R25 compact backbone (`rxrobust_lite_d_no_dac_refined`) and SAT-07 mixed-orbit protocol as the baseline family. Treat B2 as the current global metric reference, B3b as the satellite-scene reference, then verify whether Fishr, MixStyle removal, and small lambda sweeps can combine their strengths without adding seed variance.

**Tech Stack:** `train.py`, WiSig `rx_day`, 200 epochs, Primary Score (`0.35 * test_overall + 0.65 * strict_udu`), satellite evaluation on `clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit`.

---

## Key Decision

Only run `--seed 1337`. Do not schedule any repeat seed in this round.

The strongest baseline target is not plain B2, even though B2 is currently best on Primary and strict_udu. The target is a B3b-centered optimized baseline:

```text
R25 compact + mixed_orbit satellite training + cls_only + Fishr
```

Rationale:

- B2 is the current global best: Primary `88.06`, strict_udu `86.66%`.
- B3b is the best satellite-scene model across all five satellite scenarios:
  `clear_leo=45.82`, `low_elev_leo=45.74`, `rain_leo=43.37`, `storm_mp=38.60`, `mixed_orbit=42.05`.
- The document shows `lambda_sat_cons` is harmful for strict_udu, while B3b's pure satellite classification loss improves strict_udu by `+0.47pp` over B0.
- Fishr improves global OOD behavior, but B2 underperforms B3b on `storm_mp` and `mixed_orbit`, so Fishr should be fused with `cls_only`, not kept in the original consistency setting.

## Fixed Common Args

Every experiment in this plan must use:

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
  --slim_group rxrobust_lite_d_no_dac_refined \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 20 \
  --seed 1337
```

Report checkpoints in this priority:

1. `best_primary_ood_model.pth` for model selection and headline comparison.
2. `best_strict_udu_model.pth` as supporting OOD evidence.
3. `latest_model.pth` and E180-E200 stability as final-epoch robustness evidence.
4. Never select by `best_val_model.pth` alone, because val_tx is not a reliable OOD proxy.

## Experiment Groups

### Task 1: Reference Anchors

**Files:**
- Read: `analysis_report_optimization.md`
- Read: `logs/cvs_rffi_staged/B2_stable_sat07_fishr_20260514_162420.log`
- Read: `logs/cvs_rffi_staged/B3b_stable_sat07_cls_only_20260514_162420.log`

- [ ] **Step 1: Keep B2 as the global reference**

Reference values:

```text
B2 = R25 + Fishr + mixed_orbit sat
Primary: 88.06
strict_udu: 86.66%
test_overall: 90.66%
storm_mp: 37.92
mixed_orbit: 40.64
```

- [ ] **Step 2: Keep B3b as the satellite reference**

Reference values:

```text
B3b = R25 + cls_only + mixed_orbit sat
Primary: 87.94
strict_udu: 86.47%
test_overall: 90.68%
clear_leo: 45.82
low_elev_leo: 45.74
rain_leo: 43.37
storm_mp: 38.60
mixed_orbit: 42.05
```

- [ ] **Step 3: Use these acceptance gates**

Expected: a candidate must beat B3b on satellite mean or beat B2 on strict_udu/Primary to be considered stronger.

```text
Primary >= 88.06, or strict_udu >= 86.66
Satellite mean > B3b satellite mean
storm_mp >= 38.60 preferred, minimum >= 38.00
mixed_orbit >= 42.05 preferred, minimum >= 40.80
No collapse in test_overall: keep >= 90.50
```

### Task 2: Main Fusion Group

**Files:**
- Run: `train.py`
- Output: `runs/baseline_seed1337/X1_fishr_cls_only/`

- [ ] **Step 1: Run X1, the main candidate**

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
  --run_name X1_fishr_cls_only \
  --latest_save_path runs/baseline_seed1337/X1_fishr_cls_only/latest_model.pth \
  --best_save_path runs/baseline_seed1337/X1_fishr_cls_only/best_val_model.pth \
  --best_primary_save_path runs/baseline_seed1337/X1_fishr_cls_only/best_primary_ood_model.pth \
  --best_unseen_day_unseen_rx_save_path runs/baseline_seed1337/X1_fishr_cls_only/best_strict_udu_model.pth \
  --slim_group rxrobust_lite_d_no_dac_refined \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.10 \
  --lambda_sat_cons 0.00 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --seed 1337
```

Expected:

```text
strict_udu >= 86.70
Primary >= 88.10
Satellite scenes close to or above B3b
```

- [ ] **Step 2: Interpret X1**

Expected decisions:

```text
If X1 beats B2 strict_udu and preserves B3b satellite strength, promote X1 as the new strongest baseline.
If X1 improves strict_udu but hurts storm_mp/mixed_orbit, keep B3b as the satellite baseline and use X1 only as global OOD baseline.
If X1 fails both B2 and B3b, Fishr conflicts with cls_only and should be tuned downward.
```

### Task 3: MixStyle Interaction Group

**Files:**
- Run: `train.py`
- Output: `runs/baseline_seed1337/X2_fishr_cls_only_no_mixstyle/`

- [ ] **Step 1: Run X2, the aggressive candidate**

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
  --run_name X2_fishr_cls_only_no_mixstyle \
  --latest_save_path runs/baseline_seed1337/X2_fishr_cls_only_no_mixstyle/latest_model.pth \
  --best_save_path runs/baseline_seed1337/X2_fishr_cls_only_no_mixstyle/best_val_model.pth \
  --best_primary_save_path runs/baseline_seed1337/X2_fishr_cls_only_no_mixstyle/best_primary_ood_model.pth \
  --best_unseen_day_unseen_rx_save_path runs/baseline_seed1337/X2_fishr_cls_only_no_mixstyle/best_strict_udu_model.pth \
  --slim_group rxrobust_lite_d_no_dac_refined \
  --no_use_mixstyle \
  --use_sat_consistency \
  --sat_train_scenario mixed_orbit \
  --sat_cons_start_epoch 20 \
  --lambda_sat_cls 0.10 \
  --lambda_sat_cons 0.00 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --seed 1337
```

Expected:

```text
Primary >= 88.10 if the no-MixStyle effect stacks cleanly
storm_mp >= 38.60 if MixStyle was suppressing harsh satellite robustness
```

- [ ] **Step 2: Interpret X2**

Expected decisions:

```text
If X2 beats X1 on storm_mp/mixed_orbit without hurting strict_udu, promote X2.
If X2 raises satellite scenes but lowers strict_udu/test_overall, reserve it for satellite-focused reporting only.
If X2 collapses, do not combine no_mixstyle with Fishr + cls_only; use B4 only as mechanism evidence.
```

### Task 4: Fishr Lambda Around B3b

**Files:**
- Run: `train.py`
- Output: `runs/baseline_seed1337/X3_fishr001_cls_only/`
- Output: `runs/baseline_seed1337/X4_fishr003_cls_only/`

- [ ] **Step 1: Run X3 with weaker Fishr**

Use the X1 command with these replacements:

```bash
--run_name X3_fishr001_cls_only
--latest_save_path runs/baseline_seed1337/X3_fishr001_cls_only/latest_model.pth
--best_save_path runs/baseline_seed1337/X3_fishr001_cls_only/best_val_model.pth
--best_primary_save_path runs/baseline_seed1337/X3_fishr001_cls_only/best_primary_ood_model.pth
--best_unseen_day_unseen_rx_save_path runs/baseline_seed1337/X3_fishr001_cls_only/best_strict_udu_model.pth
--lambda_fishr 0.01
```

Expected:

```text
Better satellite preservation than X1 if Fishr=0.02 is too strong.
```

- [ ] **Step 2: Run X4 with stronger Fishr**

Use the X1 command with these replacements:

```bash
--run_name X4_fishr003_cls_only
--latest_save_path runs/baseline_seed1337/X4_fishr003_cls_only/latest_model.pth
--best_save_path runs/baseline_seed1337/X4_fishr003_cls_only/best_val_model.pth
--best_primary_save_path runs/baseline_seed1337/X4_fishr003_cls_only/best_primary_ood_model.pth
--best_unseen_day_unseen_rx_save_path runs/baseline_seed1337/X4_fishr003_cls_only/best_strict_udu_model.pth
--lambda_fishr 0.03
```

Expected:

```text
Higher strict_udu only if B3b still has unused gradient-invariance headroom.
```

- [ ] **Step 3: Do not run Fishr=0.05 initially**

Reason:

```text
The document already warns that stacked regularization can fail.
Fishr=0.05 should be held back unless X4 improves strict_udu without satellite-scene regression.
```

### Task 5: Satellite Classification Weight Around B3b

**Files:**
- Run: `train.py`
- Output: `runs/baseline_seed1337/X5_fishr_cls012/`
- Output: `runs/baseline_seed1337/X6_fishr_cls008/`

- [ ] **Step 1: Run X5 with stronger satellite CE**

Use the X1 command with these replacements:

```bash
--run_name X5_fishr_cls012
--latest_save_path runs/baseline_seed1337/X5_fishr_cls012/latest_model.pth
--best_save_path runs/baseline_seed1337/X5_fishr_cls012/best_val_model.pth
--best_primary_save_path runs/baseline_seed1337/X5_fishr_cls012/best_primary_ood_model.pth
--best_unseen_day_unseen_rx_save_path runs/baseline_seed1337/X5_fishr_cls012/best_strict_udu_model.pth
--lambda_sat_cls 0.12
--lambda_sat_cons 0.00
```

Expected:

```text
Possible improvement on storm_mp and rain_leo.
Watch for test_overall dropping below 90.50.
```

- [ ] **Step 2: Run X6 with baseline satellite CE**

Use the X1 command with these replacements:

```bash
--run_name X6_fishr_cls008
--latest_save_path runs/baseline_seed1337/X6_fishr_cls008/latest_model.pth
--best_save_path runs/baseline_seed1337/X6_fishr_cls008/best_val_model.pth
--best_primary_save_path runs/baseline_seed1337/X6_fishr_cls008/best_primary_ood_model.pth
--best_unseen_day_unseen_rx_save_path runs/baseline_seed1337/X6_fishr_cls008/best_strict_udu_model.pth
--lambda_sat_cls 0.08
--lambda_sat_cons 0.00
```

Expected:

```text
Useful control to determine whether B3b's gain came from removing consistency or also from increasing lambda_sat_cls to 0.10.
```

### Task 6: Start-Epoch Sanity Group

**Files:**
- Run: `train.py`
- Output: `runs/baseline_seed1337/X7_fishr_cls_only_start30/`

- [ ] **Step 1: Run X7 with delayed satellite start**

Use the X1 command with these replacements:

```bash
--run_name X7_fishr_cls_only_start30
--latest_save_path runs/baseline_seed1337/X7_fishr_cls_only_start30/latest_model.pth
--best_save_path runs/baseline_seed1337/X7_fishr_cls_only_start30/best_val_model.pth
--best_primary_save_path runs/baseline_seed1337/X7_fishr_cls_only_start30/best_primary_ood_model.pth
--best_unseen_day_unseen_rx_save_path runs/baseline_seed1337/X7_fishr_cls_only_start30/best_strict_udu_model.pth
--sat_cons_start_epoch 30
```

Expected:

```text
Smoother early representation learning.
Run this only after X1 completes, unless spare GPU time is available.
```

## Priority Queue

Run in this order:

```text
Tier 0: X1
Tier 1: X2, X3, X4
Tier 2: X5, X6
Tier 3: X7
```

If GPU budget is limited, run only:

```text
X1: Fishr + cls_only
X2: Fishr + cls_only + no_mixstyle
X3: Fishr=0.01 + cls_only
```

## Reporting Template

For each run, report:

```text
Run:
Primary:
strict_udu:
test_overall:
best_epoch:
clear_leo:
low_elev_leo:
rain_leo:
storm_mp:
mixed_orbit:
satellite_mean:
E180-E200 strict_udu mean:
E180-E200 strict_udu std:
Decision:
```

Promotion rules:

```text
Global strongest baseline: highest Primary with strict_udu >= B2 and test_overall >= 90.50.
Satellite strongest baseline: highest satellite_mean with storm_mp >= 38.60 and mixed_orbit >= 42.05.
Final recommended baseline: prefer the model that wins global unless satellite_mean trails B3b by more than 0.30pp.
```

## Self-Review

Spec coverage:

```text
Uses only seed=1337: yes.
Centers B3b cls_only as the satellite-best hypothesis: yes.
Keeps B2 as global reference, not default final answer: yes.
Designs validation experiment groups: yes.
Avoids other seeds: yes.
```

Placeholder scan:

```text
No TBD/TODO entries.
Every experiment has concrete argument changes and output paths.
```

Risk note:

```text
The historical seed sensitivity remains scientifically important, but it is intentionally excluded from this execution plan per the single-seed constraint.
```
