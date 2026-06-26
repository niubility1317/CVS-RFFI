# B3b-Centered Asymmetric Satellite Baseline Matrix

Date: 2026-05-16

## Goal

Continue the best-baseline search from B3b, with the primary target still being cross-domain generalization:

```text
Primary Score = 0.35 * test_overall + 0.65 * strict_udu
strict_udu = test_unseen_day_unseen_rx
```

The matrix treats B3b as the anchor because it is the strongest satellite-view baseline across all five satellite scenes, while B2 remains the global strict_udu reference because Fishr produced the best overall score. The central hypothesis is:

```text
B3b cls_only + carefully tuned Fishr + more satellite training views + controlled dual-backbone asymmetry
can improve both strict_udu and satellite-scene robustness without over-regularizing identity features.
```

## Baseline References

```text
B2:
  R25 compact + mixed_orbit satellite training + Fishr=0.02
  Primary 88.06, strict_udu 86.66, test_overall 90.66

B3b:
  R25 compact + mixed_orbit satellite training + cls_only
  Primary 87.94, strict_udu 86.47, test_overall 90.68
  Satellite: clear 45.82, low 45.74, rain 43.37, storm 38.60, mixed 42.05
```

## Implementation Notes

The launcher uses the manual expansion of `rxrobust_lite_d_no_dac_refined` instead of `--slim_group rxrobust_lite_d_no_dac_refined`. This is deliberate: the current preset function overwrites later manual branch/MixStyle overrides. Manual expansion keeps no-MixStyle and asymmetric branch probes honest.

The training interface now supports:

```bash
--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
```

When this is set, training batches cycle through the listed satellite scenarios. This gives true multi-satellite-view training without duplicating the full clean batch.

## Experiment Groups

### CORE: B3b plus Fishr/MixStyle surface

Purpose: answer the highest-value question first: whether B3b's satellite robustness and B2's Fishr strict_udu gain combine.

```text
N01 B3b manual reproduction
N02 Fishr=0.01
N03 Fishr=0.015
N04 Fishr=0.02
N05 Fishr=0.02, no MixStyle
N06 Fishr=0.01, no MixStyle
N07 Fishr=0.015, satellite cls=0.12
N08 Fishr=0.015, satellite cls=0.08
```

Recommendation: run this group first. Promote N03/N04 if strict_udu beats B2 or if it preserves B3b satellite mean. Promote N05/N06 only if no-MixStyle improves storm/mixed without hurting test_overall.

### MULTISAT: more satellite training views

Purpose: test whether B3b's all-scene satellite strength improves when the training view is not only `mixed_orbit`.

```text
N09 rain+storm cls-only
N10 rain+storm + Fishr=0.015
N11 low+rain+storm + Fishr=0.015
N12 all five satellite views + Fishr=0.015
N13 mixed+rain + Fishr=0.02
N14 mixed+storm + Fishr=0.02
N15 low+rain+storm + Fishr=0.015 + no MixStyle
N16 all five views + Fishr=0.01 + no MixStyle
N17 all five views + cls=0.12 + Fishr=0.01
N18 all five views + tiny consistency=0.005 + Fishr=0.01
```

Recommendation: N12 is the main multi-view candidate. N11 is the conservative hard-LEO candidate. N18 checks whether consistency loss becomes useful only when the satellite view itself is diversified.

### SCENARIO: single-view satellite controls

Purpose: preserve interpretability. If a multi-view run wins, these controls tell which view carried the gain.

```text
N19 rain_leo + Fishr=0.015
N20 storm_mp + Fishr=0.015
N21 low_elev_leo + Fishr=0.015
N22 clear_leo + Fishr=0.015
N23 rain_leo cls-only
N24 storm_mp cls-only
```

Recommendation: rain and storm are the two most important views. Rain is the strict_udu proxy; storm is the bottleneck scene.

### ASYM: dual-backbone asymmetric feature decoupling

Purpose: the current B3b route is already asymmetric: ID backbone uses `no_dac`, domain backbone uses `no_stats` plus RCN statistics. This group tests whether the asymmetry is really helping or simply inherited from the preset.

```text
N25 default asymmetry, RCN strength=0.20, all satellite views
N26 default asymmetry, RCN strength=0.50, all satellite views
N27 default asymmetry, RCN enhancer off, all satellite views
N28 symmetric no-DAC domain backbone, all satellite views
N29 swapped ID no-stats / domain no-DAC, all satellite views
N30 ID no-DAC,no-stats / domain no-stats, all satellite views
N31 ID no-DAC / full domain backbone, all satellite views
N32 ID no-stats / full domain backbone, all satellite views
```

Recommendation: N25/N26 are low-risk RCN strength probes. N28 is the key asymmetry ablation. N31 checks whether the domain path needs more capacity under satellite multi-view pressure.

### SEED: minimum reliability check

Purpose: the report shows seed variance around 3.3pp, so a final claim needs at least one strong candidate verified beyond seed 1337.

```text
N33 seed=2027 for Fishr=0.015
N34 seed=42 for Fishr=0.015
N35 seed=2027 for all-five multi-satellite candidate
N36 seed=42 for all-five multi-satellite candidate
```

Do not over-interpret seed=1337-only gains below 0.3pp. Treat them as candidate generation, not proof.

## Launch Commands

The script defaults to `CORE`, so a plain launch only starts the first eight core experiments.

Dry-run the first wave:

```bash
bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh --plan CORE --dry-run
```

Run the recommended first wave on GPUs 0-7:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 PYTHON_BIN=python3 \
bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh
```

Run core plus multi-satellite candidates:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 PYTHON_BIN=python3 \
bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh --plan CORE,MULTISAT
```

Run the full queue:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 PYTHON_BIN=python3 \
bash code/scripts/run_b3b_asym_sat_baseline_8gpu.sh --plan FULL
```

The scheduler writes:

```text
logs/b3b_asym_sat_baseline/scheduler_<timestamp>.log
logs/b3b_asym_sat_baseline/queue_<plan>_<timestamp>.tsv
runs/b3b_asym_sat_baseline/<experiment_id>/
```

## Promotion Rules

```text
Global winner:
  Primary > 88.06 and strict_udu >= 86.66

Satellite winner:
  satellite mean > B3b satellite mean and storm_mp >= 38.60

Balanced winner:
  Primary >= 88.00, strict_udu >= 86.50, test_overall >= 90.50,
  storm_mp >= 38.60, mixed_orbit >= 42.05

Reject or demote:
  test_overall < 90.20
  strict_udu < 86.00
  storm improves but mixed_orbit drops more than 1pp
  no-MixStyle improves satellite scenes but collapses global strict_udu
```

## Recommended Run Order

1. Run `CORE`.
2. If N03/N04 are competitive, run `MULTISAT`.
3. If N12/N15/N16 show satellite improvement, run `ASYM`.
4. Run `SCENARIO` for attribution only after a multi-view candidate looks real.
5. Run `SEED` only for the best one or two candidates, or run the included four validation jobs if GPU time is available.
