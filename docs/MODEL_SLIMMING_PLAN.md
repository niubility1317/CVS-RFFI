# CVS-RFFI Model Slimming Plan

## Conclusion First

The main slimming route should stay on `no_dac`.

For a `no_dac` model, the dedicated DAC branch, DAC-only training view, and DAC auxiliary losses are not necessary. They add compute and training complexity while the experiment record shows the best overall routes already remove the DAC branch.

Keep this distinction:

- Remove: structural DAC branch, DAC-only auxiliary view, `loss_cls_dac`, `loss_dac_reg`.
- Optional stress test only: weak DAC-like random impairment as part of generic augmentation.
- Keep: time branch, frequency branch, stats/RCN cues, PA branch for now.

The current code already enforces the important part: when `branch_ablation` contains `no_dac`, `align_training_with_branch_ablation()` calls `zero_dac_path()`, disabling `enable_dac_aux`, `aug_p_dac`, and DAC loss weights.

## Evidence From Previous Experiments

| Finding | Decision |
| --- | --- |
| R19 `lite_b + no_dac` is the best balanced route. | Use as anchor. |
| R25 `lite_d + no_dac` has the best parameter efficiency. | Use as compact deployment candidate. |
| Removing time/frequency causes collapse. | Never remove in main slimming. |
| Removing DAC does not hurt and often helps. | Remove DAC by default. |
| Removing stats with DAC hurts. | Keep stats/RCN domain cues. |
| GroupCE improves Worst-RX but may trade OOD. | Use targeted preset. |
| Fishr improves Primary OOD in SAT runs. | Use targeted preset. |
| Too many regularizers together can hurt. | Test one factor at a time before combining. |

## Slimming Stages

### Stage A: Anchors And Controls

Purpose: re-establish the best known route and the upper bound.

| Preset | Role |
| --- | --- |
| `slim_r19_anchor` | Main balanced target. |
| `slim_full_upper_bound` | Full-branch Lite-C upper bound, not for deployment. |
| `slim_no_dac_no_stats_guard` | Guardrail showing why stats should stay. |

### Stage B: Deployment Candidates

Purpose: reduce parameters without giving up R19-level behavior.

| Preset | Role |
| --- | --- |
| `slim_r25_compact` | Best compact candidate from prior analysis. |
| `slim_lite_d_lowmix` | Lower MixStyle compact boundary. |
| `slim_lite_e_no_dac_probe` | Aggressive tiny model probe; likely needs KD if accuracy drops. |

### Stage C: Robustness Trade-Offs

Purpose: choose based on the deployment metric.

| Preset | Role |
| --- | --- |
| `slim_r19_groupce006` | Worst-RX oriented route. |
| `slim_r19_fishr002` | Primary OOD oriented route. |
| `slim_r25_fishr002` | Tests whether Fishr still helps in compact Lite-D. |

### Stage D: Risk Boundary

Purpose: learn how far slimming can go.

| Preset | Role |
| --- | --- |
| `slim_no_dac_no_pa_probe` | Removes both defect branches; high-risk boundary. |
| `slim_no_domain_enhancer` | Tests whether RCN domain enhancer is worth training overhead. |

## Recommended Run Order

1. `slim_r19_anchor`
2. `slim_r25_compact`
3. `slim_r19_groupce006`
4. `slim_r19_fishr002`
5. `slim_r25_fishr002`
6. `slim_no_domain_enhancer`
7. `slim_lite_d_lowmix`
8. `slim_lite_e_no_dac_probe`
9. `slim_no_dac_no_pa_probe`
10. `slim_no_dac_no_stats_guard`
11. `slim_full_upper_bound`

## Success Gate

A candidate is acceptable only if it meets all of these:

- Primary OOD is within 0.5 percentage points of R19, or parameter count drops enough to justify the loss.
- Strict unseen-day/unseen-RX does not fall by more than 0.7 percentage points.
- Worst-RX remains above 84 percent unless the preset is explicitly OOD-oriented.
- Training has no collapse and no repeated unsafe backward spikes.
- SAT robustness does not regress when SGC satellite-channel view is used.

## DAC View Decision

For the current mainline `no_dac` route:

```text
DAC branch: no
DAC-only auxiliary view: no
DAC auxiliary losses: no
DAC-like weak generic augmentation: optional probe only
```

The reason is simple: if the backbone no longer has a DAC feature path, a DAC-only auxiliary view has no dedicated branch to supervise. It mostly adds extra forward passes and can reintroduce shortcut pressure. PA-only view still makes sense because PA remains a live branch in the best routes.

## Practical Commands

Run the whole slimming matrix:

```bash
bash run_model_slimming_experiments.sh
```

Run one preset:

```bash
SLIM_PRESETS=slim_r25_compact bash run_model_slimming_experiments.sh
```

Use SGC satellite-channel view on top of a slimming preset:

```bash
python train.py --preset slim_r19_anchor --stage sgc_augment --train_sat_channel --train_sat_scenario mixed_orbit --sat_view_source main --lambda_feat 1.0 --lambda_res 0.01 --epochs 100 --wisig_train_ratio 0.2
```

