# SSDG SSL Experiment Plan

## Goal

Add semi-supervised SSDG experiments without changing the existing dual-network architecture. The supervised labeled split is 0.1. Those samples keep transmitter, receiver, and date labels. The remaining train-days/train-receivers samples, excluding validation, become unlabeled for transmitter ID while still carrying receiver/date domain labels.

## Pseudo-Label Guard

Pseudo labels are admitted only when all checks pass:

- Weak-view transmitter confidence is above `ssl_pseudo_threshold`.
- Strong perturbed view predicts the same class and is above `ssl_consistency_threshold`.
- Per-sample EMA pseudo class matches the instant class and is above `ssl_ema_threshold`.
- The same high-confidence class appears for at least `ssl_min_streak` rounds.

Training logs print `seen`, `selected`, `correct`, and `correct_ratio` under `[SSDG-SSL]`. The true transmitter label is used only for this audit statistic, not as a training target.

## Preset Groups

| Preset | Base | Purpose |
| --- | --- | --- |
| `ssdg_r19_pseudo_cons` | R19 Lite-B no-DAC | Main SSDG route: conservative pseudo labels plus weak/strong consistency. |
| `ssdg_r19_pseudo_cons_strict` | R19 Lite-B no-DAC | Higher thresholds and longer streak, for noisy unlabeled pools. |
| `ssdg_r25_pseudo_cons` | R25 Lite-D no-DAC | Compact SSDG route for parameter-efficient deployment. |
| `ssdg_r19_pseudo_cons_fishr` | R19 Lite-B no-DAC | Tests whether SSL and Fishr domain gradient alignment are complementary. |

## Launch

Run all presets:

```bash
bash run_ssdg_experiments.sh
```

Run one preset:

```bash
SSDG_PRESETS=ssdg_r19_pseudo_cons_strict bash run_ssdg_experiments.sh
```

Manual command:

```bash
python -u train.py \
  --dataset wisig \
  --wisig_domain rx_day \
  --preset ssdg_r19_pseudo_cons \
  --epochs 200 \
  --batch_size 256 \
  --primary_udu_weight 0.65 \
  --latest_save_path ssdg_runs/ssdg_r19_pseudo_cons/latest_model.pth \
  --best_save_path ssdg_runs/ssdg_r19_pseudo_cons/best_model.pth
```

## Notes

- SSDG is opt-in through `--use_ssdg_ssl` or an `ssdg_*` preset.
- `--wisig_train_ratio` is forced to `--ssdg_train_ratio` when SSDG is enabled; default is `0.1`.
- The model structure, branch ablation, SGC adapter, and existing supervised losses are unchanged unless the chosen preset already changes them.
