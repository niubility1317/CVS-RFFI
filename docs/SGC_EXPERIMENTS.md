# SGC-Adapter Experiments

This root-level CVS-RFFI workspace supports SGC-Adapter through `train.py`.

## Presets

| Preset | Purpose |
| --- | --- |
| `sgc_lite_b_no_dac` | Full SGC-Adapter on the Lite-B no-DAC baseline. |
| `sgc_lite_b_no_dac_no_amp` | Disable RMS amplitude normalization. |
| `sgc_lite_b_no_dac_no_freq` | Disable CFO/Doppler compensation. |
| `sgc_lite_b_no_dac_no_spec` | Disable spectral interference suppression. |
| `sgc_lite_b_no_dac_no_res` | Disable residual channel compensation. |
| `sgc_baseline_no_adapter` | Lite-B no-DAC baseline without SGC-Adapter. |

## Three Stages

Source training:

```bash
bash run_sgc_experiments.sh source
```

Satellite-channel augmentation:

```bash
bash run_sgc_experiments.sh augment
```

Adapter-only adaptation:

```bash
bash run_sgc_experiments.sh adapt
```

Run one ablation preset:

```bash
SGC_PRESET=sgc_lite_b_no_dac_no_freq bash run_sgc_experiments.sh source
```

Run all SGC presets for one stage:

```bash
RUN_ALL_ABLATIONS=1 bash run_sgc_experiments.sh source
```

## Direct Train.py Examples

```bash
python train.py --preset sgc_lite_b_no_dac --stage source --epochs 200 --wisig_train_ratio 0.2
```

```bash
python train.py --preset sgc_lite_b_no_dac --stage sgc_augment --source_ckpt sgc_runs/sgc_lite_b_no_dac/source/best_model.pth --train_sat_channel --train_sat_scenario mixed_orbit --lambda_feat 1.0 --lambda_res 0.01 --epochs 100 --wisig_train_ratio 0.2
```

```bash
python train.py --preset sgc_lite_b_no_dac --stage sgc_adapt --source_ckpt sgc_runs/sgc_lite_b_no_dac/augment/best_model.pth --pseudo_label_threshold 0.85 --lambda_proto 1.0 --lambda_cons 0.5 --lambda_ent 0.01 --lambda_res 0.01 --adapt_lr 1e-4 --adapt_epochs 50 --wisig_train_ratio 0.2
```
