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
| `sgc_lite_b_no_dac_no_amp_freq` | Combined amplitude + frequency compensation ablation. |
| `sgc_lite_b_no_dac_residual_only` | Residual compensation only; disables explicit channel-front-end blocks. |
| `sgc_lite_b_no_dac_light` | Smaller SGC-Adapter on Lite-B for satellite-side parameter budget checks. |
| `sgc_lite_d_no_dac` | Full SGC-Adapter on compact Lite-D no-DAC backbone. |
| `sgc_lite_d_no_dac_light` | Smallest SGC candidate: Lite-D plus light SGC-Adapter. |
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

This stage adds an explicit satellite-ground channel training view. By default
the view is built from the main augmented IQ view and then passed through
`apply_sat_gnd_channel_batch(...)`.

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

By default the launcher evaluates target-domain loaders with overlaid
satellite-ground channels and prints `[SAT-TEST]` lines each epoch. Disable it with:

```bash
ENABLE_SAT_TARGET_EVAL=0 bash run_sgc_experiments.sh source
```

Run a full source -> augment -> adapt chain for every SGC preset:

```bash
PRESET_GROUPS=sgc SGC_STAGES=source,augment,adapt bash run_all_preset_experiments.sh
```

Use a focused quick matrix:

```bash
SGC_PRESETS=sgc_lite_b_no_dac,sgc_lite_b_no_dac_light,sgc_baseline_no_adapter \
EPOCHS_SGC_SOURCE=20 EPOCHS_SGC_AUGMENT=10 ADAPT_EPOCHS=5 \
bash run_all_preset_experiments.sh
```

## Direct Train.py Examples

```bash
python train.py --preset sgc_lite_b_no_dac --stage source --epochs 200 --wisig_train_ratio 0.2
```

```bash
python train.py --preset sgc_lite_b_no_dac --stage sgc_augment --source_ckpt sgc_runs/sgc_lite_b_no_dac/source/best_model.pth --train_sat_channel --train_sat_scenario mixed_orbit --sat_view_source main --lambda_feat 1.0 --lambda_res 0.01 --epochs 100 --wisig_train_ratio 0.2
```

```bash
python train.py --preset sgc_lite_b_no_dac --stage sgc_adapt --source_ckpt sgc_runs/sgc_lite_b_no_dac/augment/best_model.pth --pseudo_label_threshold 0.85 --lambda_proto 1.0 --lambda_cons 0.5 --lambda_ent 0.01 --lambda_res 0.01 --adapt_lr 1e-4 --adapt_epochs 50 --wisig_train_ratio 0.2
```
