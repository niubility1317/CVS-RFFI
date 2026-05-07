# All Preset Experiment Launcher

Use `run_all_preset_experiments.sh` to launch SGC preset experiments. The script still accepts optional legacy `slim` and `ssdg` group names, but the safe default is now SGC-only because this workspace request focuses on the satellite-ground-channel processor.

## Included Presets

SGC presets:

- `sgc_lite_b_no_dac`
- `sgc_lite_b_no_dac_no_amp`
- `sgc_lite_b_no_dac_no_freq`
- `sgc_lite_b_no_dac_no_spec`
- `sgc_lite_b_no_dac_no_res`
- `sgc_lite_b_no_dac_no_amp_freq`
- `sgc_lite_b_no_dac_residual_only`
- `sgc_lite_b_no_dac_light`
- `sgc_lite_d_no_dac`
- `sgc_lite_d_no_dac_light`
- `sgc_baseline_no_adapter`

## Common Commands

Dry run:

```bash
DRY_RUN=1 bash run_all_preset_experiments.sh
```

Run all SGC presets for the default source stage:

```bash
bash run_all_preset_experiments.sh
```

Run one compact SGC subset:

```bash
SGC_PRESETS=sgc_lite_b_no_dac,sgc_lite_b_no_dac_light,sgc_lite_d_no_dac_light bash run_all_preset_experiments.sh
```

Run SGC full staged chain:

```bash
PRESET_GROUPS=sgc SGC_STAGES=source,augment,adapt bash run_all_preset_experiments.sh
```

Use selected GPUs:

```bash
GPU_IDS=0,1,2,3 bash run_all_preset_experiments.sh
```

Disable satellite-overlaid target-domain evaluation:

```bash
ENABLE_SAT_TARGET_EVAL=0 bash run_all_preset_experiments.sh
```

Evaluate strict target only, or use more satellite scenarios:

```bash
SAT_TARGET_ON=target_strict SAT_TARGET_SCENARIOS=clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit bash run_all_preset_experiments.sh
```

## Notes

- `PRESET_GROUPS` defaults to `sgc`. Legacy `slim` and `ssdg` launch branches remain in the script, but this restored implementation validates the SGC path.
- SGC defaults to `SGC_STAGES=source`; set `source,augment,adapt` for the full chain.
- Training launchers enable `[SAT-TEST]` evaluation by default: target-domain test loaders are overlaid with configured satellite-ground channel scenarios each epoch.
- Logs are written to `logs/`; SGC checkpoints are written to `sgc_runs/`.
