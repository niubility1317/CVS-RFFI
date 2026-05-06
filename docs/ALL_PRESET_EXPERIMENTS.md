# All Preset Experiment Launcher

Use `run_all_preset_experiments.sh` to launch all root-level preset experiments added so far.

## Included Presets

SGC presets:

- `sgc_lite_b_no_dac`
- `sgc_lite_b_no_dac_no_amp`
- `sgc_lite_b_no_dac_no_freq`
- `sgc_lite_b_no_dac_no_spec`
- `sgc_lite_b_no_dac_no_res`
- `sgc_baseline_no_adapter`

Model slimming presets:

- `slim_r19_anchor`
- `slim_r25_compact`
- `slim_r19_groupce006`
- `slim_r19_fishr002`
- `slim_r25_fishr002`
- `slim_no_domain_enhancer`
- `slim_lite_d_lowmix`
- `slim_lite_e_no_dac_probe`
- `slim_no_dac_no_pa_probe`
- `slim_no_dac_no_stats_guard`
- `slim_full_upper_bound`

SSDG presets:

- `ssdg_r19_pseudo_cons`
- `ssdg_r19_pseudo_cons_strict`
- `ssdg_r25_pseudo_cons`
- `ssdg_r19_pseudo_cons_fishr`

## Common Commands

Dry run:

```bash
DRY_RUN=1 bash run_all_preset_experiments.sh
```

Run all preset groups:

```bash
bash run_all_preset_experiments.sh
```

Run only slimming and SSDG:

```bash
PRESET_GROUPS=slim,ssdg bash run_all_preset_experiments.sh
```

Run SGC full staged chain:

```bash
PRESET_GROUPS=sgc SGC_STAGES=source,augment,adapt bash run_all_preset_experiments.sh
```

Use selected GPUs:

```bash
GPU_IDS=0,1,2,3 bash run_all_preset_experiments.sh
```

## Notes

- `PRESET_GROUPS` accepts `sgc`, `slim`, and `ssdg`.
- SGC defaults to `SGC_STAGES=source`; set `source,augment,adapt` for the full chain.
- Slimming defaults to 0.2 labeled train ratio.
- SSDG presets force a 0.1 labeled train ratio inside `train.py`.
- Logs are written to `logs/`; checkpoints are written to `sgc_runs/`, `slimming_runs/`, and `ssdg_runs/`.
