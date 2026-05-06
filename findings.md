# Findings

## Experiment Analysis

- R19 (`lite_b`, `no_dac`, same_tx_crossdomain MixStyle p=0.15 strength=0.65) is the best balanced route: strong overall, Primary OOD, strict unseen-day/unseen-RX, and stable training.
- R25 (`lite_d`, `no_dac`) is the best parameter-efficiency candidate: much smaller than R19 with only a small OOD gap.
- R21/R05 variants are useful when Worst-RX is the main target.
- Removing time or frequency branches is unsafe and can collapse training.
- Removing the DAC branch did not hurt performance in the best runs; it often improved parameter efficiency and stability.
- Removing stats together with DAC hurt performance in R07/R17, so stats/RCN domain cues should generally stay.
- Satellite-channel consistency helps SAT robustness but can trade off base OOD if over-weighted.

## DAC Decision

- Structural DAC branch: remove for the main slimming route.
- DAC-only auxiliary view and DAC auxiliary losses: not needed for `no_dac` models; the code already disables them through `align_training_with_branch_ablation()` and `zero_dac_path()`.
- DAC-style random impairment inside general augmentation: only useful as a low-probability robustness stressor, not as a dedicated DAC branch/view. For strict no-DAC slimming presets, keep `aug_p_dac=0`.

