# DADDA Paper Reproduction Fix Triage Summary

Scope: DADDA method only for the paper "Cross-Receiver Radio Frequency Fingerprint Identification Based on Domain Adaptation With Dynamic Distribution Alignment". This is closed-set single-source UDA on WiSig ManySig and is not CVS Stage2, LEO deployment, or open-set evidence.

Local full report: `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\report.md`

## Confirmed Fixes

| Area | Resolution |
|---|---|
| MMD/LMMD kernel | Use one shared source-target batch bandwidth for all kernel terms. |
| Normalization | Paper-faithful formal runs must keep `normalize=true`; `normalize=false` requires explicit ablation flag. |
| Sample count | Interpret ManySig paper setting as `6 x 4000 = 24000` samples per receiver domain, not 4000 total samples per receiver. |
| Model width diagnostic | Added ResNet18-width diagnostic config with `base_channels=64`, `feature_dim=512`, `multiscale_dim=512`, and `detach_target_probabilities=false`. |
| LMMD scale | Use class-wise `LMMD_sum` in the DADDA dynamic joint loss, matching the class-wise local distance used by the dynamic factor. |
| Seed policy | Reset the base seed before each Table II task/method model initialization so row results are independent of lane order. |
| Dynamic alpha gradient | Treat Eq. (5) dynamic `alpha` as a per-batch scalar weight (`detach_dynamic_alpha=true`) instead of a differentiable optimization path. |

## Final DADDA-Only Table II Comparison

Final run prefix: `dadda_alphadetach_table2_20260709_1558`.

Configuration: `conv2d_paper`, ResNet18-width diagnostic model, `paper_domain_sample_count=null`, `normalize=true`, `detach_target_probabilities=false`, `detach_dynamic_alpha=true`, `alpha_mode=dynamic`, per-row seed reset, `epochs=100`, `batch_size=128`, seed `20260709`, DADDA method only.

| Task | Final DADDA % | Previous seed-reset % | Original DADDA % | Paper DADDA % | Delta vs paper pp |
|---|---:|---:|---:|---:|---:|
| `1-1->8-8` | 93.80 | 85.67 | 72.17 | 97.15 | -3.35 |
| `8-8->1-1` | 91.13 | 80.30 | 70.42 | 95.47 | -4.35 |
| `19-2->1-1` | 89.39 | 89.35 | 77.44 | 90.65 | -1.26 |
| `1-1->19-2` | 92.03 | 86.96 | 79.46 | 98.03 | -6.01 |
| `20-1->2-1` | 79.33 | 78.85 | 78.76 | 79.53 | -0.21 |
| `2-1->20-1` | 92.35 | 88.05 | 76.83 | 80.53 | +11.82 |
| `7-14->2-19` | 91.87 | 83.28 | 65.92 | 91.33 | +0.54 |
| `2-19->7-14` | 93.82 | 90.11 | 78.54 | 97.28 | -3.46 |
| `1-19->2-19` | 99.05 | 96.30 | 72.18 | 98.99 | +0.06 |
| `2-19->1-19` | 94.94 | 95.62 | 73.41 | 98.15 | -3.21 |
| `14-7->7-7` | 74.75 | 62.42 | 49.34 | 89.39 | -14.64 |
| `7-7->14-7` | 82.01 | 78.62 | 65.23 | 93.79 | -11.78 |
| **Mean** | **89.54** | **84.63** | **71.64** | **92.52** | **-2.99** |

## Interpretation

The fixes improved the DADDA-only mean from the original local `71.64%` to `89.54%`, within `2.99 pp` of the paper mean `92.52%`. The largest remaining outlier is still `14-7->7-7`, now `74.75%` versus paper `89.39%`, improved by `+12.32 pp` from the seed-reset result. The `alpha.detach` correction directly reduced the target-class absorption failures, while target-probability detaching did not improve the stubborn rows. The remaining gap is unlikely to be caused by receiver mapping, target-label leakage, sample-count cap, crop length, normalization, LMMD scale, lane-order RNG, fixed alpha, or target-batch BN.

Artifacts:

| Artifact | Path |
|---|---|
| Narrow fixed full table | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\full_fixed_table2_20260709_0135\` |
| ResNet18-width final full table | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\resnet18_width_table2_20260709_0300\` |
| LMMD-sum seed-reset full table | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\lmmdsum_seedreset_table2_20260709_1240\` |
| Alpha-detach full table raw JSON | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\dadda_alphadetach_table2_raw_20260709_1558.json` |
| Final alpha-detach aggregated CSV | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\dadda_alphadetach_table2_summary_20260709_1558.csv` |
| Targeted per-class confusion | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\targeted_gapfix_confusion_20260709_1526.json` |
