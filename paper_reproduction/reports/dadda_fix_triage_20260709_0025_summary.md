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

## Final DADDA-Only Table II Comparison

Final run prefix: `dadda_resnet18_width_table2_20260709_0300`.

Configuration: `conv2d_paper`, ResNet18-width diagnostic model, `paper_domain_sample_count=null`, `normalize=true`, `detach_target_probabilities=false`, `epochs=100`, `batch_size=128`, seed `20260709`, DADDA method only.

| Task | Final DADDA % | Previous DADDA % | Paper DADDA % | Delta vs paper pp |
|---|---:|---:|---:|---:|
| `1-1->8-8` | 83.57 | 72.17 | 97.15 | -13.58 |
| `8-8->1-1` | 61.12 | 70.42 | 95.47 | -34.35 |
| `19-2->1-1` | 89.16 | 77.44 | 90.65 | -1.49 |
| `1-1->19-2` | 82.71 | 79.46 | 98.03 | -15.32 |
| `20-1->2-1` | 77.64 | 78.76 | 79.53 | -1.89 |
| `2-1->20-1` | 90.80 | 76.83 | 80.53 | +10.27 |
| `7-14->2-19` | 81.18 | 65.92 | 91.33 | -10.15 |
| `2-19->7-14` | 91.14 | 78.54 | 97.28 | -6.14 |
| `1-19->2-19` | 93.45 | 72.18 | 98.99 | -5.54 |
| `2-19->1-19` | 93.38 | 73.41 | 98.15 | -4.77 |
| `14-7->7-7` | 62.42 | 49.34 | 89.39 | -26.97 |
| `7-7->14-7` | 78.42 | 65.23 | 93.79 | -15.37 |
| **Mean** | **82.08** | **71.64** | **92.52** | **-10.44** |

## Interpretation

The fixes improved the DADDA-only mean from `71.64%` to `82.08%`, but the reproduction still does not match the paper mean of `92.52%`. The remaining gap is unlikely to be caused by receiver mapping, target-label leakage, sample-count cap, normalization, or the DADDA alpha direction. The most likely remaining causes are unreleased author implementation details: exact modified ResNet18 stem/stride/BN behavior, preprocessing crop/windowing, or stochastic/data-order differences.

Artifacts:

| Artifact | Path |
|---|---|
| Narrow fixed full table | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\full_fixed_table2_20260709_0135\` |
| ResNet18-width final full table | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\resnet18_width_table2_20260709_0300\` |
| Aggregated CSV | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\dadda_resnet18_width_table2_summary.csv` |
