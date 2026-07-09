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

## Final DADDA-Only Table II Comparison

Final run prefix: `dadda_lmmdsum_seedreset_table2_20260709_1240`.

Configuration: `conv2d_paper`, ResNet18-width diagnostic model, `paper_domain_sample_count=null`, `normalize=true`, `detach_target_probabilities=false`, `alpha_mode=dynamic`, per-row seed reset, `epochs=100`, `batch_size=128`, seed `20260709`, DADDA method only.

| Task | Final DADDA % | Previous DADDA % | Paper DADDA % | Delta vs paper pp |
|---|---:|---:|---:|---:|
| `1-1->8-8` | 85.67 | 72.17 | 97.15 | -11.48 |
| `8-8->1-1` | 80.30 | 70.42 | 95.47 | -15.17 |
| `19-2->1-1` | 89.35 | 77.44 | 90.65 | -1.30 |
| `1-1->19-2` | 86.96 | 79.46 | 98.03 | -11.07 |
| `20-1->2-1` | 78.85 | 78.76 | 79.53 | -0.68 |
| `2-1->20-1` | 88.05 | 76.83 | 80.53 | +7.52 |
| `7-14->2-19` | 83.28 | 65.92 | 91.33 | -8.05 |
| `2-19->7-14` | 90.11 | 78.54 | 97.28 | -7.17 |
| `1-19->2-19` | 96.30 | 72.18 | 98.99 | -2.69 |
| `2-19->1-19` | 95.62 | 73.41 | 98.15 | -2.53 |
| `14-7->7-7` | 62.42 | 49.34 | 89.39 | -26.97 |
| `7-7->14-7` | 78.62 | 65.23 | 93.79 | -15.17 |
| **Mean** | **84.63** | **71.64** | **92.52** | **-7.90** |

## Interpretation

The fixes improved the DADDA-only mean from `71.64%` to `84.63%`, but the reproduction still does not match the paper mean of `92.52%`. The remaining largest outlier is `14-7->7-7` at `62.42%` versus paper `89.39%`. Fixed-alpha diagnostics collapsed this pair to `16.67%`, so the paper's dynamic alpha direction should be retained. The remaining gap is unlikely to be caused by receiver mapping, target-label leakage, sample-count cap, crop length, normalization, LMMD scale, or lane-order RNG. The most likely remaining causes are unreleased author implementation details around exact modified ResNet18/BN behavior or target pseudo-label dynamics for the `14-7/7-7` receiver pair.

Artifacts:

| Artifact | Path |
|---|---|
| Narrow fixed full table | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\full_fixed_table2_20260709_0135\` |
| ResNet18-width final full table | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\resnet18_width_table2_20260709_0300\` |
| LMMD-sum seed-reset final full table | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\lmmdsum_seedreset_table2_20260709_1240\` |
| Final aggregated CSV | `E:\type10-7\automation_reports\CV-SincNet\dadda_fix_triage_20260709_0025\dadda_lmmdsum_seedreset_table2_summary.csv` |
