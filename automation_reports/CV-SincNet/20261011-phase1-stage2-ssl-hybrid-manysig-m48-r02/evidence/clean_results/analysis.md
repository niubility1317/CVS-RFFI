# Fixed-parent SSL hybrid mechanisms: S0–S15 E40 clean test

Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.
P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.

| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 80.526 | 1.320 | 79.052 | 79.950 |
| H0 | 3 | 80.886 | 1.517 | 79.135 | 80.371 |
| S0 | 3 | 80.793 | 1.801 | 78.723 | 80.309 |
| S1 | 3 | 80.782 | 1.583 | 79.009 | 80.289 |
| S2 | 3 | 80.830 | 1.272 | 79.385 | 80.413 |
| S3 | 3 | 80.741 | 1.558 | 78.990 | 80.246 |
| S4 | 3 | 79.921 | 1.090 | 78.773 | 79.457 |
| S5 | 3 | 80.819 | 1.265 | 79.384 | 80.402 |
| S6 | 3 | 80.610 | 1.900 | 78.446 | 80.138 |
| S7 | 3 | 80.492 | 1.881 | 78.367 | 80.005 |
| S8 | 3 | 80.772 | 1.022 | 79.636 | 80.389 |
| S9 | 3 | 80.720 | 1.169 | 79.389 | 80.313 |
| S10 | 3 | 80.710 | 0.980 | 79.581 | 80.318 |
| S11 | 3 | 81.019 | 1.266 | 79.569 | 80.630 |
| S12 | 3 | 80.670 | 2.019 | 78.346 | 80.207 |
| S13 | 3 | 81.027 | 1.076 | 79.823 | 80.652 |
| S14 | 3 | 80.771 | 0.556 | 80.175 | 80.401 |
| S15 | 3 | 81.179 | 0.459 | 80.880 | 80.847 |

## Every seed

| Arm | Model seed | Accuracy (%) |
|---|---:|---:|
| S0 | 2026092701 | 81.651 |
| S1 | 2026092701 | 81.287 |
| S2 | 2026092701 | 81.323 |
| S3 | 2026092701 | 81.257 |
| S4 | 2026092701 | 80.049 |
| S5 | 2026092701 | 81.302 |
| S6 | 2026092701 | 81.379 |
| S7 | 2026092701 | 81.164 |
| S8 | 2026092701 | 81.061 |
| S9 | 2026092701 | 81.189 |
| S10 | 2026092701 | 81.204 |
| S11 | 2026092701 | 81.579 |
| S12 | 2026092701 | 81.663 |
| S13 | 2026092701 | 81.360 |
| S14 | 2026092701 | 80.862 |
| S15 | 2026092701 | 80.950 |
| S0 | 2026092702 | 78.723 |
| S1 | 2026092702 | 79.009 |
| S2 | 2026092702 | 79.385 |
| S3 | 2026092702 | 78.990 |
| S4 | 2026092702 | 78.773 |
| S5 | 2026092702 | 79.384 |
| S6 | 2026092702 | 78.446 |
| S7 | 2026092702 | 78.367 |
| S8 | 2026092702 | 79.636 |
| S9 | 2026092702 | 79.389 |
| S10 | 2026092702 | 79.581 |
| S11 | 2026092702 | 79.569 |
| S12 | 2026092702 | 78.346 |
| S13 | 2026092702 | 79.823 |
| S14 | 2026092702 | 80.175 |
| S15 | 2026092702 | 80.880 |
| S0 | 2026092703 | 82.004 |
| S1 | 2026092703 | 82.051 |
| S2 | 2026092703 | 81.780 |
| S3 | 2026092703 | 81.975 |
| S4 | 2026092703 | 80.941 |
| S5 | 2026092703 | 81.770 |
| S6 | 2026092703 | 82.006 |
| S7 | 2026092703 | 81.944 |
| S8 | 2026092703 | 81.618 |
| S9 | 2026092703 | 81.581 |
| S10 | 2026092703 | 81.345 |
| S11 | 2026092703 | 81.908 |
| S12 | 2026092703 | 82.001 |
| S13 | 2026092703 | 81.896 |
| S14 | 2026092703 | 81.276 |
| S15 | 2026092703 | 81.707 |
| P0 | 2026092701 | 80.928 |
| P0 | 2026092702 | 79.052 |
| P0 | 2026092703 | 81.599 |
| H0 | 2026092701 | 81.733 |
| H0 | 2026092702 | 79.135 |
| H0 | 2026092703 | 81.792 |

## Paired differences

| Comparison | Seeds | Mean (pp) | Positive seeds |
|---|---:|---:|---:|
| S1 − P0 | 3 | 0.256 | 2 |
| S0 − H0 | 3 | -0.094 | 1 |
| S1 − S0 | 3 | -0.010 | 2 |
| S2 − S1 | 3 | 0.047 | 2 |
| S3 − S1 | 3 | -0.041 | 0 |
| S4 − S1 | 3 | -0.861 | 0 |
| S5 − S1 | 3 | 0.037 | 2 |
| S6 − S1 | 3 | -0.172 | 1 |
| S7 − S1 | 3 | -0.291 | 0 |
| S8 − S1 | 3 | -0.011 | 1 |
| S9 − S1 | 3 | -0.063 | 1 |
| S10 − S1 | 3 | -0.073 | 1 |
| S11 − S1 | 3 | 0.237 | 2 |
| S12 − S1 | 3 | -0.112 | 1 |
| S13 − S1 | 3 | 0.244 | 2 |
| S14 − S1 | 3 | -0.011 | 1 |
| S15 − S1 | 3 | 0.397 | 1 |
| S5 − S2 | 3 | -0.011 | 0 |
| S5 − S3 | 3 | 0.078 | 2 |
| S7 − S6 | 3 | -0.119 | 0 |
| S8 − S5 | 3 | -0.047 | 1 |
| S9 − S8 | 3 | -0.052 | 1 |
| S10 − S8 | 3 | -0.062 | 1 |
| S11 − S5 | 3 | 0.200 | 3 |
| S12 − S1 | 3 | -0.112 | 1 |
| S13 − S5 | 3 | 0.208 | 3 |
| S14 − S9 | 3 | 0.052 | 1 |
| S15 − S13 | 3 | 0.153 | 1 |
| S13 − S12 | 3 | 0.357 | 1 |

Primary S13 success requires all three seeds and clean mean >=84%.
All 16 preregistered arms are reported. A complete arm meeting 84% is recognized separately; target scores never select or change training.
S1-P0 includes the optimizer, schedule, sampling, and additional training. Each treatment-S1 estimates the added mechanisms under equal optimizer-update budget.
SAM uses extra forward/backward passes; equal update budget is not equal compute. Averaging arms also change the fixed tail schedule.
Three seed differences are descriptive evidence, not an automatic statistical significance claim.
Parent, shared preparation, incremental training, and test costs are referenced separately in resources.json.
