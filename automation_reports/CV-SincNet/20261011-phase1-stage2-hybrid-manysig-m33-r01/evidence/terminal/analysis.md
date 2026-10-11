# Fixed-parent hybrid mechanisms: H0–H10 E40 clean test

Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.
P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.

| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 80.526 | 1.320 | 79.052 | 79.950 |
| H0 | 3 | 80.886 | 1.517 | 79.135 | 80.371 |
| H1 | 3 | 80.688 | 1.349 | 79.173 | 80.226 |
| H2 | 3 | 80.872 | 1.522 | 79.115 | 80.355 |
| H3 | 3 | 79.888 | 0.963 | 78.780 | 79.416 |
| H4 | 3 | 80.698 | 1.350 | 79.181 | 80.236 |
| H5 | 3 | 80.469 | 1.807 | 78.383 | 79.982 |
| H6 | 3 | 80.791 | 1.485 | 79.078 | 80.266 |
| H7 | 3 | 80.433 | 1.847 | 78.302 | 79.943 |
| H8 | 3 | 80.721 | 1.153 | 79.435 | 80.302 |
| H9 | 3 | 80.644 | 1.334 | 79.112 | 80.216 |
| H10 | 3 | 80.666 | 1.047 | 79.465 | 80.238 |

## Every seed

| Arm | Model seed | Accuracy (%) |
|---|---:|---:|
| H0 | 2026092701 | 81.733 |
| H1 | 2026092701 | 81.133 |
| H2 | 2026092701 | 81.737 |
| H3 | 2026092701 | 80.362 |
| H4 | 2026092701 | 81.145 |
| H5 | 2026092701 | 81.454 |
| H6 | 2026092701 | 81.595 |
| H7 | 2026092701 | 81.446 |
| H8 | 2026092701 | 81.070 |
| H9 | 2026092701 | 81.261 |
| H10 | 2026092701 | 81.149 |
| H0 | 2026092702 | 79.135 |
| H1 | 2026092702 | 79.173 |
| H2 | 2026092702 | 79.115 |
| H3 | 2026092702 | 78.780 |
| H4 | 2026092702 | 79.181 |
| H5 | 2026092702 | 78.383 |
| H6 | 2026092702 | 79.078 |
| H7 | 2026092702 | 78.302 |
| H8 | 2026092702 | 79.435 |
| H9 | 2026092702 | 79.112 |
| H10 | 2026092702 | 79.465 |
| H0 | 2026092703 | 81.792 |
| H1 | 2026092703 | 81.760 |
| H2 | 2026092703 | 81.764 |
| H3 | 2026092703 | 80.520 |
| H4 | 2026092703 | 81.768 |
| H5 | 2026092703 | 81.570 |
| H6 | 2026092703 | 81.701 |
| H7 | 2026092703 | 81.551 |
| H8 | 2026092703 | 81.659 |
| H9 | 2026092703 | 81.558 |
| H10 | 2026092703 | 81.385 |
| P0 | 2026092701 | 80.928 |
| P0 | 2026092702 | 79.052 |
| P0 | 2026092703 | 81.599 |

## Paired differences

| Comparison | Seeds | Mean (pp) | Positive seeds |
|---|---:|---:|---:|
| H0 − P0 | 3 | 0.360 | 3 |
| H1 − H0 | 3 | -0.198 | 1 |
| H2 − H0 | 3 | -0.014 | 1 |
| H3 − H0 | 3 | -0.999 | 0 |
| H4 − H0 | 3 | -0.188 | 1 |
| H5 − H0 | 3 | -0.418 | 0 |
| H6 − H0 | 3 | -0.095 | 0 |
| H7 − H0 | 3 | -0.453 | 0 |
| H8 − H0 | 3 | -0.165 | 1 |
| H9 − H0 | 3 | -0.243 | 0 |
| H10 − H0 | 3 | -0.220 | 1 |
| H4 − H1 | 3 | 0.009 | 3 |
| H4 − H2 | 3 | -0.175 | 2 |
| H7 − H5 | 3 | -0.036 | 0 |
| H8 − H4 | 3 | 0.023 | 1 |
| H9 − H8 | 3 | -0.078 | 1 |
| H10 − H8 | 3 | -0.055 | 2 |

Primary H9 success requires all three seeds and clean mean >=84%.
All 11 preregistered arms are reported. A complete arm meeting 84% is recognized separately; target scores never select or change training.
H0-P0 includes the optimizer, schedule, sampling, and additional training. Each treatment-H0 estimates the added mechanisms under equal optimizer-update budget.
SAM uses extra forward/backward passes; equal update budget is not equal compute. Averaging arms also change the fixed tail schedule.
Three seed differences are descriptive evidence, not an automatic statistical significance claim.
Parent, shared preparation, incremental training, and test costs are referenced separately in resources.json.
