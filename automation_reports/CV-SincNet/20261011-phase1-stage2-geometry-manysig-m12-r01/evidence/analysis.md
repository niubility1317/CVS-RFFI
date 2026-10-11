# Fixed-parent feature geometry: B0–B3 E40 clean test

Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.
P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.

| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 80.526 | 1.320 | 79.052 | 79.950 |
| B0 | 3 | 80.913 | 1.510 | 79.170 | 80.399 |
| B1 | 3 | 80.747 | 1.290 | 79.257 | 80.289 |
| B2 | 3 | 80.833 | 1.308 | 79.332 | 80.469 |
| B3 | 3 | 80.943 | 1.208 | 79.551 | 80.574 |

## Every seed

| Arm | Model seed | Accuracy (%) |
|---|---:|---:|
| B0 | 2026092701 | 81.755 |
| B1 | 2026092701 | 81.479 |
| B2 | 2026092701 | 81.732 |
| B3 | 2026092701 | 81.705 |
| B0 | 2026092702 | 79.170 |
| B1 | 2026092702 | 79.257 |
| B2 | 2026092702 | 79.332 |
| B3 | 2026092702 | 79.551 |
| B0 | 2026092703 | 81.814 |
| B1 | 2026092703 | 81.505 |
| B2 | 2026092703 | 81.434 |
| B3 | 2026092703 | 81.573 |
| P0 | 2026092701 | 80.928 |
| P0 | 2026092702 | 79.052 |
| P0 | 2026092703 | 81.599 |

## Paired differences

| Comparison | Seeds | Mean (pp) | Positive seeds |
|---|---:|---:|---:|
| B0 − P0 | 3 | 0.387 | 3 |
| B1 − B0 | 3 | -0.166 | 1 |
| B2 − B1 | 3 | 0.086 | 2 |
| B3 − B2 | 3 | 0.110 | 2 |
| B3 − B0 | 3 | 0.030 | 1 |
| B3 − P0 | 3 | 0.416 | 2 |

Primary success requires all three B3 seeds and clean mean ≥84%; no best-arm replacement.
B0−P0 includes the changed optimizer, schedule, sampling, and additional training. B3−B0 estimates the combined mechanism increment.
B1−B0 combines deterministic comparison and finite-risk L/T; B2−B1 isolates the conditional policy; B3−B2 isolates full160 R.
Three seed differences are descriptive evidence, not an automatic statistical significance claim.
Parent training, shared preparation, incremental training, and test costs are referenced separately in resources.json.
