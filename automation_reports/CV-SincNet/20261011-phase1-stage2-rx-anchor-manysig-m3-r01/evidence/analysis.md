# Matched receiver anchor: Z3 E40 clean test with fixed reused controls

Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.
P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.

| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 80.526 | 1.320 | 79.052 | 79.950 |
| B0 | 3 | 80.913 | 1.510 | 79.170 | 80.399 |
| B2 | 3 | 80.833 | 1.308 | 79.332 | 80.469 |
| B3 | 3 | 80.943 | 1.208 | 79.551 | 80.574 |
| S3 | 3 | 80.947 | 1.252 | 79.504 | 80.584 |
| Z3 | 3 | 80.986 | 1.267 | 79.527 | 80.626 |

## Every seed

| Arm | Model seed | Accuracy (%) |
|---|---:|---:|
| Z3 | 2026092701 | 81.814 |
| Z3 | 2026092702 | 79.527 |
| Z3 | 2026092703 | 81.616 |
| P0 | 2026092701 | 80.928 |
| P0 | 2026092702 | 79.052 |
| P0 | 2026092703 | 81.599 |
| B0 | 2026092701 | 81.755 |
| B2 | 2026092701 | 81.732 |
| B3 | 2026092701 | 81.705 |
| S3 | 2026092701 | 81.749 |
| B0 | 2026092702 | 79.170 |
| B2 | 2026092702 | 79.332 |
| B3 | 2026092702 | 79.551 |
| S3 | 2026092702 | 79.504 |
| B0 | 2026092703 | 81.814 |
| B2 | 2026092703 | 81.434 |
| B3 | 2026092703 | 81.573 |
| S3 | 2026092703 | 81.588 |

## Paired differences

| Comparison | Seeds | Mean (pp) | Positive seeds |
|---|---:|---:|---:|
| B0 − P0 | 3 | 0.387 | 3 |
| Z3 − B0 | 3 | 0.072 | 2 |
| Z3 − B2 | 3 | 0.153 | 3 |
| Z3 − B3 | 3 | 0.043 | 2 |
| Z3 − S3 | 3 | 0.039 | 3 |
| Z3 − P0 | 3 | 0.459 | 3 |

Primary success requires all three Z3 seeds and clean mean ≥84%; no best-arm replacement.
B0−P0 includes the changed optimizer, schedule, sampling, and additional training. Z3−B0 estimates the combined mechanism increment.
Z3−S3 removes only the fixed RX correction target (alpha=0), retaining c and the complete R estimator. B0/B2/B3/S3 predictions are reused from their registered E40 runs.
Three seed differences are descriptive evidence, not an automatic statistical significance claim.
Parent training, shared preparation, incremental training, and test costs are referenced separately in resources.json.
