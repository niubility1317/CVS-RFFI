# Stable receiver geometry: S3 E40 clean test with fixed reused controls

Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.
P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.

| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 80.526 | 1.320 | 79.052 | 79.950 |
| B0 | 3 | 80.913 | 1.510 | 79.170 | 80.399 |
| B2 | 3 | 80.833 | 1.308 | 79.332 | 80.469 |
| B3 | 3 | 80.943 | 1.208 | 79.551 | 80.574 |
| S3 | 3 | 80.947 | 1.252 | 79.504 | 80.584 |

## Every seed

| Arm | Model seed | Accuracy (%) |
|---|---:|---:|
| S3 | 2026092701 | 81.749 |
| S3 | 2026092702 | 79.504 |
| S3 | 2026092703 | 81.588 |
| P0 | 2026092701 | 80.928 |
| P0 | 2026092702 | 79.052 |
| P0 | 2026092703 | 81.599 |
| B0 | 2026092701 | 81.755 |
| B2 | 2026092701 | 81.732 |
| B3 | 2026092701 | 81.705 |
| B0 | 2026092702 | 79.170 |
| B2 | 2026092702 | 79.332 |
| B3 | 2026092702 | 79.551 |
| B0 | 2026092703 | 81.814 |
| B2 | 2026092703 | 81.434 |
| B3 | 2026092703 | 81.573 |

## Paired differences

| Comparison | Seeds | Mean (pp) | Positive seeds |
|---|---:|---:|---:|
| B0 − P0 | 3 | 0.387 | 3 |
| S3 − B0 | 3 | 0.033 | 1 |
| S3 − B2 | 3 | 0.114 | 3 |
| S3 − B3 | 3 | 0.004 | 2 |
| S3 − P0 | 3 | 0.420 | 2 |

Primary success requires all three S3 seeds and clean mean ≥84%; no best-arm replacement.
B0−P0 includes the changed optimizer, schedule, sampling, and additional training. S3−B0 estimates the combined mechanism increment.
S3−B2 adds stable RX R; S3−B3 changes only the frozen R estimator. B0/B2/B3 predictions are reused from the registered E40 control run.
Three seed differences are descriptive evidence, not an automatic statistical significance claim.
Parent training, shared preparation, incremental training, and test costs are referenced separately in resources.json.
