# Report physical SSL: C0–C4/C6–C8 E40 clean test

Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.
P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.

| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3 | 80.526 | 1.320 | 79.052 | 79.950 |
| C0 | 3 | 80.273 | 1.149 | 78.953 | 79.784 |
| C1 | 3 | 80.002 | 1.566 | 78.194 | 79.528 |
| C2 | 3 | 80.061 | 1.983 | 77.772 | 79.585 |
| C3 | 3 | 80.100 | 1.456 | 78.423 | 79.620 |
| C4 | 3 | 80.004 | 1.298 | 78.526 | 79.569 |
| C6 | 3 | 80.033 | 1.586 | 78.214 | 79.551 |
| C7 | 3 | 80.153 | 1.748 | 78.134 | 79.701 |
| C8 | 3 | 80.288 | 1.222 | 79.096 | 79.868 |

## Every seed

| Arm | Model seed | Accuracy (%) |
|---|---:|---:|
| C0 | 2026092701 | 81.046 |
| C1 | 2026092701 | 80.933 |
| C2 | 2026092701 | 81.247 |
| C3 | 2026092701 | 80.846 |
| C4 | 2026092701 | 80.527 |
| C6 | 2026092701 | 80.760 |
| C7 | 2026092701 | 81.172 |
| C8 | 2026092701 | 80.230 |
| C0 | 2026092702 | 78.953 |
| C1 | 2026092702 | 78.194 |
| C2 | 2026092702 | 77.772 |
| C3 | 2026092702 | 78.423 |
| C4 | 2026092702 | 78.526 |
| C6 | 2026092702 | 78.214 |
| C7 | 2026092702 | 78.134 |
| C8 | 2026092702 | 79.096 |
| C0 | 2026092703 | 80.821 |
| C1 | 2026092703 | 80.880 |
| C2 | 2026092703 | 81.165 |
| C3 | 2026092703 | 81.031 |
| C4 | 2026092703 | 80.958 |
| C6 | 2026092703 | 81.126 |
| C7 | 2026092703 | 81.152 |
| C8 | 2026092703 | 81.537 |
| P0 | 2026092701 | 80.928 |
| P0 | 2026092702 | 79.052 |
| P0 | 2026092703 | 81.599 |

## Paired differences

| Comparison | Seeds | Mean (pp) | Positive seeds |
|---|---:|---:|---:|
| C8 − P0 | 3 | -0.239 | 1 |
| C1 − C0 | 3 | -0.271 | 1 |
| C2 − C0 | 3 | -0.212 | 2 |
| C3 − C0 | 3 | -0.173 | 1 |
| C4 − C0 | 3 | -0.270 | 1 |
| C6 − C0 | 3 | -0.240 | 1 |
| C7 − C0 | 3 | -0.121 | 2 |
| C8 − C0 | 3 | 0.014 | 2 |
| C2 − C1 | 3 | 0.059 | 2 |
| C3 − C1 | 3 | 0.098 | 2 |
| C4 − C1 | 3 | 0.001 | 2 |
| C6 − C1 | 3 | 0.031 | 2 |
| C7 − C1 | 3 | 0.150 | 2 |
| C8 − C1 | 3 | 0.285 | 2 |
| C8 − C2 | 3 | 0.226 | 2 |
| C8 − C4 | 3 | 0.284 | 2 |
| C8 − C7 | 3 | 0.135 | 2 |

Primary C8 success requires all three seeds and clean mean >=84%.
All 8 preregistered arms are reported. A complete arm meeting 84% is recognized separately; target scores never select or change training.
C8-P0 includes the optimizer, schedule, sampling, and additional training. C8-C1 estimates the fixed physical/frequency combination; other preregistered differences estimate the added mechanisms under equal optimizer-update budget.
Extra physical forwards and conditional snapshot fits change compute; equal identity update budget is not equal compute.
Three seed differences are descriptive evidence, not an automatic statistical significance claim.
Parent, shared preparation, incremental training, and test costs are referenced separately in resources.json.
