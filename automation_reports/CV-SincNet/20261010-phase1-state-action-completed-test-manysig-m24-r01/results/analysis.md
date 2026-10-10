# State conditioned actions: fixed legacy-gating cosine comparison: clean and nine channel views

All 24 controls were preregistered and trained from scratch. Every view contains 168000 physical queries.
All predictions preceded truth access. The source selection controller does not consume these results.
Each arm has four paired model seeds. Fixed E200 checkpoints; no target feedback.

| Stage | Arm | View | Seeds | Accuracy (%) | SD (pp) | Macro-F1 (%) | SD (pp) |
|---|---|---|---:|---:|---:|---:|---:|
| multi_state_action | native | clean | 4 | 80.908 | 0.772 | 80.437 | 0.781 |
| multi_state_action | native | practical_high | 4 | 79.040 | 0.606 | 78.720 | 0.630 |
| multi_state_action | native | practical_mid | 4 | 76.995 | 0.534 | 76.638 | 0.559 |
| multi_state_action | native | practical_low_suburban | 4 | 74.825 | 0.566 | 74.394 | 0.581 |
| multi_state_action | native | practical_high_urban | 4 | 76.036 | 0.528 | 75.667 | 0.555 |
| multi_state_action | native | practical_mid_urban | 4 | 63.596 | 0.115 | 62.910 | 0.137 |
| multi_state_action | native | practical_low_urban | 4 | 53.319 | 0.232 | 52.375 | 0.182 |
| multi_state_action | native | leo_clear_weak | 4 | 65.227 | 1.908 | 65.232 | 1.683 |
| multi_state_action | native | leo_low_elev_weak | 4 | 61.900 | 1.763 | 61.960 | 1.504 |
| multi_state_action | native | leo_rain_weak | 4 | 62.085 | 1.526 | 62.004 | 1.283 |
| multi_state_action | real_views | clean | 4 | 80.848 | 0.777 | 80.414 | 0.803 |
| multi_state_action | real_views | practical_high | 4 | 78.818 | 0.530 | 78.541 | 0.568 |
| multi_state_action | real_views | practical_mid | 4 | 76.789 | 0.470 | 76.486 | 0.513 |
| multi_state_action | real_views | practical_low_suburban | 4 | 74.616 | 0.443 | 74.247 | 0.482 |
| multi_state_action | real_views | practical_high_urban | 4 | 75.854 | 0.408 | 75.540 | 0.442 |
| multi_state_action | real_views | practical_mid_urban | 4 | 63.444 | 0.262 | 62.845 | 0.232 |
| multi_state_action | real_views | practical_low_urban | 4 | 53.212 | 0.382 | 52.343 | 0.289 |
| multi_state_action | real_views | leo_clear_weak | 4 | 64.864 | 1.739 | 64.968 | 1.614 |
| multi_state_action | real_views | leo_low_elev_weak | 4 | 61.534 | 1.655 | 61.701 | 1.537 |
| multi_state_action | real_views | leo_rain_weak | 4 | 61.644 | 1.414 | 61.685 | 1.314 |
| multi_state_action | L | clean | 4 | 81.185 | 0.761 | 80.731 | 0.790 |
| multi_state_action | L | practical_high | 4 | 78.904 | 0.484 | 78.609 | 0.513 |
| multi_state_action | L | practical_mid | 4 | 76.874 | 0.462 | 76.557 | 0.495 |
| multi_state_action | L | practical_low_suburban | 4 | 74.727 | 0.419 | 74.353 | 0.445 |
| multi_state_action | L | practical_high_urban | 4 | 75.929 | 0.386 | 75.599 | 0.417 |
| multi_state_action | L | practical_mid_urban | 4 | 63.556 | 0.253 | 62.945 | 0.208 |
| multi_state_action | L | practical_low_urban | 4 | 53.279 | 0.361 | 52.400 | 0.276 |
| multi_state_action | L | leo_clear_weak | 4 | 65.075 | 2.034 | 65.134 | 1.836 |
| multi_state_action | L | leo_low_elev_weak | 4 | 61.731 | 1.923 | 61.865 | 1.739 |
| multi_state_action | L | leo_rain_weak | 4 | 61.935 | 1.704 | 61.926 | 1.517 |
| multi_state_action | LT | clean | 4 | 80.992 | 0.814 | 80.551 | 0.823 |
| multi_state_action | LT | practical_high | 4 | 78.894 | 0.645 | 78.617 | 0.699 |
| multi_state_action | LT | practical_mid | 4 | 76.852 | 0.527 | 76.554 | 0.575 |
| multi_state_action | LT | practical_low_suburban | 4 | 74.704 | 0.492 | 74.347 | 0.527 |
| multi_state_action | LT | practical_high_urban | 4 | 75.926 | 0.553 | 75.615 | 0.611 |
| multi_state_action | LT | practical_mid_urban | 4 | 63.461 | 0.154 | 62.865 | 0.194 |
| multi_state_action | LT | practical_low_urban | 4 | 53.207 | 0.230 | 52.344 | 0.171 |
| multi_state_action | LT | leo_clear_weak | 4 | 64.927 | 1.966 | 64.990 | 1.735 |
| multi_state_action | LT | leo_low_elev_weak | 4 | 61.607 | 1.850 | 61.757 | 1.616 |
| multi_state_action | LT | leo_rain_weak | 4 | 61.734 | 1.597 | 61.759 | 1.366 |
| multi_state_action | L_EG | clean | 4 | 80.938 | 1.014 | 80.514 | 1.047 |
| multi_state_action | L_EG | practical_high | 4 | 78.780 | 0.489 | 78.496 | 0.525 |
| multi_state_action | L_EG | practical_mid | 4 | 76.773 | 0.364 | 76.471 | 0.395 |
| multi_state_action | L_EG | practical_low_suburban | 4 | 74.614 | 0.323 | 74.253 | 0.346 |
| multi_state_action | L_EG | practical_high_urban | 4 | 75.810 | 0.352 | 75.494 | 0.388 |
| multi_state_action | L_EG | practical_mid_urban | 4 | 63.468 | 0.234 | 62.884 | 0.170 |
| multi_state_action | L_EG | practical_low_urban | 4 | 53.213 | 0.432 | 52.371 | 0.340 |
| multi_state_action | L_EG | leo_clear_weak | 4 | 64.997 | 2.186 | 65.054 | 1.937 |
| multi_state_action | L_EG | leo_low_elev_weak | 4 | 61.710 | 2.096 | 61.844 | 1.844 |
| multi_state_action | L_EG | leo_rain_weak | 4 | 61.875 | 1.837 | 61.881 | 1.593 |
| multi_state_action | LT_EG | clean | 4 | 81.069 | 1.017 | 80.643 | 1.072 |
| multi_state_action | LT_EG | practical_high | 4 | 79.024 | 0.417 | 78.777 | 0.470 |
| multi_state_action | LT_EG | practical_mid | 4 | 76.999 | 0.405 | 76.723 | 0.459 |
| multi_state_action | LT_EG | practical_low_suburban | 4 | 74.833 | 0.341 | 74.497 | 0.390 |
| multi_state_action | LT_EG | practical_high_urban | 4 | 76.051 | 0.347 | 75.769 | 0.396 |
| multi_state_action | LT_EG | practical_mid_urban | 4 | 63.566 | 0.288 | 62.996 | 0.263 |
| multi_state_action | LT_EG | practical_low_urban | 4 | 53.290 | 0.435 | 52.434 | 0.364 |
| multi_state_action | LT_EG | leo_clear_weak | 4 | 64.840 | 2.198 | 64.897 | 2.013 |
| multi_state_action | LT_EG | leo_low_elev_weak | 4 | 61.471 | 2.056 | 61.591 | 1.873 |
| multi_state_action | LT_EG | leo_rain_weak | 4 | 61.638 | 1.847 | 61.632 | 1.690 |

## Paired accuracy differences

| Comparison | View | Mean (pp) | SD (pp) | Positive seeds |
|---|---|---:|---:|---:|
| real_views − native | clean | -0.060 | 0.085 | 0/4 |
| real_views − native | practical_high | -0.222 | 0.167 | 0/4 |
| real_views − native | practical_mid | -0.206 | 0.149 | 0/4 |
| real_views − native | practical_low_suburban | -0.208 | 0.162 | 0/4 |
| real_views − native | practical_high_urban | -0.183 | 0.216 | 1/4 |
| real_views − native | practical_mid_urban | -0.152 | 0.236 | 2/4 |
| real_views − native | practical_low_urban | -0.108 | 0.181 | 2/4 |
| real_views − native | leo_clear_weak | -0.363 | 0.203 | 0/4 |
| real_views − native | leo_low_elev_weak | -0.367 | 0.333 | 1/4 |
| real_views − native | leo_rain_weak | -0.441 | 0.297 | 0/4 |
| real_views − native | six_view_mean | -0.180 | 0.182 | 0/4 |
| L − native | clean | 0.276 | 0.182 | 4/4 |
| L − native | practical_high | -0.136 | 0.334 | 2/4 |
| L − native | practical_mid | -0.121 | 0.291 | 1/4 |
| L − native | practical_low_suburban | -0.098 | 0.335 | 2/4 |
| L − native | practical_high_urban | -0.107 | 0.314 | 2/4 |
| L − native | practical_mid_urban | -0.040 | 0.271 | 2/4 |
| L − native | practical_low_urban | -0.041 | 0.195 | 2/4 |
| L − native | leo_clear_weak | -0.151 | 0.160 | 1/4 |
| L − native | leo_low_elev_weak | -0.169 | 0.258 | 1/4 |
| L − native | leo_rain_weak | -0.150 | 0.262 | 1/4 |
| L − native | six_view_mean | -0.090 | 0.285 | 2/4 |
| LT − native | clean | 0.084 | 0.150 | 3/4 |
| LT − native | practical_high | -0.146 | 0.204 | 1/4 |
| LT − native | practical_mid | -0.143 | 0.153 | 1/4 |
| LT − native | practical_low_suburban | -0.121 | 0.254 | 2/4 |
| LT − native | practical_high_urban | -0.110 | 0.173 | 1/4 |
| LT − native | practical_mid_urban | -0.135 | 0.145 | 1/4 |
| LT − native | practical_low_urban | -0.112 | 0.070 | 0/4 |
| LT − native | leo_clear_weak | -0.299 | 0.181 | 0/4 |
| LT − native | leo_low_elev_weak | -0.293 | 0.298 | 1/4 |
| LT − native | leo_rain_weak | -0.351 | 0.277 | 1/4 |
| LT − native | six_view_mean | -0.128 | 0.160 | 1/4 |
| L_EG − native | clean | 0.029 | 0.321 | 2/4 |
| L_EG − native | practical_high | -0.260 | 0.175 | 0/4 |
| L_EG − native | practical_mid | -0.222 | 0.180 | 0/4 |
| L_EG − native | practical_low_suburban | -0.211 | 0.245 | 1/4 |
| L_EG − native | practical_high_urban | -0.226 | 0.188 | 0/4 |
| L_EG − native | practical_mid_urban | -0.128 | 0.228 | 1/4 |
| L_EG − native | practical_low_urban | -0.106 | 0.226 | 2/4 |
| L_EG − native | leo_clear_weak | -0.229 | 0.282 | 1/4 |
| L_EG − native | leo_low_elev_weak | -0.191 | 0.459 | 1/4 |
| L_EG − native | leo_rain_weak | -0.210 | 0.424 | 1/4 |
| L_EG − native | six_view_mean | -0.192 | 0.194 | 0/4 |
| LT_EG − native | clean | 0.161 | 0.245 | 2/4 |
| LT_EG − native | practical_high | -0.016 | 0.370 | 3/4 |
| LT_EG − native | practical_mid | 0.004 | 0.274 | 3/4 |
| LT_EG − native | practical_low_suburban | 0.008 | 0.302 | 3/4 |
| LT_EG − native | practical_high_urban | 0.015 | 0.339 | 3/4 |
| LT_EG − native | practical_mid_urban | -0.030 | 0.267 | 2/4 |
| LT_EG − native | practical_low_urban | -0.030 | 0.218 | 2/4 |
| LT_EG − native | leo_clear_weak | -0.387 | 0.308 | 0/4 |
| LT_EG − native | leo_low_elev_weak | -0.429 | 0.370 | 0/4 |
| LT_EG − native | leo_rain_weak | -0.447 | 0.395 | 0/4 |
| LT_EG − native | six_view_mean | -0.008 | 0.286 | 3/4 |
| LT − L | clean | -0.193 | 0.105 | 0/4 |
| LT − L | practical_high | -0.010 | 0.217 | 1/4 |
| LT − L | practical_mid | -0.022 | 0.157 | 1/4 |
| LT − L | practical_low_suburban | -0.023 | 0.162 | 1/4 |
| LT − L | practical_high_urban | -0.003 | 0.220 | 1/4 |
| LT − L | practical_mid_urban | -0.095 | 0.173 | 1/4 |
| LT − L | practical_low_urban | -0.072 | 0.144 | 1/4 |
| LT − L | leo_clear_weak | -0.148 | 0.282 | 2/4 |
| LT − L | leo_low_elev_weak | -0.124 | 0.310 | 2/4 |
| LT − L | leo_rain_weak | -0.201 | 0.311 | 1/4 |
| LT − L | six_view_mean | -0.038 | 0.172 | 1/4 |
| L_EG − L | clean | -0.247 | 0.278 | 1/4 |
| L_EG − L | practical_high | -0.124 | 0.212 | 1/4 |
| L_EG − L | practical_mid | -0.101 | 0.229 | 2/4 |
| L_EG − L | practical_low_suburban | -0.113 | 0.229 | 2/4 |
| L_EG − L | practical_high_urban | -0.119 | 0.182 | 1/4 |
| L_EG − L | practical_mid_urban | -0.088 | 0.129 | 0/4 |
| L_EG − L | practical_low_urban | -0.066 | 0.159 | 2/4 |
| L_EG − L | leo_clear_weak | -0.078 | 0.208 | 1/4 |
| L_EG − L | leo_low_elev_weak | -0.021 | 0.283 | 1/4 |
| L_EG − L | leo_rain_weak | -0.060 | 0.212 | 1/4 |
| L_EG − L | six_view_mean | -0.102 | 0.178 | 1/4 |
| LT_EG − LT | clean | 0.077 | 0.261 | 3/4 |
| LT_EG − LT | practical_high | 0.130 | 0.331 | 3/4 |
| LT_EG − LT | practical_mid | 0.147 | 0.205 | 3/4 |
| LT_EG − LT | practical_low_suburban | 0.129 | 0.234 | 3/4 |
| LT_EG − LT | practical_high_urban | 0.125 | 0.322 | 2/4 |
| LT_EG − LT | practical_mid_urban | 0.105 | 0.215 | 3/4 |
| LT_EG − LT | practical_low_urban | 0.083 | 0.217 | 2/4 |
| LT_EG − LT | leo_clear_weak | -0.087 | 0.365 | 1/4 |
| LT_EG − LT | leo_low_elev_weak | -0.136 | 0.383 | 1/4 |
| LT_EG − LT | leo_rain_weak | -0.096 | 0.460 | 1/4 |
| LT_EG − LT | six_view_mean | 0.120 | 0.253 | 3/4 |

Per-row/view/RX/TX confusion matrices and metrics: scores.json/csv. Resources and timings: resources.json.
Phase2, K, adaptation and new-class metrics: N/A. This is an exposed benchmark; no claim of a new blind test.
