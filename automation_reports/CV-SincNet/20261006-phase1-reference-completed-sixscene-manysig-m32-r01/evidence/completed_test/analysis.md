# Completed reference_response models: clean and six full views

All 32 models were fixed by completion time, not target metrics. Every view contains 168000 physical queries.
All predictions preceded truth access. The source selection controller does not consume these results.
Unequal seed counts describe this completion snapshot, not a complete stage comparison. Single-seed SD is N/A.

| Stage | Arm | View | Seeds | Accuracy (%) | SD (pp) | Macro-F1 (%) | SD (pp) |
|---|---|---|---:|---:|---:|---:|---:|
| r2 | bridge | clean | 4 | 79.095 | 0.651 | 78.569 | 0.464 |
| r2 | bridge | practical_high | 4 | 78.148 | 0.954 | 77.949 | 1.132 |
| r2 | bridge | practical_mid | 4 | 76.557 | 0.860 | 76.308 | 1.007 |
| r2 | bridge | practical_low_suburban | 4 | 74.728 | 0.833 | 74.420 | 0.921 |
| r2 | bridge | practical_high_urban | 4 | 75.796 | 0.878 | 75.506 | 1.033 |
| r2 | bridge | practical_mid_urban | 4 | 65.499 | 0.722 | 64.791 | 0.709 |
| r2 | bridge | practical_low_urban | 4 | 56.121 | 0.650 | 55.051 | 0.515 |
| r2 | ema | clean | 4 | 78.710 | 1.608 | 78.204 | 1.397 |
| r2 | ema | practical_high | 4 | 77.684 | 1.191 | 77.493 | 1.191 |
| r2 | ema | practical_mid | 4 | 76.135 | 1.182 | 75.896 | 1.161 |
| r2 | ema | practical_low_suburban | 4 | 74.286 | 1.250 | 73.987 | 1.189 |
| r2 | ema | practical_high_urban | 4 | 75.357 | 1.190 | 75.093 | 1.161 |
| r2 | ema | practical_mid_urban | 4 | 65.239 | 1.097 | 64.612 | 0.882 |
| r2 | ema | practical_low_urban | 4 | 55.989 | 0.846 | 55.024 | 0.448 |
| r2 | pseudo | clean | 4 | 78.256 | 0.845 | 77.514 | 0.491 |
| r2 | pseudo | practical_high | 4 | 76.349 | 1.858 | 75.951 | 1.374 |
| r2 | pseudo | practical_mid | 4 | 74.704 | 1.842 | 74.281 | 1.332 |
| r2 | pseudo | practical_low_suburban | 4 | 72.850 | 1.676 | 72.404 | 1.170 |
| r2 | pseudo | practical_high_urban | 4 | 74.022 | 1.855 | 73.598 | 1.336 |
| r2 | pseudo | practical_mid_urban | 4 | 63.989 | 1.521 | 63.343 | 0.958 |
| r2 | pseudo | practical_low_urban | 4 | 55.024 | 1.274 | 54.147 | 0.693 |
| r2 | twostage | clean | 4 | 79.240 | 1.363 | 78.776 | 1.188 |
| r2 | twostage | practical_high | 4 | 77.940 | 1.211 | 77.816 | 1.262 |
| r2 | twostage | practical_mid | 4 | 76.294 | 1.187 | 76.125 | 1.202 |
| r2 | twostage | practical_low_suburban | 4 | 74.431 | 1.242 | 74.211 | 1.205 |
| r2 | twostage | practical_high_urban | 4 | 75.530 | 1.270 | 75.321 | 1.277 |
| r2 | twostage | practical_mid_urban | 4 | 65.142 | 1.171 | 64.555 | 0.947 |
| r2 | twostage | practical_low_urban | 4 | 55.825 | 0.976 | 54.891 | 0.573 |
| r3 | all_dg | clean | 2 | 77.006 | 1.477 | 75.989 | 1.356 |
| r3 | all_dg | practical_high | 2 | 76.637 | 1.393 | 75.975 | 1.468 |
| r3 | all_dg | practical_mid | 2 | 75.049 | 1.377 | 74.358 | 1.502 |
| r3 | all_dg | practical_low_suburban | 2 | 73.213 | 1.467 | 72.482 | 1.645 |
| r3 | all_dg | practical_high_urban | 2 | 74.329 | 1.349 | 73.621 | 1.505 |
| r3 | all_dg | practical_mid_urban | 2 | 64.022 | 0.934 | 63.174 | 1.345 |
| r3 | all_dg | practical_low_urban | 2 | 54.807 | 0.473 | 53.864 | 1.012 |
| r3 | base | clean | 3 | 77.775 | 1.547 | 76.843 | 1.698 |
| r3 | base | practical_high | 3 | 75.152 | 1.809 | 74.654 | 1.065 |
| r3 | base | practical_mid | 3 | 73.582 | 1.708 | 73.073 | 0.983 |
| r3 | base | practical_low_suburban | 3 | 71.750 | 1.548 | 71.229 | 0.890 |
| r3 | base | practical_high_urban | 3 | 72.822 | 1.844 | 72.327 | 1.157 |
| r3 | base | practical_mid_urban | 3 | 62.931 | 1.512 | 62.298 | 1.061 |
| r3 | base | practical_low_urban | 3 | 54.175 | 1.281 | 53.288 | 0.927 |
| r3 | cons | clean | 2 | 77.460 | 0.489 | 76.707 | 0.605 |
| r3 | cons | practical_high | 2 | 76.401 | 1.599 | 75.863 | 1.786 |
| r3 | cons | practical_mid | 2 | 74.876 | 1.448 | 74.298 | 1.693 |
| r3 | cons | practical_low_suburban | 2 | 73.250 | 1.435 | 72.650 | 1.703 |
| r3 | cons | practical_high_urban | 2 | 74.066 | 1.326 | 73.481 | 1.595 |
| r3 | cons | practical_mid_urban | 2 | 64.008 | 0.856 | 63.270 | 1.436 |
| r3 | cons | practical_low_urban | 2 | 54.953 | 0.435 | 54.114 | 1.208 |
| r3 | domain | clean | 3 | 75.883 | 3.324 | 75.117 | 3.253 |
| r3 | domain | practical_high | 3 | 76.036 | 2.040 | 75.619 | 1.937 |
| r3 | domain | practical_mid | 3 | 74.459 | 1.951 | 74.007 | 1.865 |
| r3 | domain | practical_low_suburban | 3 | 72.733 | 1.894 | 72.242 | 1.827 |
| r3 | domain | practical_high_urban | 3 | 73.664 | 1.772 | 73.202 | 1.731 |
| r3 | domain | practical_mid_urban | 3 | 63.519 | 1.065 | 62.835 | 1.302 |
| r3 | domain | practical_low_urban | 3 | 54.610 | 0.584 | 53.692 | 1.067 |
| r3 | fishr | clean | 2 | 78.500 | 0.348 | 77.882 | 0.209 |
| r3 | fishr | practical_high | 2 | 77.990 | 0.743 | 77.631 | 0.792 |
| r3 | fishr | practical_mid | 2 | 76.307 | 0.867 | 75.872 | 0.960 |
| r3 | fishr | practical_low_suburban | 2 | 74.423 | 0.930 | 73.910 | 1.069 |
| r3 | fishr | practical_high_urban | 2 | 75.512 | 0.629 | 75.068 | 0.727 |
| r3 | fishr | practical_mid_urban | 2 | 64.986 | 0.564 | 64.118 | 0.880 |
| r3 | fishr | practical_low_urban | 2 | 55.700 | 0.503 | 54.500 | 0.988 |
| r3 | group_ce | clean | 2 | 77.095 | 0.816 | 76.333 | 0.931 |
| r3 | group_ce | practical_high | 2 | 75.603 | 1.491 | 75.082 | 1.791 |
| r3 | group_ce | practical_mid | 2 | 74.087 | 1.484 | 73.554 | 1.862 |
| r3 | group_ce | practical_low_suburban | 2 | 72.340 | 1.655 | 71.796 | 2.080 |
| r3 | group_ce | practical_high_urban | 2 | 73.255 | 1.285 | 72.727 | 1.691 |
| r3 | group_ce | practical_mid_urban | 2 | 63.024 | 0.553 | 62.478 | 1.236 |
| r3 | group_ce | practical_low_urban | 2 | 53.937 | 0.322 | 53.324 | 1.119 |
| r3 | orth | clean | 2 | 77.571 | 0.914 | 76.828 | 1.049 |
| r3 | orth | practical_high | 2 | 76.847 | 1.584 | 76.334 | 1.769 |
| r3 | orth | practical_mid | 2 | 75.290 | 1.561 | 74.714 | 1.781 |
| r3 | orth | practical_low_suburban | 2 | 73.477 | 1.690 | 72.829 | 1.913 |
| r3 | orth | practical_high_urban | 2 | 74.434 | 1.401 | 73.874 | 1.658 |
| r3 | orth | practical_mid_urban | 2 | 64.140 | 0.771 | 63.289 | 1.271 |
| r3 | orth | practical_low_urban | 2 | 55.026 | 0.609 | 53.915 | 1.167 |

Per-row/view/RX/TX confusion matrices and metrics: scores.json/csv. Resources and timings: resources.json.
Phase2, K, adaptation and new-class metrics: N/A. This is an exposed benchmark; no claim of a new blind test.
