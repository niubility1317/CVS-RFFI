# Dual evidence: fixed legacy-gating cosine comparison: clean and six full views

All 16 controls were preregistered and trained from scratch. Every view contains 168000 physical queries.
All predictions preceded truth access. The source selection controller does not consume these results.
Each arm has four paired model seeds. Fixed E200 checkpoints; no target feedback.

| Stage | Arm | View | Seeds | Accuracy (%) | SD (pp) | Macro-F1 (%) | SD (pp) |
|---|---|---|---:|---:|---:|---:|---:|
| dual_evidence | legacy_cosine | clean | 4 | 80.878 | 0.763 | 80.417 | 0.786 |
| dual_evidence | legacy_cosine | practical_high | 4 | 78.971 | 0.550 | 78.657 | 0.591 |
| dual_evidence | legacy_cosine | practical_mid | 4 | 76.937 | 0.476 | 76.592 | 0.513 |
| dual_evidence | legacy_cosine | practical_low_suburban | 4 | 74.749 | 0.478 | 74.340 | 0.506 |
| dual_evidence | legacy_cosine | practical_high_urban | 4 | 76.003 | 0.382 | 75.649 | 0.425 |
| dual_evidence | legacy_cosine | practical_mid_urban | 4 | 63.499 | 0.174 | 62.842 | 0.173 |
| dual_evidence | legacy_cosine | practical_low_urban | 4 | 53.229 | 0.328 | 52.315 | 0.269 |
| dual_evidence | raw_dual | clean | 4 | 70.836 | 1.166 | 69.887 | 1.042 |
| dual_evidence | raw_dual | practical_high | 4 | 72.394 | 1.702 | 72.210 | 1.635 |
| dual_evidence | raw_dual | practical_mid | 4 | 71.803 | 1.622 | 71.642 | 1.574 |
| dual_evidence | raw_dual | practical_low_suburban | 4 | 70.976 | 1.539 | 70.834 | 1.508 |
| dual_evidence | raw_dual | practical_high_urban | 4 | 71.499 | 1.633 | 71.372 | 1.578 |
| dual_evidence | raw_dual | practical_mid_urban | 4 | 66.069 | 1.210 | 65.997 | 1.207 |
| dual_evidence | raw_dual | practical_low_urban | 4 | 59.747 | 1.009 | 59.487 | 1.053 |
| dual_evidence | curvature_dual | clean | 4 | 60.007 | 0.932 | 58.714 | 0.969 |
| dual_evidence | curvature_dual | practical_high | 4 | 64.061 | 2.064 | 62.831 | 2.148 |
| dual_evidence | curvature_dual | practical_mid | 4 | 62.934 | 1.847 | 61.795 | 1.899 |
| dual_evidence | curvature_dual | practical_low_suburban | 4 | 61.511 | 1.703 | 60.418 | 1.736 |
| dual_evidence | curvature_dual | practical_high_urban | 4 | 62.698 | 1.881 | 61.547 | 1.951 |
| dual_evidence | curvature_dual | practical_mid_urban | 4 | 56.158 | 1.243 | 55.067 | 1.175 |
| dual_evidence | curvature_dual | practical_low_urban | 4 | 49.540 | 0.764 | 48.336 | 0.655 |
| dual_evidence | curvature_interaction | clean | 4 | 57.170 | 0.832 | 56.002 | 1.073 |
| dual_evidence | curvature_interaction | practical_high | 4 | 62.713 | 0.498 | 61.577 | 0.702 |
| dual_evidence | curvature_interaction | practical_mid | 4 | 61.600 | 0.430 | 60.558 | 0.600 |
| dual_evidence | curvature_interaction | practical_low_suburban | 4 | 60.338 | 0.329 | 59.368 | 0.458 |
| dual_evidence | curvature_interaction | practical_high_urban | 4 | 61.356 | 0.397 | 60.307 | 0.639 |
| dual_evidence | curvature_interaction | practical_mid_urban | 4 | 55.045 | 0.363 | 54.154 | 0.586 |
| dual_evidence | curvature_interaction | practical_low_urban | 4 | 48.612 | 0.394 | 47.634 | 0.549 |

## Paired accuracy differences

| Comparison | View | Mean (pp) | SD (pp) | Positive seeds |
|---|---|---:|---:|---:|
| raw_dual − legacy_cosine | clean | -10.042 | 0.726 | 0/4 |
| raw_dual − legacy_cosine | practical_high | -6.577 | 2.038 | 0/4 |
| raw_dual − legacy_cosine | practical_mid | -5.134 | 1.845 | 0/4 |
| raw_dual − legacy_cosine | practical_low_suburban | -3.772 | 1.710 | 0/4 |
| raw_dual − legacy_cosine | practical_high_urban | -4.503 | 1.814 | 0/4 |
| raw_dual − legacy_cosine | practical_mid_urban | 2.571 | 1.038 | 4/4 |
| raw_dual − legacy_cosine | practical_low_urban | 6.519 | 0.781 | 4/4 |
| raw_dual − legacy_cosine | six_view_mean | -1.816 | 1.509 | 1/4 |
| curvature_dual − legacy_cosine | clean | -20.871 | 1.293 | 0/4 |
| curvature_dual − legacy_cosine | practical_high | -14.910 | 1.965 | 0/4 |
| curvature_dual − legacy_cosine | practical_mid | -14.004 | 1.813 | 0/4 |
| curvature_dual − legacy_cosine | practical_low_suburban | -13.238 | 1.747 | 0/4 |
| curvature_dual − legacy_cosine | practical_high_urban | -13.304 | 1.869 | 0/4 |
| curvature_dual − legacy_cosine | practical_mid_urban | -7.340 | 1.350 | 0/4 |
| curvature_dual − legacy_cosine | practical_low_urban | -3.688 | 0.853 | 0/4 |
| curvature_dual − legacy_cosine | six_view_mean | -11.081 | 1.580 | 0/4 |
| curvature_interaction − legacy_cosine | clean | -23.708 | 0.852 | 0/4 |
| curvature_interaction − legacy_cosine | practical_high | -16.258 | 0.939 | 0/4 |
| curvature_interaction − legacy_cosine | practical_mid | -15.337 | 0.799 | 0/4 |
| curvature_interaction − legacy_cosine | practical_low_suburban | -14.411 | 0.710 | 0/4 |
| curvature_interaction − legacy_cosine | practical_high_urban | -14.646 | 0.744 | 0/4 |
| curvature_interaction − legacy_cosine | practical_mid_urban | -8.454 | 0.338 | 0/4 |
| curvature_interaction − legacy_cosine | practical_low_urban | -4.617 | 0.279 | 0/4 |
| curvature_interaction − legacy_cosine | six_view_mean | -12.287 | 0.547 | 0/4 |
| curvature_dual − raw_dual | clean | -10.829 | 1.816 | 0/4 |
| curvature_dual − raw_dual | practical_high | -8.333 | 3.426 | 0/4 |
| curvature_dual − raw_dual | practical_mid | -8.869 | 3.129 | 0/4 |
| curvature_dual − raw_dual | practical_low_suburban | -9.465 | 2.919 | 0/4 |
| curvature_dual − raw_dual | practical_high_urban | -8.801 | 3.156 | 0/4 |
| curvature_dual − raw_dual | practical_mid_urban | -9.911 | 2.133 | 0/4 |
| curvature_dual − raw_dual | practical_low_urban | -10.207 | 1.426 | 0/4 |
| curvature_dual − raw_dual | six_view_mean | -9.265 | 2.695 | 0/4 |
| curvature_interaction − curvature_dual | clean | -2.837 | 1.552 | 0/4 |
| curvature_interaction − curvature_dual | practical_high | -1.348 | 2.091 | 1/4 |
| curvature_interaction − curvature_dual | practical_mid | -1.333 | 1.797 | 1/4 |
| curvature_interaction − curvature_dual | practical_low_suburban | -1.173 | 1.645 | 1/4 |
| curvature_interaction − curvature_dual | practical_high_urban | -1.342 | 1.837 | 1/4 |
| curvature_interaction − curvature_dual | practical_mid_urban | -1.113 | 1.165 | 0/4 |
| curvature_interaction − curvature_dual | practical_low_urban | -0.928 | 0.627 | 0/4 |
| curvature_interaction − curvature_dual | six_view_mean | -1.206 | 1.522 | 0/4 |

Per-row/view/RX/TX confusion matrices and metrics: scores.json/csv. Resources and timings: resources.json.
Phase2, K, adaptation and new-class metrics: N/A. This is an exposed benchmark; no claim of a new blind test.
