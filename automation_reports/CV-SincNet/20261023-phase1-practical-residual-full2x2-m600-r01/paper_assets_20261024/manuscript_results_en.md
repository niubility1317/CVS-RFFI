# Paper-ready experimental results

## Protocol and comparison design

We evaluated four complete comparison methods (RIEI, DRIFT-v2, IB-RFF and SDG) under a preregistered 2×2 design: channel augmentation off/on and ordinary pseudo-labeling off/on. All methods retained their native objectives, classifiers and training budgets. Channel augmentation used the practical ResidualChannel in residual/post-sync/no-equalization mode, with low suburban, mid urban and low urban sampled with probabilities 0.40, 0.35 and 0.25. Application probabilities followed 0.30/0.60/0.80 over the registered training-budget intervals. Ordinary pseudo-labeling used the current model, confidence threshold 0.95 and weight 1. PAMF supervised controls and native semi-supervised T1 were evaluated separately; native T1 was not replaced by ordinary pseudo-labeling.

Training used D1–D3. Three disjoint evaluation populations tested unseen dates (source receivers on D0), unseen receivers (target receivers on D1–D3), and their combination (target receivers on D0). D0 precedes the training dates, so this experiment does not establish future-date extrapolation. Each row was frozen across all populations, raw/matching-DURA routes, clean and six practical scenes before independent scoring. We report three-F-seed means and sample SD, averaging scenes within each seed. These SDs condition on one G per contract and role and are not confidence intervals.

## Main comparison

For F0 cross-receiver generalization, native PAMF-T1 with matching DURA reached 78.21 ± 0.84% S6 accuracy, compared with 67.12 ± 0.97% for SDG Full with both augmentation and ordinary pseudo-labeling. The difference in means was 11.09 percentage points. The corresponding PAMF supervised channel-on system reached 76.48 ± 0.45%. The external D recipe was fixed in advance rather than selected by target accuracy. All A/B/C/D cells remain available in the complete tables; the highlighted D comparison does not imply that D is optimal for every method.

## Generalization and limitations

Performance varied across receiver partitions. Native PAMF-T1 achieved 66.96 ± 0.43% and 74.53 ± 0.15% on the F1 and F2 cross-receiver S6 endpoints. On the F2 joint date/receiver shift, T1 reached 74.49 ± 0.26% whereas the supervised channel-on recipe reached 74.65 ± 1.14%. Thus, the evidence does not support uniformly positive semi-supervised gains. Source-count results use only m=1,2,3,4,6 under natural budgets and fixed six-target populations; no m=5 result is interpolated, and F0 is not inserted into the curve. ManyRx uses ten classes and is reported separately from six-class ManySig.

Augmentation and pseudo-labeling effects are method- and condition-dependent. All negative effects and the 61 external PL rows with zero accepted pseudo-labels are retained. A/C restoration uses G-L, while B/D and native T1 use G-LU; a restored comparison across PL cells therefore also changes the restorer's source-IQ permissions and is a complete-system comparison. Raw within-method 2×2 contrasts provide the cleaner augmentation/PL attribution. Existing PAMF module and structure studies retain their original all-target-date population and cannot be relabeled as the new cross-receiver endpoint.

## Computation and reporting boundaries

The 600 new F models required 246.37 shared-worker elapsed hours; 27 reused F sources account for 46.26 historical hours. Four new complete G lineages required 18.42 hours. These costs are not exclusive GPU hours or wall-clock completion time. Cached classifier prediction time is not online G+F latency. Full native PAMF acceptance and per-F realized channel counts are unavailable in the compact telemetry and remain missing. Existing dedicated timing and mechanism evidence is preserved separately, without claiming new practical-residual latency measurements.
