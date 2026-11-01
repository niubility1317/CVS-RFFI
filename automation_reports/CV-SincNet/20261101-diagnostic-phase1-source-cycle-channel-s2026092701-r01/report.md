# Phase1 source periodic pseudo-pilot channel probe

Status: repeated source-only experiments; no candidate advanced to Phase1 test. A 20-sample complex template learned from 2,048 unlabeled clean U_s source records was compared with 1,024 validation and 1,024 holdout source records. Each record was granted its best circular shift and complex phase against the learned template, so the alignment is an oracle diagnostic. Similarity median was 0.9800 on receiver 4/6 day 3 validation and 0.8873 on receiver 8 day 3 holdout (10th percentile 0.7867). These values do not prove a known pilot or stable template across receivers.

The ensuing restoration pilot therefore uses each record's own DURA periodic component as a pseudo-pilot. Received IQ is aligned to DURA without clean reference, relative phase drift is estimated from lag 20, and 20 harmonic ratios produce a regularized interpolated frequency response. A prior-regularized inverse of received IQ is blended with DURA. A difference gate preserves near-clean input. This is an approximate inference procedure, not a known physical channel estimate. DURA and Tri-SIG-TIME SSL are frozen; clean source IQ serves only simulation and later paired waveform quality.

The source pilot contains 536 validation IDs from receivers 4/6 day 3 and 280 holdout IDs from receiver 8 day 3. IDs overlap earlier source diagnostics and frozen experts saw source records. A predeclared grid of four inversion regularizers and three blend weights used mixed L_s true targets and confident U_s clean-teacher pseudo-targets for validation. Changes are counts relative to DURA; RX totals are across the three impaired views.

| Regularizer | Blend | Low suburban | Mid urban | Low urban | Total | RX4 total | RX6 total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0.25 | -12 | -9 | -14 | -35 | -16 | -19 |
| 0.3 | 0.50 | -48 | -41 | -37 | -126 | -49 | -77 |
| 0.3 | 1.00 | -119 | -105 | -117 | -341 | -122 | -219 |
| 1.0 | 0.25 | +0 | -2 | -10 | -12 | -9 | -3 |
| 1.0 | 0.50 | -14 | -10 | -14 | -38 | -18 | -20 |
| 1.0 | 1.00 | -47 | -40 | -38 | -125 | -48 | -77 |
| 3.0 | 0.25 | -2 | +1 | -2 | -3 | -2 | -1 |
| 3.0 | 0.50 | +0 | +0 | -9 | -9 | -10 | +1 |
| 3.0 | 1.00 | -7 | -5 | -15 | -27 | -14 | -13 |
| 10.0 | 0.25 | +0 | +1 | +1 | +2 | +3 | -1 |
| 10.0 | 0.50 | -1 | +0 | +2 | +1 | +2 | -1 |
| 10.0 | 1.00 | +0 | +1 | -6 | -5 | -4 | -1 |

The original rule required nonnegative receiver-group total and selected DURA: the otherwise best combination (regularizer 10, blend 0.25) has total +2 but RX6 total -1. A supplemental tolerance of 0.5% per receiver-group total was chosen after seeing these validation results; it selected that candidate. This selection is post-hoc and its source holdout is not independent model confirmation. Each holdout view has 272 valid targets, including 72 real L_s labels and 200 teacher pseudo-targets. IQ MSE uses all 280 records and independently optimizes shift ±16 and complex gain against paired source clean IQ.

| View | DURA correct | Candidate correct | L_s DURA | L_s candidate | DURA aligned MSE | Candidate aligned MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 272 | 272 | 72 | 72 | 0.00004085 | 0.00004085 |
| practical_low_suburban | 266 | 266 | 70 | 70 | 0.02743456 | 0.02746898 |
| practical_mid_urban | 261 | 260 | 70 | 69 | 0.04777574 | 0.04782382 |
| practical_low_urban | 247 | 246 | 66 | 66 | 0.09643116 | 0.09641706 |

Low urban loses one mixed-target correct decision; mid urban loses one true-label decision. Low-urban mean aligned IQ MSE decreases only slightly, while other impaired views worsen. This evidence is insufficient to advance the candidate. The template fit was independently rerun with identical JSON; a second tolerance run reproduced the complete candidate-validation and holdout result. Synthetic identity-channel and four-record source checks passed. Candidate predictions were fixed before source transmitter labels; no U_s transmitter labels, validation IQ or test IQ were read. The simulator is not validated real satellite data, and the Phase1 development test had previously been observed.
