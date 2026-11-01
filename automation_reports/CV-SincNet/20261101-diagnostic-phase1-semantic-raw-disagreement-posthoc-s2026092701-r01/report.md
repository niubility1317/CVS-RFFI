# Post-hoc Phase1 raw/semantic disagreement diagnostic

This is an analysis of previously observed Phase1 development-test labels. It does not define or validate a deployable gate; future model choices informed by it are not blind. The frozen ground classifier is Tri-SIG-TIME SSL. Fixed raw and semantic-consensus decisions for all seven views were read and checked before joining truth; the same independently completed ID list was verified against DURA.

For each view, the table considers records where the raw classifier predicts class 5 and the semantic-restoration classifier predicts a different class. It reports the hypothetical paired change from switching all of those records to raw. The gate uses only predictions but was chosen *after* observing test-group results, so these are descriptive counts, not expected generalization performance. One view contains 168,000 records.

| View | Disagreements | Repairs | Damages | Net correct | Net percentage points |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 65 | 12 | 45 | -33 | -0.0196 |
| practical_high | 2115 | 137 | 1905 | -1768 | -1.0524 |
| practical_mid | 3076 | 633 | 2224 | -1591 | -0.9470 |
| practical_low_suburban | 4097 | 1120 | 2527 | -1407 | -0.8375 |
| practical_high_urban | 3131 | 805 | 2089 | -1284 | -0.7643 |
| practical_mid_urban | 8664 | 3699 | 3592 | +107 | +0.0637 |
| practical_low_urban | 13868 | 5768 | 5118 | +650 | +0.3869 |

In the low urban view this hypothetical rule nets 650 correct decisions (+0.3869 percentage points). In the high view it loses 1,768 (-1.0524 percentage points), so switching by predicted class alone is unreliable and cannot achieve the overall goal. Full counts for all six raw-predicted classes are in the JSON. Repeated read-only execution reproduced the same results.
