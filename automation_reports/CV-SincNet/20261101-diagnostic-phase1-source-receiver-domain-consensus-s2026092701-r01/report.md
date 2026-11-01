# Phase1 source receiver-disjoint consensus controller

Status: source-only diagnostic with matched control and independent heldout replays; neither candidate advanced to target test. Both controllers use frozen Tri-SIG-TIME SSL and the same four frozen signal experts. Their 698–256–6 class estimate may select only an expert whose frozen classifier predicts the estimated class; otherwise DURA is retained.

The source controller split uses receiver IDs 1, 3 and 4 for fitting (4,932 records), receiver 6 for validation (1,621) and receiver 8 for holdout (1,639). This receiver separation applies only to the new controller: the frozen ground classifier and experts saw source records. IDs may overlap prior source diagnostics. Source-simulator channels are not validated real satellite data. The complete L_s pool has 210 true examples per class for each source receiver; the sampled subset is not assumed perfectly balanced.

The experimental controller adds a receiver-discrimination head with gradient reversal (weight 0.2); a matched control has weight 0. Both use the same sampled IDs, channel seeds, architecture, initialization and optimization schedule. This is a focused source adaptation of the general idea of separating receiver nuisance information, rather than a reproduction of published architectures; see [Pan et al.](https://arxiv.org/abs/2510.09405) and [Xie et al.](https://arxiv.org/abs/2208.02724). Source receiver 6 validation selected epoch 2 and confidence 0.7 for both (net +40 and +39 over three views, respectively). Receiver 8 holdout has 1,586 valid records per view: 402 L_s true labels and 1,184 confident U_s clean-teacher pseudo-targets.

| Source holdout view | DURA correct | Adversarial correct | Adversarial repairs/damages | Control correct | Control repairs/damages |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 1584 | 1585 | 1/0 | 1585 | 1/0 |
| practical_low_suburban | 1545 | 1545 | 6/6 | 1545 | 7/7 |
| practical_mid_urban | 1492 | 1481 | 9/20 | 1478 | 10/24 |
| practical_low_urban | 1413 | 1403 | 11/21 | 1404 | 14/23 |

Despite validation gains, the adversarial model changes the three heldout simulated views by 0, -11 and -10 correct mixed-target decisions; the control changes them by 0, -14 and -9. Clean teacher agreement changes by -3 out of all 1,639 holdout IDs in both runs. Independent replay regenerated every heldout source view and reproduced all counts; no U_s transmitter labels, validation IQ or test IQ were read. Previously observed Phase1 development-test results motivated this research, so any later test evaluation is nonblind.
