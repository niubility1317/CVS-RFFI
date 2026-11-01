# Phase1 source cycle-constrained residual network

Status: repeated source-only trained-network experiment; no network advanced to holdout or Phase1 test. Frozen DURA supplies the anchor and frozen Tri-SIG-TIME SSL remains the final classifier. The observed approximately 20-sample source periodicity was used to define a sample-specific phase-aligned decomposition. The network receives DURA, DURA's nonperiodic residual and received IQ aligned to DURA without paired clean IQ. A 2D convolutional model predicts only the zero-mean-per-cycle residual correction; the anchor's periodic component is held fixed in its estimated phase coordinates. A received-to-DURA difference gate suppresses the correction on near-clean signals.

For source training only, source clean IQ is aligned to the DURA anchor using the source pair and supplies the desired residual correction. Training minimizes paired residual MSE with a correction-energy penalty; no transmitter labels train this network. The source sample has 1,626 training IDs from receivers 1/3/4 days 1/2, 536 validation IDs from receivers 4/6 day 3, and 280 holdout IDs from receiver 8 day 3. The period and network design follow earlier inspected source diagnostics and these sample IDs overlap those diagnostics. Frozen experts saw source records. Simulated satellite conditions have not been validated against an actual satellite link.

Source validation uses a mixture of real L_s labels and high-confidence U_s clean-teacher pseudo-targets. Checkpoints at epochs 2/5/10 and correction strengths 0.25/0.5/1.0 were fixed before scoring, with receiver-group, per-view and clean-teacher safeguards. Their changes in source correct decisions relative to DURA were:

| Epoch | Strength | Low suburban | Mid urban | Low urban | Total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.25 | +1 | +0 | -2 | -1 |
| 2 | 0.50 | +1 | +0 | -5 | -4 |
| 2 | 1.00 | +3 | +1 | -4 | +0 |
| 5 | 0.25 | +1 | -1 | -1 | -1 |
| 5 | 0.50 | +1 | +0 | -4 | -3 |
| 5 | 1.00 | +2 | -2 | +0 | +0 |
| 10 | 0.25 | +1 | -1 | -2 | -2 |
| 10 | 0.50 | +1 | +0 | -2 | -1 |
| 10 | 1.00 | +2 | -1 | -1 | +0 |

No candidate obtained positive aggregate validation improvement, so the predeclared rule retained DURA and the source holdout was not scored for a selected model. Paired training loss decreased from 0.012960 to 0.012737 but did not translate into a validation classification improvement. Synthetic and four-record real-source checks verified zero initialization, zero-mean cycle constraint after an update, finite gradients and frozen-base classification. A second complete training run matched the entire JSON. All candidate predictions were fixed before source transmitter labels were read; no U_s transmitter labels, validation IQ or test IQ were read. Earlier observed Phase1 development-test results motivated the work, so later use of that test is nonblind.
