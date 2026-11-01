# Phase1 source cycle network with frozen-base feature loss

Status: repeated source-only trained-network experiment; no network advanced to holdout or Phase1 test. Frozen DURA supplies the anchor and frozen Tri-SIG-TIME SSL remains the final classifier. The phase-aligned 20-sample decomposition and 2D cycle-constrained network match the earlier source cycle network: the predicted correction has zero cycle mean under the anchor's estimated phase, and an input/DURA difference guard protects near-clean signals. Raw received IQ is aligned to DURA without reference clean IQ at inference.

The training target includes source clean IQ aligned to DURA and its frozen Tri-SIG-TIME SSL feature embedding. The loss combines paired residual MSE, correction-energy regularization and cosine feature discrepancy; gradients flow through the unchanged frozen base only to the restoration network. Source clean IQ is unavailable at inference. Source sample selection, simulation, optimizer schedule and validation safeguards match the earlier cycle-only network. The 1,626 training IDs come from receiver 1/3/4 days 1/2; 536 validation IDs use receiver 4/6 day 3; 280 source holdout IDs use receiver 8 day 3. The period and network were designed after previous source inspection, these IDs overlap earlier diagnostics, and frozen experts saw source records. Simulated satellite conditions have not been validated against a real satellite link.

Source validation combines true L_s labels and confident U_s clean-teacher pseudo-targets. Three checkpoints and three correction strengths were fixed before scoring. The table shows changes in correct decisions versus DURA; the last two columns are totals across impaired views for the separate validation receivers.

| Epoch | Strength | Low suburban | Mid urban | Low urban | Total | RX4 total | RX6 total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.25 | +0 | -2 | -1 | -3 | +0 | -3 |
| 2 | 0.50 | +0 | -2 | -1 | -3 | +1 | -4 |
| 2 | 1.00 | -1 | -4 | -5 | -10 | -4 | -6 |
| 5 | 0.25 | -1 | -2 | +2 | -1 | +2 | -3 |
| 5 | 0.50 | +0 | -2 | +3 | +1 | +3 | -2 |
| 5 | 1.00 | -1 | -1 | -1 | -3 | +1 | -4 |
| 10 | 0.25 | -1 | -2 | +2 | -1 | +2 | -3 |
| 10 | 0.50 | -1 | -2 | +2 | -1 | +3 | -4 |
| 10 | 1.00 | -1 | -1 | +1 | -1 | +1 | -2 |

Epoch 5/strength 0.5 gives aggregate +1 but RX6 loses 2, so it fails the predeclared receiver-group safeguard. The fixed selector retains DURA and does not score a selected model on the source holdout. A four-record real-source smoke explicitly verified nonzero finite feature-loss gradient into the restoration network and no parameter gradient in the frozen base. A second complete run matched the result JSON. All candidates were fixed before source transmitter labels; no U_s transmitter labels, validation IQ or test IQ were read. Earlier Phase1 development-test findings had been observed, so any future use of that test is nonblind.
