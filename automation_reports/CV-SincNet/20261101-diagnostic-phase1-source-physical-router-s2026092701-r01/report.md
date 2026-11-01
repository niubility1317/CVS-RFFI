# Phase1 physical-channel source router diagnostic

Status: source-only exploratory diagnostic, independently replayed; no new target-test evaluation. The classifier is frozen Tri-SIG-TIME SSL. The research direction was informed by already observed Phase1 test expert-oracle results; that test is not blind for subsequent choices.

The router observes the four frozen experts' six-class probabilities and raw/DURA log energies. The 4,096 source physical IDs exclude the 13,312 IDs used in the preceding source probes. Split: 2,456 fitting, 820 validation and 820 holdout IDs. The frozen experts themselves trained on source records. Clean source windows are independently passed through the practical residual-channel simulator in three scenarios, with 25 MHz sample rate, 2.462 GHz carrier, 30 dB reference SNR, no equalization and source RX/day-specific sessions. This is a simulation, not validated real satellite data.

All expert outputs were fixed before joining source L_s true labels and U_s high-confidence clean-teacher pseudo-labels. The fitting set had 9,584 valid view-observations and 417 potentially repairable DURA errors. Source validation chose epoch 20 and probability gap 0.6, giving net +8 across the three simulated corrupted views subject to the registered clean-teacher guard and at most four validation damages in any scenario. Holdout has 800 valid examples per view: 207 L_s true and 593 U_s pseudo-targets.

| Source view | DURA correct | Router correct | Repairs | Damages | Four-expert oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 800 | 800 | 0 | 0 | 800 |
| practical_low_suburban | 781 | 782 | 1 | 0 | 798 |
| practical_mid_urban | 760 | 760 | 1 | 1 | 789 |
| practical_low_urban | 693 | 688 | 1 | 6 | 771 |

On the low-urban holdout, the router changes 693 correct to 688: one repair and six damages. It provides no consistent benefit across these source views and does not justify a formal test. The four-expert oracle requires the source target and is not deployable. Its 771 correct low-urban predictions describe only the upper bound among these four already fixed predictions; it does not prove a trainable selector can reach that value.

The saved router and independently regenerated physical-channel views were replayed on all 820 heldout IDs, reproducing the paired results and restricted oracle counts. No U_s transmitter labels, validation IQ, or test IQ were read. JSON and scripts preserve seeds, source-model hashes, weights and split/target details.
