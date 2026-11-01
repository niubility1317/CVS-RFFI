# Phase1 rich-feature physical-channel source router

Status: source-only exploratory diagnostic, independently replayed; no new target-test evaluation. The classifier is frozen Tri-SIG-TIME SSL. The research direction was informed by previously observed Phase1 test expert-oracle results; that test is not blind for subsequent choices.

The router has 378 inputs: four six-class expert probability vectors, two log energies, normalized frozen-classifier embeddings for raw and DURA outputs, and 16-bin normalized spectral powers for raw and DURA residual IQ. A 378–128–4 MLP selects raw, DURA, boundary or feature-conditioned output. The 4,096 source physical IDs exclude 17,408 IDs used in earlier probes. Split: 2,456 fitting, 820 validation and 820 holdout IDs. Frozen experts themselves trained on source records.

Clean source windows are independently passed through the practical residual-channel simulator in low suburban, mid urban and low urban settings, at 25 MHz, 2.462 GHz, 30 dB reference SNR and no equalization, with source RX/day-specific hardware sessions. This is an artificial simulation, not validated real satellite data. Expert outputs were fixed before joining source L_s true labels and U_s confident clean-teacher pseudo-labels. The fitting set had 9,560 valid view-observations and 428 potentially repairable DURA errors. Source validation selected epoch 2 and probability gap 0.6, net +6 over the three simulated views subject to the clean-teacher guard and at most four validation damages in any scenario. Holdout has 796 valid examples per view, comprising 215 true L_s and 581 confident U_s pseudo-targets.

| Source view | DURA correct | Router correct | Repairs | Damages | Four-expert oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 795 | 795 | 0 | 0 | 795 |
| practical_low_suburban | 768 | 769 | 2 | 1 | 787 |
| practical_mid_urban | 733 | 737 | 8 | 4 | 779 |
| practical_low_urban | 684 | 692 | 12 | 4 | 761 |

Holdout changes are +1, +4 and +8 correct in the three physical-channel views, with no clean mixed-target change. These small gains justify an exploratory Phase1 test, but do not establish a gain on its six test scenes. The oracle requires the source target and is not deployable. Its 761 low-urban correct predictions are only an upper bound over these four fixed experts, not evidence that the learned router can achieve it.

The saved router and independently regenerated physical-channel inputs were replayed on all 820 heldout IDs, reproducing paired results and restricted oracle counts. No U_s transmitter labels, validation IQ or test IQ were read. JSON and scripts retain seeds, frozen-model hashes, weights and split/target details.
