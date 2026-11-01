# Phase1 source semantic-consensus diagnostic

Status: independently replayed source-only exploratory diagnostic. The frozen ground classifier is Tri-SIG-TIME SSL. This design follows previously observed Phase1 test results, so subsequent testing is not blind.

A 698–256–6 network infers a class from four fixed experts' frozen-classifier embeddings and posterior vectors, two log energies, and normalized raw/residual spectral bins. If confidence is at least 0.9 and the inferred class differs from DURA, the network selects an expert whose frozen classifier agrees, prioritizing feature, boundary and raw. Otherwise it uses DURA. The frozen classifier makes the final prediction. Epoch 20 and threshold 0.9 were selected on source validation, yielding net +28 across three simulated source views.

The source split contains 4,912 fitting, 1,640 validation and 1,640 holdout IDs, disjoint within this experiment. Some IDs may overlap earlier diagnostics; the fixed experts trained on source records. Expert predictions were fixed before joining source labels. Each holdout view contains 398 true L_s labels and 1,198 confident U_s clean-teacher pseudo-targets. The physical simulator has not been validated as real satellite data.

| Source view | DURA correct | Model correct | Repairs | Damages | L_s DURA | L_s model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 1594 | 1594 | 0 | 0 | 396 | 396 |
| practical_low_suburban | 1543 | 1549 | 11 | 5 | 383 | 385 |
| practical_mid_urban | 1474 | 1479 | 12 | 7 | 361 | 363 |
| practical_low_urban | 1379 | 1400 | 26 | 5 | 335 | 345 |

Clean-teacher agreement changes by -2 on 1,640 holdout IDs. The saved result is a source diagnostic, not test accuracy or evidence that the 3% goal is met. Independent replay regenerated all source views and reproduced every paired count; no U_s transmitter labels, validation IQ or test IQ were read.
