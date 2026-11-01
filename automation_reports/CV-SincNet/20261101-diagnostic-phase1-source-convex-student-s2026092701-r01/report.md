# Phase1 source convex waveform student

Status: independently replayed source-only exploratory diagnostic; no target-test measurement. This idea follows a previously observed Phase1 test expert-oracle result, so that test is not blind for subsequent choices.

The 378–96–4 network uses frozen Tri-SIG-TIME SSL embeddings, expert probabilities, log energies and normalized raw/residual spectra to predict softmax weights for raw, DURA, boundary and feature-conditioned signals. It is trained end to end through the frozen classifier using L_s true labels and high-confidence U_s clean-teacher pseudo-targets, plus a conservative DURA preference. Validation selects a fixed interpolation strength between DURA and the convex mixture. Inputs are from independent practical residual-channel simulations, not validated real satellite data. The frozen experts previously trained on source records.

The 2,048 source physical IDs exclude 21,760 earlier probe IDs and split into 1,228 training, 410 validation and 410 holdout IDs. All frozen expert waveforms and outputs were fixed before joining source labels. Training uses 4,788 valid view-observations. Source validation selected epoch 5 and interpolation strength 0.5, with net +5 across three simulated views subject to a clean-teacher guard and per-view damage constraints. Holdout has 401 valid examples per view: 104 true L_s and 297 confident U_s pseudo-targets.

| Source view | DURA correct | Mixture correct | Repairs | Damages | True L_s: DURA → mixture |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 401 | 401 | 0 | 0 | 104 → 104 |
| practical_low_suburban | 395 | 396 | 2 | 1 | 104 → 104 |
| practical_mid_urban | 371 | 374 | 8 | 5 | 94 → 97 |
| practical_low_urban | 349 | 347 | 14 | 16 | 91 → 87 |

The low-urban holdout loses two mixed-target correct predictions: the true-label subset loses four, while the pseudo-label subset gains two. The other two corrupted views change by +1 and +3; the clean mixed-target count is unchanged, and clean-teacher agreement is unchanged across all 410 holdout IDs. This small, inconsistent result does not justify formal Phase1 test evaluation. The separate source oracle was privileged and does not establish that this student can recover its weights.

The fixed network and independently regenerated source views were replayed on all 410 heldout IDs with identical paired counts. JSON and scripts preserve seeds, weights and source split/target details. No U_s transmitter labels, validation IQ or test IQ were read.
