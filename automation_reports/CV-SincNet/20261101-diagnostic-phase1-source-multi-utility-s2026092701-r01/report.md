# Phase1 source multi-candidate repair/damage utility diagnostic

Status: independently replayed source-only exploratory diagnostic; no new target-test evaluation. The ground classifier is frozen Tri-SIG-TIME SSL. This idea follows previously observed Phase1 test expert-oracle analysis; subsequent use of that test is not blind.

The 698–128–4 network estimates source repair and damage probabilities for switching from DURA separately to raw or feature-conditioned outputs. Inputs are four expert posterior vectors, two log energies, four normalized frozen-classifier embeddings and normalized raw/residual spectral bins. It chooses the candidate with the highest estimated repair-minus-damage score only when the score exceeds a fixed source-selected threshold; otherwise it uses DURA. The 4,912 fitting, 1,640 validation and 1,640 holdout source physical IDs are disjoint within this experiment. Some IDs overlap earlier source diagnostics, and the fixed experts themselves trained on source records. Source views use independently seeded practical residual-channel simulation, not validated real satellite data.

Expert outputs were fixed before joining L_s true labels and high-confidence U_s clean-teacher pseudo-labels. Fitting used 19,156 valid view-observations, including 3,621 raw/DURA and 1,883 feature/DURA disagreements. Corresponding repair/damage events were 281/2,759 for raw and 670/980 for feature. Source validation selected epoch 10 and utility threshold 0.08, net +38 over three simulated views under clean-teacher and per-view constraints. Holdout has 1,605 valid observations per view, comprising 410 L_s true and 1,195 U_s pseudo-targets.

| Source view | DURA correct | Utility model correct | Repairs | Damages | Raw choices | Feature choices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 1601 | 1601 | 0 | 0 | 0 | 0 |
| practical_low_suburban | 1564 | 1566 | 8 | 6 | 1 | 15 |
| practical_mid_urban | 1467 | 1469 | 11 | 9 | 3 | 22 |
| practical_low_urban | 1380 | 1385 | 27 | 22 | 5 | 54 |

The three simulated holdout views gain 2, 2 and 5 correct mixed-target decisions; the clean mixed-target count is unchanged. Clean-teacher agreement changes by -1 over all 1,640 holdout IDs. In low urban only five valid records select raw. These small source gains do not establish a test gain or achievement of the 3% goal. A separate post-hoc three-prediction test oracle has a low-urban upper bound of 85.6113% but requires test truth and is not deployable.

The fixed network and independently regenerated source views were replayed on all 1,640 holdout IDs with identical paired counts. JSON and scripts preserve seeds, weights, split and target details. No U_s transmitter labels, validation IQ or test IQ were read in this source experiment.
