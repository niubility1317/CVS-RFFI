# Phase1 source pairwise switch-utility diagnostic

Status: independently replayed source-only exploratory diagnostic; no new target-test evaluation. The ground classifier is frozen Tri-SIG-TIME SSL. This idea follows previously observed Phase1 test expert-oracle analysis; subsequent use of that test is not blind.

The network learns, on source examples where switching changes correctness, whether switching from DURA to the fixed feature-conditioned expert repairs or damages the prediction. It uses 698 features: four expert posterior vectors, two log energies, four normalized frozen-classifier embeddings and normalized raw/residual spectral bins. A 698–128–1 MLP supplies a switch score. Source training, validation and holdout contain 4,912, 1,640 and 1,640 distinct physical IDs respectively. Some IDs overlap earlier source diagnostics, and the fixed experts themselves trained on source records. Source views are independently seeded practical residual-channel simulations; these are not validated real satellite data.

All expert outputs were fixed before joining L_s true labels and high-confidence U_s clean-teacher pseudo-labels. Of the fitting view-observations, 1,625 had decisive repair/damage outcomes: 688 repairs and 937 damages. Validation selected epoch 20 and threshold 0.55, net +9 across three simulated views subject to clean-teacher and per-view constraints. Holdout has 1,603 valid observations per view: 430 true L_s and 1,173 confident U_s pseudo-targets.

| Source view | DURA correct | Utility model correct | Repairs | Damages | Switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 1600 | 1602 | 2 | 0 | 2 |
| practical_low_suburban | 1548 | 1560 | 15 | 3 | 20 |
| practical_mid_urban | 1502 | 1507 | 17 | 12 | 35 |
| practical_low_urban | 1387 | 1386 | 32 | 33 | 87 |

The low-urban view has one net damage; low-suburban and mid-urban gain 12 and 5. Clean mixed-target correctness increases by two, while clean-teacher agreement changes by -3 over all 1,640 holdout IDs. This is not consistent across scenes. In addition, this two-choice family is bounded by the previously observed DURA/boundary/feature oracle on low urban and cannot by itself reach the stricter 3% target. The oracle requires truth and is not deployable; its bound applies only to the specified already fixed predictions.

The saved network and independently regenerated source views were replayed on all 1,640 heldout IDs with identical paired counts. JSON and scripts preserve seeds, weights, split and target details. No U_s transmitter labels, validation IQ or test IQ were read in the source experiment.
