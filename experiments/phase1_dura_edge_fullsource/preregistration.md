# Fixed reflection and edge-protected fusion: expanded source comparison

Replay the two already-frozen DURA-anchored spectral fusion models on the established 11,761 source-only holdout records. They use the same ordered 8192 source fit IDs, frozen spectral expert, protected source objective, view schedule and five epochs, but were trained separately using different edge handling. Reproduce the frozen expert tensors, original 672 pilot decisions, same raw/DURA predictions and gate, and prior fixed source comparator; compare both models to DURA and to each other. This diagnostic was designed after prior ground development results and does not train or modify any model.

Use original epoch-19 source views and first run a 64-record real-source smoke. Verify exact checkpoints, matching ordered fit IDs and disjoint holdout. Freeze all 11,761 predictions and their SHA-256 before connecting 1047 permitted L_s identities; U_s clean-F pseudo targets are separate and are not identity truth. Independently recount 16 groups and 96 comparisons. No U_s truth, V, ground IQ or target feedback. Results are previously explored simulations, not blind or real satellite evidence.

Source replay: 20261103-phase1-dura-edge-fullsource-s2026092701-r01. All other registered evaluations continue independently.
