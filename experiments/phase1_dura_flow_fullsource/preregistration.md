# Fixed context and DURA conditional-flow models: expanded source comparison

After both source models finish training and freeze their 672 source predictions, replay the already-fixed ordinary DURA context model and four-step DURA conditional flow model on the established 11,761 source-only holdout records. They use the same ordered 8192 source fit IDs, five epochs, DURA anchor and frozen Tri-SIG-TIME SSL, but differ in architecture, rollout and source objective; this is not a one-factor ablation. The flow prediction and score are not used to cancel either registered ground test.

Run a 64-record real-source smoke, verify exact checkpoints, ordered fit IDs, heldout disjointness, the previously fixed 672 pilot decisions and identical raw/DURA/gate decisions. Freeze all 11,761 predictions and SHA-256 before connecting 1047 permitted L_s identities. Clean-F pseudo targets are not true identities. Independently recount 16 groups and 96 comparisons. No U_s identity truth, V, ground IQ or feedback. Previously explored simulations are neither blind nor real-satellite validation.

Source replay: 20261103-phase1-dura-flow-fullsource-s2026092701-r01. Ground evaluations continue independently.
