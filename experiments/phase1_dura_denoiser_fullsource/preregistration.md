# Fixed DURA denoiser versus context: expanded source pair

Replay the two fixed source-only DURA-correction models on the established 11761 source holdout records. The models share the same ordered 8192 source fit IDs and five epochs, frozen DURA and Tri-SIG-TIME SSL. Their architecture and optimization objectives differ, so this is exploratory fixed-model comparison, not a single-factor ablation. These source records have been explored previously.

Require both fixed model checkpoints, disjoint source holdouts, matching raw, DURA and gate, a 64-record real-source smoke and reproduction of all 672 original fixed predictions. Fix and hash all 11761 predictions before connecting 1047 permitted L_s identities, then independently recount 16 groups and 96 comparisons. Do not read U_s identity truth, V or ground data. Pseudo targets are not identity labels; these results are neither a blind test nor real-satellite evidence. Other registered experiments continue unchanged.

Source replay: 20261103-phase1-dura-denoiser-fullsource-s2026092701-r01.
