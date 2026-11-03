# Fixed edge and three-way fusion: expanded source comparison

Replay the two fixed models on the established 11,761 source-only holdout records. Both have five epochs, the same ordered 8,192 source fit IDs and a frozen spectral expert. Their architectures and objectives differ, so this is an exploratory fixed-model comparison, not a single-factor ablation. These data have been explored previously.

Verify exact model versions and expert tensors, fit IDs, holdout disjointness, raw, DURA and gate predictions, and reproduce all 672 fixed pilot decisions. Require a 64-record source smoke and fix all predictions and hashes before connecting 1,047 permitted L_s identities. Recount 16 groups and 96 comparisons independently. Pseudo targets are not identity truth. Never read U_s truth, V or ground data; the result is neither a blind test nor real satellite performance. Ground tests continue independently regardless of source results.

Source replay: 20261103-phase1-dura-triple-fullsource-s2026092701-r01.
