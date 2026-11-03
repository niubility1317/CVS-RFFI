# Source matched pair: iterative time/frequency versus independent multiscale

Compare fixed independent multiscale and iterative time/frequency models on the same 11,761 Phase1 source holdout records. Both use the same ordered 8,192 fit records, three cached views, random seed, optimizer, aligned source objective and five-epoch budget. Architectures and compute differ; this is not a single-factor architecture ablation. Neither model loads DURA.

Before replay require the original model's fixed source score and existing 11,761-record independent pair audit, and the iterative model's fixed source score and independent source audit. Verify fit IDs, raw decisions, blind gate and both original 672 fixed source decisions. Fix all pair predictions and hashes before scoring 1,047 allowed L_s identities and frozen-reader pseudo classes. Independently recount 16 groups, 96 comparisons and 480 fields. Do not access U_s truth, V or ground data. Prior explored source data cannot establish blind or in-orbit performance.

Run 20261103-phase1-iterative-multiscale-fullsource-s2026092701-r01.
