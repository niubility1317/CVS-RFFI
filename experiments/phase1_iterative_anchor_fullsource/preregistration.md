# Standalone versus DURA-anchored iterative time/frequency recovery

Compare the independently trained standalone and DURA-anchored three-step models on the same 11,761 explored Phase1 source holdout records. Both use the same architecture, ordered source fit IDs, view cache, seed, optimizer and five-epoch budget. Initial estimate and source objective differ; the result is a comparison of complete systems, not a single-factor ablation.

Require each model's fixed source predictions and separate source independent audit. Confirm same fit records, each 672-record original source pilot, raw and DURA base decisions and blind gate. On inactive gate the standalone output must be raw IQ and the DURA-anchored output must be DURA; use one blind noise estimate per active batch. Fix all predictions and hashes before connecting 1,047 legal L_s identities and frozen-reader pseudo classes, then independently recount 16 groups, 96 comparisons and 480 fields. Do not read U_s truth, V or ground data; source records are explored and cannot establish blind or in-orbit performance.

Run 20261103-phase1-iterative-anchor-fullsource-s2026092701-r01.
