# Matched source replay: wide physical fusion versus local projection teacher

Replay the fixed wide model and the local-teacher model on the same 11,761 explored Phase1 source holdout records. They use identical architecture, ordered 8,192 fit IDs, view cache, optimizer, seed and five epochs, but independently trained objectives and paths. Share a single frozen physical posterior per active batch, verify DURA/raw decisions and gate against the fixed comparator, and reproduce each model's 672 fixed source predictions before finalizing all pair predictions.

After all predictions and hashes are fixed, score only the 1,047 allowed L_s identities and separately clean-reader pseudo classes. Independently recount 16 groups, 96 comparisons and 480 fields; never read U_s truth, V or ground data. Previously explored source data are not a blind or in-orbit test. Seven-view ground tests remain separate.

Run: 20261103-phase1-local-teacher-fullsource-s2026092701-r01.
