# Matched augmented DURA models with and without waveform protection

Replay two already-frozen source models on the established 11,761 source-only holdout records: the augmented DURA context model and its new DURA-protected counterpart. Both use identical architecture, source view schedule, seed, ordered 8192 fit IDs and augmentation generator; the intended training difference is the 5x positive DURA-relative aligned-IQ degradation penalty. Compare both to the frozen DURA prediction and each other. The data have already been explored, and this is a source diagnostic, not a target adaptation or blind test.

Use the original epoch-19 source views without added augmentation. Before truth access, verify both exact checkpoint identities, ordered fit IDs and heldout disjointness, reference raw/DURA/gate decisions and the 672 fixed pilot predictions. First run a 64-record actual-source smoke. Fix all 11,761 predictions and their SHA-256 before connecting 1,047 legitimate L_s identities. Report high-confidence clean-F pseudo targets separately, and independently recount all 16 groups and 96 comparisons by sets. Do not read U_s identity truth, V, ground data or target feedback. Ground evaluations remain independent and unchanged.

Source replay: 20261103-phase1-dura-augprotect-fullsource-s2026092701-r01.
