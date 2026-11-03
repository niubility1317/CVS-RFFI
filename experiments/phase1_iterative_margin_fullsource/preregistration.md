# DURA iterative recognition-margin source pair

Compare independently trained baseline and recognition-margin DURA iterative time/frequency models on the same 11,761 previously explored source-holdout records. Both models have identical architecture and ordered 8,192 source-fit IDs and use the same frozen DURA anchor, blind gate, noise estimator and Tri-SIG-TIME SSL reader. The added source recognition-margin loss changes the optimization trajectory; the result compares complete training recipes.

Wait for both fixed 672-prediction source scores and independent audits. Replay each source pilot, require matching raw/DURA/gate and identical ordered fit IDs, then fix all 11,761 pair predictions before reading permitted L_s identity metadata. Compare each network against DURA and against the other using identical source records, including 96 comparisons and an independent set-based recount. Never read U_s truth, V or ground labels. Previously explored source outcomes are not a blind or in-orbit test; seven-scene ground tests remain independent of this source analysis.

Run 20261103-phase1-iterative-margin-fullsource-s2026092701-r01.
