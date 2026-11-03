# Independent projector versus multiscale network: expanded source pair

Replay the two fixed source-only models on the established 11761 source holdout records. Both are independent of DURA and use the same ordered 8192 fitting records and five source epochs; architectures, objectives and computation differ, so the comparison is exploratory and not a single-factor ablation. DURA's previously fixed source predictions are used only as a comparator. The Tri-SIG-TIME reader and existing source records were previously explored.

Require source predictions and legal L_s scores fixed for both models and a passing independent source audit for the projector before replay. Verify exact frozen reader tensors, fit IDs and holdout disjointness, raw reader and gate outputs, and all 672 originally fixed predictions; require a 64-record actual-source smoke. Fix and hash all 11761 pair predictions before connecting 1047 legal L_s identities. Independently recount 16 groups and 96 comparisons. Read no U_s truth, V or ground data; pseudo labels are not identity truth. Do not interrupt ground tests or infer independent blind or real-satellite performance.

Run: 20261103-phase1-projector-multiscale-fullsource-s2026092701-r01.
