# Fixed As64 allocation, E30

Only oneB01 seed2026092701:originalP43 ownAsFINAL20 student+ownEMA plus10epochs/2220updates, cumulative30/6660. Reuse originalNoAs30 andAs30 frozen readouts.

CurrentL128 loader,originalpositions,primaryordinary128,ORD128anchors,allCE label smoothing.01,allKL,rawpair,6L768NN andtotalLscale1.25 stay unchanged. Only thebool128 As mask changes:private CPU priorities then TXxRX feasiblewaterfill selects64 originalpositions withoutreplacement. Missing/saturated cells are handled fromavailablepositions,no added/resampled/deduplicated packet andno Asreordering tofixedpositions. Itsordinary64 complement changes withthemask.

Thegroupcanonicalorder comesfrommemberpriorities ratherthannumericclass IDs;eachwaterfillround privately permutes activegroups. Classrenaming preserves exactsamegroups/mask. Core batchpermutation equivalence carriespositionassociated priorities andsameprivate roundRNG;fresh draws onnewpositions are notclaimed toselectsamePIDs. Exactpriorityties useoriginalposition astiebreaker. NoPIDuniqueness gate. No global NumPy/Python/Torch/CUDA RNG draws are introduced by theCPU selector;whole U/optimization trajectory equality isnotclaimed.

Streams are reference_steps,shape_steps,unit_steps,allocation_steps andtraining_sequence(gzipJSONL). allocation_steps TX/RX/day/sample_ids coverall128;scene/state/lock areactualselected64 inorder. cells androunds exposeavailablecounts,oldparity andnew quotas/selections. CPU selectorwall andhookwall areseparateoverlapping timings,notadditive;actualfiniteAs exposure changes andnotpuregradientvariance.

Native14 keys are7concat(actualfull6Ltotal) and7allocation. NoextraNN orCE,CPU allocation/logging cost recorded. Launcherfirststep:ownparent checkpointnoquery2NN+actual6L768NN=770NN,PASSimmediately trains. All7fixedpredictions thenindependenttruthlastscore remainmandatory evenifsourcecriteriafail. Sourcefuturequalification:fullV newAs-originalNoAs>=-.1pp andtwo-drawmidlow newAs-originalAs>=+.5pp;noautomaticE50/groups/seed/parent scans.
