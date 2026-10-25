**Fig01_F0_cross_date_raw** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross_date. External cell D is fixed, not target-selected. Sup uses G-L; external PL/REPL use G-LU. Complete-recipe differences are not single-component causal effects.

**Fig02_F0_cross_date_restored** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross_date. External cell D is fixed, not target-selected. Sup uses G-L; external PL/REPL use G-LU. Complete-recipe differences are not single-component causal effects.

**Fig03_F0_cross_rx_raw** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross_rx. External cell D is fixed, not target-selected. Sup uses G-L; external PL/REPL use G-LU. Complete-recipe differences are not single-component causal effects.

**Fig04_F0_cross_rx_restored** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross_rx. External cell D is fixed, not target-selected. Sup uses G-L; external PL/REPL use G-LU. Complete-recipe differences are not single-component causal effects.

**Fig05_F0_cross_both_raw** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross_both. External cell D is fixed, not target-selected. Sup uses G-L; external PL/REPL use G-LU. Complete-recipe differences are not single-component causal effects.

**Fig06_F0_cross_both_restored** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross_both. External cell D is fixed, not target-selected. Sup uses G-L; external PL/REPL use G-LU. Complete-recipe differences are not single-component causal effects.

**Fig07_source_cross_rx_restored** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. Cross-RX endpoint, matching restoration. Source curve uses natural budgets and fixed six target RX, m=1,2,3,4,6 only; F0 is not a five-source point. Label and RX partitions retain their registered physical roles. All nonmonotonic values remain.

**Fig08_labels_cross_rx_restored** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. Cross-RX endpoint, matching restoration. Source curve uses natural budgets and fixed six target RX, m=1,2,3,4,6 only; F0 is not a five-source point. Label and RX partitions retain their registered physical roles. All nonmonotonic values remain.

**Fig09_receivers_cross_rx_restored** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. Cross-RX endpoint, matching restoration. Source curve uses natural budgets and fixed six target RX, m=1,2,3,4,6 only; F0 is not a five-source point. Label and RX partitions retain their registered physical roles. All nonmonotonic values remain.

**Fig10_ManyRx_macroF1** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. ManyRx cross-RX Macro-F1 uses all ten registered TX, with unequal cell sizes; no pooling with six-class ManySig.

**Fig11_F0_seven_views_restored** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross-RX; practical channel views are simulations. Clean is also restored; there is no scene-aware bypass.

**Fig12_core_conditional_effects** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. Full 111 minus 011/101/110. A is supervised CRIC, B supervised paired KL, C the entire REPL U objective. Full conditional effects may differ from eight-cell marginal effects. C0 opens no U loader.

**Fig13_core_eight_cells** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross-RX; 000 is a matched 768-L-forward multi-view CE skeleton, not strict 384-forward CE. A fixed G-LU still uses U when F has C0.

**Fig14_REPL_inner_mechanisms** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. F0 cross-RX, fixed E50 and default head. No-extrapolation reuses the exact matching historical Joint160 F; no native EMA/gate ablations are relabeled as REPL.

**Fig15_REPL_structure** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. REPL structure layer. REPL has768 L packet forwards/update and U; strict CE has384 and no U/EMA/CRIC/KL. Compare variants within layer, not as an exposure-matched cross-layer causal contrast.

**Fig16_CE_structure** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. Strict CE structure layer. REPL has768 L packet forwards/update and U; strict CE has384 and no U/EMA/CRIC/KL. Compare variants within layer, not as an exposure-matched cross-layer causal contrast.

**Fig17_existing_G_with_REPL** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. Existing G weights only; no G retraining. Default uses fixed G1 across F seeds, whereas seven variants use their paired G seeds. This is not pure G variance or a fully replicated copy-by-KD factorial.

**Fig18_m3_crossed_seed** m3-natural cross-RX S6. Each cell is one frozen F/G pair, without a within-cell SD. Three diagonal pairs are three complete-system seeds; nine cells are not nine independent replicas. Numerals are authoritative.

**Fig19_measured_throughput** Exclusive RTX3090, FP32/TF32off, five rounds per batch. F is eager and GF uses the established optimized graph engine; their difference is an execution-path benchmark, not an isolated algorithmic speedup from adding G. Cached classification is excluded.

**Fig20_REPL_structure_latency** Five rounds are timing repetitions, not model seeds. Points are pooled wall medians and bars are SD of round medians. F eager and GF optimized are disclosed separately. Co-resident baseline/incremental memory are retained; no isolated-model VRAM claim.

**Fig21_CE_structure_latency** Five rounds are timing repetitions, not model seeds. Points are pooled wall medians and bars are SD of round medians. F eager and GF optimized are disclosed separately. Co-resident baseline/incremental memory are retained; no isolated-model VRAM claim.

**Fig22_source_physical_sensitivity** Identical 720 labeled-source packets, 90 TX/RX/day strata, 11 interventions, three F seeds. No U/target or fitted probe. A synthetic PA perturbation is not assumed identity-preserving; this is sensitivity evidence, not complete hardware disentanglement.

**Fig23_PAMF_fairness_fixed_G** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. New matched PAMF Sup/ordinary-PL F evaluated with both G-L and G-LU on all 11 contracts. PL F still uses U with G-L; only Sup F plus G-L is a completely L-only system. Negative effects remain.

**Fig24_riei_Full_four_cells** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. RIEI Full + PL on F0 cross-RX. All four cells are Full, not CE-only. Existing verified external aggregates are reused; A/C restoration uses G-L, B/D G-LU, so restored cross-PL contrasts also change G permissions.

**Fig25_drift_Full_four_cells** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. DRIFT-v2 Full + PL on F0 cross-RX. All four cells are Full, not CE-only. Existing verified external aggregates are reused; A/C restoration uses G-L, B/D G-LU, so restored cross-PL contrasts also change G permissions.

**Fig26_ibrff_Full_four_cells** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. IB-RFF Full + PL on F0 cross-RX. All four cells are Full, not CE-only. Existing verified external aggregates are reused; A/C restoration uses G-L, B/D G-LU, so restored cross-PL contrasts also change G permissions.

**Fig27_sdg_Full_four_cells** Three F seeds, mean and sample SD; S6 is averaged within seed. G is fixed by contract/role unless stated. SD is not a confidence interval or significance test. SDG Full + PL on F0 cross-RX. All four cells are Full, not CE-only. Existing verified external aggregates are reused; A/C restoration uses G-L, B/D G-LU, so restored cross-PL contrasts also change G permissions.
