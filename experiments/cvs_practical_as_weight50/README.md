# Fixed weighted FINAL30 to E50 continuation

Only the training budget extends. Two rows load their own strict weighted FINAL30 student and EMA from the pinned weighted30 release, then train20 additional epochs with reset AdamW/cosine20. Total trajectory is20+10+20, not continuous50 and not50 extra. The optimizer matches the original E50 budget tail:lr2e-4,weight_decay1e-4,minimumLR1e-6.

The .5 primary ordinary/.125 mixed ordinary64/.375 selected64/.25 raw/.1 ORD128 objective, full physical As64, NoAs actual raw64, six neural calls and21 native metric fields remain unchanged. New local steps1..4440 map to cumulative6661..11100 and epochs31..50. Each row has3409920 labeled neural forwards; actual satellite observations are284160 for B01 and0 for B00.

Ordinary augmentation still freezes epoch13 using the strict original20 ancestor args (schedule_args_epochs20), not weighted30 native args epochs10. The EMA comes from own weighted30; temporary caches and optimizer counters reset. The original fixed default head jointly trains. No new U route, source RX holdout, Ldev or model branch is introduced.

Runtime requires project,run_id,parent_release,parent_commit for weighted30 3befd47c8041d81f9430eda53562d0821624ff04 and reference_release/reference_commit for original P43 19aedc3604fed9c193237dd92366164178af0983. New runner bootstraps the actual pinned weighted30 runtime and its original chain. Original budget50 models are readout controls only.

Local checks:python -m experiments.cvs_practical_as_weight50.checks;python experiments/cvs_practical_as_weight50/check_emitted.py experiments/cvs_practical_as_weight50. Launcher real smoke loads both own weighted30 checkpoints (4 NN) and executes one actual6L synthetic L step per row (1536 NN). PASS immediately trains. Unchanged grouped loss equivalence is reused; real probes verify full weighted objective and default head/Sinc/backbone gradients without updates or real IQ.

Eight logical frozen rows cover original/weighted × NoAs/As × E30/E50. Only two weighted50 rows train or predict again. Both automatically complete legal source readout, seven fixed predictions and independent exec truth-last scores, irrespective of source scientific outcome. Future fixed02/03 eligibility alone uses source As50-minus-ownAs30 fullVclean>=0pp and two-draw midlow mean>=+.5pp. No automatic seed expansion or retry.

Complete native/student/teacher costs, physical L/U sequence, reference/shape/weight/unit steps, source strata and actual NoAs raw64 versus As(raw64) differences are retained. Full source V F1 or per-packet errors are not fabricated. One seed does not prove repeated double improvement.
