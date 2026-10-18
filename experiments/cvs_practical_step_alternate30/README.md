# Fixed step-alternating practical continuation

Only mixed-slot membership changes during the last 10 epochs of original As20+10. Odd cumulative steps use full practical for all 128 labeled packets; even steps reuse the generated ordinary IQ and skip the channel. Each step keeps six identity forwards and the same .5/.5/.25 RDrop plus original .1 ORD128 CROSS_RX loss. There are 142,080 practical observations in the tail, matching original As30 in total; equal exposure per physical packet is not claimed.

Pinned original P43 As20 student and own EMA, L6300/U56700/V27000 and all five source RX stay fixed. Original As30 and NoAs30 are frozen readout controls. No additional U view, label access, head fitting, parameters or deployment branches are introduced.

Local checks: python -m experiments.cvs_practical_step_alternate30.checks; python experiments/cvs_practical_step_alternate30/check_emitted.py experiments/cvs_practical_step_alternate30.

Publisher: python -m experiments.cvs_practical_step_alternate30.publish. Private runtime stays in local_artifacts/F_As_stepALT_E30_20261018/runtime.json. The single ordinary-account preflight verifies actual source-parent/control metadata and capacity. Dispatcher allocates a worker under the existing shared GPU lock with max3 total processes and at least12000MB free. One archive comparison and remote compile precede launch. There is no retry.

Launcher first runs strict parent checks, one real checkpoint no-query smoke and two actual F L-step synthetic probes:128/0 As,1536 training sample forwards,zero updates and real IQ. PASS immediately continues training, fixed FINAL30, legal source readout, all seven fixed predictions and independent exec truth-last scoring. Scientific failure never skips formal evaluation.

Retained artifacts include full native logs, physical L/U sequence, every128/0 exposure and source stratum, unchanged channel metadata and actual student/teacher counters. Automatic summary stays preliminary pending complete independent same-row analysis. Scores never choose a seed, parameter or checkpoint.
