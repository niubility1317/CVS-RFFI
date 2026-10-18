# Fixed existing mixed-slot supervision reallocation

Two own-original E20+10 rows preserve all6L calls, original64 mask, generated IQ, channel, dropout/RDrop, primary ordinary .5, raw .25, .1 ORD128 CROSS_RX and hidden-U rules. Only the existing mixed128 loss changes from .25 ordinary64 + .25 selected64 to .125 ordinary64 + .375 selected64, with group means for both CE and symmetricKL. NoAs selected64 preserve the actual original NoAs identity raw64 IQ; the matched weighted NoAs row is required.

Original NoAs30 and As30 are frozen readout controls. Both own original20 student and EMA states are checked strictly against actual P43 configuration and source role handles. reference_config.json stores the actual original B00/B01 stage20 and source configs.

Local checks: python -m experiments.cvs_practical_as_weight30.checks; python experiments/cvs_practical_as_weight30/check_emitted.py experiments/cvs_practical_as_weight30. Checks execute the actual grouping helper with scalar/gradient mocks and verify the exact native metric schema. Native concat/total_L is always the full6L total; no shape/RD native metric is implied. Actual raw values are weight/raw_RD, raw_CE and raw_KL.

Launcher first checks both real own20 checkpoints without query (4 sample forwards total), then one actual6L synthetic L-step per row (1536 forwards). Equal .25/.25 group weights must reproduce original RD/CE/KL loss and gradients with real Torch. Fixed candidate .125/.375 is then used in both real probes. No optimizer updates or real IQ in smoke. PASS directly trains.

Both fixed FINAL30 rows automatically produce legal source readouts, seven fixed formal predictions and independent exec truth-last scores. Only future E50 or seed expansion uses source fullV AsWeight-minus-originalAs clean >=-.1 absolutepp and source two-draw midlow mean >=+.5pp. Source failure never skips formal. There is no automatic retry or expansion.

Complete native logs, physical L/U sequence, reference/shape/weight/unit steps, actual teacher/student counters and legal source strata are retained. Source fullV F1 or paired error counts are not fabricated. One seed cannot establish repeated double improvement.
