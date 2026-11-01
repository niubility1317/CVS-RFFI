# Phase1 source clean-guarded estimated-channel fusion

Status: source-only diagnostic; no variant advanced to Phase1 test. Tri-SIG-TIME SSL and DURA are frozen. The source-trained estimated channel, DURA waveform and received IQ are the only inference inputs. Source clean data and simulated channel traces are used to simulate source views, train the earlier estimator and measure aligned waveform fidelity, never as inference inputs.

Five clean-guard strengths gave tied impaired-view classification scores on the initial source validation. Its original rule picked the first, unguarded candidate. The tie rule was revised after inspecting initial holdout results: prefer source clean-teacher agreement, then the stronger guard. This selects scale 0.2. The initial tie replay is not independent. The guard was then fixed in a written specification before the new 8,192 source IDs and simulation seeds were scored. The source selector validation uses 1,079 receiver 4/6 day 3 IDs and the holdout uses 573 receiver 8 day 3 IDs. These exclude the 2,048 estimator-pilot and 8,192 preceding guard-sample IDs, but may overlap other source probes. Frozen experts saw source data; only the selector/estimator split is receiver/day-disjoint. The simulator is not validated real satellite data.

The new holdout has 560 valid mixed targets per view, comprising 148 L_s true labels and 412 confident U_s clean-teacher pseudo-targets. MSE uses all 573 source pairs and independently aligns each waveform to source clean IQ by shifts from -16 to +16 and complex gain over the central 224 samples; it is one fidelity diagnostic, not proof of satellite recovery.

| View | DURA correct | Guarded correct | Repairs | Damages | L_s DURA | L_s guarded | DURA aligned MSE | Guarded aligned MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 560 | 560 | 0 | 0 | 148 | 148 | 0.00003994 | 0.00004177 |
| practical_low_suburban | 554 | 548 | 0 | 6 | 144 | 142 | 0.02468809 | 0.02741902 |
| practical_mid_urban | 537 | 537 | 5 | 5 | 140 | 139 | 0.05377673 | 0.05609127 |
| practical_low_urban | 485 | 489 | 8 | 4 | 127 | 128 | 0.09574032 | 0.09808186 |

Low urban gains 4 mixed-target decisions and 1 true-label decision, but low suburban loses 6 and its true-label subset loses 2. The clean aligned MSE is 0.00003994 for DURA, 0.00004177 guarded, and 0.00136083 ungated. The guard strongly reduces clean degradation, while every impaired view has worse mean aligned MSE than DURA. The fresh fixed-source diagnostic was run twice with matching complete JSON. All candidate predictions were fixed before source labels were read; no U_s transmitter labels, validation IQ or test IQ were read. Phase1 development-test results had previously been observed, so future test use is nonblind.
