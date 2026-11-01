# Phase1 source estimated-channel restoration and fusion

Status: repeated source-only diagnostics. Neither variant advanced to Phase1 test. Frozen Tri-SIG-TIME SSL is the final classifier; DURA and the source-trained channel estimator remain frozen. At inference, channel estimation, phase/shift alignment and inversion use only received IQ and DURA output. The estimator's simulated-channel and source-clean traces were used as training targets, not inference inputs. The exact reflected convolution operator and prior-consistent inverse passed a synthetic check; both integrated variants passed real-source smoke checks.

The 271 source validation IDs use receivers 4 and 6 on day 3; 139 holdout IDs use receiver 8 on day 3. The estimator was fitted on receivers 1, 3 and 4 on days 1 and 2, and its checkpoint was selected on the same validation groups used here, so validation is not independent for the combined system. The holdout receiver/day is disjoint only for the estimator; frozen experts saw source records and IDs may overlap earlier probes. The simulator is not validated real satellite data. All candidate predictions were fixed before source transmitter labels were joined.

First, four estimated-channel inverses and the aligned prior alone were evaluated. All five had negative total source-validation net change, so the fixed selection rule retained DURA; heldout scores were not selected or reported. A second variant aligns each of those outputs to DURA with a per-record complex gain and integer shift, then tries four blend fractions. Among the 20 prespecified combinations, source validation selected candidate 15: inverse regularization multiplier 10 and blend 0.5, net +4 across three validation views. Multiple validation comparisons and the small holdout limit the evidence.

The holdout has 134 valid targets per view: 34 L_s true labels and 100 confident U_s clean-teacher pseudo-targets. Waveform quality uses all 139 paired source IDs and independently aligns each waveform to clean source IQ by integer shift from -16 to +16 and complex gain over the central 224 samples. This is one limited metric, not proof of physical fidelity.

| Source holdout view | DURA correct | Fusion correct | Repairs | Damages | L_s DURA | L_s fusion | DURA aligned IQ MSE | Fusion aligned IQ MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 133 | 134 | 1 | 0 | 33 | 34 | 0.000038 | 0.001486 |
| practical_low_suburban | 129 | 130 | 1 | 0 | 32 | 32 | 0.027054 | 0.029849 |
| practical_mid_urban | 120 | 121 | 1 | 0 | 30 | 30 | 0.066073 | 0.068009 |
| practical_low_urban | 117 | 118 | 2 | 1 | 29 | 28 | 0.091735 | 0.094442 |

The three impaired holdout views each gain one mixed-target correct decision, while low-urban true L_s correct decisions fall from 29 to 28. Clean mixed-target accuracy rises by one, but clean-teacher agreement falls by 2 out of 139. The selected fusion's mean aligned IQ MSE rises in every view; all 139 clean samples worsen under this metric. This evidence is insufficient for target testing. Repeated runs reproduced the original result JSON; a separate regeneration reproduced it while measuring paired-source waveform quality. No U_s transmitter labels, validation IQ or test IQ were read. Prior observed Phase1 development-test findings motivated this work; future use of that test is nonblind.
