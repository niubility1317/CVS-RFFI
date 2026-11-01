# Phase1 source simulated SNR and classification errors

Status: repeated source-only post-hoc stratification, not an inference algorithm. On 538 receiver 8/day 3 source records, frozen Tri-SIG-TIME SSL classified the simulated received IQ and frozen DURA output before source transmitter labels were joined. Low-urban source targets are 516 valid records: 150 true L_s labels and 366 confident U_s clean-teacher pseudo-targets. F and DURA saw source records, and IDs may overlap earlier diagnostics. This source subset cannot establish Phase1 test or real satellite performance.

SNR is the simulator's signal-to-added-noise power before AGC, a privileged post-hoc measurement rather than total or externally measured SNR. The model never receives this value for inference. The DURA waveform MSE independently optimizes complex gain and ±16-sample shift against paired source clean IQ, over all records in each bin. Bins are lower-inclusive and upper-exclusive.

| Simulated added-noise SNR (dB) | All IDs | Valid targets | Received correct | DURA correct | L_s DURA correct | DURA aligned IQ MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| -∞ to 0 | 88 | 82 | 26 | 56 | 16 / 22 | 0.274109 |
| 0 to 5 | 145 | 140 | 62 | 117 | 35 / 40 | 0.151384 |
| 5 to 10 | 67 | 62 | 36 | 53 | 18 / 22 | 0.093318 |
| 10 to 15 | 20 | 19 | 15 | 19 | 6 / 6 | 0.042580 |
| 15 to 20 | 36 | 35 | 33 | 35 | 10 / 10 | 0.020908 |
| 20 to 25 | 120 | 118 | 115 | 118 | 28 / 28 | 0.010216 |
| 25 to 30 | 62 | 60 | 58 | 60 | 22 / 22 | 0.006978 |

All 58 DURA errors among the 516 valid low-urban targets occurred below 10 dB (226/284 correct); all 232 valid records at 10 dB or above were correct. Among the 150 true L_s labels, all 15 DURA errors also occurred below 10 dB. The clean-teacher pseudo-target construction may inflate apparent accuracy, the source subset is small, and the association does not prove the same distribution or causal effect on the previously observed development test.

Simulator lock-state stratification gives DURA 172/223 valid decisions for unlocked, 71/78 for degraded and 215/215 for locked. Lock is simulator metadata, not a measured deployment input. A four-record source smoke checked all views and bin accounting; a second full run matched JSON. No U_s transmitter labels, validation IQ or test IQ were read. The simulator is not validated against a real satellite link.
