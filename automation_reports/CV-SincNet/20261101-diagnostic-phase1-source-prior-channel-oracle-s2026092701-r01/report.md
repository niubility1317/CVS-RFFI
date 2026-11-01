# Privileged Phase1 source DURA-prior channel consistency

This is a small source-only oracle diagnostic, not a deployable restoration network. It uses the true simulated channel, residual frequency, AGC and source-clean RMS. It also uses clean and noiseless paired traces to fit the channel phase and to align DURA's output with the clean source signal. None of those privileged values is available at deployment. The finite reflected-boundary linear operator is verified against the simulator's forward filter before a regularized inverse is solved with DURA's aligned waveform as a prior. Frozen Tri-SIG-TIME SSL makes all predictions.

The 64 source IDs match the exact-channel boundary check; 63 are valid in each view, including 16 L_s true labels and 47 confident U_s clean-teacher pseudo-targets. Predictions were fixed before labels were joined. Four fixed regularization strengths (1, 10, 100, 1000 times the nominal noise-to-gain term) were all recorded; results are not a mathematical upper bound. Fixed experts saw source records, and simulated channels are not validated real satellite data.

| Source view | Valid | DURA correct | Privileged aligned DURA prior correct | Prior plus channel correction correct, four strengths |
| --- | ---: | ---: | ---: | --- |
| practical_low_suburban | 63 | 60 | 60 | 61 / 61 / 61 / 60 |
| practical_mid_urban | 63 | 57 | 58 | 55 / 60 / 58 / 58 |
| practical_low_urban | 63 | 53 | 55 | 52 / 57 / 54 / 55 |

In the low urban source view, strength 10 yields 57/63 versus DURA 53/63 and the privileged aligned prior 55/63. The strength-10 change consists of 5 repairs and 1 damage. This is a small mixed-target sample, with the clean-pair alignment itself accounting for part of the observed difference; it does not demonstrate an inference-time gain or physical fidelity. The identical run was repeated and the saved JSON matched. No U_s transmitter labels, validation IQ or test IQ were read. Earlier observed Phase1 test results motivated the exploration, so future development-test evaluations are nonblind.
