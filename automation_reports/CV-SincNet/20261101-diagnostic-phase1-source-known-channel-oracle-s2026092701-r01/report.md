# Phase1 source known-channel inverse diagnostic

This source-only diagnostic asks whether a simple inverse is promising even with unavailable simulator internals. It uses the simulator's true 62-tap residual kernel, residual carrier frequency, AGC and paired-clean RMS. A second variant also estimates a complex phase/gain using clean and noiseless paired traces. None of these privileged inputs is available at deployment; the four fixed regularization strengths are shown individually in JSON. Selecting their best result after viewing labels would add further oracle information. This finite inverse is not an upper bound. Tri-SIG-TIME SSL and DURA are frozen.

The 256 source IDs are the same as the earlier spectral search; each view has 251 valid records, including 64 L_s true labels and 187 U_s confident clean-teacher pseudo-targets. Predictions were fixed before joining source targets. The simulator is not validated real satellite data and experts saw source records during their own training.

| Source view | Raw correct | DURA correct | Best of 4 FFT inverses | Best of 4 FFT inverses with privileged phase fit |
| --- | ---: | ---: | ---: | ---: |
| practical_low_suburban | 224 | 245 | 226 | 226 |
| practical_mid_urban | 189 | 229 | 202 | 200 |
| practical_low_urban | 157 | 217 | 152 | 164 |

The best-of-four columns use source labels for this summary and are not deployable. In low urban even the privileged phase-fitted variants reach at most 164/251, versus DURA 217/251. A repeated run reproduces all counts. On all 256 records, the mean aligned paired-clean IQ MSE for the best-noise-regularization inverse (multiplier 1) is about 0.3198 versus 0.0575 for DURA; alignment independently fits integer shifts from -16 to +16 and complex gain over the central 224 samples. The phase fit cannot improve this gain-invariant metric by construction.

The inverse uses a frequency-domain reflected-padding approximation, while the simulator applies finite linear convolution with reflected prehistory. Its poor result does not rule out a better boundary-aware or nonlinear channel model. No true U_s transmitter labels, validation IQ, or test IQ were read. Earlier Phase1 test analysis motivated this experiment, so further test development is not blind.
