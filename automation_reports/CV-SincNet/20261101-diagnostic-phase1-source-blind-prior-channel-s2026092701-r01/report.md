# Phase1 source blind prior-guided channel diagnostic

This source-only exploratory method fits a regularized complex FIR from the frozen DURA output to the received IQ of the same record, then solves one reflected-boundary linear inverse with that DURA waveform as the prior. At inference the computation uses only received IQ and frozen DURA; it does not use simulator channel parameters, paired clean IQ or labels. Frozen Tri-SIG-TIME SSL evaluates the result. The per-record deterministic solve estimates channel coefficients, but makes no gradient or model-weight updates.

The initial source experiment used 256 IDs (251 valid: 64 L_s true, 187 confident U_s clean-teacher pseudo-targets) and nine estimated taps. It was independently repeated with identical JSON. On the low urban view DURA scored 217/251; the four correction strengths scored 125, 179, 217 and 217. On low suburban and mid urban, no strength improved over DURA consistently. All source predictions were fixed before joining source targets.

The following comparison uses the same 64-ID subset for both 9- and 62-tap estimates. Each view has 63 valid targets: 16 true L_s and 47 U_s pseudo-targets. The four values use residual multipliers 0.1, 1, 10 and 100, in that order. The fit ratio is median mean-square residual divided by DURA input power; it is not a calibrated noise or channel estimate.

| Source view | Valid | DURA | 9-tap correct, 4 strengths | 62-tap correct, 4 strengths | 9-tap median fit ratio | 62-tap median fit ratio |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| clean | 63 | 63 | 63 / 63 / 63 / 63 | 63 / 63 / 63 / 63 | 0.0003 | 0.0005 |
| practical_low_suburban | 63 | 60 | 41 / 46 / 59 / 60 | 59 / 59 / 60 / 61 | 0.5194 | 0.0034 |
| practical_mid_urban | 63 | 57 | 35 / 46 / 56 / 57 | 55 / 59 / 59 / 57 | 0.5687 | 0.0039 |
| practical_low_urban | 63 | 53 | 34 / 41 / 53 / 53 | 50 / 51 / 52 / 51 | 0.6334 | 0.1017 |

Although 62 taps substantially reduce this in-record fit residual, they do not improve low-urban source accuracy on the matched subset. These are exploratory source diagnostics without an independent validation split; IDs overlap earlier probes, fixed experts saw source records, and the 62-tap run has not been independently repeated. No option has advanced to test. The simulator is not validated as real satellite data, and previous Phase1 development-test analysis motivated this exploration. No true U_s transmitter labels, validation IQ or test IQ were read.
