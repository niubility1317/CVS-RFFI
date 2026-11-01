# Privileged Phase1 source spectral-mixture diagnostic

This is a source-only, truth-assisted diagnostic, not a deployable network or a target-test performance result. Per record, 30 gradient steps optimize 16 frequency-band convex combinations of four fixed expert signals against its source target. The best successful iterate is selected with that target, and the original DURA decision is retained if no successful iterate is found. The procedure uses ground truth (L_s) or the clean-teacher pseudo-target (U_s) at inference and therefore cannot be deployed as evaluated. It is a finite search, not a mathematical upper bound. Tri-SIG-TIME SSL stays frozen.

The same 256 source IDs and independently seeded simulated channels are used as the earlier scalar convex mixture diagnostic. Of these, 251 are valid: 64 true L_s labels and 187 confident clean-teacher pseudo-targets. Experts saw source records in their own training, and the simulator is not validated real satellite data. The test-oracle observation had already motivated this line of research.

| Source view | DURA correct | Scalar search correct | Spectral search correct | Spectral successes beyond all 4 experts | Spectral repairs | Median relative waveform change on repairs | Direct clean IQ MSE improved on repairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| practical_low_suburban | 245 | 250 | 251 | 1 | 6 | 0.1549 | 0/6 |
| practical_mid_urban | 229 | 246 | 251 | 4 | 22 | 0.1767 | 0/22 |
| practical_low_urban | 217 | 242 | 251 | 9 | 34 | 0.1693 | 0/34 |

An independent repeat reproduced all original counts and measured waveform quality. Relative waveform change is RMS(mixture − DURA)/RMS(DURA). The clean IQ comparison uses the paired normalized source waveform and direct sample-wise complex MSE without additional alignment. All 62 spectral repairs had equal or higher MSE than DURA under this particular measure. This does not establish perceptual or physical fidelity, but it is a concrete warning that the label-optimized classification gains are not evidence of signal restoration. No test IQ or true U_s transmitter labels were read.
