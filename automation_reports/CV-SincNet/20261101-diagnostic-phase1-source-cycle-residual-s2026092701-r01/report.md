# Phase1 source cycle-residual paired diagnostic

Status: repeated source-only analysis, not an inference method. The 20-sample, phase-drift-aware cycle projection was discovered from earlier unlabeled source records. The complementary nonperiodic residual accounts for 57.11% of the paired clean source waveform energy in this 280-record receiver 8/day 3 sample. This is one energy measure on this sample, not an assertion that all nonperiodic energy is transmitter fingerprint.

For each simulated view, DURA and received IQ were independently aligned to paired clean source IQ by a source-only best shift between -16 and +16 and complex gain. The clean, aligned DURA and aligned received waveforms were decomposed using their own phase-drift-aware cycle projection. The residual-error metric is the sum of squared complex residual error divided by total clean residual energy. For a diagnostic oracle only, each record's received/DURA residual interpolation coefficient is optimized using its paired clean residual and constrained to [0, 1]. Both clean alignment and this choice are unavailable during deployment; the oracle is not an upper bound on nonlinear or differently conditioned restoration.

| Source view | Median DURA/clean residual correlation | DURA relative residual MSE | Ideal paired blend relative residual MSE | Ideal error reduction | Records improving >1% |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.999980 | 0.000248 | 0.000000 | 100.000% | 280 |
| practical_low_suburban | 0.987241 | 0.058801 | 0.058405 | 0.674% | 39 |
| practical_mid_urban | 0.979300 | 0.118239 | 0.116977 | 1.068% | 42 |
| practical_low_urban | 0.911123 | 0.188143 | 0.186656 | 0.790% | 50 |

For low urban the DURA/clean residual correlation median is 0.9111 and even this privileged per-record interpolation reduces aggregate residual MSE by only 0.790%. This limits the case for directly mixing received and DURA residuals; it does not exclude learning a nonlinear estimator or recovering other components. The source sample and frozen experts have appeared in other diagnostics. The simulator is not validated real satellite data. No U_s transmitter labels, validation IQ or test IQ were read. Synthetic complex-gain alignment checks passed, and an independent full replay matched the result JSON exactly.
