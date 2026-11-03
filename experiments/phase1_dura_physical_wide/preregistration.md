# Wider continuous DURA/physical posterior fusion

Use the same architecture, source fit sample, view generation, frozen DURA and physical posterior, reader-guided objective, optimization budget and seed as the registered phase1_dura_physical_fusion r02, but double the final complex fusion correction so each real and imaginary coefficient can approach magnitude 1 instead of .5. The output remains one IQ waveform and the inactive gate returns DURA. The wider variant is independently initialized and trained; no modification of the already running narrow experiment.

This variant is prompted by a previously explored source-only pseudo-class interpolation diagnostic. That diagnostic used 26 selected source views and fixed global scalar coefficients; it cannot establish performance or an upper bound for this time-varying model. Training reads paired source IQ and frozen-reader pseudo features but no TX identity, U_s truth, V or ground data. Inference uses one current IQ and fixed models without target feedback. Fix source and all seven Phase1 ground predictions before authorized truth-last scoring and independent audits. Explored simulated data are not a blind test or in-orbit result.

Source experiment: 20261103-phase1-dura-physical-wide-source-s2026092701-r01. Ground experiment: 20261103-phase1-dura-physical-wide-ground-test-s2026092701-r01.
