# Phase1 complex gain-invariant multiresolution restoration

Matched source ablation of the complex-mask network: optimize a differentiable optimal-complex-gain IQ error on the central samples, retain a deviation penalty, and choose the frozen model by the same ±64-sample aligned paired-clean IQ MSE used for post-selection assessment. Selection remains source-only, before F weights or TX labels are read. The network remains a bounded two-resolution complex spectral filter with a fixed blind gate.

Prior Phase1 development and the original source outcomes informed this design. No independent blind-test or real-satellite claim is justified. If a candidate passes fixed source selection, the pipeline seals all seven Phase1 views before independent scoring and same-record DURA comparison.
