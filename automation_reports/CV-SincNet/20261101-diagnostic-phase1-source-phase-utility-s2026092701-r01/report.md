# Phase1 source phase repair/damage utility selector

Status: source-only diagnostic; rejected before Phase1 test. Twelve global carrier rotations of frozen DURA outputs are classified by the frozen Tri-SIG-TIME SSL. A pointwise MLP predicts whether choosing a phase that disagrees with DURA repairs, damages, or neutrally changes the source decision. Features contain frozen classifier embeddings and probabilities, source-input/DURA energies and phase angle. At inference only input IQ and fixed models are used; the final prediction is that of an actual phase-rotated waveform.

The 4,096 sampled IDs exclude the earlier semantic, estimator and guard pilot samples, but may overlap other source probes. The selector trains on 1,646 source IDs from receivers 1/3/4 on days 1/2; validation uses 527 from receivers 4/6 on day 3; holdout uses 288 from receiver 8 on day 3. Frozen experts saw source data, and earlier Phase1 development-test results motivated this exploration; neither source holdout nor any later test establishes satellite generalization. The simulator is not validated real satellite data. Phase candidates and their classifier outputs were fixed before source labels were joined.

There were 17,957 disagreeing source-training phase examples: 1,819 neutral, 1,134 repairs and 15,004 damages. The square-root class-weighted loss and approximate posterior correction were fixed before this run. Source validation across 30 epoch/threshold combinations, with receiver-group and clean-consistency guards, selected epoch 2, threshold zero, and net +1 over three views. This small validation gain is subject to selection noise. Each source holdout view has 280 valid mixed targets: 68 L_s true labels and 212 confident U_s clean-teacher pseudo-targets.

| View | DURA correct | Utility correct | Repairs | Damages | L_s DURA | L_s utility | Switches on valid IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 280 | 279 | 0 | 1 | 68 | 67 | 1 |
| practical_low_suburban | 269 | 267 | 0 | 2 | 62 | 61 | 3 |
| practical_mid_urban | 256 | 255 | 1 | 2 | 67 | 66 | 4 |
| practical_low_urban | 238 | 236 | 4 | 6 | 59 | 57 | 14 |

The chosen model loses 2/280 in low urban (including 2/68 true labels), 2 in low suburban, 1 in mid urban and 1 in clean. A separate fixed-model run recomputed all holdout phase outputs and selections before joining holdout source labels, and matched every holdout count and clean-teacher net. The original real-source smoke exercised all four views without labels. No U_s transmitter labels, validation IQ or test IQ were read. The observed deterioration rules out advancing this particular selector to complete Phase1 testing.
