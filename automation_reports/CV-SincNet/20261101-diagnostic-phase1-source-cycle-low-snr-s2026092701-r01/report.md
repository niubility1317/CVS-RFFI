# Phase1 source low-SNR cycle network and blind noise proxy

Status: source-only exploratory network and proxy; neither advanced to Phase1 test. The trained cycle-constrained network modifies only DURA's nonperiodic component under its estimated 20-sample phase coordinates. Its dilated 2D convolutions expand the cycle-grid receptive field. Source training combines paired clean-IQ residual error, correction regularization, frozen Tri-SIG-TIME SSL feature similarity and frozen-base class-probability distillation. Simulated added-noise SNR below 10 dB receives 4x training weight: 1,649 of 4,878 impaired training examples. Neither clean source IQ nor simulator SNR is an inference input; DURA and the base classifier remain frozen.

Source selection covered 1,626 training IDs, 536 receiver 4/6 day 3 validation IDs and 280 receiver 8/day 3 holdout IDs. These IDs overlap other inspected cycle experiments; frozen experts saw source data. Checkpoints and strengths were scored on validation with group and clean guards; epoch 2/strength 0.25 was selected at total +1. Each source holdout view has 272 valid mixed targets: 72 real L_s and 200 confident U_s clean-teacher pseudo-targets. MSE uses all 280 pairs and separately optimizes ±16 sample shift and complex gain.

| View | DURA correct | Network correct | Repairs | Damages | L_s DURA | L_s network | DURA aligned MSE | Network aligned MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 272 | 272 | 0 | 0 | 72 | 72 | 0.00004085 | 0.00004085 |
| practical_low_suburban | 266 | 266 | 0 | 0 | 72 | 72 | 0.02705416 | 0.02706128 |
| practical_mid_urban | 256 | 255 | 2 | 3 | 64 | 64 | 0.06029429 | 0.06030350 |
| practical_low_urban | 229 | 232 | 3 | 0 | 56 | 56 | 0.10211454 | 0.10212277 |

Low urban gains 3 pseudo-target decisions with no true-label gain; mid urban loses one and mean IQ error rises slightly in every impaired view. A fixed-model replay matched all holdout counts and MSE after deliberately replacing paired-clean training targets and simulator SNR metadata with dummy values in inference. Four-record real-source smoke verified finite gradients and frozen base parameters. These small mixed results are insufficient to advance the network.

Separately, a blind proxy estimates relative noise by fitting DURA to received IQ using a complex gain and a ±64-sample shift over the central 128 samples, then converting the residual-energy fraction to a dB-like score. It uses no clean IQ or simulator SNR. On an already observed receiver 8/day 3 source sample of 538 IDs, low-urban proxy rank correlation with simulated added-noise SNR was 0.9437. The 10 dB proxy threshold flags all 300 simulated-SNR-below-10 dB records and 7 of 238 higher-SNR records; this threshold was chosen after inspecting that same source sample and is not an independent test.

An initial proxy used only ±16 samples and is superseded because simulator kernels have dominant delay 24-32; its JSON is retained for traceability. The corrected proxy passed a synthetic 61-sample-shift test and a full repeat with identical JSON. Simulated SNR and source clean IQ were used only to train or score diagnostics as stated. Candidate predictions were fixed before source transmitter labels; no U_s transmitter labels, validation IQ or test IQ were read. The simulator is not validated against a real satellite link, and prior Phase1 development-test findings were observed.
