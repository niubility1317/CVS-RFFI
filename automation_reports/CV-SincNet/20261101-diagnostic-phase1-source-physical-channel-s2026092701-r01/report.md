# Phase1 source physical-channel probe

Status: source-only exploratory diagnostic. This is not a Phase1 satellite-test result and cannot establish performance on real satellite data. The idea follows already observed Phase1 test expert-oracle results, so later uses of that same test are not blind.

On 512 fresh source physical IDs (128 L_s and 384 U_s), the clean source windows are processed with independent seeds through the Phase1 practical residual-channel simulator using 25 MHz sample rate, 2.462 GHz carrier, 30 dB reference SNR, no equalization, and source RX/day-specific sessions. The three source stress scenarios are low suburban, mid urban and low urban. Fixed experts and the noise-trained guarded router use frozen Tri-SIG-TIME SSL. The experts themselves trained on source records. All four-view predictions were fixed before source labels were joined. Each view has 502 valid mixed-target observations: 128 true L_s and 374 confident U_s clean-teacher pseudo-labels.

| Source view | DURA correct | Guarded router correct | Three restored-expert union | Four-expert union | Raw choices | Median DURA log energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 501 | 501 | 502 | 502 | 0 | -0.6933 |
| practical_low_suburban | 486 | 486 | 499 | 499 | 1 | -0.7034 |
| practical_mid_urban | 458 | 458 | 488 | 494 | 0 | -0.7080 |
| practical_low_urban | 428 | 428 | 472 | 478 | 2 | -0.7210 |

The current gate almost never selects raw here; at low urban it gives one repair and one damage. The fixed restored experts have additional complementary predictions, but the union is a privileged source-target oracle, not a deployable model. The DURA median log energies are approximately -0.69 to -0.72 in these four views, while the current gate requires at most -1.0. This source-only observation motivates training and validating selection directly on independently generated physical-channel source examples; it does not prove improvement on the test set.

The script records physical-ID keyed channel seeds, RX/day session construction, parameters and frozen model provenance. The JSON includes view accounting and confidence scope. No U_s transmitter labels, validation IQ or test IQ were read.
