# Phase1 fresh-source blind-gated low-SNR network

Status: fixed fresh-source evaluation and metadata-isolation replay; no target test. The source-trained cycle residual network at epoch 2/strength 0.25 and the blind proxy threshold 10 dB were chosen after earlier source observations, then the exact script, model hashes, threshold and seeds were recorded before this fresh run. The per-record proxy fits DURA to received IQ with complex gain and ±64-sample shift on the central 128 samples; a low proxy enables the network, while the original difference guard protects clean signals. Inference uses only received IQ, frozen DURA, the fixed network and frozen Tri-SIG-TIME SSL. It does not use paired clean IQ, simulator SNR, transmitter labels or other records.

The 8,192 newly sampled source IDs exclude the entire 4,096-ID cycle-network sample and 8,192-ID SNR/proxy sample; channel and receiver-hardware random seeds are new. Other past diagnostics may overlap and frozen experts saw source records. Receiver 4/6 day 3 yields 1,117 validation IDs, receiver 8 day 3 yields 505 holdout IDs. Each holdout view has 490 valid targets: 125 true L_s and 365 confident U_s clean-teacher pseudo-targets. Source clean is used to simulate and assess waveform quality only. This is not a blind Phase1 test or real satellite validation.

| Holdout view | DURA correct | Network without SNR gate | Blind-gated correct | L_s DURA | L_s gated | DURA aligned MSE | Gated aligned MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 489 | 489 | 489 | 124 | 124 | 0.00005299 | 0.00005299 |
| practical_low_suburban | 475 | 475 | 475 | 121 | 121 | 0.02709373 | 0.02709372 |
| practical_mid_urban | 459 | 461 | 461 | 112 | 113 | 0.06086725 | 0.06087054 |
| practical_low_urban | 431 | 431 | 431 | 110 | 109 | 0.09955235 | 0.09955756 |

In low urban the blind-gated network is 431/490 versus DURA 431/490; its true-label subset falls from 110/125 to 109/125. In source validation low urban it falls from 967/1102 to 961/1102 (true-label 250/283 to 245/283). The source holdout mid-urban view gains two mixed-target decisions including one true label, but low-urban waveform MSE rises slightly. This network is not advanced to full Phase1 evaluation.

The blind proxy itself generalizes within this simulator sample: on low-urban source validation it marks 624/624 records with simulated added-noise SNR below 10 dB, plus 32 false positives among 493 higher-SNR records; on receiver 8 holdout it marks 297/297 low-SNR records plus 6 false positives among 208 higher-SNR records. These counts use simulator SNR solely as post-hoc reference; the 10 dB gate was chosen from an earlier source sample and the simulator is not validated as a real satellite link.

An independent rerun replaced all simulator SNR metadata with zeros and retained dummy paired-clean training targets. All classification, gate and waveform-quality results matched; only the reference-only SNR confusion summaries changed. The fixed hashes and exclusion assertions were checked, and a four-record real-source smoke checked clean gating and confusion accounting. All candidate waveforms were fixed before source transmitter labels were read; no U_s transmitter labels, validation IQ or test IQ were read. Earlier Phase1 development-test results had been observed, so subsequent use of that test is nonblind.
