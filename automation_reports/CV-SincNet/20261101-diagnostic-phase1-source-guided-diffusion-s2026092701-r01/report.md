# Phase1 source guided diffusion bridge pilot

Status: independently replayed source-only exploratory model; not advanced to Phase1 test because low-urban holdout showed no net accuracy gain. Tri-SIG-TIME SSL and DURA remain frozen. The 32-channel conditional noise network uses received IQ and DURA as context and performs four deterministic DDIM-like denoising steps. A gate based only on the normalized received/DURA difference reduces changes when these signals agree.

Compared with the prior pilot, the source training uses 10, 20 and 30 dB reference-SNR augmentation on training records only. Some noisy candidates are centered on DURA rather than the paired clean signal, and the loss combines noise prediction with clean-source teacher-logit consistency and a clean-record reconstruction penalty. Frozen teacher weights never update; a real-model source smoke confirmed finite gradients through its input. The validation/holdout simulator remains at the original fixed 30 dB reference setting.

Source fitting uses receivers 1, 3 and 4 on days 1 and 2 (1,642 IDs); validation uses receivers 4 and 6 on day 3 (551); holdout uses receiver 8 on day 3 (262); 1,641 records are excluded. This split is disjoint for the new model only: frozen experts saw source records, and IDs may overlap earlier diagnostics. The simulator is not validated real satellite data. Source validation requires nonnegative combined gain for each receiver group plus clean and per-view safeguards; it selected epoch 12 and blend 0.5, net +8 across the three validation views.

Each holdout view has 258 valid targets: 76 true L_s and 182 confident U_s clean-teacher pseudo-targets. IQ MSE covers all 262 source IDs with independent complex-gain and integer-shift alignment between -16 and +16 samples on the central 224 samples. It is a limited fidelity diagnostic, not a proof of physical restoration.

| Source holdout view | DURA correct | Guided diffusion correct | Repairs | Damages | DURA aligned IQ MSE | Guided aligned IQ MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 258 | 258 | 0 | 0 | 0.000042 | 0.000042 |
| practical_low_suburban | 252 | 253 | 1 | 0 | 0.023437 | 0.023540 |
| practical_mid_urban | 238 | 239 | 1 | 0 | 0.058937 | 0.059079 |
| practical_low_urban | 225 | 225 | 1 | 1 | 0.089546 | 0.089580 |

The three simulated views change by +1, +1 and 0 mixed-target decisions; clean classification and teacher-agreement counts remain unchanged. Clean aligned IQ MSE rises from approximately 0.0000416 to 0.0000425, a smaller increase than the prior pilot's 0.0001242, though 235 of 262 clean samples still worsen under this metric. Mean aligned IQ MSE also increases slightly in each impaired view. An independent replay reproduced all counts and measured waveform quality. No U_s transmitter labels, validation IQ or test IQ were read. Earlier Phase1 development-test observations informed the work, so future evaluation on that set is nonblind.
