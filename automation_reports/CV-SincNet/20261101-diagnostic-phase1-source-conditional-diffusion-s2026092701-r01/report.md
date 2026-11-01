# Phase1 source conditional diffusion restoration pilot

Status: independently replayed source-only pilot; not advanced to Phase1 test because heldout low-urban accuracy did not improve and clean waveform fidelity worsened. The ground classifier is frozen Tri-SIG-TIME SSL, and DURA is frozen. The small conditional noise network receives a noisy candidate, the DURA output, the received IQ and a noise level; it is trained on paired clean source IQ with an epsilon-prediction objective. Deterministic four-step DDIM-like sampling starts from DURA, then a source-validated blend combines the generated waveform with DURA. This is an exploratory implementation, not a reproduction of published work; see [Yin et al.](https://arxiv.org/abs/2503.05514) and [Song et al.](https://arxiv.org/abs/2010.02502).

Source fitting uses receivers 1, 3 and 4 on days 1 and 2 (1,642 records). Validation uses receivers 4 and 6 on day 3 (551), holdout uses receiver 8 on day 3 (262), and 1,641 other sampled records are excluded. This separation applies to the new network; the frozen experts saw source records and IDs may overlap previous probes. Source simulations are not validated satellite data. Validation requires nonnegative combined gain in each receiver group plus clean and per-view safeguards. It selected epoch 12 and blend 0.5, with net +3 mixed-target correct decisions over three views.

Each holdout view has 258 valid records: 76 L_s true labels and 182 confident U_s clean-teacher pseudo-targets. IQ MSE uses all 262 heldout paired source records and independently optimizes a complex gain and integer shift from -16 to +16 over the central 224 samples; this is a diagnostic metric, not a proof of physical fidelity.

| Source holdout view | DURA correct | Diffusion correct | Repairs | Damages | Aligned IQ MSE DURA | Aligned IQ MSE diffusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 258 | 258 | 0 | 0 | 0.000042 | 0.000124 |
| practical_low_suburban | 252 | 253 | 1 | 0 | 0.023437 | 0.023445 |
| practical_mid_urban | 238 | 239 | 1 | 0 | 0.058937 | 0.058971 |
| practical_low_urban | 225 | 224 | 1 | 2 | 0.089546 | 0.089542 |

The three heldout simulated views change by +1, +1 and -1 mixed-target decisions; clean classification count and clean-teacher agreement remain unchanged. Mean clean aligned IQ MSE increases, and all 262 clean records have higher error under this metric. An independent replay regenerated all heldout source outputs before joining labels and reproduced every classification count. No U_s transmitter labels, validation IQ or test IQ were read. Previously observed Phase1 development-test results informed this exploration, so future evaluations on that test are nonblind.
