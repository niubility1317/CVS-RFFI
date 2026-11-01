# Phase1 source diffusion start-step selection

Status: independently repeated source-only diagnostic. It reuses the frozen 32-channel guided diffusion model and frozen Tri-SIG-TIME SSL; no model training occurs. No option advanced to target test. Motivated by [Yin et al.](https://arxiv.org/html/2503.05514v1), the experiment compares fixed starting noise levels and simple rules using the normalized received/DURA difference as a proxy for impairment. The ratio is not a calibrated SNR estimate.

The source-validation pool has 1,072 IDs from receivers 4 and 6 on day 3; holdout has 562 IDs from receiver 8 on day 3. They were sampled from source IDs excluding the earlier consensus and 4,096-ID diffusion pilot samples, but may overlap other diagnostics; frozen experts saw source records. Source channels are simulated, not validated real satellite data. All 18 combinations of six step rules and three blend values were generated and their frozen-classifier decisions fixed before source labels were joined. Source validation requires nonnegative three-view net change in both receiver groups plus clean and per-view safeguards.

Validation selected fixed start 0.1 and blend 1.0 with net +10; several rules tied on validation and the prespecified iteration order selected the first. Thus the selected rule is a fixed start, not an adaptive mapping. The matched control uses the original guided model's fixed start 0.3 and blend 0.5. The holdout has 549 valid targets per view: 150 true L_s labels and 399 confident U_s clean-teacher pseudo-targets.

| Source holdout view | DURA correct | Selected correct | Repairs | Damages | Fixed 0.3 / blend 0.5 correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 548 | 548 | 0 | 0 | 548 |
| practical_low_suburban | 529 | 531 | 3 | 1 | 531 |
| practical_mid_urban | 514 | 512 | 0 | 2 | 514 |
| practical_low_urban | 475 | 472 | 0 | 3 | 472 |

The selected option changes the three heldout views by +2, -2 and -3 mixed-target correct decisions; clean classification is unchanged. The original fixed-start control changes them by +2, 0 and -3. Full source results and all validation options are in JSON. Re-running the complete diagnostic produced identical JSON, but no separate waveform-quality evaluation was performed for this rejected option. No U_s transmitter labels, validation IQ or test IQ were read. Prior Phase1 development-test findings informed exploration, so future evaluations on that test are nonblind.
