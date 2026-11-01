# Phase1 source semantic-gated diffusion diagnostic

Status: repeated source-only diagnostic; not advanced to target test. It combines the fixed guided-diffusion pilot with the fixed semantic-consensus model. On each received signal the generated waveform is used only if its frozen Tri-SIG-TIME SSL class differs from DURA and matches the high-confidence semantic estimate. Otherwise DURA is retained. No model weights or target-time labels are updated; the two original source-trained model weights and frozen experts are preserved in this directory for reproducibility.

Source validation has 1,072 IDs from receivers 4 and 6 on day 3; holdout has 562 IDs from receiver 8 on day 3. These IDs exclude samples used by the prior 8,192-ID semantic-consensus and 4,096-ID diffusion pilots, but may overlap other source probes; frozen ground classifier and experts saw source records. The simulator is not validated real satellite data. All source predictions for DURA, guided diffusion, semantic consensus and the gate were fixed before source labels were joined. Source validation selects among six confidence thresholds, requiring nonnegative net gain in each validation receiver group plus clean and per-view safeguards; it selected 0.7, net +4 over three views.

The holdout has 549 valid targets per view: 150 L_s true labels and 399 confident U_s clean-teacher pseudo-targets. The two controls are the ungated guided diffusion output and the independently fixed semantic-consensus expert selector.

| Source holdout view | DURA correct | Gate correct | Gate repairs | Gate damages | Ungated diffusion correct | Semantic control correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 548 | 548 | 0 | 0 | 548 | 548 |
| practical_low_suburban | 529 | 531 | 2 | 0 | 531 | 530 |
| practical_mid_urban | 514 | 515 | 1 | 0 | 514 | 514 |
| practical_low_urban | 475 | 473 | 0 | 2 | 472 | 475 |

The gate changes the three heldout views by +2, +1 and -2 mixed-target decisions. In low urban the ungated diffusion changes by -3 and semantic control by 0. Clean classification is unchanged. Full rerun produced identical JSON; no waveform fidelity evaluation was performed for this rejected option. No U_s transmitter labels, validation IQ or test IQ were read. Prior Phase1 development-test results informed this research; further evaluations on that set are nonblind.
