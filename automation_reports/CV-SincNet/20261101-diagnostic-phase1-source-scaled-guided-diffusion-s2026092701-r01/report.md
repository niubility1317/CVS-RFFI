# Phase1 source scaled guided diffusion bridge

Status: independently replayed source-only exploratory model; not advanced to Phase1 test because low-urban and mid-urban source holdout accuracy worsened. Tri-SIG-TIME SSL and DURA remain frozen. Relative to the earlier guided pilot, the network increases from 32 to 48 channels and from three to five residual blocks, is trained for up to 20 epochs, and uses 8,192 sampled source IDs. This changes data and model scale together and is not a controlled attribution of their individual effects.

The source sample excludes IDs from the earlier 8,192-ID semantic-consensus probe and 4,096-ID diffusion pilots, but may overlap other earlier diagnostics; frozen experts saw source records. Fitting uses receivers 1, 3 and 4 on days 1 and 2 (3,196 IDs), validation uses receivers 4 and 6 on day 3 (1,072), holdout uses receiver 8 on day 3 (562), and 3,362 are excluded by the domain split. Only the new model has receiver/day-disjoint holdout. Source channels are simulated, not validated real satellite data.

The model uses four deterministic DDIM-like steps, DURA/received conditioning, a clean-protection gate, source-only 10/20/30 dB reference-SNR augmentation, clean-source teacher-logit consistency, and a clean-record reconstruction penalty. Source validation requires nonnegative combined gain in both receiver groups plus clean and per-view safeguards. It selected epoch 15 and blend 1.0, net +12 over three validation views (+5 and +7 by group). The real frozen-base source smoke confirmed finite gradients and no teacher updates.

Each holdout view has 549 valid targets: 150 L_s true and 399 confident U_s clean-teacher pseudo-targets. IQ MSE covers all 562 source records with independent complex-gain and integer-shift alignment from -16 to +16 over the central 224 samples; it is a limited metric, not proof of physical restoration.

| Source holdout view | DURA correct | Model correct | Repairs | Damages | DURA aligned IQ MSE | Model aligned IQ MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 548 | 548 | 0 | 0 | 0.000042 | 0.000043 |
| practical_low_suburban | 529 | 533 | 5 | 1 | 0.026465 | 0.026544 |
| practical_mid_urban | 514 | 510 | 3 | 7 | 0.063117 | 0.063268 |
| practical_low_urban | 475 | 471 | 5 | 9 | 0.098369 | 0.098388 |

The three impaired views change by +4, -4 and -4 mixed-target correct decisions, with clean classification and teacher-agreement counts unchanged. Mean aligned IQ MSE increases slightly in all impaired views and in clean. An independent replay regenerated all heldout source views and reproduced every accuracy count. No U_s transmitter labels, validation IQ or test IQ were read. Previously observed Phase1 development-test findings informed exploration; further use of that test is nonblind.
