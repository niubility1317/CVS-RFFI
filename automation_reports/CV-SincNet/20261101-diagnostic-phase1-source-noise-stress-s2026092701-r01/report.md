# Phase1 source-noise stress diagnostic

Status: source-only exploratory diagnostic. This is not a satellite-test result and does not evaluate a deployable selector. The fixed classifier is Tri-SIG-TIME SSL, and the four fixed experts are raw, DURA, boundary, and feature-conditioned. The experiment was motivated by an already observed test-set expert oracle, so later results on that same test set would be exploratory.

The sample consists of 512 source physical IDs from the router's previous heldout set. For source views 01 and 20, independent seeded Gaussian noise is added at 15 dB or 5 dB relative to the received signal power; each noised signal is rescaled to its pre-addition RMS. This is an artificial source stress, not a verified model of the satellite channel. All fixed experts had trained on source records. All seven view predictions were fixed before joining labels. The evaluation uses 492 valid observations per view: 209 true L_s labels and 283 high-confidence clean-teacher U_s pseudo-labels. No U_s transmitter labels, validation-set IQ, or test-set IQ were read.

| Source view | DURA correct | Raw correct | Three restored-expert union | Four-expert union | Raw-unique correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 465 | 414 | 478 | 479 | 1 |
| 01 + 15 dB | 451 | 398 | 474 | 476 | 2 |
| 01 + 5 dB | 351 | 265 | 405 | 456 | 51 |
| 20 | 473 | 433 | 486 | 486 | 0 |
| 20 + 15 dB | 469 | 414 | 485 | 487 | 2 |
| 20 + 5 dB | 351 | 300 | 397 | 450 | 53 |

The union requires target knowledge and cannot be deployed. The strong-noise cases contain substantially more raw-unique source examples than the unaugmented cases; they motivate, but do not prove, that noise-augmented source training can teach a router to exploit those examples. The JSON records all paired totals. The source stress script records random seeds, frozen checkpoint hashes and timing of the label join.
