# Phase1 noise-augmented source router diagnostic

Status: source-only exploratory result, independently replayed; no new Phase1 test evaluation and no claim of a satellite gain. Ground classifier: fixed Tri-SIG-TIME SSL. This idea follows an already observed Phase1 test expert-oracle analysis; that same test is not a blind evaluation for future choices.

The 26–64–4 router selects between raw, DURA, boundary and feature-conditioned waveforms by four frozen-classifier posterior vectors and raw/DURA log energies. It is trained on a newly selected set of 4,096 source physical IDs that excludes the previous 512-record and 4,096-record probes. Split by physical ID: 2,456 fitting, 820 validation and 820 holdout. The frozen experts were themselves trained on source records, so holdout is only independent of router fitting. There are seven views per ID: clean, source 01 and 20, and each corrupted view with 15 dB or 5 dB added white Gaussian noise relative to received signal power, followed by RMS restoration. This stress is artificial and not a verified satellite channel.

L_s targets are true labels; U_s targets are clean-teacher pseudo-labels only when confidence is at least 0.95. Expert outputs were fixed before joining source labels. Training had 16,702 valid view-observations and 1,358 DURA-error observations repairable by one of the four fixed predictions. Validation selected epoch 10 and probability gap 0.2, net +136 across all six corrupted validation views with a clean-teacher guard. Each holdout view has 801 valid observations, comprising 196 true L_s and 605 confident U_s pseudo-targets.

| Source view | DURA correct | Router correct | Repairs | Damages | Raw expert choices |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 800 | 799 | 0 | 1 | 0 |
| 01 | 771 | 757 | 14 | 28 | 4 |
| 01_plus_15dB | 755 | 738 | 14 | 31 | 4 |
| 01_plus_5dB | 600 | 633 | 121 | 88 | 126 |
| 20 | 770 | 758 | 14 | 26 | 1 |
| 20_plus_15dB | 755 | 740 | 18 | 33 | 2 |
| 20_plus_5dB | 586 | 642 | 121 | 65 | 121 |

At 5 dB added noise, the router obtains +33 and +56 net correct predictions over DURA, but the two unaugmented views lose 14 and 12, and the two 15 dB views lose 17 and 15. The clean mixed-target count falls by one; clean-teacher agreement is 819→820 over all 820 holdout IDs. Therefore the network has not shown a reliable all-condition improvement. The oracle in the replay requires privileged source targets and is only an upper bound for these four fixed predictions.

The saved router was independently reloaded and all seven views recomputed, with identical paired counts. The replay reconstructs the original per-ID seeded noise from the complete source ordering. The audit verifies source split sizes, provenance flags, all seven view counts, repair/damage accounting and restricted oracle identities. No U_s transmitter labels, validation-set IQ, or test-set IQ were used.
