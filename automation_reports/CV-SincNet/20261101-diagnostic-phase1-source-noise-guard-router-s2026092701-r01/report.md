# Phase1 noise-guarded source router diagnostic

Status: source-only exploratory diagnostic, independently replayed. Ground classifier: frozen Tri-SIG-TIME SSL. This idea follows already observed Phase1 test expert-oracle analysis; later use of that same test cannot be described as a blind evaluation.

The source-trained 26–64–4 router selects among raw, DURA, boundary and feature-conditioned waveforms using their four classifier-posterior vectors and raw/DURA log energies. A frozen guard allows a non-DURA selection only when log DURA mean-square energy is at most -1.0 and the router's alternative probability is at least its DURA probability. The guard was selected on source validation only.

The 4,096 source IDs exclude all 512 + 4,096 + 4,096 IDs from earlier probes. Split by physical ID: 2,456 fitting, 820 validation, 820 holdout. The fixed experts had trained on source records, so holdout is only for router selection. Seven views per ID comprise clean, simulated source views 01 and 20, and each with independently seeded 15 dB and 5 dB additive Gaussian noise relative to received signal power, followed by RMS restoration. This artificial stress is not a validated satellite channel.

L_s uses true labels; U_s uses clean-teacher pseudo-labels only when confidence is at least 0.95. Expert outputs were fixed before joining source labels. The fitting set had 16,786 valid view-observations, of which 1,310 DURA errors were repairable by the fixed experts. Across 200 predeclared combinations of epoch, probability gap and energy limit, validation maximized net improvement in the two 5 dB views, requiring each unaugmented or 15 dB view to lose no more than four records and clean-teacher agreement to lose no more than 1%. Epoch 20, probability gap 0 and energy limit -1.0 were selected; the two strong-noise validation views gained 131 net predictions. Holdout has 799 valid examples per view: 189 true L_s and 610 confident U_s pseudo-targets.

| Source view | DURA correct | Router correct | Repairs | Damages | Raw expert choices |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 796 | 796 | 0 | 0 | 0 |
| 01 | 765 | 764 | 0 | 1 | 2 |
| 01_plus_15dB | 743 | 742 | 0 | 1 | 3 |
| 01_plus_5dB | 574 | 631 | 63 | 6 | 116 |
| 20 | 774 | 775 | 1 | 0 | 1 |
| 20_plus_15dB | 750 | 749 | 0 | 1 | 2 |
| 20_plus_5dB | 595 | 658 | 69 | 6 | 120 |

On holdout, the two 5 dB views gain 57 and 63 net correct predictions. Unaugmented views change by -1 and +1, the 15 dB views each lose 1, and the clean mixed-target count is unchanged. Clean-teacher agreement on all 820 physical IDs is 818→818. These results support a Phase1 exploratory test, not a claim that any satellite scene has met the 3% target.

The saved router was independently reloaded and all seven views recomputed with identical paired counts. The replay reconstructs the original per-ID seeded noise from the complete source ordering. The audit checks selection over the recorded validation grid, source split sizes, provenance flags, all view counts, repair/damage accounting and the restricted oracle identities. The oracle requires privileged target knowledge and is not deployable. No U_s transmitter labels, validation-set IQ, or test-set IQ were used.
