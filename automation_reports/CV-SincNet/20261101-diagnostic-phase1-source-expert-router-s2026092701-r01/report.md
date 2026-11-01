# Phase1 source-only expert-router diagnostic

Status: independently replayed source diagnostic; not a deployed model and no target-test evaluation. The frozen ground classifier is Tri-SIG-TIME SSL. This idea was developed after observing Phase1 test-set expert-oracle results, so any later evaluation on that same test set is exploratory, not blind.

The router selects among raw, DURA, boundary-student, and feature-conditioned expert predictions. Its inputs are the four six-class posterior vectors and two log energies. It is a 26–64–4 MLP, trained only with source records. Ground L_s uses actual labels; U_s uses frozen clean-teacher pseudo-labels only when confidence is at least 0.95. All expert predictions were fixed before joining source labels. The source split contains 2,456 fitting, 820 validation and 820 held-out physical IDs. All fixed experts had previously trained on these source records, so the holdout applies to the router alone. Three observations per ID use clean and two simulated source views.

Across fitting observations, 7,194 were valid and 120 offered an expert rescue of a DURA error. Validation selected epoch 5 and threshold 0, giving net +6 over the two corrupted views and no clean-teacher loss. Source holdout had 791 valid examples per view: 209 labeled L_s and 582 confident pseudo-labeled U_s. The clean-teacher totals below use all 820 IDs, not just the 791 valid mixed-target examples.

| View | DURA correct | Router correct | Repairs | Damages | Raw choices | Feature choices | Privileged four-expert union | Raw-unique correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 790 | 790 | 0 | 0 | 0 | 0 | 790 | 0 |
| 01 | 752 | 756 | 8 | 4 | 0 | 19 | 775 | 5 |
| 20 | 763 | 760 | 4 | 7 | 0 | 17 | 783 | 2 |

The union is an inaccessible oracle that knows the source target and is restricted to the four fixed predictions; it is not a realizable router result. The independently reloaded router did not select raw, and its held-out improvement did not transfer consistently between the two source views. This does not establish an effective selector and does not justify claiming a Phase1 test gain. Source clean-teacher agreement was 820→820, 771→775, and 782→777 over all 820 IDs for clean, view 01, and view 20 respectively; these figures measure agreement with the frozen teacher rather than true transmitter accuracy.

The replay reconstructs all four expert predictions, reloads the saved router without refitting, verifies every held-out paired count, and computes the source-only restricted oracle after predictions are fixed. The audit checks split, provenance, accounting and oracle identities. JSON files retain the exact model and numerical evidence. No U_s transmitter labels, validation-set IQ, or test-set IQ were read in these source diagnostics.
