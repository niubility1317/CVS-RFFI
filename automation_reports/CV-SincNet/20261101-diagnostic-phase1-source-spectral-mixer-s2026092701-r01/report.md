# Phase1 source frequency-band mixture diagnostic

Status: independently replayed source-only exploratory diagnostic; not advanced to target test because holdout benefit is inconsistent. Ground classifier Tri-SIG-TIME SSL and all four experts are frozen. This idea follows previously observed Phase1 test results; any later use of those test data would not be blind.

The 378–128–64 network emits four nonnegative expert weights for each of 16 frequency bands. Weights sum to one per band. It mixes the corresponding complex FFT coefficients, then inverts the FFT; the final waveform is 25% mixture and 75% DURA according to source validation. This creates a new waveform, but frequency-domain convexity alone does not prove that the output is a physically faithful restoration.

The 2,048 source IDs are disjoint across fitting (1,228), validation (410) and holdout (410) for this probe. These IDs overlap the earlier scalar convex mixer probe; experts saw source records during their own training. Source training used 4,784 valid view-observations, 20 epochs; source validation selected epoch 10 and coefficient 0.25 (net +10 across three simulated views). The source simulator is not validated real satellite data.

Each holdout view contains 400 valid observations: 101 true L_s labels and 299 confident U_s clean-teacher pseudo-targets. Expert outputs were fixed before source labels were joined.

| Source view | DURA correct | Mixer correct | Repairs | Damages | L_s DURA | L_s mixer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 400 | 400 | 0 | 0 | 101 | 101 |
| practical_low_suburban | 392 | 390 | 3 | 5 | 101 | 101 |
| practical_mid_urban | 375 | 375 | 5 | 5 | 93 | 94 |
| practical_low_urban | 346 | 347 | 10 | 9 | 87 | 89 |

The simulated holdout views change by -2, 0 and +1 mixed-target correct decisions; clean mixed-target count is unchanged and clean-teacher agreement changes by -1 among all 410 holdout IDs. This small, inconsistent source result does not support a test launch or the 3% performance goal. The fixed model and regenerated heldout views independently replayed every count. No U_s transmitter labels, validation IQ or test IQ were read.
