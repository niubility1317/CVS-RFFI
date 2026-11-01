# Phase1 source paired-waveform carrier phase regression

Status: source-only exploratory diagnostic; no model advanced to Phase1 test. A convolutional regressor observes received IQ and frozen DURA IQ and predicts a continuous global phase correction. Source clean waveforms provide a complex correlation phase target for training and later quality measurement, but are not inference inputs. A received-to-DURA difference guard of 0.2 leaves near-clean signals unchanged. The resulting physically rotated waveform is classified by frozen Tri-SIG-TIME SSL; no transmitter label is used by the regressor.

The initial source protocol used 1,646 training IDs, 527 validation IDs and 288 holdout IDs. Training used receiver 1/3/4 days 1/2; validation used receiver 4/6 day 3; holdout used receiver 8 day 3. These source IDs were previously inspected in other phase experiments and the frozen models saw source records, so the holdout is disjoint only for regressor fitting and is not a blind population test. Simulated satellite conditions have not been validated against actual satellite data.

The original strict validation rule required nonnegative per-view change for each validation receiver and nonnegative clean-teacher change. It retained DURA. At epoch 2, the alternative had +3 net low-urban mixed-target correct decisions across validation, but receiver 4 lost one mid-urban decision. After examining these results, a supplemental replay allowed per-receiver scenario loss up to 0.5% of that receiver's source validation IDs while preserving nonnegative aggregate per receiver and clean-teacher agreement. This post-hoc rule selected epoch 2, confidence threshold 0.1; it is not independent model selection. Training losses and all validation counts in this replay match the original experiment exactly.

On the 288 held-out IDs, each view contains 280 valid mixed targets: 68 L_s true transmitter labels and 212 confident U_s clean-teacher pseudo-targets. Phase error is the clean-to-restored complex correlation-weighted mean of 1 minus the cosine of the phase difference, without a separate time-shift search; it covers all 288 records. It diagnoses global phase alignment only and does not establish full waveform fidelity.

| View | DURA correct | Corrected correct | Repairs | Damages | L_s DURA | L_s corrected | Baseline phase error | Corrected phase error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 280 | 280 | 0 | 0 | 68 | 68 | 0.00000020 | 0.00000020 |
| practical_low_suburban | 269 | 270 | 1 | 0 | 65 | 65 | 0.00144334 | 0.00146468 |
| practical_mid_urban | 261 | 262 | 1 | 0 | 63 | 63 | 0.00484817 | 0.00488702 |
| practical_low_urban | 244 | 246 | 2 | 0 | 60 | 60 | 0.00500521 | 0.00503938 |

The low-urban gain is 2/280 mixed-target decisions, with no gain among 68 true-label records. The independently recomputed fixed-model replay matches all selected holdout classification counts and computes holdout-only phase error. These results are small and exploratory; the validation-selection tolerance was chosen after observing results. All candidate waveforms and frozen classifier outputs were fixed before source transmitter labels were read. No U_s transmitter labels, validation IQ or test IQ were read. Prior observed Phase1 development-test findings motivated exploration; future use of that test is nonblind.
