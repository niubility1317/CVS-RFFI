# Phase1 source joint-phase selector

Status: repeated source-only diagnostic; no selector advanced to Phase1 test. Frozen DURA yields 12 global-phase waveforms, each classified with frozen Tri-SIG-TIME SSL. A source-trained classifier pools the mean and standard deviation of all phase embeddings and probabilities, together with input/DURA probabilities and signal energy. It can select only a real phase waveform predicted as its desired class; a fixed received-to-DURA difference gate of 0.2 leaves near-clean signals untouched. This differs from the earlier independent pointwise repair/damage classifier, but it is still a waveform candidate selector, not evidence of newly recovered signal content.

The source records overlap previously inspected phase experiments. The selector uses 1,646 source-train IDs from receivers 1/3/4 on days 1/2 and 527 validation IDs from receivers 4/6 on day 3. A separate receiver 8/day 3 source holdout has 288 IDs but no selected model was evaluated there. The frozen F and DURA experts previously saw source records. Source targets combine real L_s labels and high-confidence U_s clean-teacher pseudo-targets. Simulated satellite conditions have not been validated against a real link.

The fixed protocol tried 5 training epochs and 7 confidence thresholds, requiring nonnegative change in every impaired scenario separately for both validation receiver groups and no loss in clean-teacher agreement. It selected DURA unless a candidate had positive aggregate validation change under those guards. Across all 35 combinations the largest aggregate change was zero, so DURA remained selected. Representative mixed-target validation changes below count correct records relative to DURA; they are not test accuracies.

| Epoch | Confidence threshold | Low suburban | Mid urban | Low urban | Total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.5 | +0 | -1 | +0 | -1 |
| 2 | 0.7 | +0 | -1 | +0 | -1 |
| 2 | 0.8 | +0 | -1 | +0 | -1 |
| 2 | 0.9 | +0 | +0 | +0 | +0 |
| 5 | 0.5 | +0 | -3 | -1 | -4 |
| 5 | 0.7 | +0 | -1 | +0 | -1 |
| 5 | 0.8 | +0 | -1 | +0 | -1 |
| 5 | 0.9 | +0 | +0 | +0 | +0 |
| 10 | 0.5 | +0 | -4 | +1 | -3 |
| 10 | 0.7 | +0 | -2 | +0 | -2 |
| 10 | 0.8 | +0 | -1 | +0 | -1 |
| 10 | 0.9 | +0 | +0 | +0 | +0 |
| 20 | 0.5 | -1 | -5 | -1 | -7 |
| 20 | 0.7 | +0 | -3 | +0 | -3 |
| 20 | 0.8 | +0 | -2 | +0 | -2 |
| 20 | 0.9 | +0 | +0 | +0 | +0 |
| 30 | 0.5 | -2 | -7 | -2 | -11 |
| 30 | 0.7 | +0 | -3 | +0 | -3 |
| 30 | 0.8 | +0 | -2 | +0 | -2 |
| 30 | 0.9 | +0 | +0 | +0 | +0 |

The complete source experiment was independently rerun and produced the identical JSON. A four-record real-source smoke checked each view's pooled feature shape and candidate-class masking without labels. All phase and frozen classifier outputs were fixed before source transmitter labels were read; U_s transmitter labels, validation IQ and test IQ were not read. Earlier observed Phase1 development-test results motivated this exploration, so any future use of that test is nonblind.
