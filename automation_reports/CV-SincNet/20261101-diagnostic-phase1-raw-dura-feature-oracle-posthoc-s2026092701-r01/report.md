# Phase1 post-hoc raw/DURA/feature decision oracle

This is a previously observed single-seed Phase1 ground-test analysis, not a deployable model or blind test. The selector is given transmitter truth and may choose only among the already fixed raw, DURA and feature-conditioned predicted labels. It cannot change the waveform or produce a new class. All fixed predictions, matching physical IDs and frozen Tri-SIG-TIME SSL ancestry were checked before the truth join.

| View | DURA accuracy | Privileged three-decision oracle |
| --- | ---: | ---: |
| clean | 86.6155% | 87.0560% |
| practical_high | 84.3435% | 90.5696% |
| practical_mid | 83.0161% | 90.3095% |
| practical_low_suburban | 81.3220% | 89.9518% |
| practical_high_urban | 83.0494% | 90.4685% |
| practical_mid_urban | 75.7060% | 88.7792% |
| practical_low_urban | 68.8012% | 85.6113% |

The raw clean ground reference is 86.5387%; the stricter 3% relative-loss target is 83.9425%. For low urban, DURA is 68.8012% and this inaccessible oracle is 85.6113%. A deployable selector restricted to these three predictions would need at least 90.07% of the oracle's available **net** gain to reach the target. Any damages increase the gross number of repairs required. This bound does not apply to networks that generate new waveforms or predictions, and does not establish that such a selector can be learned.

The other oracle counts were matched exactly against the prior independently checked four-expert analysis. Source and data caveats, method and all counts are in the accompanying script and JSON. Because test labels informed exploratory design, any subsequent same-test result remains developmental rather than blind.
