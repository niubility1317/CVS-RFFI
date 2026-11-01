# Phase1 source pseudo-pilot channel-shape diagnostic

Status: repeated source-only diagnostic. For 280 receiver 8/day 3 source records per simulated view, the earlier periodic pseudo-pilot method estimated frequency response from received IQ and frozen DURA only. The simulator's true 62-tap residual kernel was used solely as a privileged post-hoc source reference. Shape error is normalized squared frequency-response error after independently choosing a complex gain and integer delay in [-96, 96] for each estimate and reference. The flat-response baseline receives the same oracle alignment. This tests only the simulator's convolutional residual kernel; residual frequency offsets and other receiver impairments are not included, and the metric is not a classification result.

| Source view | Pseudo-pilot mean shape NMSE | Flat-response mean shape NMSE | Pseudo-pilot better | Median true kernel peak index |
| --- | ---: | ---: | ---: | ---: |
| practical_low_suburban | 0.129277 | 0.033255 | 0 / 280 | 24 |
| practical_mid_urban | 0.207062 | 0.109980 | 5 / 280 | 24 |
| practical_low_urban | 0.302916 | 0.195655 | 4 / 280 | 24 |

The flat response is more accurate under this metric for 276/280 low-urban records, indicating that the current pseudo-pilot estimate is not a reliable approximation of the simulator kernel shape. The metric's oracle delay/gain alignment and simulator-only truth are unavailable at deployment, and this does not exclude other estimators or nonlinear channel effects.

An initial diagnostic allowed only ±16 samples and yielded errors near 1.0. It is invalid for this comparison because the true kernel's dominant tap appears at sample 24-32; its JSON is preserved as superseded evidence. The corrected range covers the 62-tap kernel and additional possible input-alignment offsets. Synthetic tests check both unknown gain and a 61-sample delay; a second complete wide-delay run matched JSON exactly. No U_s transmitter labels, validation IQ or test IQ were read. The simulator is not validated real satellite data, and frozen experts saw source records.
