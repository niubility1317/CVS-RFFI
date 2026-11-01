# Phase1 source supervised channel-estimator feasibility pilot

Status: source-only supervised simulator experiment, independently replayed; not a restoration or classification result. The estimator sees received IQ, frozen DURA output and received amplitude. Tri-SIG-TIME SSL and DURA remain frozen; no transmitter labels are used. Supervised channel targets come from the simulated 62-tap residual kernel, AGC, residual frequency and a global complex fit using paired source clean/noiseless traces. The target is an approximation of a time-varying, IQ-impaired channel, not an exact physical channel label. Simulator parameters and clean traces are unavailable to the estimator at inference.

Of 2,048 sampled source IDs, model fitting uses receivers 1, 3 and 4 on days 1 and 2 (806), validation uses receivers 4 and 6 on day 3 (271), and holdout uses receiver 8 on day 3 (139); 832 are excluded. This split is disjoint for the estimator only: frozen experts saw source records, and sample IDs may overlap earlier probes. Simulated channels are not validated real satellite data. Epoch 15 was chosen by the mean impaired-view source validation error. No U_s transmitter labels, validation IQ or test IQ were read.

The 124 real/imaginary channel components are normalized to unit Euclidean norm per record; the kernel NMSE is mean squared Euclidean error, so a zero predictor scores 1. The common template is the mean normalized kernel over all four training views. The stronger same-view template averages training kernels from the matching source view and therefore assumes the scenario is known; it is a privileged comparison, not necessarily an inference-time baseline. Frequency error measures the residual-frequency head.

| Source holdout view | IDs | Estimator kernel NMSE | Common-template NMSE | Same-view template NMSE | Frequency MAE, Hz |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 139 | 0.0212 | 0.7984 | 0.0000 | 32.08 |
| practical_low_suburban | 139 | 0.0605 | 0.4334 | 0.2080 | 132.95 |
| practical_mid_urban | 139 | 0.1189 | 0.7244 | 0.6508 | 185.82 |
| practical_low_urban | 139 | 0.1471 | 0.8692 | 0.8051 | 258.82 |

In this finite source holdout the estimator beats both templates on the simulated effective-kernel metric; for low urban it scores approximately 0.147 versus the same-view template's 0.805. Two independent regenerations reproduce the selected model's source metrics, one of which adds the scenario-specific template. This result does not establish signal restoration, physical channel estimation in real satellite data, or classifier improvement. A separate source-only integration is needed before any Phase1 test. Prior observed Phase1 development-test findings motivated this work, so future use of that test is nonblind.
