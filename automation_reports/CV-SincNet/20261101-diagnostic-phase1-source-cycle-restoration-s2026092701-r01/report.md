# Phase1 source periodic structure and cycle projection

Status: repeated source-only exploratory diagnostics; fixed cycle candidates were not advanced to Phase1 test. Frozen DURA and Tri-SIG-TIME SSL are unchanged. The Phase1 input is an already-equalized 256-sample WiSig IQ window; the experiment does not assume or establish the type or alignment of its preamble.

On 2,048 unlabeled U_s source records from receivers 4/6/8 on day 3, normalized complex autocorrelation shows a striking 20-sample structure. Each row gives median magnitude of per-record correlation at the lag and magnitude of the mean complex correlation divided by mean magnitude. It is a second-order observation, not proof that every record contains the same known pilot or that this component carries transmitter identity.

| Lag (samples) | Median correlation magnitude | Across-record phase coherence |
| ---: | ---: | ---: |
| 20 | 0.643911 | 0.996713 |
| 40 | 0.679087 | 0.996772 |
| 60 | 0.637700 | 0.993734 |
| 80 | 0.646947 | 0.991859 |
| 100 | 0.613599 | 0.987869 |
| 120 | 0.552136 | 0.982008 |

In the simulated low-urban view, at lag 40 the mean absolute error against paired source clean correlation is 0.177901 for received IQ and 0.032673 after DURA. A second full run matched the complete correlation JSON; a synthetic constant-wave test verified the estimator. This finding motivated a phase-drift-aware 20-sample projection: estimate per-period phase from DURA, average each of 20 residue classes after de-rotation, then interpolate or extrapolate between that periodic component and the original DURA waveform. Six fixed coefficients (-0.3, -0.1, -0.03, 0.03, 0.1, 0.3) were evaluated; a 0.2 received-versus-DURA difference gate preserves near-clean signals. The phase-drift projection and four real source views passed smoke checks.

The projection probe used 536 source validation IDs on receiver 4/6 day 3 and 280 source holdout IDs on receiver 8 day 3. Its sampled IDs exclude the named earlier semantic, channel-estimator, clean-guard and phase-utility samples, but may overlap other probes. Frozen experts saw source records. Candidates were fixed before source transmitter labels were joined. Selection used real L_s labels and high-confidence U_s clean-teacher pseudo-targets. Per-view and receiver-group safeguards were fixed before the source score. All six candidates had negative aggregate validation changes, so the prescribed rule retained DURA and no candidate was selected for holdout or test.

| Coefficient | Low suburban | Mid urban | Low urban | Total |
| ---: | ---: | ---: | ---: | ---: |
| -0.30 | -19 | -25 | -32 | -76 |
| -0.10 | -6 | -9 | -16 | -31 |
| -0.03 | -1 | -1 | +0 | -2 |
| +0.03 | +2 | +1 | -4 | -1 |
| +0.10 | +1 | +1 | -4 | -2 |
| +0.30 | -19 | -21 | -30 | -70 |

A separate diagnostic regenerated candidate outputs and matched every source-validation per-view change. It then calculated label-assisted oracles on the source holdout. In low urban, the baseline was 229/272 mixed-target correct (56/72 true L_s); using source targets to select the best among the six fixed candidates and DURA could reach 243/272 (60/72 L_s). Requiring an altered candidate also to reduce its paired clean-IQ MSE after independent ±16-sample shift and complex-gain alignment lowers the ideal count to 230/272 (56/72 L_s). Thus 13 of the 14 potential mixed-target repairs have no fidelity-improving candidate under this single metric. The oracle is impossible at inference, is not a general upper bound on new waveforms, and is not a Phase1 test result. This supports caution about directly suppressing the nonperiodic residual, not a general claim about the usefulness of periodic structure.

All source labels were joined after fixed candidate predictions; no U_s transmitter labels, validation IQ or test IQ were read. Clean source IQ was used to simulate and measure source samples only. The period was chosen after observing the earlier unlabeled source probe, and prior Phase1 development-test results had already been observed. The simulator is not validated against a real satellite link.
