# Phase1 source cross-receiver and cross-day consensus controller

Status: independently replayed source-only diagnostic. The controller did not advance to Phase1 target test because its source heldout result deteriorated. Tri-SIG-TIME SSL and the four signal experts are frozen; the new model only chooses an expert whose frozen classifier agrees with its class estimate.

The controller is fitted on source receivers 1, 3 and 4 from days 1 and 2 (3,275 sampled IDs). Receiver 4 on day 3 and receiver 6 on day 3 jointly form validation (1,093); receiver 8 on day 3 forms holdout (544); 3,280 other sampled records are excluded. The holdout receiver and day are disjoint from the new controller's fitting, not from the frozen base and experts, which saw source records. Sample IDs may overlap earlier probes. The physical source simulator is not validated satellite data.

The 698–256–6 source model uses separate gradient-reversal heads for receiver (weight 0.2) and day (weight 0.1). Source validation requires nonnegative combined gain for both receiver groups, and per-group clean and per-view safeguards. Epoch 30 and confidence 0.9 were selected; validation net is +9 correct decisions over the three simulated views (+7 for receiver 4 and +2 for receiver 6). The source holdout contains 527 valid records per view: 127 true L_s labels and 400 confident U_s clean-teacher pseudo-targets.

| Source holdout view | DURA correct | Model correct | Repairs | Damages | L_s DURA | L_s model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 527 | 527 | 0 | 0 | 127 | 127 |
| practical_low_suburban | 513 | 512 | 1 | 2 | 126 | 125 |
| practical_mid_urban | 501 | 496 | 4 | 9 | 118 | 116 |
| practical_low_urban | 461 | 457 | 6 | 10 | 112 | 109 |

The three simulated holdout views change by -1, -5 and -4 mixed-target correct decisions; clean mixed-target count and clean-teacher agreement remain unchanged. An independent replay regenerated all 544 heldout source IDs and reproduced every count. No U_s transmitter labels, validation IQ or test IQ were read. Prior observed Phase1 development-test findings informed this direction; future use of that test is nonblind.
