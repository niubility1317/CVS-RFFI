# Phase1 source batch prototypes and aligned-target semantics

Status: two repeated source-only exploratory diagnostics, no model advanced to Phase1 test. In the batch experiment, each same-receiver/day/same-view group builds six frozen-base feature prototypes from high-confidence DURA outputs. A record is excluded from its own prototype. Candidate selection uses only unlabeled waveforms in that group, actual candidate F predictions and the frozen models; it needs a sufficiently large receiver-grouped batch and is transductive. It never uses paired clean or another scenario as an inference input. The output is an actual frozen-expert waveform, classified by frozen Tri-SIG-TIME SSL, and a difference gate protects near-clean input.

The fixed source selection rule tested 15 seed-confidence/margin combinations on 536 receiver 4/6 day 3 source IDs, with per-group and clean safeguards. It selected seed confidence 0.9 and margin 0.01, aggregate +4 over three source validation views. The receiver 8/day 3 source holdout contains 280 physical IDs; each view has 272 valid mixed targets, comprising 72 true L_s labels and 200 high-confidence U_s clean-teacher pseudo-targets. Source IDs overlap previous diagnostics and frozen experts saw source data. The simulated satellite is not a validated real link.

| Source view | DURA correct | Batch correct | Repairs | Damages | L_s DURA | L_s batch | DURA aligned IQ MSE | Batch aligned IQ MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 272 | 272 | 0 | 0 | 72 | 72 | 0.000041 | 0.000041 |
| practical_low_suburban | 268 | 268 | 0 | 0 | 72 | 72 | 0.021819 | 0.024533 |
| practical_mid_urban | 256 | 256 | 0 | 0 | 67 | 67 | 0.054567 | 0.055546 |
| practical_low_urban | 235 | 236 | 1 | 0 | 58 | 58 | 0.101432 | 0.102413 |

Low urban gains one pseudo-target decision and no true-label decision. Aligned IQ error is measured on all 280 records, including 8 records excluded from mixed-target classification; it increases for the impaired views. The benefit is too small and fidelity evidence unfavorable, so the method is not advanced to target evaluation.

Separately, the paired clean waveform used as an aligned training target for the prior cycle network was classified by the frozen base and compared with the original clean teacher decision. Predictions were fixed before source transmitter labels were joined. In low-urban source training, 1,576 of 1,593 valid records agreed with the original clean decision; on the 411 true L_s records, original clean correct decisions were 410 versus 401 after alignment. Validation agreement was 529/532 and source-holdout agreement 270/272. Alignment is source training machinery only and uses paired source clean IQ, not an inference input. These changes may affect feature-based supervision but do not by themselves explain all observed failures.

Both complete diagnostics were independently rerun with identical JSON. The prototype smoke used 36 real source records and checked leave-one-out algebra, minimum prototype size and all four views. All candidate and target-alignment decisions were fixed before transmitter labels; no U_s transmitter labels, validation IQ or test IQ were read. Prior Phase1 development-test results were observed, so later use of that test is nonblind.
