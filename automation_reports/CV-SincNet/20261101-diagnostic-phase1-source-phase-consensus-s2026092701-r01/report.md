# Phase1 source global-phase candidate diagnostic

Status: repeated source-only exploratory diagnostic; no candidate advanced to target test. Twelve fixed global phase rotations of DURA are classified by the unchanged Tri-SIG-TIME SSL. A prior source-trained semantic network estimates a class from four frozen expert features; a practical rule selects the phase candidate with the highest frozen-classifier probability of that class only when semantic confidence is at least 0.9 and the candidate actually predicts that class. The diagnostic also evaluates phase-vote rules added after the first source results were observed. None uses transmitter labels at inference.

The source holdout has 562 physical IDs from receiver 8 on day 3; each view has 549 valid targets, including 150 L_s true labels and 399 confident U_s clean-teacher pseudo-targets. These IDs exclude the earlier 8,192-ID semantic and 4,096-ID diffusion pilot samples, but may overlap other probes. The semantic network was trained on other source IDs from the same receiver/day domains, and frozen experts saw source records. The simulator is not validated real satellite data. All fixed phase and expert decisions were produced before source labels were joined.

The phase oracle uses source targets to choose the best among 12 rotations per record. It is not deployable, not a general upper bound, and cannot be compared as a trained method. “Beyond four experts” counts oracle successes on records where none of the four original fixed expert decisions was correct. The phase-vote column selects a winning class only when at least seven rotations agree and the semantic estimate agrees with confidence at least 0.9; the 11-vote column uses phase votes alone.

| Source holdout view | DURA correct | Phase oracle | Oracle successes beyond 4 experts | Fixed semantic phase rule | 7 votes + semantic | 11 votes alone |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 548 | 549 | 0 | 546 | 547 | 544 |
| practical_low_suburban | 529 | 547 | 5 | 530 | 529 | 525 |
| practical_mid_urban | 514 | 537 | 1 | 513 | 513 | 511 |
| practical_low_urban | 475 | 531 | 17 | 477 | 478 | 478 |

In low urban the truth-dependent oracle reaches 531/549 versus DURA 475/549, including 17 successes beyond the four original experts. The fixed semantic phase rule reaches 477/549, while both the 7-vote-plus-semantic and 11-vote rules reach 478/549. The latter rules worsen at least one other view or clean classification; the 11-vote rule reduces clean correct decisions by 4. This source sample has been inspected during exploration, no independent source-validation split selected these voting thresholds, and source positive changes are too small to justify target testing. The full diagnostic was repeated with identical JSON. No U_s transmitter labels, validation IQ or test IQ were read. Prior observed Phase1 development-test findings motivated exploration, so future use of that test is nonblind.
