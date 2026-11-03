# Standalone iterative time/frequency recovery

A standalone IQ restorer performs three shared-weight refinement steps. Each step processes canonical-phase received and current IQ in time and full-frequency domains; pooled temporal self-attention adds global context. Separate zero-initialized additive IQ outputs are transformed back and accumulated; unlike purely multiplicative bounded spectral corrections, the model can synthesize corrections in either domain. Restore an initial received window without loading DURA.

Use the same Phase1 source records, three cached views, optimizer, random seed, five epochs and aligned-IQ plus frozen-reader objective as the multiscale-token model. Training uses paired source IQ and frozen reader features/pseudoclasses without TX identities, U_s truth, V or ground data. Inference takes one received IQ and fixed components; no clean reference or online target feedback. The frozen blind gate decides whether the model is applied. Predictions must be fixed before source L_s truth-last scoring and all seven ground views fixed before independent ground scoring. Explored source and simulated development test data are not independent blind or in-orbit results.

Source 20261103-phase1-iterative-timefreq-source-s2026092701-r01. Ground 20261103-phase1-iterative-timefreq-ground-test-s2026092701-r01.
