# Standalone iterative time/frequency recovery with expanded source fit

Use the identical architecture, source simulation views, optimizer, random seed, frozen Tri-SIG-TIME SSL, blind estimator, gate and aligned source objective as the 8,192-record standalone iterative model. Construct a deterministic nested fit set of 32,768 eligible source records: include the original 8,192 and add 24,576 from eligible remaining source records, excluding the frozen physical-prior fit and established source holdouts. Independently train five epochs on three source views; this changes both total optimization steps and source record coverage and is not a pure sample-coverage ablation.

Training uses paired Phase1 source IQ and frozen reader features without reading TX identity, U_s truth, V or ground data. Inference uses one received IQ and fixed components, no DURA or clean reference. Fix predictions before truth-last source scoring and all seven ground views before independent scoring. Source and ground data have previously been explored; no blind-test or in-orbit claim.

Source 20261103-phase1-iterative-large-source-s2026092701-r01. Ground 20261103-phase1-iterative-large-ground-test-s2026092701-r01.
