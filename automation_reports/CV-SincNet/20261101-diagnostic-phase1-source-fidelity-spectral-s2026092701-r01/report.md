# Phase1 source fidelity-constrained spectral mixer

Status: source-only exploratory result. No candidate passed the preselected source-validation rule, so DURA was retained; no holdout or target evaluation was performed and no candidate weights were saved. Tri-SIG-TIME SSL and the four expert models are frozen. This probe follows previously observed Phase1 test analysis; future evaluations on that development set are not blind.

The 378–128–64 source network mixes four expert signals independently in 16 complex frequency bands. Training uses the classification objective plus penalties for sample-wise complex IQ MSE to the paired normalized source clean waveform, including a penalty when the mix has greater direct MSE than DURA. This MSE has no additional alignment; the training penalty alone does not establish waveform fidelity.

The 2,048 physical source IDs split into 1,228 fitting, 410 validation and 410 holdout records, disjoint within this probe but overlapping earlier source mixer diagnostics. Fixed experts saw source records. There were 4,788 valid fitting views using L_s true labels and confident U_s clean-teacher pseudo-targets. Simulated channels are not validated real satellite data. Across the prespecified epochs and blend fractions, no configuration had a positive net result on the three source-validation views while meeting clean and per-view safeguards. One configuration tied at net zero; the fixed selection rule requires strictly positive net benefit and therefore retained DURA.

The exploratory script and complete validation output are preserved. The numerical results were not independently replayed because no candidate weights were selected. No true U_s transmitter labels, validation IQ, or test IQ were read.
