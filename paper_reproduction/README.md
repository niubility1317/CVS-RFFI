# Paper Reproduction Modules

This directory keeps paper-faithful reproduction components and the separate CVS Stage2-C extension path.
The original paper-only training queues, paper parity tests, local PDF evidence paths, unresolved repro notes, and non-CVS smoke configs are intentionally excluded from the GitHub release.

Paper-faithful scope:

- `protonet_cda/`: model code used by the CVS-aligned adapter.
- `feature_separation_crossrx/`: model/loss code used by the CVS-aligned adapter.
- `receiver_agnostic_twostage_uda/`: paper-faithful Bao et al. GLOBECOM 2023 components for DANN, LMMD, uncertainty fine-tuning selection, and dry-run protocol validation.
- `mitigating_receiver_impact_da/`: paper-faithful Liu Yang et al. IEEE IoT Journal 2024 components for DV-KL domain alignment, CPL adaptive pseudo-labeling, class weighting, GAD min-max training, and dry-run protocol validation.
- `configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json`: paper-faithful dry-run configuration for the closed-set WiSig ManySig receiver-ratio matrix.
- `configs/mitigating_receiver_impact_da_manysig_paper_faithful.json`: paper-faithful dry-run configuration for the IoTJ 2024 closed-set WiSig ManySig cross-receiver/cross-day matrix.

CVS extension scope:

- `cvs_aligned/`: CVS Stage2-C protocol, metrics, and evaluation adapter.
- `configs/*_cvs_stage2c_*.json`: sanitized CVS Stage2-C example configs.

Boundary:

- These files do not claim paper reproduction completion.
- Any result claim must name the CVS split, target receiver, K-shot support/query protocol, satellite/stress view, seed, and full same-row metrics.
- Receiver-Agnostic Two-stage UDA dry-runs and Mitigating Receiver Impact DA dry-runs are paper-faithful closed-set cross-receiver checks only. They are not CVS Stage2-A/B/C, satellite/LEO deployment, open-set, or new-class registration evidence.

Log separation:

- Reproduction and comparison-method logs, including RIEI, DRIFT, Fedbase, ProtoNet CDA, Feature Separation, CVCNN, and TIFS, must use `paper_reproduction/logs/`.
- Reproduction and comparison run outputs, checkpoints, and structured results must use `paper_reproduction/runs/`.
- CVS mainline, Phase1/Phase2, Stage2, spaceborne, and CV-SincNet optimization logs must stay under `logs/cvs/`.
- Do not write reproduction or comparison-method logs to the top-level `logs/` directory.
