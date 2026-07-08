# CVS-aligned paper baseline adapters

This directory keeps only the baseline components needed by the CVS Stage2-C extension path.
The original paper-only training queues, paper parity tests, local PDF evidence paths, unresolved repro notes, and non-CVS smoke configs are intentionally excluded from the GitHub release.

Kept scope:

- `cvs_aligned/`: CVS Stage2-C protocol, metrics, and evaluation adapter.
- `protonet_cda/`: model code used by the CVS-aligned adapter.
- `feature_separation_crossrx/`: model/loss code used by the CVS-aligned adapter.
- `configs/*_cvs_stage2c_*.json`: sanitized CVS Stage2-C example configs.

Boundary:

- These files do not claim paper reproduction completion.
- Any result claim must name the CVS split, target receiver, K-shot support/query protocol, satellite/stress view, seed, and full same-row metrics.

Log separation:

- Reproduction and comparison-method logs, including RIEI, DRIFT, Fedbase, ProtoNet CDA, Feature Separation, CVCNN, and TIFS, must use `paper_reproduction/logs/`.
- Reproduction and comparison run outputs, checkpoints, and structured results must use `paper_reproduction/runs/`.
- CVS mainline, Phase1/Phase2, Stage2, spaceborne, and CV-SincNet optimization logs must stay under `logs/cvs/`.
- Do not write reproduction or comparison-method logs to the top-level `logs/` directory.
