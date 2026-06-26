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
