# Paper Reproduction Modules

This directory keeps paper-faithful reproduction components and the separate CVS Stage2-C extension path.
The original paper-only training queues, paper parity tests, local PDF evidence paths, unresolved repro notes, and non-CVS smoke configs are intentionally excluded from the GitHub release.

Paper-faithful scope:

- `protonet_cda/`: model code used by the CVS-aligned adapter.
- `feature_separation_crossrx/`: model/loss code used by the CVS-aligned adapter.
- `receiver_agnostic_twostage_uda/`: paper-faithful Bao et al. GLOBECOM 2023 components for DANN, LMMD, uncertainty fine-tuning selection, and dry-run protocol validation.
- `mitigating_receiver_impact_da/`: paper-faithful Liu Yang et al. IEEE IoT Journal 2024 components for DV-KL domain alignment, CPL adaptive pseudo-labeling, class weighting, GAD min-max training, and dry-run protocol validation.
- `dadda_cross_receiver/`: paper-faithful Feng et al. IEEE Internet of Things Journal 2025 components for closed-set cross-receiver DADDA with a 1-D ResNet18-style `G_f` approximation, a 1-D multiscale `G_m` approximation, MMD, LMMD, and dynamic adaptive factor.
- `orthogonal_incremental_sei/`: paper-faithful closed-set FSCIL-SEI components for orthogonal pseudo targets, base-stage CE/contrastive/center losses, incremental classifier-weight calibration, and A/H/F metrics.
- `csil_class_incremental_iot/`: paper-faithful ADS-B class-incremental CSIL components aligned to the official MATLAB repository, including shifted zero-bias cosine fingerprints, channel expansion, KD/EWC helpers, masked SGD update, and dry-run protocol validation.
- `configs/receiver_agnostic_twostage_uda_manysig_paper_faithful.json`: paper-faithful dry-run configuration for the closed-set WiSig ManySig receiver-ratio matrix.
- `configs/mitigating_receiver_impact_da_manysig_paper_faithful.json`: paper-faithful dry-run configuration for the IoTJ 2024 closed-set WiSig ManySig cross-receiver/cross-day matrix.
- `configs/dadda_cross_receiver_manysig_paper_faithful.json`: paper-faithful dry-run configuration for the DADDA closed-set WiSig ManySig receiver-transfer matrix.
- `configs/orthogonal_incremental_sei_smoke.json`: synthetic dry-run configuration for the orthogonal-space FSCIL-SEI wiring check.
- `configs/orthogonal_incremental_sei_wisig.json`: paper-protocol WiFi/WiSig configuration skeleton for closed-set FSCIL; it is not CVS Stage2 evidence.
- `configs/csil_adsb_paper_faithful.json`: ADS-B CSIL dry-run configuration for the original paper protocol; it is not CVS Stage2 evidence.
- `paper_original_matrix.md`: release-level ledger mapping paper items to implemented, smoke-only, and pending reproduction evidence.

CVS extension scope:

- `cvs_aligned/`: CVS Stage2-C protocol, metrics, and evaluation adapter.
- `configs/*_cvs_stage2c_*.json`: sanitized CVS Stage2-C example configs.

Boundary:

- These files do not claim paper reproduction completion.
- Any result claim must name the CVS split, target receiver, K-shot support/query protocol, satellite/stress view, seed, and full same-row metrics.
- Receiver-Agnostic Two-stage UDA dry-runs, Mitigating Receiver Impact DA dry-runs, and DADDA dry-runs/smoke runs are paper-faithful closed-set cross-receiver checks only. They are not CVS Stage2-A/B/C, satellite/LEO deployment, open-set, or new-class registration evidence.
- DADDA currently supports source-only and proposed DADDA rows; the paper's DANN, DAN, DSAN, WD, DCORAL, CDAN baselines, SNR robustness, ablations, visualizations, kernel sweep, and timing table remain pending.
- Orthogonal Incremental SEI is a paper-faithful closed-set FSCIL baseline. It does not define disjoint `R_s/R_t`, target-old support, target-new support under LEO view, or unknown-query rejection; any CVS use must live in `cvs_aligned/` with explicit `cvs_extension=true`.

Log separation:

- Reproduction and comparison-method logs, including RIEI, DRIFT, Fedbase, ProtoNet CDA, Feature Separation, CVCNN, and TIFS, must use `paper_reproduction/logs/`.
- Reproduction and comparison run outputs, checkpoints, and structured results must use `paper_reproduction/runs/`.
- CVS mainline, Phase1/Phase2, Stage2, spaceborne, and CV-SincNet optimization logs must stay under `logs/cvs/`.
- Do not write reproduction or comparison-method logs to the top-level `logs/` directory.

Receiver-Agnostic Two-stage UDA diagnostics:

```bash
python -m paper_reproduction.receiver_agnostic_twostage_uda.train \
  --config paper_reproduction/configs/receiver_agnostic_twostage_uda_manysig_n607_formal.json \
  --formal \
  --source-receiver-counts 6 \
  --methods dann,dann_lmmd \
  --transductive-target-eval \
  --lmmd-lambda 0.05 \
  --lmmd-layers features \
  --stage2-lr 0.0001 \
  --reset-stage2-optimizer \
  --max-stage2-steps 500
```

Use `--transductive-target-eval` when reproducing the paper-style UDA reporting pool: all target receiver samples are available as unlabeled target-domain data, and labels are used only for reporting. The held-out target split remains the default for stricter internal validation. `--lmmd-lambda`, `--lmmd-layers`, `--stage2-lr`, and `--reset-stage2-optimizer` are diagnostic knobs because the paper does not publish the LMMD trade-off weight, the concrete layer set `L`, or the stage2 optimizer policy.

For LMMD-sensitive diagnosis, reuse one DANN base and sweep reset-optimizer LMMD variants:

```bash
python -m paper_reproduction.receiver_agnostic_twostage_uda.train \
  --config paper_reproduction/configs/receiver_agnostic_twostage_uda_manysig_n607_formal.json \
  --formal \
  --source-receiver-counts 6 \
  --lmmd-grid \
  --lmmd-grid-lambdas 0.005,0.01 \
  --lmmd-grid-layers features,activations \
  --lmmd-grid-steps 250,500 \
  --lmmd-grid-lrs 0.0001 \
  --transductive-target-eval
```

DADDA dry-run:

```bash
python -m paper_reproduction.dadda_cross_receiver.train \
  --config paper_reproduction/configs/dadda_cross_receiver_manysig_paper_faithful.json \
  --dry-run
```

CSIL dry-run:

```bash
python -m paper_reproduction.csil_class_incremental_iot.train \
  --config paper_reproduction/configs/csil_adsb_paper_faithful.json \
  --dry-run --formal
```

CSIL outputs are ADS-B class-incremental protocol checks only. They are not CVS Stage2-A/B/C, satellite/LEO deployment, open-set, or new-class registration evidence.
