# Phase1 GPU0_A Joint-Safe And Prototype/Mask Audit Traceability

Date: 2026-06-29

Objective: repair the late-pseudo checkpoint cliff risk in `PHASE1_CEN51_REPAIR_RXFLOOR_GROUPFISHR_LATEPSEUDO_GPU0_A` without changing the Phase1 source-only SSDG protocol in `项目.md`, and connect prototype/mask/feature-space modules as verifiable audit-only telemetry before enabling active losses.

Key anomaly: the GPU0_A `E198->E199` transition showed a joint regression in strict UDU, receiver floor, and satellite metrics while pseudo-label precision remained near 99.9%, pointing to checkpoint/PAIC multi-objective variance rather than pseudo-label contamination.

| ID | Requirement | Target files | Status | Verification | Notes |
|----|-------------|--------------|--------|--------------|-------|
| JS-01 | Add `joint_safe` checkpoint selection over `val_tx`, strict UDU, receiver floor, satellite aggregate mean/floor, and satellite strict mean/floor. | `code/SSDG/train_ssdg.py`;`code/cvsrffi/ssdg_guard.py` | verified | `test_joint_safe_score_prefers_balanced_satellite_epoch_over_clean_only_peak`;dry-run argparse | Default `clean_val_tx` behavior remains unchanged unless the new metric or guard is enabled. |
| JS-02 | Treat missing required joint-safe metrics as unsafe so failed/empty satellite eval cannot write `latest_safe_ssdg.pth`. | `code/SSDG/train_ssdg.py`;`code/cvsrffi/ssdg_guard.py` | verified | `test_joint_safe_requires_satellite_metrics_when_guarded`;`13 passed` guard/matrix tests | Review follow-up from subagent `019f0f6d-2ab3-7031-a99a-af8c375f5ab1`. |
| JS-03 | Add late-pseudo one-epoch rollback guard and `[JOINT-GUARD]`/`[SAFE-CKPT]` logging. | `code/SSDG/train_ssdg.py`;`code/cvsrffi/ssdg_guard.py` | verified | `test_one_epoch_drop_guard_catches_gpu0_style_cliff`;`py_compile` | Synthetic record covers the GPU0_A `E198->E199` cliff pattern. |
| JS-04 | Add PAIC variance guard over weighted satellite CE, satellite consistency, domain loss, gradient norm, and pseudo reliable ratio. | `code/SSDG/train_ssdg.py`;`code/cvsrffi/ssdg_guard.py` | verified | `test_paic_variance_guard_catches_high_variance_without_pseudo_precision_collapse` | Guard can block best update and apply short satellite-pressure cooldown. |
| PM-01 | Add Stage A prototype/mask/geometry audit-only CLI flags and telemetry markers while keeping losses at zero. | `code/SSDG/train_ssdg.py` | verified | dry-run argparse;matrix validator | Expected markers: `[PROTO-TX]`,`[PROTO-RX]`,`[MASK]`,`[BATCH-GEOM]`,`[TXRX-ANOVA]`. |
| PM-02 | Verify module reachability for `phase2_prototypes`, `feature_masks`, `tx_rx_geometry`, `balanced_tx_rx_sampler`, and `open_world_head`. | `code/cvsrffi/*`;`code/tests/*` | verified | `16 passed` module tests | This is telemetry/module reachability only, not a performance claim. |
| PM-03 | Stage B/C/D active prototype/mask/geometry losses. | `code/SSDG/train_ssdg.py`;`code/cvsrffi/*` | deferred | pending | Non-zero active loss weights fail closed until training wiring is explicitly implemented and verified. |
| PM-04 | Ensure masked SupCon weights cannot pass validator as audit-only. | `tools/optimizer_validate_matrix.py`;`code/tests/test_phase1_ground_proto_mask_matrix.py` | verified | `test_validator_rejects_masked_supcon_active_loss_in_audit_only_row` | Review follow-up from subagent `019f0f6d-2ab3-7031-a99a-af8c375f5ab1`. |
| OW-01 | Add `eval_open_world.py` and `phase2_adapt.py`. | `code/eval_open_world.py`;`code/phase2_adapt.py` | deferred | pending | Reserved for Stage E after Phase1 joint-safe plus Stage A audit are stable. |
| MX-01 | Add four GPU0_A-derived candidates: `SOFTPSEUDO_190X10`, `EMA_KEEP15`, `SATSOFT_NO_CONS`, and `GROUPSOFT_190X10`. | `tools/spaceborne_fewshot_da_matrix.py` | verified | generated `PHASE1_GPU0_JOINTSAFE4`;validator `PASS` | All four use `--best_metric joint_safe` and explicit guard flags. |
| MX-02 | Make validator distinguish audit-only rows from active training rows. | `tools/optimizer_validate_matrix.py` | verified | `test_phase1_gpu0_jointsafe4_matrix_has_guarded_audit_only_rows`;negative validator tests | Rows with non-zero prototype/mask/geometry/SupCon weights require verified active training wiring. |

Generated artifacts:

| Artifact | Path |
|---|---|
| Matrix | `E:\type10-7\automation_reports\CV-SincNet\phase1_gpu0_jointsafe_proto_audit_20260629\matrix.json` |
| Report | `E:\type10-7\automation_reports\CV-SincNet\phase1_gpu0_jointsafe_proto_audit_20260629\report.md` |
| Launcher | `E:\type10-7\code\scripts\launch_phase1_gpu0_jointsafe_proto_audit_20260629.sh` |
| Snapshot | `E:\type10-7\code\snapshots\phase1_gpu0_jointsafe_proto_audit_20260629` |

Final local verification summary:

- `py_compile`: passed for guard, training entrypoint, matrix generator, and validator.
- Guard/matrix tests: `13 passed`.
- Prototype/mask/open-world/balanced-sampler tests: `16 passed`.
- `train_ssdg.py --dry_run`: parsed `label_epochs=190 pseudo_epochs=10 total_epochs=200`.
- Matrix validator: `PASS`, four launchable Phase1 rows.
- N607 status: no remote launch or sync was performed in this implementation turn.
