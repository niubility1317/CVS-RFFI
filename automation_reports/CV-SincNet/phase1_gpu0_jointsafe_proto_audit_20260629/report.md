# phase1_gpu0_jointsafe_proto_audit_20260629

## Objective

Validate Phase1 source-only ground DG representation optimization together with deployment-time Phase2 few-shot adaptation paths.
Phase1 rows track source-domain TX prototypes, receiver feature distribution, mask auxiliaries, and TX/RX geometry without using target receiver samples.
Ground model default: BEX02_fishr002_mixed_e170 (${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth).

## Candidate Matrix

| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |
|---|---:|---|---:|---|---|---|---|---|
| `PHASE1_GPU0_JOINTSAFE_SOFTPSEUDO_190X10` | 0 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE_EMA_KEEP15` | 1 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE_SATSOFT_NO_CONS` | 2 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE_GROUPSOFT_190X10` | 3 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |

## Verification Contract

- Framework match: the ground-stage model is the existing generalized CVS-RFFI metric space checkpoint; this launcher only covers satellite-deployed few-shot procedures.
- No semi-supervised target adaptation is used: target receiver adaptation is labeled-only and commands set `--entropy_weight 0`, `--consistency_weight 0`, and `--pseudo_weight 0`.
- `CVS-SFE` is a feature-level validation over frozen `z_id` prototypes; the support features stand for samples already affected by `H_sg o R_sat`, and it must report `full_accuracy`, `accepted_accuracy`, `coverage`, `new_class_accuracy`, `old_class_accuracy`, and `unknown_rejection_rate`.
- `CVS-FTRC` uses target receiver support after explicit star-ground channel synthesis (`--target_channel_view satellite`) and is not strict DG; it must be reported separately from source-only DG tables.
- OA-MSE rows are staged as Stage2-A MSE-lite, Stage2-B MSE-subspace, and Stage2-C OA-MSE-Head; unknown query samples are eval-only and cannot fit thresholds.
- Future star-ground augmentation uses `star_ground_channel_impl=simplified_leo_residual` with `leo_clear_weak`, `leo_low_elev_weak`, and `leo_rain_weak`; legacy five-scenario LEO is control-only unless explicitly marked.
- OA-MSE launchable rows must carry the combined onboard adaptation bundle: Weibull EVT, target adapter, pseudo-unknown energy, seen-new evidence gate, ambiguous-only Siamese verifier, accepted-only online update, and Stage2 receiver-domain separation.
- Gate and adapter variants must record their candidate-level parameters in `matrix.json`; rollback decisions are deployment gates, not post-hoc notes.
- Any accepted-only metric must be shown with its full denominator and coverage.
- Satellite metrics are stress-test metrics unless real in-orbit IQ is explicitly used.

## Launch

Run local/remote dry-run first:

```bash
bash code/scripts/launch_phase1_gpu0_jointsafe_proto_audit_20260629.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.

## Local Implementation And Verification

This run is a Phase1 source-only GPU0_A repair matrix. It does not change the `项目.md` protocol and does not use target receiver samples for training or model selection.

Changed local runtime files:

| File | Purpose |
|---|---|
| `code/SSDG/train_ssdg.py` | Adds `joint_safe` checkpoint selection, one-epoch protected-metric drop guard, PAIC late-variance guard, safe checkpoint paths, and Stage A prototype/mask/geometry audit-only CLI/log markers. |
| `code/cvsrffi/ssdg_guard.py` | Pure guard utilities for protected metric extraction, joint-safe scoring, one-epoch cliff detection, and PAIC variance detection. |
| `tools/spaceborne_fewshot_da_matrix.py` | Adds `PHASE1_GPU0_JOINTSAFE4` with four GPU0_A-derived candidates. |
| `tools/optimizer_validate_matrix.py` | Adds validator checks for `joint_safe` guard CLI and audit-only vs active prototype/mask loss state. |
| `code/cvsrffi/phase2_prototypes.py`, `feature_masks.py`, `tx_rx_geometry.py`, `balanced_tx_rx_sampler.py`, `open_world_head.py` | Required Stage A module reachability for prototype/mask/geometry/open-world audit markers. |

Generated artifacts:

| Artifact | Path |
|---|---|
| Matrix | `E:\type10-7\automation_reports\CV-SincNet\phase1_gpu0_jointsafe_proto_audit_20260629\matrix.json` |
| Launcher | `E:\type10-7\code\scripts\launch_phase1_gpu0_jointsafe_proto_audit_20260629.sh` |
| Traceability | `E:\type10-7\analysis\phase1_gpu0_jointsafe_proto_audit_traceability_20260629.md` |
| Pre-edit snapshot | `E:\type10-7\code\snapshots\phase1_gpu0_jointsafe_proto_audit_20260629` |

Verification commands run locally:

```powershell
conda run --no-capture-output -n ssr-gpu python -m py_compile code\cvsrffi\ssdg_guard.py code\SSDG\train_ssdg.py tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py
conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_ssdg_guard.py code\tests\test_phase1_ground_proto_mask_matrix.py -q
conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase2_prototypes.py code\tests\test_feature_masks.py code\tests\test_tx_rx_geometry.py code\tests\test_open_world_head.py code\tests\test_balanced_tx_rx_sampler.py -q
conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --output_dir tmp_dry_runs\joint_safe_argparse --dry_run --epochs 200 --label_epochs 190 --best_metric joint_safe --enable_joint_safe_guard true --paic_guard_enabled true --use_phase2_ground_prototypes true --use_feature_masks true --use_txrx_geometry_losses true --phase1_distribution_audit_only true --lambda_tx_proto 0 --lambda_mask_aux 0 --lambda_txrx_rect 0
conda run --no-capture-output -n ssr-gpu python tools\optimizer_validate_matrix.py automation_reports\CV-SincNet\phase1_gpu0_jointsafe_proto_audit_20260629\matrix.json
```

Verification result: focused guard/matrix tests passed (`13 passed`), prototype/mask/open-world/balanced-sampler tests passed (`16 passed`), matrix validator returned `PASS` with four launchable Phase1 rows. Parallel `conda run` attempts hit a Windows temporary-file lock, so final verification was run serially.

Review follow-up:

| Finding | Fix | Verification |
|---|---|---|
| Missing satellite statistics could still be treated as a safe checkpoint if drop/PAIC did not fire. | Added required joint-safe metric checking; missing `sat_mean_tx`, `sat_floor_tx`, `sat_strict_mean`, or `sat_strict_floor` now marks the epoch unsafe and blocks `latest_safe_ssdg.pth`/best update. | `test_joint_safe_requires_satellite_metrics_when_guarded`;`13 passed` |
| Validator did not include masked SupCon weights in active loss detection. | Added `lambda_tx_supcon_masked` and `lambda_rx_supcon_masked` to the active-weight set. | `test_validator_rejects_masked_supcon_active_loss_in_audit_only_row`;`13 passed` |

Git-backed mirror: implementation files and tests were copied to `E:\type10-7\github_publish\CVS-RFFI-repo` on branch `codex/cvs-rffi-release-20260626` for versioned review/commit.
