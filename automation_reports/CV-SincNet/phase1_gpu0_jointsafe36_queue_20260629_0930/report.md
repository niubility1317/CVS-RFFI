# phase1_gpu0_jointsafe36_queue_20260629_0930

## Objective

Validate Phase1 source-only ground DG representation optimization together with deployment-time Phase2 few-shot adaptation paths.
Phase1 rows track source-domain TX prototypes, receiver feature distribution, mask auxiliaries, and TX/RX geometry without using target receiver samples.
Ground model default: BEX02_fishr002_mixed_e170 (${ROOT}/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth).

## Candidate Matrix

| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |
|---|---:|---|---:|---|---|---|---|---|
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0` | 0 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1` | 1 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2` | 2 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3` | 3 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4` | 4 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5` | 5 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6` | 6 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7` | 7 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_FISHRSOFT_S8` | 0 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_BASE_S0` | 1 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SEED_S1` | 2 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_TAU88_S2` | 3 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SHORT195_S3` | 4 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_MID188_S4` | 5 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_SATLOW_S5` | 6 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINSOFT_S6` | 7 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_DOMAINFIRM_S7` | 0 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_EMA_KEEP15_FISHRSOFT_S8` | 1 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_BASE_S0` | 2 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SEED_S1` | 3 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_TAU88_S2` | 4 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SHORT195_S3` | 5 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_MID188_S4` | 6 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_SATLOW_S5` | 7 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINSOFT_S6` | 0 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_DOMAINFIRM_S7` | 1 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_SATSOFT_NO_CONS_FISHRSOFT_S8` | 2 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_BASE_S0` | 3 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SEED_S1` | 4 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_TAU88_S2` | 5 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SHORT195_S3` | 6 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_MID188_S4` | 7 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_SATLOW_S5` | 0 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINSOFT_S6` | 1 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_DOMAINFIRM_S7` | 2 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |
| `PHASE1_GPU0_JOINTSAFE36_GROUPSOFT_190X10_FISHRSOFT_S8` | 3 | Safe-SSDG-CVS-R01 | 0 | `logit_calibration` | `source_only_ground_training_no_target_receiver` | `Y_old_source_only` | `gpu0_a_late_pseudo_repair+joint_safe_checkpoint+phase1_proto_mask_audit` | joint_safe_score,strict_udu,receiver_floor,sat_mean_3,sat_floor_3,sat_strict_mean_3,pseudo_precision,paic_guard |

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
bash code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.

## Local Launch Preparation

This run implements the user request for 36 queued Phase1 SSDG experiments with at most two active experiments per GPU. The launcher uses `STAGE2_MAX_ACTIVE_PER_GPU=2` and, by default, includes existing `nvidia-smi pmon` GPU processes in the per-GPU occupancy count through `INCLUDE_EXTERNAL_GPU_PROCS=1`.

Protocol boundary:

- Phase: Phase1 source-only SSDG.
- Training receivers: source receivers only, `CEN51_TRAIN_RXS=0,1,2,3,4,5,6`.
- Target receiver usage: forbidden during training and model selection.
- Prototype/mask/geometry modules: audit-only telemetry, all active loss weights remain zero.
- Checkpoint policy: `joint_safe` with required satellite metric presence, one-epoch protected metric drop guard, and PAIC variance guard.
- Claim boundary: queued training launch only; no deployment success claim.

Experiment design:

| Axis | Values |
|---|---|
| Base families | `softpseudo_190x10`,`ema_keep15`,`satsoft_no_cons`,`groupsoft_190x10` |
| Sweeps per family | `base_s0`,`seed_s1`,`tau88_s2`,`short195_s3`,`mid188_s4`,`satlow_s5`,`domainsoft_s6`,`domainfirm_s7`,`fishrsoft_s8` |
| Total candidates | 36 |
| GPU distribution | GPU0-GPU3: 5 rows each; GPU4-GPU7: 4 rows each |
| Concurrent cap | 2 active GPU compute processes per GPU including external jobs when visible to `nvidia-smi pmon` |
| Primary metrics | `joint_safe_score`, strict UDU, receiver floor, satellite mean/floor, satellite strict mean/floor, pseudo reliability, PAIC guard markers |

Local files changed or generated:

| File | Purpose |
|---|---|
| `tools/spaceborne_fewshot_da_matrix.py` | Adds `PHASE1_GPU0_JOINTSAFE36`, 36 candidate variants, and queue launcher external GPU process accounting. |
| `code/tests/test_phase1_ground_proto_mask_matrix.py` | Adds tests for 36-row queue distribution and validator pass. |
| `code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh` | Generated queue launcher. |
| `automation_reports/CV-SincNet/phase1_gpu0_jointsafe36_queue_20260629_0930/matrix.json` | 36-row validated matrix. |
| `automation_reports/CV-SincNet/phase1_gpu0_jointsafe36_queue_20260629_0930/report.md` | This report and launch record. |

Local verification:

```powershell
conda run --no-capture-output -n ssr-gpu python -m py_compile tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py code\SSDG\train_ssdg.py code\cvsrffi\ssdg_guard.py
conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_ground_proto_mask_matrix.py code\tests\test_ssdg_guard.py -q
conda run --no-capture-output -n ssr-gpu python tools\spaceborne_fewshot_da_matrix.py --plan PHASE1_GPU0_JOINTSAFE36 --run-id phase1_gpu0_jointsafe36_queue_20260629_0930
conda run --no-capture-output -n ssr-gpu python tools\optimizer_validate_matrix.py automation_reports\CV-SincNet\phase1_gpu0_jointsafe36_queue_20260629_0930\matrix.json
bash -n code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh
bash code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh --dry-run
```

Results: `py_compile` passed; focused tests passed (`14 passed`, one `.pytest_cache` permission warning); validator returned `PASS` with 36 launchable rows; `bash -n` passed; launcher dry-run printed 36 commands without launching.

Scheduler PID correction: before remote launch, the generated launcher was tightened to start `"${CMD[@]}"` directly in the background instead of through a parenthesized subshell. This keeps the recorded PID aligned with the training process for per-GPU queue accounting. Local re-verification after this change: `py_compile` passed, focused tests passed (`14 passed`), `bash -n` passed, and dry-run counted 36 candidates.

Remote scheduler correction: first remote start showed `[SPACEBORNE-FSDA-WAIT] gpu=0 active=0 external=2 total=2 max=2` before launching any candidate because `nvidia-smi pmon` counted the display `Xorg` process. The launcher was corrected to count only pmon rows with type `C`, so graphics processes no longer consume the training concurrency budget.

Planned remote sync:

| Local | Remote |
|---|---|
| `code/SSDG/train_ssdg.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py` |
| `code/cvsrffi/*.py` guard/prototype/mask/geometry modules | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/` |
| `tools/spaceborne_fewshot_da_matrix.py` | `/home/szu2070436088/2510044040/CV-SincNet/tools/spaceborne_fewshot_da_matrix.py` |
| `tools/optimizer_validate_matrix.py` | `/home/szu2070436088/2510044040/CV-SincNet/tools/optimizer_validate_matrix.py` |
| `code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh` |
| `automation_reports/CV-SincNet/phase1_gpu0_jointsafe36_queue_20260629_0930/*` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase1_gpu0_jointsafe36_queue_20260629_0930/` |

Planned remote launch command after preflight, sync, and remote dry-run:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && \
mkdir -p logs/phase1_gpu0_jointsafe36_queue_20260629_0930 && \
RUN_ID=phase1_gpu0_jointsafe36_queue_20260629_0930 STAGE2_MAX_ACTIVE_PER_GPU=2 INCLUDE_EXTERNAL_GPU_PROCS=1 \
nohup bash code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh \
> logs/phase1_gpu0_jointsafe36_queue_20260629_0930/scheduler.out 2>&1 & echo $!
```

## Remote Launch Record

Remote launch timestamp: 2026-06-29 local host time, after N607 direct preflight.

N607 preflight and initial occupancy:

| Check | Result |
|---|---|
| Direct SSH preflight | PASS; N607 reachable directly, project root visible, 8 RTX3090 GPUs visible. |
| Existing training occupancy | One active `stage2_spaceborne_h06_phase1_floorrepair_20260628_221940` `train_ssdg.py` compute process per GPU. |
| Queue capacity rule | Keep at most two GPU compute training processes per GPU including existing external `C` processes from `nvidia-smi pmon`. |
| Initial launch decision | Launch one new queued candidate per GPU; keep the remaining 28 candidates queued behind the per-GPU cap. |

Remote sync and verification:

| Item | Result |
|---|---|
| Synced files | Training entrypoint, `cvsrffi` guard/prototype/mask/geometry modules, matrix generator, validator, queue launcher, matrix, and report. |
| Remote `py_compile` | PASS. |
| Remote matrix validator | PASS, 36 launchable rows, 0 issues. |
| Remote `bash -n` | PASS. |
| Remote launcher dry-run | PASS, 36 candidates printed without launching. |

The live scheduler was launched with `setsid` so the queue remains on N607 after the bounded SSH command exits:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && \
mkdir -p logs/phase1_gpu0_jointsafe36_queue_20260629_0930 && \
setsid env RUN_ID=phase1_gpu0_jointsafe36_queue_20260629_0930 STAGE2_MAX_ACTIVE_PER_GPU=2 INCLUDE_EXTERNAL_GPU_PROCS=1 \
bash code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh \
> logs/phase1_gpu0_jointsafe36_queue_20260629_0930/scheduler.out 2>&1 < /dev/null &
```

Live scheduler state:

| Field | Value |
|---|---|
| Scheduler process | `706167 bash code/scripts/launch_phase1_gpu0_jointsafe36_queue_20260629_0930.sh` |
| Scheduler log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gpu0_jointsafe36_queue_20260629_0930/scheduler.out` |
| Launched candidates so far | 8 |
| Queue wait marker | `[SPACEBORNE-FSDA-WAIT] gpu=0 active=1 external=1 total=2 max=2` |
| Interpretation | The queue is working as intended: GPU0 already has one queued training process plus one existing external training process, so the next GPU0 candidate waits until a slot frees. |

Initial launched candidates:

| GPU | Candidate | PID | Log |
|---|---|---:|---|
| 0 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0` | 706177 | `logs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_BASE_S0.out` |
| 1 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1` | 706186 | `logs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SEED_S1.out` |
| 2 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2` | 706195 | `logs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_TAU88_S2.out` |
| 3 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3` | 706204 | `logs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3.out` |
| 4 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4` | 706213 | `logs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_MID188_S4.out` |
| 5 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5` | 706222 | `logs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SATLOW_S5.out` |
| 6 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6` | 706294 | `logs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINSOFT_S6.out` |
| 7 | `PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7` | 706366 | `logs/phase1_gpu0_jointsafe36_queue_20260629_0930/PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_DOMAINFIRM_S7.out` |

N607 `nvidia-smi pmon -c 1` startup snapshot showed exactly two compute `C` processes per GPU: the existing floorrepair process and the newly launched queued process. The display `Xorg` process appears as type `G` and is intentionally ignored by the training concurrency guard.

Startup log health:

| Check | Result |
|---|---|
| First 8 log files | Present. |
| Required config markers | `[CONFIG-JOINT-SAFE]`, `[CONFIG-PROTO-MASK]`, `[CONFIG-SAT]`, `[CONFIG-PSEUDO]`, `[CONFIG-LOSS]`, and `[SAFE-CKPT]` found. |
| Startup errors searched | No `Traceback`, `RuntimeError`, `unrecognized arguments`, CUDA OOM, or NaN markers found in the inspected head/tail windows. |
| Prototype/mask/geometry status | Audit-only telemetry is enabled; active prototype/mask/geometry loss weights remain zero. |

SSH cleanup:

- Two early remote launch attempts timed out because the detached command did not return cleanly through the Windows SSH client. Local stale `ssh.exe` clients were identified and closed before continuing.
- After the successful bounded checks, the remaining remote long-lived processes are the intended scheduler and training jobs on N607, not local SSH shells.
- No experiment result or deployment success claim is made here; this record only confirms queue launch and startup health.

## Monitor Snapshot 2026-06-29 09:25 CST

User question: whether the 36-experiment queue has finished.

Read-only N607 monitor result:

| Field | Value |
|---|---:|
| Matrix candidates | 36 |
| Launched by scheduler | 16 |
| Scheduler-complete markers | 0 |
| Scheduler-failed markers | 0 |
| Active launched jobs | 16 |
| Ended but unreported jobs | 0 |
| Pending not launched | 20 |
| GPU compute rows from `nvidia-smi pmon` | 16 |

Interpretation: the experiment queue has not finished. It is currently using the full allowed queue capacity, two active queued jobs per GPU, and has 20 candidates still waiting.

Scheduler tail showed repeated:

```text
[SPACEBORNE-FSDA-WAIT] gpu=0 active=2 external=0 total=2 max=2
```

This means the earlier external floorrepair processes have ended or are no longer visible as external compute rows, and the queue has filled both allowed slots per GPU with its own candidates.

Active launched jobs by family and progress:

| Family | Active jobs | Current epoch range |
|---|---:|---|
| `SOFTPSEUDO_190X10` | 9 | E122 to E175 |
| `EMA_KEEP15` | 7 | E120 to E123 |

No inspected launched log tail contained `Traceback`, `RuntimeError`, `unrecognized arguments`, CUDA OOM, or NaN markers. This snapshot is a monitor-only status check, not a completion report.
