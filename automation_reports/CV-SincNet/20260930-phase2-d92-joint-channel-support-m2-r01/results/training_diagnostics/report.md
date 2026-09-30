# Complete channel training log diagnostics

Status: `COMPLETE_CHANNEL_TRAINING_LOG_SCAN_VERIFIED`. Run: `20260930-phase2-d92-joint-channel-support-m2-r01`.

All listed structured records and full text logs were read and reconciled. Inner accuracy is a training metric, not independent validation.

## Actual counts

| Counter | Actual |
|---|---:|
| episodes | 160 |
| k1_episodes | 40 |
| oof_episodes | 120 |
| proxy_anchor_count | 1400 |
| sequence_paths | 1760 |
| baseline_head_fit_count | 3168 |
| channel_preparation_count | 3168 |
| channel_stage_count | 4576 |
| diagnostic_fit_count | 0 |
| trained_channel_stage_count | 936 |
| optimizer_steps | 7488 |
| inner_objective_evaluation_count | 8424 |
| inner_head_fit_count | 25272 |
| inner_factorization_count | 25272 |
| final_head_fit_count | 936 |
| final_factorization_count | 936 |
| head_fit_count | 29376 |
| factorization_count | 29376 |
| baseline_factorization_count | 3168 |
| derivative_triangular_solve_count | 44928 |
| candidate_final_events | 4576 |
| structured_nonfinite_count | 0 |
| structured_technical_failure_statuses | 0 |
| failure_artifacts | 0 |
| candidate_stages | 4576 |
| trained_channel_stages | 936 |
| skipped_channel_stages | 3640 |
| all_projected_zero_trained_stages | 0 |
| all_gradient_zero_trained_stages | 0 |
| changed_from_anchor_stages | 936 |
| identity_forward_stages | 3640 |
| trained_inner_correct_count_unchanged_stages | 352 |
| nonzero_projected_updates | 7488 |
| zero_projected_updates | 0 |
| projection_cancelled_updates | 0 |
| gradient_zero_steps | 0 |
| clipped_steps | 0 |
| box_active_steps | 150 |
| objective_increase_edges | 1265 |
| objective_decrease_edges | 6223 |
| objective_equal_edges | 0 |

## Text marker observations

| Marker | Matching lines |
|---|---:|
| error | 0 |
| warning | 0 |
| traceback | 0 |
| oom_or_killed | 0 |
| nonfinite_text | 0 |
| recovery_or_resume | 0 |
| early_stop_mention | 18308 |
| early_stop_disabled_declaration | 18308 |
| configuration | 39620 |
| environment | 4 |
| launch_command | 4 |
| early_stop_actual_message | 0 |

Marker counts overlap and can include configuration declarations; they do not automatically diagnose a failure.

## Measured objectives and parameters

`snapshots.csv` reports first/last/final and final-minus-initial statistics by scope/state/train K. `curves.csv` contains all eight update positions and final points; `stage_curves.csv` retains each physical stage. `stages.csv` retains compact parameter/anchor distances, step counts and projection/gradient statistics. Raw 736-vectors remain in the referenced original training streams.

## Boundaries

- All records and text lines scanned; full/compact/CSV/text streams reconciled. No query or outer-held scores read.
- First and last are pre-update objectives at steps 1 and 8. Final objective is after all 8 updates.
- Inner accuracy pools physical inner-held training counts; it is part of the training objective, not independent validation.
- Actual stages receive equal weight within scope/state/train K; updates are not independent experiments.
- Physical K1/single-class stages skip; ordinary zero-gradient stages still execute eight Adam updates. Unmeasured metrics remain null.
- Parameter/gradient arrays are verified coordinate by coordinate, then reduced to per-block statistics. Original full arrays remain in source logs; no spectral contraction bound applies to this method.
- The supervised objective pools held_margin_loss_sum; head_training_loss_data/ridge/total are separate closed-form head diagnostics.
- Equal aggregate correct counts do not prove identical individual predictions. Individual prediction transitions are not recorded here.
- early_stop_disabled_declaration counts configuration false; early_stop_actual_message counts explicit events/plain stopping messages separately.
- Text markers count matching lines, overlap, and include config declarations; they do not by themselves prove a failure or restart.
- Training/score wall time and RSS refer to the recorded CPU platform; GPU memory and unmeasured metrics remain null.
- Ground data payload, full fitted state RAM, and code release archive are distinct. No deployment bytes inferred from adapter vector size.
- Original Phase1 model delivery is outside incremental method payload; no deployment package was measured by this collector.

## Artifact storage

Full derived JSON is preserved locally as `training_diagnostics.json` and delivered losslessly as [training_diagnostics.json.gz](training_diagnostics.json.gz). Decompress it as UTF-8 JSON; byte-for-byte readback passed. Sizes and original location: [artifact_storage.json](artifact_storage.json). All CSV, summary and report artifacts remain uncompressed.
