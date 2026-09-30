# Complete joint training log diagnostics

Status: `COMPLETE_JOINT_TRAINING_LOG_SCAN_VERIFIED`. Run: `20260930-phase2-d92-joint-spectral-support-m2-r01`.

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
| joint_preparation_count | 3168 |
| joint_stage_count | 4576 |
| fixed_stage_count | 3168 |
| diagnostic_fit_count | 0 |
| trained_joint_stage_count | 936 |
| optimizer_steps | 7488 |
| geometry_fit_count | 3704 |
| geometry_factorization_count | 2304 |
| inner_objective_evaluation_count | 8424 |
| inner_head_fit_count | 25272 |
| inner_factorization_count | 25272 |
| final_head_fit_count | 1435 |
| final_factorization_count | 1435 |
| head_fit_count | 29875 |
| factorization_count | 29875 |
| baseline_factorization_count | 3168 |
| candidate_final_events | 7744 |
| structured_nonfinite_count | 0 |
| failure_artifacts | 0 |
| candidate_stages | 7744 |
| trained_joint_stages | 936 |
| skipped_joint_stages | 3640 |
| fixed_control_stages | 3168 |
| all_projected_zero_trained_stages | 141 |
| all_gradient_zero_trained_stages | 0 |
| changed_from_anchor_joint_stages | 795 |
| joint_identity_forward_stages | 3789 |
| nonzero_projected_updates | 6340 |
| zero_projected_updates | 1148 |
| projection_cancelled_updates | 1148 |
| gradient_zero_steps | 0 |

## Text marker observations

| Marker | Matching lines |
|---|---:|
| error | 0 |
| warning | 0 |
| traceback | 0 |
| oom_or_killed | 0 |
| nonfinite_text | 0 |
| recovery_or_resume | 0 |
| early_stop | 30980 |
| configuration | 52292 |
| environment | 4 |
| launch_command | 4 |

Marker counts overlap and can include configuration declarations; they do not automatically diagnose a failure.

## Measured objectives and parameters

`snapshots.csv` reports first/last/final and final-minus-initial statistics by scope/state/train K. `curves.csv` contains all eight update positions and final points; `stage_curves.csv` retains each physical stage. `stages.csv` retains exact theta, anchors, step counts and projection/gradient statistics. Spectral contraction uses recorded inner spectra; unrecorded final full geometry remains null.

## Boundaries

- All records and text lines scanned; full/compact/CSV/text streams reconciled. No query or outer-held scores read.
- First and last are pre-update objectives at steps 1 and 8. Final objective is after all 8 updates.
- Inner accuracy pools physical inner-held training counts; it is part of the training objective, not independent validation.
- Actual stages receive equal weight within scope/state/train K; updates are not independent experiments.
- Fixed controls are separate from learned joint stages. Skipped/no-information objectives remain null.
- Contraction bound theta_sum/2 uses the frozen normalized covariance contract. Inner spectral gain uses recorded inner geometry only; full final geometry gain is unrecorded/null.
- Text markers count matching lines, overlap, and include config declarations; they do not by themselves prove a failure or restart.
- Training/score wall time and RSS refer to the recorded CPU platform; GPU memory and unmeasured metrics remain null.
- Ground data payload, full fitted state RAM, and code release archive are distinct. No deployment bytes inferred from theta size.
- Original Phase1 model delivery is outside incremental method payload; no deployment package was measured by this collector.
