# stage2_spaceborne_h06_oldgeom_repair_20260626_022740

## Objective

Validate the two deployment-time few-shot adaptation paths from the spaceborne CVS-RFFI design:
new-TX enrollment (`CVS-SFE`) and target receiver calibration (`CVS-FTRC`).
Ground model default: CEN51_R04_H06_LOW_PROB_HYBRID_R010 (${ROOT}/runs/cen51_r04_hybrid8_strong_leo_residual_20260624_1545/CEN51_R04_H06_LOW_PROB_HYBRID_R010/latest_model.pth).

## Candidate Matrix

| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |
|---|---:|---|---:|---|---|---|---|---|
| `OA_MSE_H06_OLDGEOM48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU0_B_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU0_C_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU0_D_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU0_E_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU0_F_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU1_A_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU1_B_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU1_C_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU1_D_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU1_E_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU1_F_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU2_A_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU2_B_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU2_C_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU2_D_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU2_E_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU3_A_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU3_B_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU3_C_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU3_D_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU3_E_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU3_F_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU4_A_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU4_B_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU4_C_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU4_D_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU4_E_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU5_A_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU5_B_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU5_C_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU5_D_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU5_E_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU5_F_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU6_A_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU6_B_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU6_C_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU6_D_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU6_E_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU6_F_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU7_A_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU7_B_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU7_D_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU7_E_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDGEOM48_GPU7_F_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_geometry_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |

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
bash code/scripts/launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.


## Automation audit addendum (2026-06-26T02:31:23+0800)

### Control files read

- `E:\type10-7\AGENTS.md`: SHA256 `1bfc5fe4a493fd7cbc10d71d2997f47610a7bf6a5938af8a9ec36396331beeaf`
- `E:\type10-7\??.md`: SHA256 `d4b692a36777cf3b26badd9ba1d3bfb683141ad3fb121bb2170bc17259d9eb57`; CVS authority version `2026-06-24`
- `E:\type10-7\tools\optimizer_control_manifest.md`: SHA256 `359202b395effb7ac6e9dbf6c4598f008c8d101c41fa3b5952fd54ece63fc95f`
- `E:\type10-7\automation_reports\CV-SincNet\automation_prompt_backups\20260615_001820_stage2_closed_loop_v4\stage2_prompt.md`: SHA256 `1e105e3906f062cfd256021458f316bf565aeb5b95cff623a9691a8f62b59d24`
- `E:\type10-7\tools\optimizer_workflow_contract.md`: SHA256 `a530a63a91fec7527801f0048d6efb2659ee43d3f9caf3aec7b38dbe58d93dde`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`: SHA256 `68c0458fdd4c34bf39ef61df2ea0fb93e7ee55e2b8d9f1015e1762f69f6040c9`

### Current control decision

- State `required_next_action`: `DESIGN_H06_REPRESENTATION_OR_PROTOTYPE_GEOMETRY_REPAIR_FROM_SEPARABILITY_DIAGNOSTIC`
- Prior H06 old-primary consensus runner status: `COMPLETED_NEGATIVE_DIAGNOSTIC_OLDPRIMARY_CONSENSUS_NOT_PROMOTABLE`
- Prior separability diagnostic status: `COMPLETED_DIAGNOSTIC_NEGATIVE_FEATURE_GEOMETRY_NOT_SEPARABLE`
- This run does not relaunch `stage2_spaceborne_h06_oldprimary_consensus_repair_20260625_localcheck` in place.
- This run keeps Stage2-B old/unknown-only boundaries, uses target-old support, keeps unknown query eval-only, and excludes target-new support/query enrollment.

### Multi-role review

- Protocol role: PASS. Current H06 OLDGEOM path remains consistent with `??.md` Stage2-B old/unknown diagnostic boundary; no Stage2-C or deployment-success claim is made.
- Evidence role: `SUBAGENT_RUNTIME_UNAVAILABLE`; controller used local diagnostic artifacts and validator evidence.
- Runner role: `SUBAGENT_RUNTIME_UNAVAILABLE`; controller performed bounded local generation, validation, and launcher checks.
- Validation/supervision role: `SUBAGENT_RUNTIME_UNAVAILABLE`; controller wrote this supervision section and preserved artifacts.

### Local changes and snapshot

- Added plan `OA_MSE_H06_OLDGEOM48` in `E:\type10-7\tools\spaceborne_fewshot_da_matrix.py` for support-center geometry, target shift/halo/ring pseudo-unknowns, soft/multi-prototype consistency, and old-primary measurement gates.
- Updated `E:\type10-7\tools\optimizer_validate_matrix.py` so `OLDGEOM` candidate IDs are recognized as old/unknown-only and old-primary route rows.
- Generated launcher `E:\type10-7\code\scripts\launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh`.
- Non-git snapshot: `E:\type10-7\code\snapshots\stage2_spaceborne_h06_oldgeom_repair_20260626_022740\`.

### Local verification

- `conda run -n ssr-gpu python -m py_compile tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py`: PASS.
- `conda run -n ssr-gpu python tools\spaceborne_fewshot_da_matrix.py --plan OA_MSE_H06_OLDGEOM48 --run-id stage2_spaceborne_h06_oldgeom_repair_20260626_022740`: generated 48 candidates.
- `conda run -n ssr-gpu python tools\optimizer_validate_matrix.py automation_reports\CV-SincNet\stage2_spaceborne_h06_oldgeom_repair_20260626_022740\matrix.json --expected-count 48 --launcher code\scripts\launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh --repair-launcher-identity --output automation_reports\CV-SincNet\stage2_spaceborne_h06_oldgeom_repair_20260626_022740\artifacts\validator_final.json`: PASS, 48 launchable, 0 issues.
- `bash -n code/scripts/launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh`: PASS.
- `bash code/scripts/launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh --dry-run > automation_reports/CV-SincNet/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/artifacts/launcher_dry_run_local.txt`: PASS.
- A parallel `conda run` attempt hit Windows temp activation lock `__conda_tmp_28264`; serial retry passed, so it is not treated as verification failure.

### Hashes

- `tools/spaceborne_fewshot_da_matrix.py`: SHA256 `66AAF513455E3CC4EE45694358AE6DA0743459BD8A68380C31CCAF52AA48FCBA`
- `tools/optimizer_validate_matrix.py`: SHA256 `E6D50E31D95F229C51C38F5E19E88D3176448AA58BDFE3A65E0D29FB79D01C29`
- `automation_reports/CV-SincNet/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/matrix.json`: SHA256 `767C1E3EE739FFC2B353C163369FEFF85F77E6C79FF58E88057A9725A5150512`
- `code/scripts/launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh`: SHA256 `63AD5B8F2DAE0DC5A1B26AA98B568D404D0254EF2F41E25F2BA399F103CB4BC5`

### Planned N607 sync mapping

- `E:\type10-7\tools\spaceborne_fewshot_da_matrix.py` -> `/home/szu2070436088/2510044040/CV-SincNet/tools/spaceborne_fewshot_da_matrix.py`
- `E:\type10-7\tools\optimizer_validate_matrix.py` -> `/home/szu2070436088/2510044040/CV-SincNet/tools/optimizer_validate_matrix.py`
- `E:\type10-7\code\scripts\launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh` -> `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldgeom_repair_20260626_022740\matrix.json` -> `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/matrix.json`

### Runner command boundary

- Intended remote working directory: `/home/szu2070436088/2510044040/CV-SincNet`
- Intended launcher: `bash code/scripts/launch_stage2_spaceborne_h06_oldgeom_repair_20260626_022740.sh`
- Default remote Python: `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- Expected logs: `/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/*.out`
- Expected outputs: `/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/*/metrics.json`, `manifest.json`, `score_table.csv`
- Max active per GPU: 2; this launcher starts up to 2 jobs per GPU only if slots are available.

### Risks and interpretation boundary

- This is a repair experiment, not deployment evidence and not a paper success claim.
- Success requires old-class accuracy and unknown false accept tradeoff to improve against the negative H06 consensus and H06 separability diagnostics.
- Any positive route must be confirmed by completed metrics and report update; startup PASS or landed submit is not runner completion.


## N607 runner landing update (2026-06-26T02:38:45+08:00)

### Remote validation and sync

- Direct N607 preflight before sync: PASS; artifact `automation_reports/CV-SincNet/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/artifacts/n607_ssh_preflight_before_sync.txt`.
- Remote monitor before sync: no project training process; artifact `automation_reports/CV-SincNet/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/artifacts/n607_remote_monitor_before_sync.json`.
- Remote py_compile/validator/bash syntax/dry-run: PASS; remote validator wrote `automation_reports/CV-SincNet/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/artifacts/validator_remote.json` on N607.
- Synced files: generator, validator, launcher, matrix using direct `N607` SCP only; bridge route not used.

### Submit result

- Remote submit command timed out locally before printing PID, so it was not used as proof by itself.
- Local stale SSH client from the timeout was closed: PID 1068; final local SSH/SCP cleanup is CLEAN.
- Read-only remote evidence confirmed launcher landed and is running.
- Remote launcher shell PID: `2933287`
- Remote launcher PID: `2933289`
- Launcher log: `/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/launcher_submit.out`
- Remote runs root: `/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_spaceborne_h06_oldgeom_repair_20260626_022740`
- Remote logs root: `/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_spaceborne_h06_oldgeom_repair_20260626_022740`

### Startup/progress probe

- Probe time: `2026-06-26T02:35:11.699231+08:00`
- Active candidate processes: `16`
- Launcher launched count: `35`
- Launcher complete count: `22`
- Launcher failed count: `0`
- Metrics present: `22`
- Manifests present: `22`
- Score tables present: `22`
- Logs present: `36`
- GPU compute apps count in that probe: `0`

### State and registry

- State updated: `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`
- State backup: `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json.bak_stage2_spaceborne_h06_oldgeom_repair_20260626_022740`
- Registry appended: `E:\type10-7\automation_reports\CV-SincNet\optimizer_execution_registry.jsonl`
- Next required action: `MONITOR_H06_OLDGEOM_REPAIR_RUN_TO_COMPLETION_AND_ANALYZE_METRICS`

### Claim boundary

This is only landed submit/startup-progress evidence. It is not runner completion, not Stage2 success, not deployment evidence, and not a paper/report success claim. The next automation run must monitor completion and analyze all completed metrics before any route interpretation.


## Final runner analysis (2026-06-26T02:43:08+08:00)

### Completion status

- Remote runner completed: 48/48 metrics, 48/48 manifests, 48/48 score tables, 48/48 feature files.
- Launcher failures: 0 in final probe.
- Analysis artifact: `automation_reports/CV-SincNet/stage2_spaceborne_h06_oldgeom_repair_20260626_022740/artifacts/h06_oldgeom_postrun_analysis.json`.
- Status: `COMPLETED_NEGATIVE_DIAGNOSTIC_OLDGEOM_NOT_PROMOTABLE`.

### Key metrics

- Old-class accuracy mean/max: `0.107060` / `0.355556`.
- Unknown FAR mean/min: `0.150347` / `0.000000`.
- Unknown rejection mean/max: `0.849653` / `1.000000`.
- Coverage mean/median: `0.171267` / `0.087500`.
- Old/unknown hmean mean/max: `0.158222` / `0.428039`.
- AUROC mean: `0.599842`.
- Target-hit count at old>=0.95 and unknown FAR<=0.05: `0`.

### Comparison against prior H06 old-primary consensus

- Old mean delta: `-0.020949`.
- Old max delta: `0.000000`.
- Unknown FAR mean delta: `-0.046528`.
- Hmean mean delta: `-0.017064`.
- Hmean max delta: `-0.018513`.

### Best candidate

- Candidate: `OA_MSE_H06_OLDGEOM48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0`.
- Old accuracy: `0.327778`.
- Unknown FAR: `0.383333`.
- Coverage: `0.525000`.
- Old/unknown hmean: `0.428039`.

### Interpretation

OLDGEOM improved the mean unknown FAR versus the H06 old-primary consensus but lost old retention and did not improve best hmean. The route is not promotable. The next optimizer step should design a fresh H06 repair from this artifact rather than relaunching OLDGEOM48 or the earlier old-primary consensus run in place. Target-new remains excluded.

### State and registry finalization

- State updated: `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`.
- Final state backup: `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json.bak_stage2_spaceborne_h06_oldgeom_repair_20260626_022740_final`.
- Registry appended: `E:\type10-7\automation_reports\CV-SincNet\optimizer_execution_registry.jsonl`.
- Next required action: `DESIGN_NEXT_H06_REPAIR_FROM_OLDGEOM_NEGATIVE_DIAGNOSTIC`.

### Final claim boundary

This is a completed negative Stage2-B old/unknown diagnostic. It is not Stage2 success, not deployment evidence, not Stage2-C seen-new success, and not a paper/report success claim.
