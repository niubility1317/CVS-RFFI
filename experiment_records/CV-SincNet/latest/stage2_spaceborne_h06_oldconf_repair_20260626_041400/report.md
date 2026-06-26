# stage2_spaceborne_h06_oldconf_repair_20260626_041400

## Objective

Validate the two deployment-time few-shot adaptation paths from the spaceborne CVS-RFFI design:
new-TX enrollment (`CVS-SFE`) and target receiver calibration (`CVS-FTRC`).
Ground model default: CEN51_R04_H06_LOW_PROB_HYBRID_R010 (${ROOT}/runs/cen51_r04_hybrid8_strong_leo_residual_20260624_1545/CEN51_R04_H06_LOW_PROB_HYBRID_R010/latest_model.pth).

## Candidate Matrix

| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |
|---|---:|---|---:|---|---|---|---|---|
| `OA_MSE_H06_OLDCONF48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU0_B_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU0_C_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU0_D_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU0_E_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU0_F_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU1_A_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU1_B_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU1_C_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU1_D_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU1_E_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU1_F_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU2_A_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU2_B_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU2_C_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU2_D_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU2_E_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU3_A_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU3_B_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU3_C_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU3_D_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU3_E_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU3_F_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU4_A_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU4_B_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU4_C_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU4_D_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU4_E_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU5_A_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU5_B_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU5_C_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU5_D_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU5_E_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU5_F_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU6_A_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU6_B_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU6_C_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU6_D_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU6_E_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU6_F_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU7_A_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU7_B_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU7_D_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU7_E_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDCONF48_GPU7_F_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_conformal_retention_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |

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
bash code/scripts/launch_stage2_spaceborne_h06_oldconf_repair_20260626_041400.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.

## 2026-06-26 Automation Control Audit

Operator: Codex automation `cv-sincnet-n607-monitor-optimizer-v4-2`.

Required control files read in order:

- `E:\type10-7\AGENTS.md`
- `E:\type10-7\项目.md`
- `E:\type10-7\tools\optimizer_control_manifest.md`
- `E:\type10-7\automation_reports\CV-SincNet\automation_prompt_backups\20260615_001820_stage2_closed_loop_v4\stage2_prompt.md`
- `E:\type10-7\tools\optimizer_workflow_contract.md`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`

State authority before this run:

- `required_next_action`: `DESIGN_NEXT_H06_REPAIR_FROM_OLDGEOM_NEGATIVE_DIAGNOSTIC`
- OLDGEOM48 postrun was a completed negative diagnostic: `target_hit_count=0`, `old_class_accuracy_mean=0.1070601871`, `unknown_false_accept_rate_mean=0.1503472235`, `old_unknown_hmean_max=0.4280392136`.
- Hard boundary: do not relaunch OLDGEOM48 or the previous old-primary consensus path; keep Stage2-B old/unknown-only with target-new excluded until separation improves.

Lane monitor evidence:

- Local N607 direct preflight passed.
- Inventory artifact: `automation_reports\CV-SincNet\stage2_monitor_optimizer_h06_nextrepair_20260626_041400\artifacts\n607_training_inventory_20260626_041400.json`
- `phase1_monitor_state=1`, `phase2_monitor_state=1`, no active training processes, no ambiguous cross-lane processes.
- SSH cleanup artifact: `automation_reports\CV-SincNet\stage2_monitor_optimizer_h06_nextrepair_20260626_041400\artifacts\ssh_cleanup_after_monitor_20260626_041400.json`

Multi-role review summary:

- Protocol review: OLDGEOM48 is non-promotable and must not be relaunched; `KOLD>=1`, `KNEW=0`, Rt/Rs disjoint, unknown eval-only, no Stage2-C or seen-new claim.
- Evidence/innovation review: OLDGEOM reduced unknown FAR but further harmed old retention; next path should test support reliability/conformal or reconstruction-backed retention rather than another margin/temperature replay.
- Runner review: a valid next run needs a fresh run ID, matrix, launcher, paths, registry key, and command hash; same OLDGEOM registry identity is a hard blocker.
- Validation/supervision review: validator PASS on OLDGEOM does not authorize relaunch; the validator must recognize the new old/unknown-only plan token before launch.

Local changes for this run:

- `tools\spaceborne_fewshot_da_matrix.py`: added `OA_MSE_H06_OLDCONF48`, a support-conformal/reconstruction retention repair plan derived from OLDGEOM evidence while preserving Stage2-B old/unknown-only protocol.
- `tools\optimizer_validate_matrix.py`: added `OLDCONF` token to old/unknown-only validator checks.
- `code\scripts\launch_stage2_spaceborne_h06_oldconf_repair_20260626_041400.sh`: generated fresh launcher for 48 candidates.
- Snapshot: `E:\type10-7\code\snapshots\stage2_spaceborne_h06_oldconf_repair_20260626_041400`
- Hash artifact: `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\local_hashes_and_snapshot.json`

Local verification:

- `conda run -n ssr-gpu python -m py_compile tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py` passed.
- `conda run -n ssr-gpu python tools\spaceborne_fewshot_da_matrix.py --plan OA_MSE_H06_OLDCONF48 --run-id stage2_spaceborne_h06_oldconf_repair_20260626_041400` generated 48 candidates.
- `conda run -n ssr-gpu python tools\optimizer_validate_matrix.py automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\matrix.json --expected-count 48 --launcher code\scripts\launch_stage2_spaceborne_h06_oldconf_repair_20260626_041400.sh --repair-launcher-identity --output automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\validator_repair_identity.json` passed with `issue_count=0` and 48 launchable rows.
- `bash -n code\scripts\launch_stage2_spaceborne_h06_oldconf_repair_20260626_041400.sh` passed.
- Local launcher dry-run exited 0 and emitted 48 `SPACEBORNE-FSDA-CANDIDATE` lines.

Claim boundary before remote actions:

- This report section records a locally validated repair candidate only.
- It is not runner completion, not Stage2 success, not deployment evidence, and not Stage2-C seen-new evidence.

## Remote Sync Plan

Pre-sync N607 gate:

- `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`: passed.
- `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe tools\n607_training_inventory.py --direct-only --pretty`: passed with no active training processes and no GPU compute apps.
- Cleanup artifact: `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\ssh_cleanup_before_sync_preflight_inventory.json`

Local-to-remote mapping:

- `E:\type10-7\tools\spaceborne_fewshot_da_matrix.py` -> `/home/szu2070436088/2510044040/CV-SincNet/tools/spaceborne_fewshot_da_matrix.py`
- `E:\type10-7\tools\optimizer_validate_matrix.py` -> `/home/szu2070436088/2510044040/CV-SincNet/tools/optimizer_validate_matrix.py`
- `E:\type10-7\code\scripts\launch_stage2_spaceborne_h06_oldconf_repair_20260626_041400.sh` -> `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_stage2_spaceborne_h06_oldconf_repair_20260626_041400.sh`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\matrix.json` -> `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/stage2_spaceborne_h06_oldconf_repair_20260626_041400/matrix.json`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\report.md` -> `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/stage2_spaceborne_h06_oldconf_repair_20260626_041400/report.md`

Remote gates after sync must pass before any runner submit:

- Remote SHA256 comparison against `local_hashes_and_snapshot.json`.
- Remote `py_compile` for changed Python files.
- Remote validator with `--expected-count 48`, launcher identity repair output, and zero issues.
- Remote `bash -n` and launcher `--dry-run` candidate count check.

## Remote Runner Completion Update

Timestamp: 2026-06-26 04:38 CST. Route: direct `N607`.

Remote gate and submit evidence:

- Remote SHA256 comparison: PASS.
- Remote `py_compile`: PASS using `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`, the same Python path declared by the launcher.
- Remote validator: PASS, 48/48 launchable, 0 issues.
- Remote launcher dry-run: PASS with 48 `SPACEBORNE-FSDA-CANDIDATE` lines.
- Submit SSH command timed out locally after landing; stale local `ssh.exe` was found and killed. Follow-up read-only probes confirmed the run landed and executed, so no relaunch was attempted.
- Final remote probe: 48 run dirs, 48 metrics, 48 manifests, 48 score tables, 48 feature files, 0 launcher failures, 0 traceback/error logs, no active process lines for this run.
- Final cleanup artifact: `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\ssh_cleanup_after_remote_final_probe_analysis_fetch.json`

Pulled local artifacts:

- `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\oldconf_gate_analysis.json`
- `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\remote_launcher_submit.out`
- `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\remote_final_probe_and_analysis_stdout.txt`

Post-run metrics from 48 candidates:

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| old_class_accuracy | 0.127662 | 0.077778 | 0.000000 | 0.366667 |
| unknown_false_accept_rate | 0.193403 | 0.108333 | 0.000000 | 0.683333 |
| unknown_rejection_rate | 0.806597 | 0.891667 | 0.316667 | 1.000000 |
| coverage | 0.210417 | 0.131250 | 0.000000 | 0.654167 |
| full_accuracy | 0.297396 | 0.275000 | 0.229167 | 0.387500 |
| old_unknown_hmean | 0.172332 | 0.138334 | 0.000000 | 0.415321 |
| auroc | 0.626111 | 0.618102 | 0.492778 | 0.759444 |
| loss_initial | 10.483408 | 10.250624 | 7.026703 | 14.188387 |
| loss_final | 1.899670 | 1.858526 | 1.315039 | 2.939555 |

Best hmean candidates:

- `OA_MSE_H06_OLDCONF48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0`: old=0.327778, unknown FAR=0.433333, unknown rejection=0.566667, coverage=0.495833, hmean=0.415321, selected_alpha=0.75.
- `OA_MSE_H06_OLDCONF48_GPU6_B_MSE_SUBSPACE_KOLD5_KNEW0`: old=0.327778, unknown FAR=0.433333, unknown rejection=0.566667, coverage=0.483333, hmean=0.415321, selected_alpha=0.75.
- `OA_MSE_H06_OLDCONF48_GPU2_A_MSE_SUBSPACE_KOLD5_KNEW0`: old=0.300000, unknown FAR=0.366667, unknown rejection=0.633333, coverage=0.454167, hmean=0.407143, selected_alpha=1.0.

Comparison against OLDGEOM48:

- Old mean improved from 0.107060 to 0.127662, but remains low.
- Unknown FAR worsened from 0.150347 to 0.193403.
- Best hmean fell from 0.428039 to 0.415321.
- Target-hit count remains 0 under the old>=0.95 and unknown FAR<=0.05 deployment target.

Gate diagnosis:

- True old accepted rate: 0.216088.
- True unknown accepted rate/FAR: 0.193403.
- Old-primary consistency pass: old 0.558218 vs unknown 0.440278.
- Support-KNN pass: old 0.717940 vs unknown 0.672222.
- Drift pass: old 0.993981 vs unknown 0.993750.
- Class-envelope pass: old 0.739120 vs unknown 0.636111.
- Unknown-veto applied: old 0.518634 vs unknown 0.406944.
- Support conformal/reconstruction rejection rates remain close between old and unknown, so they did not create a usable separation boundary.

Runner conclusion:

- Status: `COMPLETED_NEGATIVE_DIAGNOSTIC_OLDCONF_NOT_PROMOTABLE`.
- This is a completed Stage2-B old/unknown-only diagnostic, not Stage2 success, not deployment evidence, not Stage2-C seen-new success, and not a paper/report success claim.
- Next action should be a fresh H06 representation/prototype or support-quality repair from the OLDGEOM+OLDCONF negative evidence. Do not relaunch OLDGEOM48, OLDCONF48, or the previous old-primary consensus path in place.

State and registry finalization:

- State updated: `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`.
- State backup: `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json.bak_stage2_spaceborne_h06_oldconf_repair_20260626_041400_final`.
- Registry appended: `E:\type10-7\automation_reports\CV-SincNet\optimizer_execution_registry.jsonl`.
- Current view refreshed: `E:\type10-7\automation_reports\CV-SincNet\current_state_view_latest_for_automation.json`.
- Required next action: `DESIGN_NEXT_H06_REPRESENTATION_REPAIR_FROM_OLDCONF_NEGATIVE_DIAGNOSTIC`.
- Final N607 inventory: `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\n607_training_inventory_final.json`, monitor_state=1 and no active training processes.
- Final report sync cleanup: `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldconf_repair_20260626_041400\artifacts\ssh_cleanup_after_final_report_sync.json`, status `CLEAN`.
