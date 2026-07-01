# phase2_old80first_head48_20260702_004317

## Objective

Validate OLD80_FIRST Phase2 target-old recovery before re-enabling open-set rejection optimization.
This matrix covers K=2/3/5/10 target-old support per class, excludes target-new support, keeps unknown query eval-only, and does not tune unknown query thresholds.
Ground model default: CEN51_R04_H06_LOW_PROB_HYBRID_R010 (${ROOT}/runs/cen51_r04_hybrid8_strong_leo_residual_20260624_1545/CEN51_R04_H06_LOW_PROB_HYBRID_R010/latest_model.pth).

## Candidate Matrix

| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |
|---|---:|---|---:|---|---|---|---|---|
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU0_A_MSE_SUBSPACE_KOLD2_KNEW0` | 0 | CVS-OA-MSE | 2 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU0_B_MSE_SUBSPACE_KOLD3_KNEW0` | 0 | CVS-OA-MSE | 3 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU0_C_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU0_D_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU0_E_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU0_F_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU1_A_MSE_SUBSPACE_KOLD2_KNEW0` | 1 | CVS-OA-MSE | 2 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU1_B_MSE_SUBSPACE_KOLD3_KNEW0` | 1 | CVS-OA-MSE | 3 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU1_C_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU1_D_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU1_E_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU1_F_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_A_MSE_SUBSPACE_KOLD2_KNEW0` | 2 | CVS-OA-MSE | 2 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_B_MSE_SUBSPACE_KOLD3_KNEW0` | 2 | CVS-OA-MSE | 3 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_C_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_D_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_E_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU3_A_MSE_SUBSPACE_KOLD2_KNEW0` | 3 | CVS-OA-MSE | 2 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU3_B_MSE_SUBSPACE_KOLD3_KNEW0` | 3 | CVS-OA-MSE | 3 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU3_C_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU3_D_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU3_E_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU3_F_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_A_MSE_SUBSPACE_KOLD2_KNEW0` | 4 | CVS-OA-MSE | 2 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_B_MSE_SUBSPACE_KOLD3_KNEW0` | 4 | CVS-OA-MSE | 3 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_C_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_D_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_E_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU5_A_MSE_SUBSPACE_KOLD2_KNEW0` | 5 | CVS-OA-MSE | 2 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU5_B_MSE_SUBSPACE_KOLD3_KNEW0` | 5 | CVS-OA-MSE | 3 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU5_C_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU5_D_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU5_E_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU5_F_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU6_A_MSE_SUBSPACE_KOLD2_KNEW0` | 6 | CVS-OA-MSE | 2 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU6_B_MSE_SUBSPACE_KOLD3_KNEW0` | 6 | CVS-OA-MSE | 3 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU6_C_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU6_D_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU6_E_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU6_F_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_A_MSE_SUBSPACE_KOLD2_KNEW0` | 7 | CVS-OA-MSE | 2 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_B_MSE_SUBSPACE_KOLD3_KNEW0` | 7 | CVS-OA-MSE | 3 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_D_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_E_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLD80FIRST_HEAD48_GPU7_F_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old80_first_head_first_recovery_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |

## Verification Contract

- Framework match: the ground-stage model is the existing generalized CVS-RFFI metric space checkpoint; this launcher only covers satellite-deployed target-old few-shot recovery.
- OLD80_FIRST phase gate: target-old old_class_accuracy must first recover to at least 0.72 and preferably reach 0.80 before unknown-FAR/open-set gate optimization is restored.
- Unknown query samples are eval-only. They must not fit thresholds, select heads, choose rollback, or become the primary objective in this phase.
- Target-new support and target-new claims are excluded in this matrix; every row is Stage2-B old/unknown-only FTRC.
- The active adaptation bundle is target adapter plus target-old support OLD80 head; Weibull/seen-new/Siamese/accepted-only open-set gates are deferred until the OLD80 phase gate is met.
- Report result tables by same-row metrics: candidate ID, K-shot, old80 head mode, old_class_accuracy, coverage, unknown_false_accept_rate, rollback/defer fields, and final verdict.
- Satellite metrics are stress-test metrics unless real in-orbit IQ is explicitly used.

## Launch

Run local/remote dry-run first:

```bash
bash code/scripts/launch_phase2_old80first_head48_20260702_004317.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.

## Local Implementation And Verification

Timestamp: 2026-07-02 00:43 Asia/Hong_Kong  
Operator: Codex  
Objective: implement OLD80_FIRST so Phase2 first restores target-old old_class_accuracy to >=0.72 and preferably 0.80 before restoring open-set rejection gates.  
Comparison target: previous Phase2 gate collapsed target-old old_class_accuracy to 20.8333%-27.5%; prior target-old-only diagnostics show >=72% average is feasible without unknown-first gating.

Changed local files:

| File | Purpose |
|---|---|
| `code/cvsrffi/spaceborne_fewshot.py` | Added query-label-free OLD80_FIRST old-class head over source old prototypes plus target-old support. |
| `code/eval_spaceborne_fewshot.py` | Added OLD80 head CLI, evaluation hook, score-table diagnostics, and telemetry. |
| `tools/spaceborne_fewshot_da_matrix.py` | Added `OA_MSE_H06_OLD80FIRST_HEAD48` matrix with K=2/3/5/10, fixed unknown threshold, no target-new support, unknown eval-only. |
| `tools/optimizer_validate_matrix.py` | Added validator category and OLD80_FIRST head-specific bundle validation. |
| `tests/test_spaceborne_fewshot_da_matrix.py` | Added matrix/validator coverage for OLD80_FIRST old-first semantics. |
| `code/tests/test_spaceborne_fewshot_oa_mse_repair.py` | Added unit test for recovering rejected old queries from support-only OLD80 head. |

Verification commands:

| Command | Result |
|---|---|
| `python -m py_compile code\cvsrffi\spaceborne_fewshot.py code\eval_spaceborne_fewshot.py tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py` | PASS |
| `python -m pytest tests\test_spaceborne_fewshot_da_matrix.py::SpaceborneFewShotDaMatrixTest::test_h06_oldrecov48_restores_target_old_recoverability_first_without_seen_new tests\test_spaceborne_fewshot_da_matrix.py::SpaceborneFewShotDaMatrixTest::test_h06_old80first_head48_prioritizes_old_accuracy_before_unknown_gate code\tests\test_spaceborne_fewshot_oa_mse_repair.py::test_old80_first_head_recovers_rejected_old_queries_from_support_only code\tests\test_spaceborne_fewshot_oa_mse_repair.py::test_oa_mse_multiproto_score_uses_same_class_support_anchor_mixture -q` | PASS, with non-fatal `.pytest_cache` WinError 5 warning |
| `python tools\optimizer_validate_matrix.py automation_reports\CV-SincNet\phase2_old80first_head48_20260702_004317\matrix.json` | PASS, 48 launchable rows |

Key configuration:

| Field | Value |
|---|---|
| Plan | `OA_MSE_H06_OLD80FIRST_HEAD48` |
| Candidate count | 48 Phase2 rows |
| K-shot old support | 2,3,5,10 per old class |
| Target-new support | excluded, `target_new_support_per_tx=0` |
| Unknown role | eval-only; cannot fit thresholds, select heads, choose rollback, or drive primary objective |
| Unknown threshold | fixed `0.96` for all rows; not an optimization axis |
| Old head modes | `support_cv_select`, `fused_centroid`, `support_knn3`, `support_centroid` |
| Apply policy | `replace_all` before open-set gate restoration |
| Phase gate | `old_class_accuracy>=0.80`; 0.72 is the minimum recovery sanity floor |
| Open-set bundle | deferred until OLD80 phase gate is met |

Planned N607 launch context:

| Item | Value |
|---|---|
| Working directory | `/home/szu2070436088/2510044040/CV-SincNet` |
| Conda/Python env | project default remote env used by launcher; verify before launch |
| Server command | `bash code/scripts/launch_phase2_old80first_head48_20260702_004317.sh` |
| Launcher | `code/scripts/launch_phase2_old80first_head48_20260702_004317.sh` |
| Run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_old80first_head48_20260702_004317/` |
| Log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_head48_20260702_004317/` |
| GPU allocation | generated as 6 candidates per GPU across GPU0-GPU7; launch only after preflight and capacity check |
| Expected outputs | `features.npz`, `metrics.json`, `manifest.json`, `score_table.csv` per candidate |
| Success criterion | same-row candidate reaches target-old old_class_accuracy >=0.72, preferably >=0.80; unknown FAR is recorded but not used as primary selector in this phase |
| Known risk | `replace_all` should intentionally raise unknown false accept rate; this is acceptable only for OLD80 recovery diagnosis and is not a deployment success claim |

Sync status: completed on 2026-07-02 00:50 Asia/Hong_Kong. N607 submission is queued behind existing GPU jobs.

## N607 Sync, Verification, And Launch Handoff

Preflight and occupancy:

| Item | Evidence |
|---|---|
| Preflight | `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1` PASS via direct `N607` target |
| Remote host | `dell-DSS8440` |
| Remote project root | `/home/szu2070436088/2510044040/CV-SincNet` |
| GPU state before submit | 8 RTX 3090 GPUs visible; each GPU had 2 active Python compute processes |
| Capacity rule | `STAGE2_MAX_ACTIVE_PER_GPU=2`, `INCLUDE_EXTERNAL_GPU_PROCS=1`; launcher must wait while external count is 2 |
| Local SSH cleanup | verified `ssh_processes=none` and `n607_established=none` after sync, validation, submit, and monitor checks |

Remote sync destinations:

| Local file | Remote file |
|---|---|
| `E:\type10-7\code\cvsrffi\spaceborne_fewshot.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/spaceborne_fewshot.py` |
| `E:\type10-7\code\eval_spaceborne_fewshot.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/eval_spaceborne_fewshot.py` |
| `E:\type10-7\tools\spaceborne_fewshot_da_matrix.py` | `/home/szu2070436088/2510044040/CV-SincNet/tools/spaceborne_fewshot_da_matrix.py` |
| `E:\type10-7\tools\optimizer_validate_matrix.py` | `/home/szu2070436088/2510044040/CV-SincNet/tools/optimizer_validate_matrix.py` |
| `E:\type10-7\tests\test_spaceborne_fewshot_da_matrix.py` | `/home/szu2070436088/2510044040/CV-SincNet/tests/test_spaceborne_fewshot_da_matrix.py` |
| `E:\type10-7\code\tests\test_spaceborne_fewshot_oa_mse_repair.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_spaceborne_fewshot_oa_mse_repair.py` |
| `E:\type10-7\code\scripts\launch_phase2_old80first_head48_20260702_004317.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_old80first_head48_20260702_004317.sh` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_head48_20260702_004317\matrix.json` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_old80first_head48_20260702_004317/matrix.json` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_head48_20260702_004317\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_old80first_head48_20260702_004317/report.md` |

Remote verification:

| Command | Result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/cvsrffi/spaceborne_fewshot.py code/eval_spaceborne_fewshot.py tools/spaceborne_fewshot_da_matrix.py tools/optimizer_validate_matrix.py` | PASS |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python tools/optimizer_validate_matrix.py automation_reports/CV-SincNet/phase2_old80first_head48_20260702_004317/matrix.json` | PASS: 48 launchable rows, 0 issues |
| `bash -n code/scripts/launch_phase2_old80first_head48_20260702_004317.sh` | PASS |

Remote file hashes after sync:

| File | SHA256 |
|---|---|
| `code/cvsrffi/spaceborne_fewshot.py` | `0b1df0e1d2c139ac53906bffe9bb48b5e834891e97c6bc22861e6213eefc9490` |
| `code/eval_spaceborne_fewshot.py` | `edba2abab702e9073fd766a41808c38a7b03d44364e2773fb9c2bb51edfdf524` |
| `tools/spaceborne_fewshot_da_matrix.py` | `64bc99fa5dddcafa9b82cef6706620263e9c9721941002fec30cca1a39f46306` |
| `tools/optimizer_validate_matrix.py` | `80093ff1f355d45c8d64911bd7d75459e4cd3784949f67d85dde45afa722028e` |
| `code/scripts/launch_phase2_old80first_head48_20260702_004317.sh` | `31037f05bb691c43be714e872f7beb22199d4f8d3d60015da3602ba215310a47` |
| `automation_reports/CV-SincNet/phase2_old80first_head48_20260702_004317/matrix.json` | `506ab5c4017d78ac7ce612a4160e842a8db2e2d09339996d221856571877bb04` |

Launch command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup env STAGE2_MAX_ACTIVE_PER_GPU=2 INCLUDE_EXTERNAL_GPU_PROCS=1 bash code/scripts/launch_phase2_old80first_head48_20260702_004317.sh > logs/phase2_old80first_head48_20260702_004317/launcher_submit.out 2>&1 &
```

Launch status:

| Item | Value |
|---|---|
| Launcher PID | `2887894` |
| Launcher status after submit | alive, parent changed to PID 1 |
| Submit log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_head48_20260702_004317/launcher_submit.out` |
| Current queue state | `QUEUED_WAITING_GPU_SLOT` |
| Latest observed wait line | `[SPACEBORNE-FSDA-WAIT] gpu=0 active=0 external=2 total=2 max=2` |
| Candidate logs at first monitor | 0 |
| Completed `metrics.json` at first monitor | 0 |

Next inspection:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
tail -n 40 logs/phase2_old80first_head48_20260702_004317/launcher_submit.out
find runs/phase2_old80first_head48_20260702_004317 -maxdepth 2 -name metrics.json | wc -l
```

Interpretation boundary: this OLD80_FIRST run is a Stage2-B old/unknown-only recovery experiment. Unknown FAR is recorded, but it is not the selection objective until target-old old_class_accuracy first recovers to at least 0.72 and preferably 0.80.

## Continuation Monitor And Scheduler Repair

Continuation timestamp: 2026-07-02 00:54 Asia/Hong_Kong  
Operator: Codex  
Objective: keep moving toward OLD80_FIRST evidence without exceeding N607 GPU packing limits.

Current N607 monitor evidence:

| Item | Value |
|---|---|
| Preflight | direct `N607` preflight PASS |
| Launcher PID | `2887894` still alive |
| Launcher elapsed at monitor | about 4 minutes 37 seconds |
| GPU occupancy | each of GPU0-GPU7 still had 2 Python compute processes |
| Latest launcher state | repeated `[SPACEBORNE-FSDA-WAIT] gpu=0 active=0 external=2 total=2 max=2` |
| Candidate logs | 0 |
| Completed metrics | 0 |
| Run dirs | 0 |
| Local SSH cleanup | verified `ssh_processes=none` and `n607_established=none` after monitor |

Scheduler issue found:

| Issue | Impact | Action |
|---|---|---|
| Generated launcher waits on the first candidate's GPU before scanning later candidates. | If GPU1-GPU7 free before GPU0, OLD80_FIRST evidence collection is delayed by GPU0 head-of-line blocking. | Repaired the matrix generator locally so future launchers use non-blocking `gpu_slot_available` and scan all remaining candidates before sleeping. |

Local scheduler repair artifacts:

| Artifact | Path | Status |
|---|---|---|
| Updated generator | `E:\type10-7\tools\spaceborne_fewshot_da_matrix.py` | local verified |
| Updated launcher test | `E:\type10-7\tests\test_spaceborne_fewshot_da_matrix.py` | local verified |
| Scheduler-safe launcher | `E:\type10-7\code\scripts\launch_phase2_old80first_head48_sched_20260702_0055.sh` | generated, not submitted |
| Scheduler-safe matrix | `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_head48_sched_20260702_0055\matrix.json` | validator PASS |
| Scheduler-safe report | `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_head48_sched_20260702_0055\report.md` | generated |

Local verification:

| Command | Result |
|---|---|
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile tools\spaceborne_fewshot_da_matrix.py` | PASS |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest tests\test_spaceborne_fewshot_da_matrix.py::SpaceborneFewShotDaMatrixTest::test_launcher_waits_for_background_candidate_jobs tests\test_spaceborne_fewshot_da_matrix.py::SpaceborneFewShotDaMatrixTest::test_h06_old80first_head48_prioritizes_old_accuracy_before_unknown_gate -q` | PASS, non-fatal `.pytest_cache` permission warning |
| `C:\Users\lh594\.conda\envs\ssr-gpu\python.exe tools\optimizer_validate_matrix.py automation_reports\CV-SincNet\phase2_old80first_head48_sched_20260702_0055\matrix.json` | PASS, 48 launchable rows |
| `bash -n ./code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh` | PASS |
| `bash ./code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh --dry-run` | PASS, emitted 48 candidate command blocks and exited |

N607 action boundary: the scheduler-safe launcher has not been submitted because the previous OLD80_FIRST launcher is already alive and waiting. Starting a second launcher against the same candidate family would risk duplicate outputs. The current safe action is monitor-only until the existing launcher either starts candidates, finishes, or is explicitly superseded.

Latest monitor update: 2026-07-02 01:00 Asia/Hong_Kong

| Item | Value |
|---|---|
| Launcher PID | `2887894` still alive |
| Launcher elapsed | about 10 minutes 53 seconds |
| Latest launcher state | still repeated `[SPACEBORNE-FSDA-WAIT] gpu=0 active=0 external=2 total=2 max=2` |
| Candidate logs | 0 |
| Completed `metrics.json` | 0 |
| Local SSH cleanup | verified `ssh_processes=none` and `n607_established=none` |

Status interpretation: no OLD80_FIRST metric evidence is available yet. The run is resource-queued, not failed.
