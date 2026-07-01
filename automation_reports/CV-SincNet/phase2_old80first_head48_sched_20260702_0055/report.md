# phase2_old80first_head48_sched_20260702_0055

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
bash code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.

## N607 Sync, Verification, And Launch Handoff

Timestamp: 2026-07-02 01:02 Asia/Hong_Kong  
Operator: Codex  
Objective: replace the GPU0-blocked OLD80_FIRST launcher with a scheduler-safe launcher that scans all remaining candidate GPUs before sleeping.

Preflight and occupancy:

| Item | Evidence |
|---|---|
| Preflight | direct `N607` preflight PASS |
| Remote host | `dell-DSS8440` |
| Remote project root | `/home/szu2070436088/2510044040/CV-SincNet` |
| GPU state | 8 RTX 3090 GPUs visible; each GPU still had 2 active Python compute processes |
| Capacity rule | `STAGE2_MAX_ACTIVE_PER_GPU=2`, `INCLUDE_EXTERNAL_GPU_PROCS=1`; no candidate starts until some GPU has fewer than 2 external compute processes |

Remote sync destinations:

| Local file | Remote file |
|---|---|
| `E:\type10-7\tools\spaceborne_fewshot_da_matrix.py` | `/home/szu2070436088/2510044040/CV-SincNet/tools/spaceborne_fewshot_da_matrix.py` |
| `E:\type10-7\tests\test_spaceborne_fewshot_da_matrix.py` | `/home/szu2070436088/2510044040/CV-SincNet/tests/test_spaceborne_fewshot_da_matrix.py` |
| `E:\type10-7\code\scripts\launch_phase2_old80first_head48_sched_20260702_0055.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_head48_sched_20260702_0055\matrix.json` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_old80first_head48_sched_20260702_0055/matrix.json` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_old80first_head48_sched_20260702_0055\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_old80first_head48_sched_20260702_0055/report.md` |

Remote verification:

| Command | Result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile tools/spaceborne_fewshot_da_matrix.py` | PASS |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python tools/optimizer_validate_matrix.py automation_reports/CV-SincNet/phase2_old80first_head48_sched_20260702_0055/matrix.json` | PASS: 48 launchable rows, 0 issues |
| `bash -n code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh` | PASS |
| `grep -q "SPACEBORNE-FSDA-WAIT-ANY" code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh` | PASS |

Remote hashes:

| File | SHA256 |
|---|---|
| `tools/spaceborne_fewshot_da_matrix.py` | `6a7d285a4f1e10d4619d7a3831ee3e6268a2f6e661ce1721c648109b895895d6` |
| `code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh` | `f0d0949bd7d90c221406fc7d56bb0c9b91187b9a76b1ba08ba9da46f7600f87c` |
| `automation_reports/CV-SincNet/phase2_old80first_head48_sched_20260702_0055/matrix.json` | `f53516c9a694854429086f755f71f20d5a548f42228062d1afc334769ed3f53b` |

Superseded old launcher:

| Item | Value |
|---|---|
| Old run ID | `phase2_old80first_head48_20260702_004317` |
| Old launcher PID | `2887894` |
| Safety check | only one `sleep 5` child; candidate logs=0, metrics=0, run dirs=0 |
| Action | terminated only the waiting bash launcher and sleep child; no Python training process was killed |

Launch command:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup env STAGE2_MAX_ACTIVE_PER_GPU=2 INCLUDE_EXTERNAL_GPU_PROCS=1 SCHEDULER_POLL_SECONDS=5 bash code/scripts/launch_phase2_old80first_head48_sched_20260702_0055.sh > logs/phase2_old80first_head48_sched_20260702_0055/launcher_submit.out 2>&1 &
```

Launch status:

| Item | Value |
|---|---|
| Launcher PID | `2896435` |
| Submit log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_old80first_head48_sched_20260702_0055/launcher_submit.out` |
| Current scheduler state | `[SPACEBORNE-FSDA-WAIT-ANY] remaining=48 max=2` |
| Candidate logs at first monitor | 0 |
| Completed `metrics.json` at first monitor | 0 |
| Local SSH cleanup | verified `ssh_processes=none` and `n607_established=none` after replacement monitor |

Interpretation boundary: no OLD80_FIRST metric evidence is available yet. This run is correctly queued behind the two-process-per-GPU safety rule and will start when any GPU has capacity, without waiting specifically for GPU0.

Latest monitor update: 2026-07-02 01:06 Asia/Hong_Kong

| Item | Value |
|---|---|
| Launcher PID | `2896435` still alive |
| Launcher elapsed | about 2 minutes 13 seconds |
| Current scheduler state | repeated `[SPACEBORNE-FSDA-WAIT-ANY] remaining=48 max=2` |
| Candidate logs | 0 |
| Completed `metrics.json` | 0 |
| Old launcher PID `2887894` | not alive |
| Local SSH cleanup | verified `ssh_processes=none` and `n607_established=none` |

Status interpretation: the scheduler repair is active and healthy, but no OLD80_FIRST metric evidence is available yet because no GPU has fallen below the two-process cap.

Latest monitor update: 2026-07-02 01:09 Asia/Hong_Kong

| Item | Value |
|---|---|
| Launcher PID | `2896435` still alive |
| Launcher elapsed | about 5 minutes 58 seconds |
| Current scheduler state | repeated `[SPACEBORNE-FSDA-WAIT-ANY] remaining=48 max=2` |
| Candidate logs | 0 |
| Completed `metrics.json` | 0 |
| Run directories | 0 |
| GPU occupancy | 16 active Python compute processes, 2 on each RTX 3090 GPU |
| Local SSH cleanup | verified `ssh_processes=none` and `n607_established=none` |

Status interpretation: no candidate has started and no metric evidence exists yet. This is not a candidate failure; the scheduler is correctly waiting because every GPU is already at the two-process safety cap.

Latest monitor update: 2026-07-02 01:12 Asia/Hong_Kong

| Item | Value |
|---|---|
| Launcher PID | `2896435` still alive |
| Launcher elapsed | about 8 minutes 55 seconds |
| Current scheduler state | repeated `[SPACEBORNE-FSDA-WAIT-ANY] remaining=48 max=2` |
| Candidate logs | 0 |
| Completed `metrics.json` | 0 |
| Run directories | 0 |
| GPU occupancy | 16 active Python compute processes, 2 on each RTX 3090 GPU |
| Recent errors | none found in scheduler grep for Traceback/ERROR/RuntimeError/unrecognized arguments |
| Local SSH cleanup | verified `ssh_processes=none`, `n607_established=none`, and `bridge_established=none` |

Status interpretation: the OLD80_FIRST queue is still healthy but has not produced evidence. The next decision point remains the first completed candidate metrics row; until then the old-class recovery objective is unverified.

Latest monitor update: 2026-07-02 01:15 Asia/Hong_Kong

| Item | Value |
|---|---|
| Launcher PID | `2896435` still alive |
| Launcher elapsed | about 11 minutes 33 seconds |
| Current scheduler state | repeated `[SPACEBORNE-FSDA-WAIT-ANY] remaining=48 max=2` |
| Candidate logs | 0 |
| Completed `metrics.json` | 0 |
| Run directories | 0 |
| GPU occupancy | 16 active Python compute processes, 2 on each RTX 3090 GPU |
| Occupying run family | `phase1_adv3_mechanism32_queue_20260701` training jobs, not OLD80_FIRST candidates |
| Example occupying candidates | `ADV3B25_SOURCE_MIX075_E200`, `ADV3B09_HARD48_E200`, `ADV3B21_SOFTCE_LOW_E200`, `ADV3B01_CORE80_STRICT_E200`, `ADV3B31_BAL_MAIN_E160` |
| Recent errors | none found in scheduler grep for Traceback/ERROR/RuntimeError/unrecognized arguments/CUDA OOM/Killed |
| Monitor note | one exploratory GPU process command probe had a shell syntax error after printing queue status; a smaller one-line read-only process query then succeeded |
| Local SSH cleanup | verified `ssh_processes=none`, `n607_established=none`, and `bridge_established=none` |

Status interpretation: OLD80_FIRST remains queued behind existing Phase1 training occupancy. No old-class recovery evidence is available yet, so the OLD80_FIRST objective remains unverified and should not be judged by FAR or any missing metrics.
