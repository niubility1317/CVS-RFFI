# stage2_spaceborne_h06_oldfuse_repair_20260626_122707

## Objective

Validate the two deployment-time few-shot adaptation paths from the spaceborne CVS-RFFI design:
new-TX enrollment (`CVS-SFE`) and target receiver calibration (`CVS-FTRC`).
Ground model default: CEN51_R04_H06_LOW_PROB_HYBRID_R010 (${ROOT}/runs/cen51_r04_hybrid8_strong_leo_residual_20260624_1545/CEN51_R04_H06_LOW_PROB_HYBRID_R010/latest_model.pth).

## Candidate Matrix

| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |
|---|---:|---|---:|---|---|---|---|---|
| `OA_MSE_H06_OLDFUSE48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU0_B_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU0_C_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU0_D_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU0_E_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU0_F_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU1_A_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU1_B_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU1_C_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU1_D_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU1_E_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU1_F_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU2_A_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU2_B_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU2_C_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU2_D_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU2_E_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU3_A_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU3_B_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU3_C_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU3_D_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU3_E_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU3_F_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU4_A_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU4_B_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU4_C_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU4_D_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU4_E_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU5_A_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU5_B_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU5_C_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU5_D_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU5_E_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU5_F_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU6_A_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU6_B_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU6_C_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU6_D_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU6_E_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU6_F_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU7_A_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU7_B_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU7_D_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU7_E_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDFUSE48_GPU7_F_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_oldqual_oldrisk_fusion_rollback_calibration_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |

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
bash code/scripts/launch_stage2_spaceborne_h06_oldfuse_repair_20260626_122707.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.

## Automation Control Audit

Run ID: `stage2_spaceborne_h06_oldfuse_repair_20260626_122707`
Automation: `cv-sincnet-n607-monitor-optimizer-v4-2`
Timestamp: `2026-06-26T12:31:18+08:00`
Operator: Codex automation thin wrapper.

Required controls read in order:

| Order | Path | Result |
|---:|---|---|
| 1 | `E:\type10-7\AGENTS.md` | PASS |
| 2 | `E:\type10-7\项目.md` | PASS |
| 3 | `E:\type10-7\tools\optimizer_control_manifest.md` | PASS |
| 4 | `E:\type10-7\automation_reports\CV-SincNet\automation_prompt_backups\20260615_001820_stage2_closed_loop_v4\stage2_prompt.md` | PASS |
| 5 | `E:\type10-7\tools\optimizer_workflow_contract.md` | PASS |
| 6 | `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json` | PASS |

Current-state authority:

| Field | Value |
|---|---|
| `required_next_action` | `DESIGN_NEXT_H06_OLDQUAL_OLDRISK_FUSION_OR_ROLLBACK_CALIBRATION_REPAIR` |
| Prior runner | `stage2_spaceborne_h06_oldrisk_repair2_20260626_103559` |
| Prior verdict | `COMPLETED_NEGATIVE_DIAGNOSTIC_OLDRISK_TRADEOFF_IMPROVED_NOT_PROMOTABLE` |
| Protocol boundary | Stage2-B old/unknown-only; `KNEW0`; target-new support excluded |

## Multi-Role Review

| Role | Summary |
|---|---|
| Protocol/state | Next route must be a fresh OLDQUAL/OLDRISK fusion or rollback calibration repair; do not relaunch unchanged OLDGEOM, OLDCONF, OLDBUDGET, OLDQUAL, OLDRISK, or old-primary consensus paths. |
| Evidence | OLDQUAL kept better old retention but had high unknown false acceptance; OLDRISK reduced unknown FAR but rolled back all 48 rows, so the new matrix must test fusion and rollback calibration rather than paper-success claims. |
| Runner/code | Use fresh run ID, launcher, matrix, registry keys, and command hashes; avoid the invalid old `regularized_small_adapter` policy and use accepted adapter policies only. |
| Validation/supervision | Validate protocol exclusion of target-new, category balance, launcher identity, and dry-run command coverage before any N607 launch. |

Prior evidence used for design:

| Prior route | Old mean | Old max | Unknown FAR mean | Unknown FAR min | Rollback | Target hits | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| OLDQUAL | 0.436 | 0.567 | 0.676 | 0.250 | n/a | 0 | Better retention but unsafe unknown false acceptance. |
| OLDRISK | 0.404 | 0.494 | 0.485 | 0.200 | 48/48 | 0 | Safer FAR direction but not deployable because all rows rollback. |

## Local Implementation And Verification

Changed local files:

| Path | Purpose |
|---|---|
| `tools\spaceborne_fewshot_da_matrix.py` | Added `OA_MSE_H06_OLDFUSE48` Stage2-B old/unknown-only generator route with `oldqual_oldrisk_fusion` and `rollback_calibration` categories. |
| `tools\optimizer_validate_matrix.py` | Added OLDFUSE old/unknown-only token handling and category balance validation. |
| `tests\test_spaceborne_fewshot_da_matrix.py` | Added focused OLDFUSE48 matrix/validator regression coverage. |
| `code\scripts\launch_stage2_spaceborne_h06_oldfuse_repair_20260626_122707.sh` | Generated launcher for 48 candidates, 6 rows per GPU. |

Local verification:

| Command | Result |
|---|---|
| `conda run -n ssr-gpu python -B -m py_compile tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py` | PASS |
| `conda run -n ssr-gpu python -B -m pytest -p no:cacheprovider -q tests\test_spaceborne_fewshot_da_matrix.py -k "h06_oldqual48 or h06_oldrisk48 or h06_oldfuse48"` | PASS, 3 passed, 36 deselected |
| `conda run -n ssr-gpu python -B -m pytest -p no:cacheprovider -q tests\test_spaceborne_fewshot_da_matrix.py` | PASS, 39 passed |
| `conda run -n ssr-gpu python -B -m pytest -p no:cacheprovider -q code\tests\test_optimizer_workflow_tools.py` | PASS, 52 passed |
| `conda run -n ssr-gpu python tools\optimizer_validate_matrix.py ... --expected-count 48 --repair-launcher-identity` | PASS, issue_count=0, launchable_count=48 |
| `bash code/scripts/launch_stage2_spaceborne_h06_oldfuse_repair_20260626_122707.sh --dry-run` | PASS by recount, candidate_count=48, cmd_count=48 |
| `conda run -n ssr-gpu python tools\optimizer_preflight_decision.py ...` | PASS local gates, `overall_status=PENDING_REMOTE_MONITOR` |

Local version state:

| Check | Result |
|---|---|
| `git status -sb` at `E:\type10-7` | Not a git repository |
| `git status -sb` at `E:\type10-7\code` | Not a git repository |
| Snapshot | `E:\type10-7\code\snapshots\stage2_spaceborne_h06_oldfuse_repair_20260626_122707\` |

Snapshot hashes:

| Path | SHA256 |
|---|---|
| `tools\spaceborne_fewshot_da_matrix.py` | `7df267e9984c918ed401f2b022b51862819ede81ae5c306f886484d8ce59dc3a` |
| `tools\optimizer_validate_matrix.py` | `84f1c4794af1298d3b10829ea99eda819e52f6d781b7b852b11bf5277eb1ecfa` |
| `tests\test_spaceborne_fewshot_da_matrix.py` | `1b95f55ac488b2bcfd791105f065e427ba337fb60582f23191dc32ce4b3b8167` |
| `code\scripts\launch_stage2_spaceborne_h06_oldfuse_repair_20260626_122707.sh` | `1de47a6d1513fc7096478f3c8f69371f9b0f73480b6043cb17eb86aa580c02fc` |
| `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldfuse_repair_20260626_122707\matrix.json` | `8a5224306ef50389fb39c96d88e388883f6586bb0547f2bcf1aad41c1178dcbe` |
| `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldfuse_repair_20260626_122707\local_dry_run.txt` | `de93bf8b16284c183f002496eb34b21d790e5108c1f30c3e7d62e757969f2d7c` |
| `automation_reports\CV-SincNet\stage2_spaceborne_h06_oldfuse_repair_20260626_122707\local_preflight_decision.json` | `59bd839f12f6b08545c9431c5612ff79143ff496f99fa6d4938de888a1336571` |

## Pre-Launch N607 Monitor

Direct N607 preflight passed before design and launch decision. Initial monitor output with PowerShell/BOM noise was discarded; the authoritative monitor used a UTF-8 base64 encoded short SSH command.

| Field | Result |
|---|---|
| Remote time | `2026-06-26T12:14:58+0800` |
| GPU compute apps | empty |
| Project training processes | empty |
| `phase1_monitor_state` | 1 |
| `phase2_monitor_state` | 1 |
| `unsafe_ambiguous_cross_lane` | 0 |
| SSH cleanup after monitor | PASS, ssh_processes=0, scp_processes=0, tcp22_connections=0 |

Remote launch remains gated on a fresh preflight, remote hash/validator/bash/dry-run checks, and live process/GPU capacity immediately before execution.

## Runner Completion Analysis

Timestamp: `2026-06-26T12:49:09+08:00`
Status: `COMPLETED_NEGATIVE_DIAGNOSTIC_OLDFUSE_ALL_ROLLBACK_NOT_PROMOTABLE`
Claim boundary: completed negative diagnostic only; not Stage2 success, not deployment evidence, not Stage2-C seen-new success.

| Completion artifact | Count |
|---|---:|
| launched candidates | 48 |
| completed candidates | 48 |
| metrics.json | 48 |
| manifest.json | 48 |
| score_table.csv | 48 |
| features.npz | 48 |
| failed markers | 0 |
| rollback triggered | 48 |
| rollback accepted | 0 |

| Scope | Count | Old mean | Old max | Unknown FAR mean | Unknown FAR min | Hmean mean | Hmean max | Rollback | Safe gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `overall` | 48 | 0.400810 | 0.522222 | 0.502431 | 0.333333 | 0.434851 | 0.567783 | 48 | 0 |
| `oldqual_oldrisk_fusion` | 24 | 0.368981 | 0.466667 | 0.515278 | 0.350000 | 0.408996 | 0.484848 | 24 | 0 |
| `rollback_calibration` | 24 | 0.432639 | 0.522222 | 0.489583 | 0.333333 | 0.460706 | 0.567783 | 24 | 0 |

| Comparison | Old mean delta | Unknown FAR mean delta | Hmean max delta | Interpretation |
|---|---:|---:|---:|---|
| vs OLDRISK | -0.002778 | 0.017361 | 0.059631 | better best hmean, worse mean FAR, still all rollback |
| vs OLDQUAL | -0.035301 | -0.173958 | 0.009734 | lower FAR than OLDQUAL, weaker old retention, still all rollback |

| Joint-ranking candidate | Candidate | Category | Kold | Old acc | Unknown FAR | Hmean | Rollback |
|---|---|---|---:|---:|---:|---:|---|
| best hmean | `OA_MSE_H06_OLDFUSE48_GPU2_E_MSE_SUBSPACE_KOLD10_KNEW0` | `rollback_calibration` | 10 | 0.494444 | 0.333333 | 0.567783 | true |
| best old | `OA_MSE_H06_OLDFUSE48_GPU4_E_MSE_SUBSPACE_KOLD10_KNEW0` | `rollback_calibration` | 10 | 0.522222 | 0.416667 | 0.551089 | true |
| lowest FAR | `OA_MSE_H06_OLDFUSE48_GPU2_E_MSE_SUBSPACE_KOLD10_KNEW0` | `rollback_calibration` | 10 | 0.494444 | 0.333333 | 0.567783 | true |

Interpretation: OLDFUSE is artifact-complete and diagnostically useful, but it is not promotable. The rollback-calibration category produced the stronger mean old retention and hmean, and the best hmean improved over OLDRISK, but every candidate still triggered rollback and no row satisfied the old>=0.90 plus unknown_FAR<=0.05 safety gate. Deployed rollback metrics lower unknown FAR on average, but collapse old retention, so they cannot be used as deployment evidence.

Recommended next action: `DESIGN_NEXT_H06_ROLLBACK_SAFE_RETENTION_REPAIR_FROM_OLDFUSE_NEGATIVE_DIAGNOSTIC`. Keep Stage2-B old/unknown-only and target-new excluded. Use OLDFUSE evidence to diagnose why rollback accepts none, then repair old retention under rollback-safe acceptance instead of relaunching OLDQUAL, OLDRISK, or OLDFUSE unchanged.

### Per-Candidate Result Table

| Candidate | Category | GPU | Kold | Policy | Old acc | Unknown FAR | Coverage | Hmean | Rollback | Deployed old | Deployed FAR | Loss final | Verdict |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| `OA_MSE_H06_OLDFUSE48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 0 | 5 | constrained_retention_risk | 0.3944 | 0.5833 | 0.6333 | 0.4053 | true | 0.1222 | 0.0833 | 0.8640 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU0_B_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 0 | 5 | constrained_retention_risk | 0.4667 | 0.5333 | 0.7125 | 0.4667 | true | 0.1333 | 0.0833 | 1.3189 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU0_C_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 0 | 5 | constrained_retention_risk | 0.3556 | 0.3667 | 0.5708 | 0.4554 | true | 0.1278 | 0.0500 | 1.2951 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU0_D_MSE_SUBSPACE_KOLD5_KNEW0` | rollback_calibration | 0 | 5 | identity_preserving_risk | 0.4389 | 0.4500 | 0.6708 | 0.4882 | true | 0.1722 | 0.0167 | 0.8361 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU0_E_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 0 | 10 | identity_preserving_risk | 0.4556 | 0.4500 | 0.5833 | 0.4983 | true | 0.1167 | 0.0333 | 1.1567 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU0_F_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 0 | 10 | identity_preserving_risk | 0.4667 | 0.5333 | 0.6458 | 0.4667 | true | 0.0778 | 0.0500 | 1.3696 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU1_A_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 1 | 5 | constrained_retention_risk | 0.3500 | 0.7833 | 0.7125 | 0.2676 | true | 0.1833 | 0.0667 | 0.9578 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU1_B_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 1 | 5 | constrained_retention_risk | 0.3889 | 0.6167 | 0.6542 | 0.3861 | true | 0.1222 | 0.0667 | 1.0688 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU1_C_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 1 | 5 | constrained_retention_risk | 0.2667 | 0.3833 | 0.5125 | 0.3723 | true | 0.1167 | 0.0167 | 1.3495 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU1_D_MSE_SUBSPACE_KOLD5_KNEW0` | rollback_calibration | 1 | 5 | identity_preserving_risk | 0.3556 | 0.4000 | 0.5542 | 0.4465 | true | 0.1278 | 0.0333 | 0.8824 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU1_E_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 1 | 10 | identity_preserving_risk | 0.5000 | 0.4667 | 0.6625 | 0.5161 | true | 0.1556 | 0.0167 | 1.1096 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU1_F_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 1 | 10 | identity_preserving_risk | 0.4111 | 0.6167 | 0.7083 | 0.3967 | true | 0.1111 | 0.0167 | 1.4045 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU2_A_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 2 | 5 | constrained_retention_risk | 0.3667 | 0.5000 | 0.5250 | 0.4231 | true | 0.1833 | 0.0667 | 0.9546 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU2_B_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 2 | 5 | constrained_retention_risk | 0.4389 | 0.5667 | 0.6417 | 0.4361 | true | 0.1556 | 0.0667 | 1.1369 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU2_C_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 2 | 5 | constrained_retention_risk | 0.3944 | 0.4333 | 0.6375 | 0.4651 | true | 0.1167 | 0.0333 | 1.2651 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU2_D_MSE_SUBSPACE_KOLD5_KNEW0` | rollback_calibration | 2 | 5 | identity_preserving_risk | 0.2056 | 0.4833 | 0.4792 | 0.2941 | true | 0.1000 | 0.0833 | 0.7998 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU2_E_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 2 | 10 | identity_preserving_risk | 0.4944 | 0.3333 | 0.5875 | 0.5678 | true | 0.0833 | 0.0333 | 1.1585 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 2 | 10 | identity_preserving_risk | 0.4111 | 0.4500 | 0.5625 | 0.4705 | true | 0.0944 | 0.0333 | 1.8252 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU3_A_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 3 | 5 | constrained_retention_risk | 0.2889 | 0.4500 | 0.5042 | 0.3788 | true | 0.1056 | 0.0667 | 0.9627 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU3_B_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 3 | 5 | constrained_retention_risk | 0.4444 | 0.4667 | 0.6875 | 0.4848 | true | 0.1722 | 0.1000 | 1.1466 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU3_C_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 3 | 5 | constrained_retention_risk | 0.3833 | 0.4500 | 0.6208 | 0.4518 | true | 0.1222 | 0.0833 | 1.3101 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU3_D_MSE_SUBSPACE_KOLD5_KNEW0` | rollback_calibration | 3 | 5 | identity_preserving_risk | 0.4667 | 0.6833 | 0.8292 | 0.3773 | true | 0.1389 | 0.0000 | 1.0229 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU3_E_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 3 | 10 | identity_preserving_risk | 0.4278 | 0.3833 | 0.6375 | 0.5051 | true | 0.1222 | 0.0333 | 2.0191 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU3_F_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 3 | 10 | identity_preserving_risk | 0.3833 | 0.4500 | 0.5292 | 0.4518 | true | 0.1333 | 0.0167 | 1.2627 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU4_A_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 4 | 5 | constrained_retention_risk | 0.3500 | 0.4667 | 0.5292 | 0.4226 | true | 0.1500 | 0.1000 | 0.8574 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU4_B_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 4 | 5 | constrained_retention_risk | 0.4000 | 0.5833 | 0.6083 | 0.4082 | true | 0.1167 | 0.0667 | 1.1174 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU4_C_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 4 | 5 | constrained_retention_risk | 0.2889 | 0.5667 | 0.4917 | 0.3467 | true | 0.0833 | 0.0333 | 1.1774 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU4_D_MSE_SUBSPACE_KOLD5_KNEW0` | rollback_calibration | 4 | 5 | identity_preserving_risk | 0.3833 | 0.5833 | 0.6292 | 0.3993 | true | 0.1111 | 0.0000 | 0.8148 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU4_E_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 4 | 10 | identity_preserving_risk | 0.5222 | 0.4167 | 0.6583 | 0.5511 | true | 0.1444 | 0.0000 | 1.2122 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 4 | 10 | identity_preserving_risk | 0.4278 | 0.3500 | 0.5333 | 0.5160 | true | 0.1000 | 0.0167 | 1.2808 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU5_A_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 5 | 5 | constrained_retention_risk | 0.4500 | 0.6167 | 0.7208 | 0.4140 | true | 0.1611 | 0.0833 | 1.2497 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU5_B_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 5 | 5 | constrained_retention_risk | 0.3611 | 0.5000 | 0.6292 | 0.4194 | true | 0.1667 | 0.0833 | 1.0668 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU5_C_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 5 | 5 | constrained_retention_risk | 0.4167 | 0.5333 | 0.6125 | 0.4403 | true | 0.1722 | 0.0167 | 1.2055 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU5_D_MSE_SUBSPACE_KOLD5_KNEW0` | rollback_calibration | 5 | 5 | identity_preserving_risk | 0.3444 | 0.4333 | 0.5333 | 0.4285 | true | 0.1056 | 0.0500 | 0.8408 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU5_E_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 5 | 10 | identity_preserving_risk | 0.4667 | 0.6333 | 0.7375 | 0.4107 | true | 0.0778 | 0.0667 | 1.0687 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU5_F_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 5 | 10 | identity_preserving_risk | 0.4056 | 0.4167 | 0.5875 | 0.4785 | true | 0.0389 | 0.0000 | 1.4485 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU6_A_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 6 | 5 | constrained_retention_risk | 0.3833 | 0.6500 | 0.6750 | 0.3659 | true | 0.1500 | 0.0667 | 0.9293 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU6_B_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 6 | 5 | constrained_retention_risk | 0.3667 | 0.4000 | 0.5708 | 0.4552 | true | 0.1333 | 0.0333 | 0.9962 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU6_C_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 6 | 5 | constrained_retention_risk | 0.3111 | 0.5000 | 0.6208 | 0.3836 | true | 0.1278 | 0.0333 | 1.5898 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU6_D_MSE_SUBSPACE_KOLD5_KNEW0` | rollback_calibration | 6 | 5 | identity_preserving_risk | 0.4722 | 0.5333 | 0.7208 | 0.4694 | true | 0.1278 | 0.0167 | 0.9012 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU6_E_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 6 | 10 | identity_preserving_risk | 0.4278 | 0.6167 | 0.7333 | 0.4043 | true | 0.1444 | 0.0167 | 1.1732 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU6_F_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 6 | 10 | identity_preserving_risk | 0.4944 | 0.5833 | 0.7042 | 0.4522 | true | 0.0722 | 0.0667 | 1.4545 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU7_A_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 7 | 5 | constrained_retention_risk | 0.3889 | 0.6500 | 0.6792 | 0.3684 | true | 0.1667 | 0.0667 | 0.9739 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU7_B_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 7 | 5 | constrained_retention_risk | 0.3500 | 0.4167 | 0.5625 | 0.4375 | true | 0.1278 | 0.0500 | 1.1179 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | oldqual_oldrisk_fusion | 7 | 5 | constrained_retention_risk | 0.2500 | 0.3500 | 0.4875 | 0.3611 | true | 0.1000 | 0.0833 | 1.2297 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU7_D_MSE_SUBSPACE_KOLD5_KNEW0` | rollback_calibration | 7 | 5 | identity_preserving_risk | 0.4722 | 0.4833 | 0.6000 | 0.4934 | true | 0.0444 | 0.0500 | 0.8474 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU7_E_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 7 | 10 | identity_preserving_risk | 0.5000 | 0.4667 | 0.7125 | 0.5161 | true | 0.1056 | 0.0333 | 1.3618 | rollback |
| `OA_MSE_H06_OLDFUSE48_GPU7_F_MSE_SUBSPACE_KOLD10_KNEW0` | rollback_calibration | 7 | 10 | identity_preserving_risk | 0.4500 | 0.5333 | 0.6417 | 0.4582 | true | 0.0833 | 0.0000 | 1.3713 | rollback |

