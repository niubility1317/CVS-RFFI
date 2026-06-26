# stage2_spaceborne_h06_oldqual_repair_20260626_082644

## Objective

Validate the two deployment-time few-shot adaptation paths from the spaceborne CVS-RFFI design:
new-TX enrollment (`CVS-SFE`) and target receiver calibration (`CVS-FTRC`).
Ground model default: CEN51_R04_H06_LOW_PROB_HYBRID_R010 (${ROOT}/runs/cen51_r04_hybrid8_strong_leo_residual_20260624_1545/CEN51_R04_H06_LOW_PROB_HYBRID_R010/latest_model.pth).

## Candidate Matrix

| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |
|---|---:|---|---:|---|---|---|---|---|
| `OA_MSE_H06_OLDQUAL48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU0_B_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU0_C_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU0_D_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU0_E_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU0_F_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU1_A_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU1_B_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU1_C_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU1_D_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU1_E_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU1_F_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU2_A_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU2_B_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU2_C_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU2_D_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU2_E_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU3_A_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU3_B_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU3_C_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU3_D_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU3_E_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU3_F_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU4_A_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU4_B_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU4_C_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU4_D_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU4_E_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU5_A_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU5_B_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU5_C_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU5_D_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU5_E_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU5_F_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU6_A_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU6_B_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU6_C_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU6_D_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU6_E_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU6_F_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU7_A_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU7_B_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU7_D_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU7_E_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDQUAL48_GPU7_F_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_support_quality_prototype_construction_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |

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
bash code/scripts/launch_stage2_spaceborne_h06_oldqual_repair_20260626_082644.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.

## Automation Run 2026-06-26 08:28 CST

控制面读取顺序：

1. `E:\type10-7\AGENTS.md`
2. `E:\type10-7\项目.md`
3. `E:\type10-7\tools\optimizer_control_manifest.md`
4. `E:\type10-7\automation_reports\CV-SincNet\automation_prompt_backups\20260615_001820_stage2_closed_loop_v4\stage2_prompt.md`
5. `E:\type10-7\tools\optimizer_workflow_contract.md`
6. `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`

本轮依据`required_next_action=DESIGN_NEXT_H06_REPRESENTATION_OR_SUPPORT_QUALITY_REPAIR_FROM_OLDBUDGET_NEGATIVE_DIAGNOSTIC`生成新路线`OA_MSE_H06_OLDQUAL48`。它不是OLDGEOM、OLDCONF、OLDBUDGET或old-primary consensus的复跑；实验语义保持Stage2-B old/unknown-only，`target_new_support_per_tx=0`，`new_tx_ids=__NONE__`，unknown transmitter仅作query评估。

本地变更与snapshot：

- `tools\spaceborne_fewshot_da_matrix.py`：新增`OA_MSE_H06_OLDQUAL48`矩阵生成逻辑，支持support-quality/prototype-geometry双类各24行。
- `tools\optimizer_validate_matrix.py`：新增`OLDQUAL`old/unknown-only识别和support-quality/prototype-geometry类别均衡检查。
- `tests\test_spaceborne_fewshot_da_matrix.py`：新增OLDQUAL48协议与validator回归测试。
- `code\scripts\launch_stage2_spaceborne_h06_oldqual_repair_20260626_082644.sh`：正式N607 launcher。
- snapshot：`E:\type10-7\code\snapshots\stage2_spaceborne_h06_oldqual_repair_20260626_082644\`

本地验证：

- RED：新增测试在实现前按预期失败，原因是`ValueError: unknown plan: OA_MSE_H06_OLDQUAL48`。
- PASS：`conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q tests/test_spaceborne_fewshot_da_matrix.py::SpaceborneFewShotDaMatrixTest::test_h06_oldqual48_constructs_support_quality_repair_without_seen_new`
- PASS：`conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q tests/test_spaceborne_fewshot_da_matrix.py`
- PASS：`conda run -n ssr-gpu python -m py_compile tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py`
- PASS：`conda run -n ssr-gpu python tools\optimizer_validate_matrix.py ... --expected-count 48 --repair-launcher-identity`
- PASS：`bash -n code/scripts/launch_stage2_spaceborne_h06_oldqual_repair_20260626_082644.sh`
- PASS：`bash code/scripts/launch_stage2_spaceborne_h06_oldqual_repair_20260626_082644.sh --dry-run`
- PASS：`conda run -n ssr-gpu python tools\optimizer_preflight_decision.py ...`返回`overall_status=PENDING_REMOTE_MONITOR`，`matrix_status=PASS`，`launcher_status=PASS`，`duplicate_status=PASS`。

多角色审查：

- Protocol：允许并要求从OLDBUDGET负诊断fork新的H06 representation/support-quality repair；阻止OLDGEOM/OLDCONF/OLDBUDGET/old-primary复跑；必须保持Stage2-B old/unknown-only和target-new排除。
- Runner/Validation：已有OLDBUDGET矩阵不是当前launch authority；新run必须有fresh run ID、matrix、launcher、registry key、command hash和validator。
- Evidence/Lookup：registry确认old-primary、OLDGEOM、OLDCONF、OLDBUDGET均为完成负诊断；OLDBUDGET旧类保留提升但unknown FAR恶化，`target_hit_count=0`。
- Supervision：禁止把`PENDING_REMOTE_MONITOR`解读成部署成功；禁止用日志替代实时进程/GPU证据；远程引用失败的grep/base64过滤探针已丢弃。

N607预监控与同步：

- direct N607 preflight PASS；bridge未使用。
- authoritative monitor采用无grep完整`ps`和`nvidia-smi`快照：8张GPU均为0%利用率、10MiB显存占用，`nvidia-smi --query-compute-apps`无compute app，完整`ps`无CV-SincNet训练/launcher/eval/export进程，仅有本次短SSH探针。
- 两条存在引用/BOM问题的精简监控探针被丢弃，不作为lane状态证据。
- 同步目标：`/home/szu2070436088/2510044040/CV-SincNet/`
- sha256一致：
  - `tools/spaceborne_fewshot_da_matrix.py`：`79cd1568bb67a1d35fc68aaf718892e6a72fa220261e3fe48d566807df7e16df`
  - `tools/optimizer_validate_matrix.py`：`055fd797e8c5c6f3e524e7c02c7c569aedb36a2ff3daec67b174d5e11ad2aac7`
  - `code/scripts/launch_stage2_spaceborne_h06_oldqual_repair_20260626_082644.sh`：`a7e46eaf39db2a0c271da74b14e63dd4961887c1ed155aa1004e17ec6fe8482a`
  - `automation_reports/CV-SincNet/stage2_spaceborne_h06_oldqual_repair_20260626_082644/matrix.json`：`5404b5fc272382a9dd711cebae7a55c7d28841aa1b8c87580717decf73963e2b`
- 远端PASS：`bash -n code/scripts/launch_stage2_spaceborne_h06_oldqual_repair_20260626_082644.sh`
- 远端PASS：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python tools/optimizer_validate_matrix.py ... --expected-count 48`
- 远端PASS：`DRY_RUN=1 bash code/scripts/launch_stage2_spaceborne_h06_oldqual_repair_20260626_082644.sh --dry-run`生成97行dry-run日志。
- 远端路径检查PASS：teacher checkpoint、`ManySig.pkl`、`ManyTx.pkl`、`export_spaceborne_features.py`、`eval_spaceborne_fewshot.py`和launcher均存在。

## Runner Completion 2026-06-26 08:38 CST

提交边界：

- 正式命令：`cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/stage2_spaceborne_h06_oldqual_repair_20260626_082644 runs/stage2_spaceborne_h06_oldqual_repair_20260626_082644 && nohup bash code/scripts/launch_stage2_spaceborne_h06_oldqual_repair_20260626_082644.sh > logs/stage2_spaceborne_h06_oldqual_repair_20260626_082644/launcher_stdout.log 2>&1 < /dev/null & echo LAUNCHER_PID=$!`
- 本地SSH提交命令超时，不能按文本返回判定成功；随后发现本地stale `ssh.exe` PID15352连接`172.31.111.215:22`，已关闭并复查无残留连接。
- 只读远端证据确认launcher已落地运行，随后完成；未重提、未kill、未清理远端产物。

完成证据：

- `metrics.json`：48/48
- `score_table.csv`：48/48
- launcher complete marker：48
- launcher failed marker：0
- 结束后run匹配进程：0
- 结束后本地`ssh.exe`和到`172.31.111.215:22`的ESTABLISHED连接：0

postrun分析：

- `old_class_accuracy_mean=0.4361111093312502`
- `old_class_accuracy_max=0.5666666626930237`
- `unknown_false_accept_rate_mean=0.6763888874168819`
- `unknown_false_accept_rate_min=0.25`
- `unknown_rejection_rate_mean=0.3236111125831182`
- `unknown_rejection_rate_max=0.75`
- `old_unknown_hmean_proxy_from_means=0.3715315867552738`
- `target_hit_count=0`
- `old05_far05_count=0`
- `loss_verdict=TRAINING_LOG_ANALYSIS_PASS`
- `optimization_verdict=DIAGNOSTIC_NEGATIVE_NO_TARGET_HIT`

结论：

`OA_MSE_H06_OLDQUAL48`完成但为负诊断，不可promote。它比OLDBUDGET进一步提高old retention，但unknown FAR严重恶化，说明support-quality/prototype construction本身没有解决receiver`20-1`old-vs-unknown分离。该结果不是Stage2成功、不是部署证据、不是Stage2-C seen-new证据。

状态更新：

- `stage2_optimizer_state.json`已更新到`current_run_id=stage2_spaceborne_h06_oldqual_repair_20260626_082644`。
- `latest_optimizer_runner_result.status=COMPLETED_NEGATIVE_DIAGNOSTIC_OLDQUAL_NOT_PROMOTABLE`。
- `required_next_action=DESIGN_NEXT_H06_UNKNOWN_SEPARABILITY_OR_QUERY_FREE_BACKGROUND_RISK_REPAIR_FROM_OLDQUAL_NEGATIVE_DIAGNOSTIC`。
- 下一轮禁止复跑OLDGEOM48、OLDCONF48、OLDBUDGET48、OLDQUAL48或old-primary consensus；必须保持Stage2-B old/unknown-only和target-new排除，并转向unknown separability、query-free background risk或receiver`20-1`old-vs-unknown representation evidence。
- registry已追加`phase2_completed_analysis`记录。

关键本地产物：

- postrun分析：`E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldqual_repair_20260626_082644\artifacts\postrun_analysis.json`
- runner result：`E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldqual_repair_20260626_082644\artifacts\optimizer_runner_result_oldqual.json`
- state备份：`E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldqual_repair_20260626_082644\artifacts\stage2_optimizer_state.before_oldqual_update.json`
- state更新脚本：`E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldqual_repair_20260626_082644\artifacts\update_state_registry_oldqual.py`
