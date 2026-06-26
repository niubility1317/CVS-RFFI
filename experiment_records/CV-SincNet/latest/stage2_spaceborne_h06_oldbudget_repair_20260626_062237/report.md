# stage2_spaceborne_h06_oldbudget_repair_20260626_062237

## Objective

Validate the two deployment-time few-shot adaptation paths from the spaceborne CVS-RFFI design:
new-TX enrollment (`CVS-SFE`) and target receiver calibration (`CVS-FTRC`).
Ground model default: CEN51_R04_H06_LOW_PROB_HYBRID_R010 (${ROOT}/runs/cen51_r04_hybrid8_strong_leo_residual_20260624_1545/CEN51_R04_H06_LOW_PROB_HYBRID_R010/latest_model.pth).

## Candidate Matrix

| ID | GPU | protocol | K | gate/adapter | target_visibility | label_set_relation | update_module | metrics |
|---|---:|---|---:|---|---|---|---|---|
| `OA_MSE_H06_OLDBUDGET48_GPU0_A_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU0_B_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU0_C_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU0_D_MSE_SUBSPACE_KOLD5_KNEW0` | 0 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU0_E_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU0_F_MSE_SUBSPACE_KOLD10_KNEW0` | 0 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU1_A_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU1_B_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU1_C_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU1_D_MSE_SUBSPACE_KOLD5_KNEW0` | 1 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU1_E_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU1_F_MSE_SUBSPACE_KOLD10_KNEW0` | 1 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU2_A_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU2_B_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU2_C_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU2_D_MSE_SUBSPACE_KOLD5_KNEW0` | 2 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU2_E_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU2_F_MSE_SUBSPACE_KOLD10_KNEW0` | 2 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU3_A_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU3_B_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU3_C_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU3_D_MSE_SUBSPACE_KOLD5_KNEW0` | 3 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU3_E_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU3_F_MSE_SUBSPACE_KOLD10_KNEW0` | 3 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU4_A_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU4_B_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU4_C_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU4_D_MSE_SUBSPACE_KOLD5_KNEW0` | 4 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU4_E_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU4_F_MSE_SUBSPACE_KOLD10_KNEW0` | 4 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU5_A_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU5_B_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU5_C_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU5_D_MSE_SUBSPACE_KOLD5_KNEW0` | 5 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU5_E_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU5_F_MSE_SUBSPACE_KOLD10_KNEW0` | 5 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU6_A_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU6_B_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU6_C_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU6_D_MSE_SUBSPACE_KOLD5_KNEW0` | 6 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU6_E_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU6_F_MSE_SUBSPACE_KOLD10_KNEW0` | 6 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU7_A_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU7_B_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU7_C_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU7_D_MSE_SUBSPACE_KOLD5_KNEW0` | 7 | CVS-OA-MSE | 5 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU7_E_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |
| `OA_MSE_H06_OLDBUDGET48_GPU7_F_MSE_SUBSPACE_KOLD10_KNEW0` | 7 | CVS-OA-MSE | 10 | `oa_mse` | `target_old_leo_support_labeled_unknown_eval_only` | `Y_T_has_explicit_nonoverlap_tx` | `h06_old_unknown_acceptance_budget_mse_subspace` | full_accuracy,coverage,old_class_accuracy,unknown_rejection_rate,unknown_false_accept_rate,auroc,fpr95,rollback_gate,selected_alpha |

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
bash code/scripts/launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh --dry-run
```

After N607 preflight and capacity check, run without `--dry-run` only if active jobs leave safe capacity.

## Automation Control Read

Run timestamp: 2026-06-26T06:22:37+08:00.

Required control files were readable in order:

- `E:\type10-7\AGENTS.md`
- `E:\type10-7\项目.md`
- `E:\type10-7\tools\optimizer_control_manifest.md`
- `E:\type10-7\automation_reports\CV-SincNet\automation_prompt_backups\20260615_001820_stage2_closed_loop_v4\stage2_prompt.md`
- `E:\type10-7\tools\optimizer_workflow_contract.md`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`

Current state authority: `OA_MSE_H06_OLDCONF48` completed as a negative diagnostic, not promotable; required next action is `DESIGN_NEXT_H06_REPRESENTATION_REPAIR_FROM_OLDCONF_NEGATIVE_DIAGNOSTIC`. This run does not relaunch `OLDGEOM48`, `OLDCONF48`, or the older old-primary consensus route.

## Monitor And Review

Local current-view command:

```powershell
conda run -n ssr-gpu python tools\optimizer_state_current_view.py
```

Result: state JSON readable, schema `stage2_optimizer_current_state_view_v1`.

N607 direct preflight command:

```powershell
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
```

Result: direct `N607` route PASS; server `dell-DSS8440`; project root visible; eight RTX 3090 GPUs visible with no training occupancy at preflight time.

Live inventory command:

```powershell
conda run -n ssr-gpu python tools\n607_training_inventory.py --direct-only --pretty
```

Result: direct route OK, `active_training_processes=[]`, `gpu_compute=[]`, `launcher_context=[]`, `unsafe_ambiguous_cross_lane=0`. Lane interpretation: Phase1 monitor state 1, Phase2 monitor state 1.

Multi-role review summary:

- Protocol review: keep Stage2-B old/unknown-only; target-new support and seen-new claims remain excluded; any Stage2-C claim is blocked.
- Evidence review: OLDGEOM and OLDCONF are negative diagnostics. OLDGEOM lowered unknown FAR but lost old retention; OLDCONF recovered some old retention but leaked unknown.
- Innovation review: next repair should alter support/prototype acceptance behavior, not stack another terminal reject gate.
- Runner review: a new route must use a new run ID, new matrix, new launcher, new registry identity, 48-row local validator PASS, and no duplicate command identity.

## Local Implementation

Changed local files:

- `E:\type10-7\tools\spaceborne_fewshot_da_matrix.py`: added `OA_MSE_H06_OLDBUDGET48`, a 48-row H06 post-OLDCONF Stage2-B old/unknown-only acceptance-budget route.
- `E:\type10-7\tools\optimizer_validate_matrix.py`: added `OLDBUDGET` to old/unknown-only and old-primary route recognition.

Generated files:

- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\matrix.json`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\artifacts\validator_repair_identity.json`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\artifacts\preflight_decision.json`
- `E:\type10-7\code\scripts\launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh`

Snapshot:

- `E:\type10-7\code\snapshots\stage2_spaceborne_h06_oldbudget_repair_20260626_062237`

Hashes:

- `tools\spaceborne_fewshot_da_matrix.py`: `7AA00A87313522FBCBEE20C3B36FA85361EF481E561AF2BDA5EBEB33D6678012`
- `tools\optimizer_validate_matrix.py`: `A756F5B7D74D229F235FA1D61D545CEA21FCCE1D24A2E41FD8A808BCC46B1AD1`
- `code\scripts\launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh`: `75B2AD96837073DBE36515E67E35BB7D5239EE05A0E72533020C6ADEF1368473`
- `matrix.json`: `1D4558552A16AC4E3148409667E838977A5818A6ED1D565BDCF23E2548A50827`
- `validator_repair_identity.json`: `7FF6FD438531C518761C8671D6C108D712CC136154634E45AC380D63F25002E5`
- `preflight_decision.json`: `C7A04AE0E47FC890EE1B4CB479E8CD376F0AE7B2D4D3D19BFCD48144B8778557`

## Local Verification

Commands and results:

```powershell
conda run -n ssr-gpu python -m py_compile tools\spaceborne_fewshot_da_matrix.py tools\optimizer_validate_matrix.py
```

PASS.

```powershell
conda run -n ssr-gpu python tools\spaceborne_fewshot_da_matrix.py --plan OA_MSE_H06_OLDBUDGET48 --run-id stage2_spaceborne_h06_oldbudget_repair_20260626_062237
```

PASS, generated 48 candidates.

```powershell
conda run -n ssr-gpu python tools\optimizer_validate_matrix.py automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\matrix.json --expected-count 48 --launcher code\scripts\launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh --repair-launcher-identity --output automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\artifacts\validator_repair_identity.json
```

PASS: `issue_count=0`, `launchable_count=48`, `runner_readiness=MATRIX_HAS_LAUNCHABLE_ROWS`; lane readiness `LANE_HAS_LAUNCHABLE_ROWS`.

```powershell
bash -n code/scripts/launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh
```

PASS.

```powershell
bash code/scripts/launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh --dry-run
```

PASS, printed 48 candidate commands only.

```powershell
conda run -n ssr-gpu python tools\optimizer_preflight_decision.py --project-root E:\type10-7 --matrix automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\matrix.json --launcher code\scripts\launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh --expected-count 48 --output automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\artifacts\preflight_decision.json
```

PASS: `overall_status=PENDING_REMOTE_MONITOR`, `duplicate_status=PASS`, `matrix_status=PASS`, `launcher_status=PASS`.

```powershell
conda run -n ssr-gpu python -m pytest -p no:cacheprovider -q code\tests\test_optimizer_workflow_tools.py
```

PASS, 52 tests.

## Remote Plan

Remote root: `/home/szu2070436088/2510044040/CV-SincNet`.

Files to sync before launch:

- `tools\spaceborne_fewshot_da_matrix.py` to `tools/spaceborne_fewshot_da_matrix.py`
- `tools\optimizer_validate_matrix.py` to `tools/optimizer_validate_matrix.py`
- `code\scripts\launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh` to `code/scripts/launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh`
- this report directory to `automation_reports/CV-SincNet/stage2_spaceborne_h06_oldbudget_repair_20260626_062237/`

Remote launch is allowed only after remote hash, py_compile, validator, bash syntax, dry-run, capacity, and SSH cleanup checks pass.

## Remote Execution And Final Status

Remote verification passed before launch:

- remote hash check matched local hashes for generator, validator, launcher, matrix, validator artifact, and preflight artifact.
- remote `py_compile` passed for `tools/spaceborne_fewshot_da_matrix.py` and `tools/optimizer_validate_matrix.py`.
- remote validator passed: 48/48 launchable, `issue_count=0`, lane readiness `LANE_HAS_LAUNCHABLE_ROWS`.
- remote `bash -n` passed.
- remote dry-run printed 48 candidate rows.
- final pre-launch live monitor showed no active project training processes and empty GPU compute state.

Remote launch command shape:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup bash code/scripts/launch_stage2_spaceborne_h06_oldbudget_repair_20260626_062237.sh > logs/stage2_spaceborne_h06_oldbudget_repair_20260626_062237_launcher.nohup.out 2>&1 < /dev/null &
```

Observed launcher PID: `3055923`.

Startup and completion evidence:

- first valid startup probe found 16 candidate run directories/logs and launcher capacity waits at `max=2` per GPU.
- final artifact count: 48 `features.npz`, 48 `metrics.json`, 48 `manifest.json`, and 48 `score_table.csv`.
- launcher complete lines: 48.
- fatal marker scan: no `Traceback`, `RuntimeError`, CUDA OOM, import error, argparse error, or missing-file marker found in candidate logs.
- final live monitor: no active training process and empty GPU compute state.
- SSH cleanup: PASS. One timed-out read-only probe left local `ssh.exe` PID 11384 earlier; it was stopped and later cleanup checks were zero for `ssh.exe`, `scp.exe`, and N607/bridge TCP 22.

Remote artifacts copied back:

- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\artifacts\remote_validator_repair_identity.json`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\artifacts\remote_dry_run.txt`
- `E:\type10-7\automation_reports\CV-SincNet\stage2_spaceborne_h06_oldbudget_repair_20260626_062237\artifacts\remote_stage2_analysis.json`

Final metrics summary from `remote_stage2_analysis.json`:

- `old_mean`: 0.2258101879
- `old_max`: 0.4722222090
- `unknown_FAR_mean`: 0.3041666641
- `unknown_FAR_min`: 0.0
- `unknown_rejection_mean`: 0.6958333359
- `target_hit_count`: 0
- `loss_initial_mean`: 10.3475839794
- `loss_final_mean`: 1.9517594551
- adapter alpha counts: 0.75 -> 29, 0.5 -> 3, 1.0 -> 16

Comparison against `OLDCONF48`:

- old retention improved: `old_mean` +0.0981481492.
- unknown false accept worsened: `unknown_FAR_mean` +0.1107638876.
- target hit did not improve: still 0.

Interpretation:

`OA_MSE_H06_OLDBUDGET48` is a completed negative diagnostic. It improved old retention but leaked too much unknown traffic, so it is not promotable and must not be used as deployment evidence or Stage2-C success. The next H06 turn should stay Stage2-B old/unknown-only with target-new excluded and move to representation/prototype geometry or support-quality construction rather than another terminal gate stack.

State and registry updates:

- `E:\type10-7\automation_reports\CV-SincNet\stage2_optimizer_state.json`
- `E:\type10-7\automation_reports\CV-SincNet\optimizer_execution_registry.jsonl`

New required next action:

`DESIGN_NEXT_H06_REPRESENTATION_OR_SUPPORT_QUALITY_REPAIR_FROM_OLDBUDGET_NEGATIVE_DIAGNOSTIC`
