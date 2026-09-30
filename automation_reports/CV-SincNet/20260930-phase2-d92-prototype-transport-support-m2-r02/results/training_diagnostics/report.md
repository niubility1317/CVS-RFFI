# PrototypeTransport完整训练日志诊断

状态：COMPLETE_TRANSPORT_TRAINING_LOG_SCAN_VERIFIED；run：20260930-phase2-d92-prototype-transport-support-m2-r02。

完整日志已核对。information stage与有接受更新的stage分别统计；初始前向、零梯度反向及拒绝trial均按实际费用记录。

stages.csv与stage_curves.csv保存所有阶段和全部训练位置；snapshots.csv与curves.csv保存分层聚合。B复用与物理训练绑定去重表独立保存，资源总量保留每次实际拟合。

| 计数 | 实际值 |
|---|---:|
| episodes | 160 |
| k1_episodes | 40 |
| oof_episodes | 120 |
| proxy_anchor_count | 1400 |
| sequence_paths | 1760 |
| baseline_head_fit_count | 3168 |
| transport_preparation_count | 3168 |
| transport_stage_count | 4576 |
| diagnostic_fit_count | 0 |
| trained_transport_stage_count | 936 |
| optimizer_steps | 3656 |
| inner_objective_evaluation_count | 5239 |
| inner_head_fit_count | 15717 |
| inner_factorization_count | 15717 |
| final_head_fit_count | 936 |
| final_factorization_count | 936 |
| head_fit_count | 19821 |
| factorization_count | 19821 |
| baseline_factorization_count | 3168 |
| derivative_triangular_solve_count | 22368 |
| optimizer_iterations | 3728 |
| backward_evaluation_count | 3728 |
| transport_forward_evaluation_count | 16653 |
| accepted_trial_count | 3656 |
| rejected_trial_count | 647 |
| trial_count | 4303 |
| trial_attempt_count | 4303 |
| prototype_construction_count | 5112 |
| prototype_distance_evaluation_count | 15336 |
| prepared_distance_evaluation_count | 10224 |
| final_score_evaluation_count | 7744 |
| final_score_physical_count | 1020600 |
| actual_stages | 4576 |
| information_stages | 936 |
| updated_stages | 936 |
| zero_update_information_stages | 0 |
| unique_B_physical_training_groups | 352 |
| repeated_B_actual_contexts | 1408 |
| B_C_inheritance_reuse_contexts | 2816 |

- All listed complete structured records and every text line are scanned. Parent held-result payloads are not deserialized; fit_trace/parent compact/query/registry/history files are never opened.
- Information stage means physical K>=2 and multiple classes; updated stage means at least one accepted update. Zero-update information stages still pay initial forward and cached backward costs.
- Costs count each actual fit once. One B fit shared by seq/reset is not counted again for either inheritance context. Repeated B physical bindings across new-count parents are reported separately and deduplicated only in labeled B statistics, not actual resource totals.
- B dedup binding uses row, scope, parent K, train K, class IDs and exact ordered training physical IDs. Context counts and parameter variants are retained; these fits are not independent evidence.
- Initial, every cached gradient, every accepted/rejected Armijo trial and final accepted cache are retained. Inner accuracy/class risks supervise training and are not independent validation.
- SOURCE validation, individual prediction transitions, deployment bytes and missing standalone timings remain N/A. Correct-count equality does not establish identical predictions.
- Regex markers overlap and may be declarations; only structured errors establish structured failure evidence. Solver bounded stopping is expected, not a performance gate.
- Collector never selects parameters, changes optimizer budgets, fits or scores; healthy runs remain untouched.
