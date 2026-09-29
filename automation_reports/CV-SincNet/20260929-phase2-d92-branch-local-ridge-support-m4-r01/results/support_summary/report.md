# BranchLocalRidge support 诊断

标准物理 OOF 与 support 内部 1-shot proxy 分别报告；proxy 的 K 表示 parent K，绝不是正式 K1。

{"parent_episodes": 4800, "k1_numerical_only": 1200, "standard_oof_parents": 3600, "proxy_parents": 3600, "proxy_anchors": 42000, "standard_factorizations": 32400, "proxy_factorizations": 126000, "actual_factorizations": 158400, "proxy_held_occurrences_per_arm": 7879200, "nominal_max_factorizations": 158400, "degenerate_factorization_reductions": 0, "episodes": 4800}

| 诊断 | parent K | 对照 | Δ旧 | Δ新 | ΔH | 通过 |
|---|---:|---|---:|---:|---:|---|
| standard | 5 | branch_ridge | +1.674 | +3.589 | +3.345 | True |
| standard | 5 | interaction_ridge | +0.833 | +1.846 | +1.764 | True |
| standard | 10 | branch_ridge | +1.490 | +3.885 | +3.211 | True |
| standard | 10 | interaction_ridge | +0.847 | +1.923 | +1.590 | True |
| standard | 20 | branch_ridge | +1.843 | +4.975 | +4.102 | True |
| standard | 20 | interaction_ridge | +1.030 | +2.453 | +2.071 | True |
| oneshot_proxy | 5 | branch_ridge | -0.109 | +0.549 | +0.387 | True |
| oneshot_proxy | 5 | interaction_ridge | -0.088 | +0.420 | +0.294 | True |
| oneshot_proxy | 10 | branch_ridge | -0.159 | +0.456 | +0.321 | True |
| oneshot_proxy | 10 | interaction_ridge | -0.089 | +0.393 | +0.312 | True |
| oneshot_proxy | 20 | branch_ridge | -0.124 | +0.550 | +0.410 | True |
| oneshot_proxy | 20 | interaction_ridge | -0.022 | +0.448 | +0.379 | True |

差值单位为百分点。全部新增类规模、旧类单独任务、模型、support seed、RX×场景和parent K分层见CSV。
真实K1无独立held证据。proxy每parent内先平均全部anchor，不将42000次anchor或重复held次数当独立样本量。

- coverage.episodes aliases parent_episodes; it is not an additional count.
- True K1 is numerical-only; no OOF or proxy performance exists.
- One-shot proxy means use all anchors within each parent first, then equal parent weights; parent K is retained.
- Exact train/held physical mappings and per-anchor confusion/NLL counts remain in full traces; no query scores are used.
- Standard OOF and proxy are separate; repeated held occurrences and support draws are correlated, not independent sample counts.
- All three arms use physical-sum ridge loss. Softmax NLL is uncalibrated descriptive evidence; it is not cross-entropy training.
- A/B screens are support development evidence, not formal K1/query generalization or automatic promotion.
