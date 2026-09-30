# BranchLocalMargin support 诊断

标准物理 OOF 与 support 内部 1-shot proxy 分别报告；proxy 的 K 表示 parent K，绝不是正式 K1。

{"parent_episodes": 4800, "k1_numerical_only": 1200, "standard_oof_parents": 3600, "proxy_parents": 3600, "proxy_anchors": 42000, "standard_factorizations": 32400, "proxy_factorizations": 126000, "actual_factorizations": 158400, "proxy_held_occurrences_per_arm": 7879200, "optimizer_steps": 204068488, "candidate_fits": 52800, "nominal_max_factorizations": 158400, "degenerate_factorization_reductions": 0, "episodes": 4800}

| 诊断 | parent K | 对照 | Δ旧 | Δ新 | ΔH | 通过 |
|---|---:|---|---:|---:|---:|---|
| standard | 5 | local_ridge | +0.632 | +0.542 | +0.660 | True |
| standard | 10 | local_ridge | +0.358 | +0.541 | +0.511 | True |
| standard | 20 | local_ridge | +0.365 | +0.642 | +0.551 | True |
| oneshot_proxy | 5 | local_ridge | +0.024 | -0.020 | +0.011 | False |
| oneshot_proxy | 10 | local_ridge | -0.028 | -0.023 | -0.026 | False |
| oneshot_proxy | 20 | local_ridge | -0.049 | -0.033 | -0.046 | False |

差值单位为百分点。全部新增类规模、旧类单独任务、模型、support seed、RX×场景和parent K分层见CSV。
真实K1无独立held证据。proxy每parent内先平均全部anchor，不将42000次anchor或重复held次数当独立样本量。

- coverage.episodes aliases parent_episodes; it is not an additional count.
- True K1 is numerical-only; no OOF or proxy performance exists.
- One-shot proxy means use all anchors within each parent first, then equal parent weights; parent K is retained.
- Exact train/held physical mappings and per-anchor confusion/NLL counts remain in full traces; no query scores are used.
- Standard OOF and proxy are separate; repeated held occurrences and support draws are correlated, not independent sample counts.
- The three controls use physical-sum ridge loss; LocalMargin uses physical-sum strongest-wrong-class squared hinge with RKHS penalty. Softmax NLL is uncalibrated descriptive evidence; it is not cross-entropy training.
- A/B screens are support development evidence, not formal K1/query generalization or automatic promotion.
