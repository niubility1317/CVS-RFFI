# BranchMetric support 诊断

标准物理 OOF 与 support 内部 1-shot proxy 分别报告；proxy 的 K 表示 parent K，绝不是正式 K1。

{"parent_episodes": 4800, "k1_numerical_only": 1200, "standard_oof_parents": 3600, "proxy_parents": 3600, "proxy_anchors": 42000, "standard_factorizations": 21600, "proxy_factorizations": 42000, "actual_factorizations": 63600, "proxy_held_occurrences_per_arm": 7879200, "episodes": 4800}

| 诊断 | parent K | 对照 | Δ旧 | Δ新 | ΔH | 通过 |
|---|---:|---|---:|---:|---:|---|
| standard | 5 | interaction_ridge | -0.226 | +1.730 | +1.340 | True |
| standard | 5 | kernel_ncm | +3.924 | +4.520 | +4.418 | True |
| standard | 10 | interaction_ridge | -0.059 | +1.918 | +1.440 | True |
| standard | 10 | kernel_ncm | +5.498 | +6.884 | +6.889 | True |
| standard | 20 | interaction_ridge | +0.098 | +2.093 | +1.539 | True |
| standard | 20 | kernel_ncm | +6.760 | +9.267 | +8.819 | True |
| oneshot_proxy | 5 | interaction_ridge | -0.665 | +0.109 | -0.131 | False |
| oneshot_proxy | 10 | interaction_ridge | -0.736 | +0.284 | -0.031 | False |
| oneshot_proxy | 20 | interaction_ridge | -0.834 | +0.265 | -0.094 | False |

差值单位为百分点。全部新增类规模、旧类单独任务、模型、support seed、RX×场景和parent K分层见CSV。
真实K1无独立held证据。proxy每parent内先平均全部anchor，不将42000次anchor或重复held次数当独立样本量。

- coverage.episodes aliases parent_episodes; it is not an additional count.
- True K1 is numerical-only; no OOF or proxy performance exists.
- One-shot proxy means use all anchors within each parent first, then equal parent weights; parent K is retained.
- Exact train/held physical mappings and per-anchor confusion/NLL counts remain in full traces; no query scores are used.
- Standard OOF and proxy are separate; repeated held occurrences and support draws are correlated, not independent sample counts.
- Metric solve objective may be negative and is not supervised regression loss. Softmax NLL is uncalibrated descriptive evidence.
- A/B screens are support development evidence, not formal K1/query generalization or automatic promotion.
