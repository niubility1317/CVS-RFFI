# BranchOrbitCE support 诊断

标准物理 OOF 与 support 内部 1-shot proxy 分别报告；proxy 的 K 表示 parent K，绝不是正式 K1。

{"parent_episodes": 4800, "k1_numerical_only": 1200, "standard_oof_parents": 3600, "proxy_parents": 3600, "proxy_anchors": 42000, "standard_factorizations": 21600, "proxy_factorizations": 84000, "actual_factorizations": 105600, "proxy_held_occurrences_per_arm": 7879200, "optimizer_steps": 5563959, "ce_fits": 105600, "episodes": 4800}

| 诊断 | parent K | 对照 | Δ旧 | Δ新 | ΔH | 通过 |
|---|---:|---|---:|---:|---:|---|
| standard | 5 | single_ridge | -8.962 | -8.078 | -8.722 | False |
| standard | 5 | single_ce | -5.146 | -3.467 | -3.918 | False |
| standard | 5 | orbit_ridge | -4.538 | -4.464 | -4.905 | False |
| standard | 10 | single_ridge | -8.566 | -9.885 | -10.117 | False |
| standard | 10 | single_ce | -5.557 | -4.481 | -4.962 | False |
| standard | 10 | orbit_ridge | -4.745 | -6.932 | -6.753 | False |
| standard | 20 | single_ridge | -7.120 | -8.946 | -8.584 | False |
| standard | 20 | single_ce | -4.757 | -4.049 | -4.307 | False |
| standard | 20 | orbit_ridge | -4.266 | -6.512 | -5.980 | False |
| oneshot_proxy | 5 | single_ridge | -5.471 | -3.893 | -4.324 | False |
| oneshot_proxy | 5 | single_ce | -5.097 | -3.590 | -3.954 | False |
| oneshot_proxy | 5 | orbit_ridge | -1.003 | -0.775 | -0.873 | False |
| oneshot_proxy | 10 | single_ridge | -5.321 | -3.848 | -4.297 | False |
| oneshot_proxy | 10 | single_ce | -5.060 | -3.443 | -3.947 | False |
| oneshot_proxy | 10 | orbit_ridge | -0.864 | -0.696 | -0.796 | False |
| oneshot_proxy | 20 | single_ridge | -5.217 | -4.048 | -4.343 | False |
| oneshot_proxy | 20 | single_ce | -5.064 | -3.674 | -4.058 | False |
| oneshot_proxy | 20 | orbit_ridge | -0.753 | -0.758 | -0.794 | False |

差值单位为百分点。全部新增类规模、旧类单独任务、模型、support seed、RX×场景和parent K分层见CSV。
真实K1无独立held证据。proxy每parent内先平均全部anchor，不将42000次anchor或重复held次数当独立样本量。

- coverage.episodes aliases parent_episodes; it is not an additional count.
- True K1 is numerical-only; no OOF or proxy performance exists.
- One-shot proxy means use all anchors within each parent first, then equal parent weights; parent K is retained.
- Exact train/held physical mappings and per-anchor confusion/NLL counts remain in full traces; no query scores are used.
- Standard OOF and proxy are separate; repeated held occurrences and support draws are correlated, not independent sample counts.
- CE heads use physical-sum cross entropy and fixed RKHS regularization; full training steps are retained, with source validation unavailable because source inputs are prohibited.
- Macro NLL uses fixed softmax of each arm's own scores, without post-hoc calibration. The runtime shared metrics helper labels all arms as uncalibrated_ridge_scores; that inherited text is inaccurate for CE arms, whose scores come from the CE head. Numeric NLL computation is unchanged and NLL is not a screening criterion.
- A/B screens are support development evidence, not formal K1/query generalization or automatic promotion.
