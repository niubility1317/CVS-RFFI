# 原 LocalRidge 注册机制诊断

完整 160 parent、4 个模型/cohort row 已校验。仅原 LocalRidge B0/C0，无 Adapter。真实 A、微调 B 和适应收益均为 N/A。

信道继承 practical residual / post_sync / noeq，fs=25 MHz；未新生成信道。仅解释显式选择的 RX×场景。

| 诊断 | K | 新类数 | B0旧 | C0旧类列 | C0全类竞争旧 | C0新 | H | 旧类决策变化 | 新类竞争损失 | 总旧损失 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | 5 | 0 | 63.333 | 63.333 | 63.333 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| oof | 5 | 2 | 63.333 | 63.333 | 61.250 | 50.000 | 53.522 | 0.000 | 2.083 | 2.083 |
| oof | 5 | 5 | 63.333 | 62.500 | 54.583 | 50.000 | 51.918 | 0.833 | 7.917 | 8.750 |
| oof | 5 | 10 | 63.333 | 63.750 | 53.333 | 48.250 | 50.361 | -0.417 | 10.417 | 10.000 |
| oof | 5 | 20 | 63.333 | 63.750 | 48.750 | 46.250 | 46.539 | -0.417 | 15.000 | 14.583 |
| oof | 10 | 0 | 73.542 | 73.542 | 73.542 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| oof | 10 | 2 | 73.542 | 73.958 | 71.667 | 51.875 | 59.451 | -0.417 | 2.292 | 1.875 |
| oof | 10 | 5 | 73.542 | 73.125 | 67.917 | 56.500 | 61.409 | 0.417 | 5.208 | 5.625 |
| oof | 10 | 10 | 73.542 | 73.542 | 65.625 | 57.875 | 61.456 | 0.000 | 7.917 | 7.917 |
| oof | 10 | 20 | 73.542 | 73.542 | 64.167 | 56.500 | 60.024 | 0.000 | 9.375 | 9.375 |
| oof | 20 | 0 | 76.250 | 76.250 | 76.250 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| oof | 20 | 2 | 76.250 | 76.875 | 74.688 | 53.125 | 61.590 | -0.625 | 2.188 | 1.562 |
| oof | 20 | 5 | 76.250 | 76.875 | 70.104 | 61.250 | 65.085 | -0.625 | 6.771 | 6.146 |
| oof | 20 | 10 | 76.250 | 76.771 | 68.021 | 63.625 | 65.703 | -0.521 | 8.750 | 8.229 |
| oof | 20 | 20 | 76.250 | 77.083 | 65.104 | 62.906 | 63.944 | -0.833 | 11.979 | 11.146 |
| proxy | 5 | 0 | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| proxy | 5 | 2 | 50.938 | 51.667 | 49.271 | 33.125 | 33.343 | -0.729 | 2.396 | 1.667 |
| proxy | 5 | 5 | 50.938 | 51.562 | 45.729 | 39.625 | 41.070 | -0.625 | 5.833 | 5.208 |
| proxy | 5 | 10 | 50.938 | 51.146 | 42.292 | 37.438 | 38.783 | -0.208 | 8.854 | 8.646 |
| proxy | 5 | 20 | 50.938 | 51.771 | 38.438 | 33.281 | 34.149 | -0.833 | 13.333 | 12.500 |
| proxy | 10 | 0 | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| proxy | 10 | 2 | 53.032 | 54.051 | 51.019 | 25.556 | 29.944 | -1.019 | 3.032 | 2.014 |
| proxy | 10 | 5 | 53.032 | 54.444 | 47.199 | 36.361 | 39.869 | -1.412 | 7.245 | 5.833 |
| proxy | 10 | 10 | 53.032 | 54.352 | 43.773 | 36.931 | 39.069 | -1.319 | 10.579 | 9.259 |
| proxy | 10 | 20 | 53.032 | 54.514 | 39.699 | 32.465 | 34.548 | -1.481 | 14.815 | 13.333 |
| proxy | 20 | 0 | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | 0.000 |
| proxy | 20 | 2 | 52.473 | 52.610 | 49.644 | 28.980 | 33.078 | -0.137 | 2.966 | 2.829 |
| proxy | 20 | 5 | 52.473 | 52.939 | 45.482 | 37.053 | 39.598 | -0.466 | 7.456 | 6.990 |
| proxy | 20 | 10 | 52.473 | 52.917 | 41.634 | 36.089 | 37.686 | -0.444 | 11.283 | 10.839 |
| proxy | 20 | 20 | 52.473 | 53.421 | 38.070 | 32.388 | 34.167 | -0.948 | 15.351 | 14.402 |

准确率为百分数，差值为百分点。恢复比例、绝对新旧差及分场景结果见 CSV；proxy 先均 anchor 后等权 parent。

实测 head fits=3168，factorizations=3168；运行墙钟 109.111 s。
各 head 的 fit 耗时之和 47.865 s，score 耗时之和 148.122 s；并发工作量之和不等于墙钟。

- B0 fits old training support; C0 independently refits all registered training support using the unchanged LocalRidge.
- C0 old-column argmax is an offline fixed-score counterfactual, never a deployed role-conditioned predictor.
- The exact count decomposition separates old-class decision reordering from added new-class competition; it does not isolate individual kernel or coefficient causes.
- OOF pools each physical held ID once before computing accuracy; unequal folds are not equally weighted.
- Proxy first averages all anchors per parent, then weights parents equally; repeated support draws are correlated.
- True K1 is numerical-only. No held accuracy or head fit is fabricated.
- Only the explicitly selected practical residual post_sync/noeq receiver/scene pairs are described; unselected scenes are not measured.
- The ideal 10/1/3 percentage-point directions are descriptive and are not advancement or stopping gates.
