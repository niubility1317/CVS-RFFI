# 固定源预测：clean 与 practical 错误转移

源 RX1-held／seed 2026092701，F＋跨 RX 同 TX／默认头，三组 final E20 的全部固定硬预测。此次仅复算原 1260 个物理样本，0 新 IQ／0 前向／0 更新，不改判据或训练。以下为描述性条件统计，不能证明传播机制原因，也不替代总体 accuracy／Macro-F1 与原固定晋级条件。

| 对照 | 场景 | 总体 accuracy 差 pp | 双方 clean 均正确：样本数 | 该集合修复／新错 | 该集合贡献 pp | 原对照 clean 正确集合内 accuracy 差 pp |
|---|---|---:|---:|---|---:|---:|
| B00 | practical_high | +20.634921 | 992 | 373/25 | +27.619048 | +34.158416 |
| B00 | practical_mid | +16.587302 | 992 | 340/48 | +23.174603 | +28.415842 |
| B00 | practical_low_suburban | +18.095238 | 992 | 340/41 | +23.730159 | +29.009901 |
| B00 | practical_high_urban | +16.349206 | 992 | 345/51 | +23.333333 | +28.415842 |
| B00 | practical_mid_urban | +11.428571 | 992 | 275/76 | +15.793651 | +19.306931 |
| B00 | practical_low_urban | +6.269841 | 992 | 185/66 | +9.444444 | +11.386139 |
| C_IID | practical_high | +1.190476 | 1061 | 28/9 | +1.507937 | +1.785714 |
| C_IID | practical_mid | +1.111111 | 1061 | 29/14 | +1.190476 | +1.409774 |
| C_IID | practical_low_suburban | +1.825397 | 1061 | 32/12 | +1.587302 | +1.879699 |
| C_IID | practical_high_urban | +0.952381 | 1061 | 26/11 | +1.190476 | +1.409774 |
| C_IID | practical_mid_urban | +0.396825 | 1061 | 19/11 | +0.634921 | +0.751880 |
| C_IID | practical_low_urban | +0.158730 | 1061 | 17/18 | -0.079365 | -0.093985 |

完整四组拆分和全部预测变化流保留在 error_transitions.json。四组为双方 clean 正确、仅 As 修复 clean、仅 As 新错 clean、双方 clean 错误；各组贡献之和严格等于总体 accuracy 差。条件集合由已有 clean 结果定义，仅解释同一源折的错误分布，未用于挑选或删除样本。

| 对照 | low urban clean 状态组 | 样本数 | practical 修复 | practical 新错 | 对总体贡献 pp |
|---|---|---:|---:|---:|---:|
| B00 | both_clean_correct | 992 | 185 | 66 | +9.444444 |
| B00 | As_clean_repaired | 75 | 6 | 14 | -0.634921 |
| B00 | As_clean_broken | 18 | 1 | 5 | -0.317460 |
| B00 | both_clean_wrong | 175 | 9 | 37 | -2.222222 |
| C_IID | both_clean_correct | 1061 | 17 | 18 | -0.079365 |
| C_IID | As_clean_repaired | 6 | 0 | 0 | +0.000000 |
| C_IID | As_clean_broken | 3 | 0 | 0 | +0.000000 |
| C_IID | both_clean_wrong | 190 | 3 | 0 | +0.238095 |

现有硬预测文件没有传播状态、接收锁定、logits 或 margin，不能据此归因于某种 LOS／补偿／噪声机制；同物理与固定信道配置不冒称 received IQ 字节已核对。所有 TX／day 负结果继续保留于原报告。本分析无新增分类头，不将条件统计当作 As、跨 seed、正式 target 或联合目标证明。
