# 固定源预测：clean 与 practical 错误转移

源 RX1-held／seed 2026092701，F＋跨 RX 同 TX／默认头，三组 final E20 的全部固定硬预测。此次仅复算原 1260 个物理样本，0 新 IQ／0 前向／0 更新，不改判据或训练。以下为描述性条件统计，不能证明传播机制原因，也不替代总体 accuracy／Macro-F1 与原固定晋级条件。

| 对照 | 场景 | 总体 accuracy 差 pp | 双方 clean 均正确：样本数 | 该集合修复／新错 | 该集合贡献 pp | 原对照 clean 正确集合内 accuracy 差 pp |
|---|---|---:|---:|---|---:|---:|
| M00 | practical_high | +18.650794 | 988 | 345/43 | +23.968254 | +29.522863 |
| M00 | practical_mid | +14.841270 | 988 | 315/56 | +20.555556 | +25.447316 |
| M00 | practical_low_suburban | +16.349206 | 988 | 318/52 | +21.111111 | +25.944334 |
| M00 | practical_high_urban | +14.126984 | 988 | 307/52 | +20.238095 | +24.751491 |
| M00 | practical_mid_urban | +9.761905 | 988 | 243/70 | +13.730159 | +16.898608 |
| M00 | practical_low_urban | +5.952381 | 988 | 175/64 | +8.809524 | +10.934394 |
| M_repeat | practical_high | -1.904762 | 1059 | 4/29 | -1.984127 | -2.345216 |
| M_repeat | practical_mid | -1.111111 | 1059 | 9/20 | -0.873016 | -1.031895 |
| M_repeat | practical_low_suburban | -1.428571 | 1059 | 4/22 | -1.428571 | -1.688555 |
| M_repeat | practical_high_urban | -1.904762 | 1059 | 3/27 | -1.904762 | -2.251407 |
| M_repeat | practical_mid_urban | -0.317460 | 1059 | 7/11 | -0.317460 | -0.375235 |
| M_repeat | practical_low_urban | -0.238095 | 1059 | 12/16 | -0.317460 | -0.375235 |

完整四组拆分和全部预测变化流保留在 error_transitions.json。四组为双方 clean 正确、仅 MC2 修复 clean、仅 MC2 新错 clean、双方 clean 错误；各组贡献之和严格等于总体 accuracy 差。条件集合由已有 clean 结果定义，仅解释同一源折的错误分布，未用于挑选或删除样本。

| 对照 | low urban clean 状态组 | 样本数 | practical 修复 | practical 新错 | 对总体贡献 pp |
|---|---|---:|---:|---:|---:|
| M00 | both_clean_correct | 988 | 175 | 64 | +8.809524 |
| M00 | MC2_clean_repaired | 76 | 6 | 10 | -0.317460 |
| M00 | MC2_clean_broken | 18 | 2 | 3 | -0.079365 |
| M00 | both_clean_wrong | 178 | 8 | 39 | -2.460317 |
| M_repeat | both_clean_correct | 1059 | 12 | 16 | -0.317460 |
| M_repeat | MC2_clean_repaired | 5 | 0 | 0 | +0.000000 |
| M_repeat | MC2_clean_broken | 7 | 0 | 0 | +0.000000 |
| M_repeat | both_clean_wrong | 189 | 1 | 0 | +0.079365 |

现有硬预测文件没有传播状态、接收锁定、logits 或 margin，不能据此归因于某种 LOS／补偿／噪声机制；同物理与固定信道配置不冒称 received IQ 字节已核对。所有 TX／day 负结果继续保留于原报告。本分析无新增分类头，不将条件统计当作 MC2、跨 seed、正式 target 或联合目标证明。
