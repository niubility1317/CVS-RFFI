# 固定源预测：clean 与 practical 错误转移

源 RX1-held／seed 2026092701，F＋跨 RX 同 TX／默认头，三组 final E20 的全部固定硬预测。此次仅复算原 1260 个物理样本，0 新 IQ／0 前向／0 更新，不改判据或训练。以下为描述性条件统计，不能证明传播机制原因，也不替代总体 accuracy／Macro-F1 与原固定晋级条件。

| 对照 | 场景 | 总体 accuracy 差 pp | 双方 clean 均正确：样本数 | 该集合修复／新错 | 该集合贡献 pp | 原对照 clean 正确集合内 accuracy 差 pp |
|---|---|---:|---:|---|---:|---:|
| B00 | practical_high | +0.714286 | 1002 | 12/7 | +0.396825 | +0.492126 |
| B00 | practical_mid | +0.952381 | 1002 | 17/2 | +1.190476 | +1.476378 |
| B00 | practical_low_suburban | +1.190476 | 1002 | 23/4 | +1.507937 | +1.870079 |
| B00 | practical_high_urban | +0.952381 | 1002 | 12/3 | +0.714286 | +0.885827 |
| B00 | practical_mid_urban | -0.079365 | 1002 | 7/6 | +0.079365 | +0.098425 |
| B00 | practical_low_urban | -0.158730 | 1002 | 10/10 | +0.000000 | +0.000000 |
| C_ordinary | practical_high | +4.206349 | 1002 | 46/14 | +2.539683 | +3.118908 |
| C_ordinary | practical_mid | +4.285714 | 1002 | 41/15 | +2.063492 | +2.534113 |
| C_ordinary | practical_low_suburban | +3.412698 | 1002 | 46/20 | +2.063492 | +2.534113 |
| C_ordinary | practical_high_urban | +3.492063 | 1002 | 42/17 | +1.984127 | +2.534113 |
| C_ordinary | practical_mid_urban | +1.746032 | 1002 | 34/24 | +0.793651 | +0.974659 |
| C_ordinary | practical_low_urban | +0.555556 | 1002 | 29/26 | +0.238095 | +0.292398 |

完整四组拆分和全部预测变化流保留在 error_transitions.json。四组为双方 clean 正确、仅 Ag 修复 clean、仅 Ag 新错 clean、双方 clean 错误；各组贡献之和严格等于总体 accuracy 差。条件集合由已有 clean 结果定义，仅解释同一源折的错误分布，未用于挑选或删除样本。

| 对照 | low urban clean 状态组 | 样本数 | practical 修复 | practical 新错 | 对总体贡献 pp |
|---|---|---:|---:|---:|---:|
| B00 | both_clean_correct | 1002 | 10 | 10 | +0.000000 |
| B00 | Ag_clean_repaired | 7 | 0 | 1 | -0.079365 |
| B00 | Ag_clean_broken | 14 | 0 | 0 | +0.000000 |
| B00 | both_clean_wrong | 237 | 3 | 4 | -0.079365 |
| C_ordinary | both_clean_correct | 1002 | 29 | 26 | +0.238095 |
| C_ordinary | Ag_clean_repaired | 7 | 0 | 1 | -0.079365 |
| C_ordinary | Ag_clean_broken | 24 | 0 | 0 | +0.000000 |
| C_ordinary | both_clean_wrong | 227 | 9 | 4 | +0.396825 |

现有硬预测文件没有传播状态、接收锁定、logits 或 margin，不能据此归因于某种 LOS／补偿／噪声机制；同物理与固定信道配置不冒称 received IQ 字节已核对。所有 TX／day 负结果继续保留于原报告。本分析无新增分类头，不将条件统计当作 Ag、跨 seed、正式 target 或联合目标证明。
