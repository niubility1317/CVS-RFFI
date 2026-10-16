# 固定源预测：clean 与 practical 错误转移

源 RX1-held／seed 2026092701，F＋跨 RX 同 TX／默认头，三组 final E20 的全部固定硬预测。此次仅复算原 1260 个物理样本，0 新 IQ／0 前向／0 更新，不改判据或训练。以下为描述性条件统计，不能证明传播机制原因，也不替代总体 accuracy／Macro-F1 与原固定晋级条件。

| 对照 | 场景 | 总体 accuracy 差 pp | 双方 clean 均正确：样本数 | 该集合修复／新错 | 该集合贡献 pp | 原对照 clean 正确集合内 accuracy 差 pp |
|---|---|---:|---:|---|---:|---:|
| B00 | practical_high | +3.015873 | 990 | 37/18 | +1.507937 | +1.892430 |
| B00 | practical_mid | +3.650794 | 990 | 38/9 | +2.301587 | +2.888446 |
| B00 | practical_low_suburban | +1.507937 | 990 | 28/18 | +0.793651 | +0.996016 |
| B00 | practical_high_urban | +2.619048 | 990 | 28/15 | +1.031746 | +1.294821 |
| B00 | practical_mid_urban | +1.825397 | 990 | 26/15 | +0.873016 | +1.095618 |
| B00 | practical_low_urban | +0.952381 | 990 | 28/18 | +0.793651 | +0.996016 |
| C_ordinary | practical_high | +6.190476 | 995 | 69/27 | +3.333333 | +4.207436 |
| C_ordinary | practical_mid | +7.380952 | 995 | 61/19 | +3.333333 | +4.109589 |
| C_ordinary | practical_low_suburban | +4.841270 | 995 | 54/24 | +2.380952 | +2.935421 |
| C_ordinary | practical_high_urban | +5.873016 | 995 | 62/27 | +2.777778 | +3.522505 |
| C_ordinary | practical_mid_urban | +3.492063 | 995 | 51/29 | +1.746032 | +2.250489 |
| C_ordinary | practical_low_urban | +0.952381 | 995 | 38/36 | +0.158730 | +0.391389 |

完整四组拆分和全部预测变化流保留在 error_transitions.json。四组为双方 clean 正确、仅 Ag 修复 clean、仅 Ag 新错 clean、双方 clean 错误；各组贡献之和严格等于总体 accuracy 差。条件集合由已有 clean 结果定义，仅解释同一源折的错误分布，未用于挑选或删除样本。

| 对照 | low urban clean 状态组 | 样本数 | practical 修复 | practical 新错 | 对总体贡献 pp |
|---|---|---:|---:|---:|---:|
| B00 | both_clean_correct | 990 | 28 | 18 | +0.793651 |
| B00 | Ag_clean_repaired | 26 | 1 | 1 | +0.000000 |
| B00 | Ag_clean_broken | 14 | 0 | 0 | +0.000000 |
| B00 | both_clean_wrong | 230 | 11 | 9 | +0.158730 |
| C_ordinary | both_clean_correct | 995 | 38 | 36 | +0.158730 |
| C_ordinary | Ag_clean_repaired | 21 | 2 | 1 | +0.079365 |
| C_ordinary | Ag_clean_broken | 27 | 3 | 1 | +0.158730 |
| C_ordinary | both_clean_wrong | 217 | 17 | 10 | +0.555556 |

现有硬预测文件没有传播状态、接收锁定、logits 或 margin，不能据此归因于某种 LOS／补偿／噪声机制；同物理与固定信道配置不冒称 received IQ 字节已核对。所有 TX／day 负结果继续保留于原报告。本分析无新增分类头，不将条件统计当作 Ag、跨 seed、正式 target 或联合目标证明。
