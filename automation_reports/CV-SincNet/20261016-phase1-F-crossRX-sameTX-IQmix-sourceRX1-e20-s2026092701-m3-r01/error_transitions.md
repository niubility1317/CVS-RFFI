# 固定源预测：clean 与 practical 错误转移

源 RX1-held／seed 2026092701，F＋跨 RX 同 TX／默认头，三组 final E20 的全部固定硬预测。此次仅复算原 1260 个物理样本，0 新 IQ／0 前向／0 更新，不改判据或训练。以下为描述性条件统计，不能证明传播机制原因，也不替代总体 accuracy／Macro-F1 与原固定晋级条件。

| 对照 | 场景 | 总体 accuracy 差 pp | 双方 clean 均正确：样本数 | 该集合修复／新错 | 该集合贡献 pp | 原对照 clean 正确集合内 accuracy 差 pp |
|---|---|---:|---:|---|---:|---:|
| B00 | practical_high | +0.079365 | 1000 | 15/10 | +0.396825 | +0.493583 |
| B00 | practical_mid | +0.317460 | 1000 | 15/6 | +0.714286 | +0.888450 |
| B00 | practical_low_suburban | +0.238095 | 1000 | 21/11 | +0.793651 | +0.987167 |
| B00 | practical_high_urban | +0.634921 | 1000 | 17/7 | +0.793651 | +0.987167 |
| B00 | practical_mid_urban | -0.714286 | 1000 | 11/16 | -0.396825 | -0.493583 |
| B00 | practical_low_urban | +0.238095 | 1000 | 18/15 | +0.238095 | +0.296150 |
| C_same | practical_high | -0.238095 | 1014 | 15/15 | +0.000000 | +0.000000 |
| C_same | practical_mid | -1.190476 | 1014 | 12/15 | -0.238095 | -0.292683 |
| C_same | practical_low_suburban | -0.158730 | 1014 | 19/10 | +0.714286 | +0.878049 |
| C_same | practical_high_urban | -0.476190 | 1014 | 14/11 | +0.238095 | +0.292683 |
| C_same | practical_mid_urban | +0.000000 | 1014 | 13/11 | +0.158730 | +0.195122 |
| C_same | practical_low_urban | +0.158730 | 1014 | 17/15 | +0.158730 | +0.195122 |

完整四组拆分和全部预测变化流保留在 error_transitions.json。四组为双方 clean 正确、仅 Ag 修复 clean、仅 Ag 新错 clean、双方 clean 错误；各组贡献之和严格等于总体 accuracy 差。条件集合由已有 clean 结果定义，仅解释同一源折的错误分布，未用于挑选或删除样本。

| 对照 | low urban clean 状态组 | 样本数 | practical 修复 | practical 新错 | 对总体贡献 pp |
|---|---|---:|---:|---:|---:|
| B00 | both_clean_correct | 1000 | 18 | 15 | +0.238095 |
| B00 | Ag_clean_repaired | 19 | 1 | 0 | +0.079365 |
| B00 | Ag_clean_broken | 13 | 0 | 0 | +0.000000 |
| B00 | both_clean_wrong | 228 | 6 | 7 | -0.079365 |
| C_same | both_clean_correct | 1014 | 17 | 15 | +0.158730 |
| C_same | Ag_clean_repaired | 5 | 0 | 0 | +0.000000 |
| C_same | Ag_clean_broken | 11 | 0 | 0 | +0.000000 |
| C_same | both_clean_wrong | 230 | 6 | 6 | +0.000000 |

现有硬预测文件没有传播状态、接收锁定、logits 或 margin，不能据此归因于某种 LOS／补偿／噪声机制；同物理与固定信道配置不冒称 received IQ 字节已核对。所有 TX／day 负结果继续保留于原报告。本分析无新增分类头，不将条件统计当作 Ag、跨 seed、正式 target 或联合目标证明。
