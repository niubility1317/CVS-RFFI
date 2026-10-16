# 固定源预测：clean 与 practical 错误转移

源 RX1-held／seed 2026092702，F＋跨 RX 同 TX／默认头，三组 final E20 的全部固定硬预测。此次仅复算原 1260 个物理样本，0 新 IQ／0 前向／0 更新，不改判据或训练。以下为描述性条件统计，不能证明传播机制原因，也不替代总体 accuracy／Macro-F1 与原固定晋级条件。

| 对照 | 场景 | 总体 accuracy 差 pp | 双方 clean 均正确：样本数 | 该集合修复／新错 | 该集合贡献 pp | 原对照 clean 正确集合内 accuracy 差 pp |
|---|---|---:|---:|---|---:|---:|
| D00 | practical_high | +20.317460 | 903 | 243/36 | +16.428571 | +21.325648 |
| D00 | practical_mid | +16.587302 | 903 | 210/49 | +12.777778 | +16.714697 |
| D00 | practical_low_suburban | +17.539683 | 903 | 233/48 | +14.682540 | +18.539866 |
| D00 | practical_high_urban | +17.063492 | 903 | 231/61 | +13.492063 | +17.771374 |
| D00 | practical_mid_urban | +11.269841 | 903 | 179/71 | +8.571429 | +11.623439 |
| D00 | practical_low_urban | +8.730159 | 903 | 155/66 | +7.063492 | +9.029779 |
| D_raw | practical_high | +20.000000 | 887 | 241/52 | +15.000000 | +20.039101 |
| D_raw | practical_mid | +16.825397 | 887 | 205/56 | +11.825397 | +16.226784 |
| D_raw | practical_low_suburban | +18.174603 | 887 | 231/49 | +14.444444 | +18.963832 |
| D_raw | practical_high_urban | +16.428571 | 887 | 219/69 | +11.904762 | +16.129032 |
| D_raw | practical_mid_urban | +12.142857 | 887 | 178/67 | +8.809524 | +12.316716 |
| D_raw | practical_low_urban | +8.730159 | 887 | 154/69 | +6.746032 | +8.797654 |

完整四组拆分和全部预测变化流保留在 error_transitions.json。四组为双方 clean 正确、仅 As 修复 clean、仅 As 新错 clean、双方 clean 错误；各组贡献之和严格等于总体 accuracy 差。条件集合由已有 clean 结果定义，仅解释同一源折的错误分布，未用于挑选或删除样本。

| 对照 | low urban clean 状态组 | 样本数 | practical 修复 | practical 新错 | 对总体贡献 pp |
|---|---|---:|---:|---:|---:|
| D00 | both_clean_correct | 903 | 155 | 66 | +7.063492 |
| D00 | As_clean_repaired | 34 | 10 | 0 | +0.793651 |
| D00 | As_clean_broken | 138 | 16 | 11 | +0.396825 |
| D00 | both_clean_wrong | 185 | 18 | 12 | +0.476190 |
| D_raw | both_clean_correct | 887 | 154 | 69 | +6.746032 |
| D_raw | As_clean_repaired | 50 | 12 | 1 | +0.873016 |
| D_raw | As_clean_broken | 136 | 15 | 10 | +0.396825 |
| D_raw | both_clean_wrong | 187 | 21 | 12 | +0.714286 |

现有硬预测文件没有传播状态、接收锁定、logits 或 margin，不能据此归因于某种 LOS／补偿／噪声机制；同物理与固定信道配置不冒称 received IQ 字节已核对。所有 TX／day 负结果继续保留于原报告。本分析无新增分类头，不将条件统计当作 Ag、跨 seed、正式 target 或联合目标证明。
