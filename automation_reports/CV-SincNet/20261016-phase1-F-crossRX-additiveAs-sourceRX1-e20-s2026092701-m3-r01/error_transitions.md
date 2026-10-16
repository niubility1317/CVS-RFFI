# 固定源预测：clean 与 practical 错误转移

源 RX1-held／seed 2026092701，F＋跨 RX 同 TX／默认头，三组 final E20 的全部固定硬预测。此次仅复算原 1260 个物理样本，0 新 IQ／0 前向／0 更新，不改判据或训练。以下为描述性条件统计，不能证明传播机制原因，也不替代总体 accuracy／Macro-F1 与原固定晋级条件。

| 对照 | 场景 | 总体 accuracy 差 pp | 双方 clean 均正确：样本数 | 该集合修复／新错 | 该集合贡献 pp | 原对照 clean 正确集合内 accuracy 差 pp |
|---|---|---:|---:|---|---:|---:|
| D00 | practical_high | +16.984127 | 982 | 359/43 | +25.079365 | +30.267062 |
| D00 | practical_mid | +14.206349 | 982 | 330/62 | +21.269841 | +25.618200 |
| D00 | practical_low_suburban | +14.206349 | 982 | 317/58 | +20.555556 | +24.530168 |
| D00 | practical_high_urban | +13.412698 | 982 | 336/66 | +21.428571 | +25.618200 |
| D00 | practical_mid_urban | +10.000000 | 982 | 264/71 | +15.317460 | +18.199802 |
| D00 | practical_low_urban | +5.714286 | 982 | 188/70 | +9.365079 | +10.781405 |
| D_raw | practical_high | +19.444444 | 991 | 365/36 | +26.111111 | +31.755424 |
| D_raw | practical_mid | +15.396825 | 991 | 334/58 | +21.904762 | +26.725838 |
| D_raw | practical_low_suburban | +15.317460 | 991 | 314/51 | +20.873016 | +25.246548 |
| D_raw | practical_high_urban | +15.158730 | 991 | 333/62 | +21.507937 | +26.429980 |
| D_raw | practical_mid_urban | +10.317460 | 991 | 263/77 | +14.761905 | +17.751479 |
| D_raw | practical_low_urban | +6.190476 | 991 | 183/69 | +9.047619 | +10.946746 |

首种子的 low urban 在双方 clean 都正确的 982 个样本中修复 188／新错 70，净增 118 个；其他三组净减 46 个，总体净增 72 个（+5.714286 个百分点）。因此，这个源折上的 practical 收益并非只伴随 clean 新修复样本增加；它仍包含明显错误转移，不能推断各 TX 普遍受益或传播因果机制。

完整四组拆分和全部预测变化流保留在 error_transitions.json。四组为双方 clean 正确、仅 As 修复 clean、仅 As 新错 clean、双方 clean 错误；各组贡献之和严格等于总体 accuracy 差。条件集合由已有 clean 结果定义，仅解释同一源折的错误分布，未用于挑选或删除样本。

| 对照 | low urban clean 状态组 | 样本数 | practical 修复 | practical 新错 | 对总体贡献 pp |
|---|---|---:|---:|---:|---:|
| D00 | both_clean_correct | 982 | 188 | 70 | +9.365079 |
| D00 | As_clean_repaired | 71 | 8 | 12 | -0.317460 |
| D00 | As_clean_broken | 29 | 1 | 10 | -0.714286 |
| D00 | both_clean_wrong | 178 | 10 | 43 | -2.619048 |
| D_raw | both_clean_correct | 991 | 183 | 69 | +9.047619 |
| D_raw | As_clean_repaired | 62 | 6 | 9 | -0.238095 |
| D_raw | As_clean_broken | 23 | 3 | 6 | -0.238095 |
| D_raw | both_clean_wrong | 184 | 8 | 38 | -2.380952 |

现有硬预测文件没有传播状态、接收锁定、logits 或 margin，不能据此归因于某种 LOS／补偿／噪声机制；同物理与固定信道配置不冒称 received IQ 字节已核对。所有 TX／day 负结果继续保留于原报告。本分析无新增分类头，不将条件统计当作 Ag、跨 seed、正式 target 或联合目标证明。
