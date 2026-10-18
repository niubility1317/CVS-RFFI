# P62 的 E30 源域背景：保留收益及负分层

本文件仅汇总已完成 SAT18 E30 的合法源聚合，不是新 E50 结果，不用于改变已启动配方、判定或预算。实际 NN/IQ/checkpoint/更新均0，未读取正式结果。modelseed2026092701，V1800物理样本×两draw；pooled每场景3600次观察，不是两个模型种子。

| 比较 | 场景 | 对照→DET准确率 | Δpp | 修复／新增 |
|---|---|---:|---:|---:|
| DET−CE | mid urban | 73.055556%→74.333333% | +1.277778 | 113／67 |
| DET−CE | low urban | 59.333333%→59.583333% | +0.250000 | 133／124 |
| DET−RD | mid urban | 72.805556%→74.333333% | +1.527778 | 145／90 |
| DET−RD | low urban | 59.083333%→59.583333% | +0.500000 | 159／141 |

mid BLOCKED/degraded(n656)：DET45.121951%，比CE+2.743902pp/净增18正确，比RD+2.591463pp/净增17。low BLOCKED/degraded(n414)：DET42.270531%，比CE−3.140097pp/净少13，比RD+.241546pp/净增1。low BLOCKED/unlocked(n1565)：DET29.904153%，比CE+.447284pp/净增7，比RD−.447284pp/净少7。没有一致覆盖所有困难状态的证据。旧聚合未保存每个分层的paired error交集，不能把净正确数虚构成修复/新增两项。

新增噪声比分层：mid的3至8dB(n718)比CE+2.785515pp；8至15dB(n293)比CE+2.047782pp；低于3dB(n383)为−.522193pp。low低于3dB(n1127)比CE−.088731pp/比RD−1.064774pp；3至8dB(n794)比CE−.503778pp；≥15dB(n1376)比CE+1.017442pp/比RD+1.671512pp。这里是signal_to_added_noise_db，不是总SNR；不能宣称DET普遍修复了噪声主导退化。

fullV27000原生clean（每RX5400，不与V1800面板混同）：

| RX | NoAs % | RD % | CE % | DET % | DET−NoAs pp |
|---|---:|---:|---:|---:|---:|
| 全体 | 98.359259 | 98.255556 | 98.329630 | 98.351852 | −.007407 |
| 1 | 99.481481 | 99.407407 | 99.462963 | 99.500000 | +.018519 |
| 3 | 95.611111 | 95.333333 | 95.462963 | 95.518519 | −.092593 |
| 4 | 98.388889 | 98.555556 | 98.592593 | 98.666667 | +.277778 |
| 6 | 99.629630 | 99.629630 | 99.685185 | 99.666667 | +.037037 |
| 8 | 98.685185 | 98.351852 | 98.444444 | 98.407407 | −.277778 |

现有背景支持mid有较明确源收益、clean总体基本保持；low收益不均匀、跨seed未验证。P62将按原固定八行比较报告E50相对自身E30和同预算CE/RD/NoAs的变化，不把更多预算或NoAs退化全部归给Dropout机制。

实际输入：[source_metrics.csv](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_satellite_dropout_E30_20261018/collection-final-r03/aggregates/source_metrics.csv>)、[source_summary.json](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_satellite_dropout_E30_20261018/collection-final-r03/aggregates/source_summary.json>)、[source_confusions.json](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_satellite_dropout_E30_20261018/collection-final-r03/aggregates/source_confusions.json>)、[source_complete.json](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_satellite_dropout_E30_20261018/collection-final-r03/aggregates/source_complete.json>)。
