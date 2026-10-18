# 连续 E30 星地增强恢复实验结果

固定默认分类头、固定正式 IQ 与 clean＋六种 practical；model seed 为 2026092701。相对同预算连续 No-As，urban 三场景均值变化 **+27.870 个百分点**，low urban **+20.955 个百分点**，mid urban **+27.245 个百分点**；clean 变化 **-0.118 个百分点**。

clean 下降不超过 0.1 个绝对百分点的阶段目标：**不满足**。这一判定仅用于报告，不改变原 source-only 选择条件或触发重跑。单个种子尚未观察到 clean 与 urban 同升；**尚无跨种子可重复双提升证据**。

## 四行正式准确率

STAGED 为原 20＋10 训练组织；CONT 为连续 30。B00 无星地增强，B01 使用原完整 practical 增强。两种训练组织均使用 6 次 L 前向、L 损失尺度 1.25、全部源 RX 与原 U 流程。只有 CONT_B01 为本次恢复新增训练。

| 行 | clean | high suburban | mid suburban | low suburban | high urban | mid urban | low urban | urban 均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STAGED_B00 | 85.248% | 42.685% | 40.995% | 39.473% | 41.440% | 35.133% | 30.140% | 35.571% |
| STAGED_B01 | 85.341% | 81.405% | 78.661% | 75.862% | 77.767% | 62.807% | 51.244% | 63.939% |
| CONT_B00 | 84.883% | 43.560% | 41.729% | 40.027% | 42.320% | 35.608% | 30.417% | 36.115% |
| CONT_B01 | 84.765% | 81.324% | 78.602% | 75.790% | 77.729% | 62.853% | 51.373% | 63.985% |

下表每格依次为宏平均 F1／最差 RX 准确率，单位均为 %。

| 环境 | STAGED_B00 | STAGED_B01 | CONT_B00 | CONT_B01 |
| --- | --- | --- | --- | --- |
| clean | 84.980% / 78.450% | 85.044% / 78.362% | 84.597% / 76.875% | 84.428% / 77.225% |
| high suburban | 39.311% / 34.304% | 81.084% / 71.908% | 40.083% / 33.296% | 80.951% / 71.671% |
| mid suburban | 37.955% / 33.521% | 78.360% / 70.213% | 38.728% / 32.325% | 78.244% / 70.025% |
| low suburban | 36.759% / 32.054% | 75.573% / 67.742% | 37.399% / 31.179% | 75.439% / 67.479% |
| high urban | 38.481% / 34.000% | 77.423% / 69.108% | 39.294% / 32.883% | 77.341% / 68.708% |
| mid urban | 33.447% / 29.496% | 62.450% / 56.458% | 34.020% / 28.775% | 62.407% / 56.304% |
| low urban | 29.185% / 25.750% | 50.887% / 46.379% | 29.674% / 25.171% | 50.889% / 46.496% |

## 配对变化与错误修复

连续 As − 连续 No-As；差值单位为绝对百分点，错误数使用相同正式物理样本。

| 环境 | Δ准确率 | Δ宏 F1 | Δ最差 RX | 修复 | 新增错误 | 净修复 |
| --- | --- | --- | --- | --- | --- | --- |
| clean | -0.118 | -0.169 | +0.350 | 3609 | 3807 | -198 |
| high suburban | +37.764 | +40.867 | +38.375 | 67124 | 3681 | 63443 |
| mid suburban | +36.873 | +39.516 | +37.700 | 66178 | 4231 | 61947 |
| low suburban | +35.763 | +38.041 | +36.300 | 64791 | 4709 | 60082 |
| high urban | +35.409 | +38.047 | +35.825 | 63810 | 4323 | 59487 |
| mid urban | +27.245 | +28.386 | +27.529 | 52647 | 6876 | 45771 |
| low urban | +20.955 | +21.215 | +21.325 | 44031 | 8826 | 35205 |

连续 As − staged As；差值单位为绝对百分点，错误数使用相同正式物理样本。

| 环境 | Δ准确率 | Δ宏 F1 | Δ最差 RX | 修复 | 新增错误 | 净修复 |
| --- | --- | --- | --- | --- | --- | --- |
| clean | -0.576 | -0.616 | -1.138 | 582 | 1549 | -967 |
| high suburban | -0.082 | -0.134 | -0.238 | 948 | 1085 | -137 |
| mid suburban | -0.059 | -0.116 | -0.187 | 1144 | 1243 | -99 |
| low suburban | -0.072 | -0.133 | -0.262 | 1385 | 1506 | -121 |
| high urban | -0.039 | -0.082 | -0.400 | 1182 | 1247 | -65 |
| mid urban | +0.046 | -0.044 | -0.154 | 1963 | 1885 | 78 |
| low urban | +0.129 | +0.002 | +0.117 | 2499 | 2283 | 216 |

## 源域证据与保持率

原 source-only 决策：`SOURCE_CRITERIA_FAIL_FIXED_CONTINUOUS_ROUTE_CLOSED`，source_qualified=`false`。该结果在正式汇总之前冻结。

| 预登记条件 | 实际判定 |
| --- | --- |
| clean_not_below_matched_NoAs | 未通过 |
| clean_vs_staged_As_within_quarter_pp | 通过 |
| each_RX_vs_staged_As_within_half_pp | 通过 |
| both_draw_mid_low_vs_staged_As_gain_one_pp | 未通过 |

| 行 | 完整源 V clean | 固定两次 draw mid urban 均值 | 固定两次 draw low urban 均值 |
| --- | --- | --- | --- |
| STAGED_B00 | 98.359% | 43.556% | 35.611% |
| STAGED_B01 | 98.256% | 72.806% | 59.083% |
| CONT_B00 | 98.333% | 43.944% | 36.556% |
| CONT_B01 | 98.256% | 72.833% | 59.056% |

固定集合取同种子连续 No-As 的最终 E30 clean 正确样本；该集合来自匹配对照，不代表共享预训练父模型。原 STAGED_B00 固定集合的结果同时保留。

| 环境 | 固定集合样本数 | No-As 保留数 | No-As 保持率 | As 保留数 | As 保持率 |
| --- | --- | --- | --- | --- | --- |
| clean | 142604 | 142604 | 100.000% | 138797 | 97.330% |
| high suburban | 142604 | 68753 | 48.213% | 128688 | 90.242% |
| mid suburban | 142604 | 65733 | 46.095% | 123952 | 86.920% |
| low suburban | 142604 | 62876 | 44.091% | 119267 | 83.635% |
| high urban | 142604 | 66712 | 46.781% | 122779 | 86.098% |
| mid urban | 142604 | 55836 | 39.155% | 98419 | 69.016% |
| low urban | 142604 | 47201 | 33.099% | 79803 | 55.961% |

以下为同一固定源域面板的 low／mid urban 分层关联。信号与新增噪声比不是总 SNR；状态、接收质量与传播损伤相互依赖，不能据此把收益唯一归因为噪声、CFO、多径或门控。正式测试没有逐样本物理状态，源域分层收益不能直接当作正式状态因果结论。

| 环境 | 分层 | 区间 | 样本数 | No-As | As | Δ对照 pp | Δstaged As pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low urban | 信号/新增噪声比 dB | 3-8 | 794 | 28.212% | 39.924% | +11.713 | +0.252 |
| low urban | 信号/新增噪声比 dB | 8-15 | 303 | 41.914% | 71.287% | +29.373 | -0.990 |
| low urban | 信号/新增噪声比 dB | <3 | 1127 | 19.343% | 27.507% | +8.163 | +0.177 |
| low urban | 信号/新增噪声比 dB | >=15 | 1376 | 54.288% | 93.241% | +38.953 | -0.145 |
| low urban | 状态/锁定 | BLOCKED/degraded | 414 | 31.401% | 42.271% | +10.870 | +0.242 |
| low urban | 状态/锁定 | BLOCKED/unlocked | 1565 | 21.342% | 30.607% | +9.265 | +0.256 |
| low urban | 状态/锁定 | LOS/locked | 1060 | 56.321% | 94.717% | +38.396 | +0.000 |
| low urban | 状态/锁定 | PARTIAL/degraded | 253 | 44.269% | 77.866% | +33.597 | -1.581 |
| low urban | 状态/锁定 | PARTIAL/locked | 307 | 46.580% | 88.274% | +41.694 | -0.651 |
| low urban | 状态/锁定 | PARTIAL/unlocked | 1 | 0.000% | 0.000% | +0.000 | +0.000 |
| mid urban | 信号/新增噪声比 dB | 3-8 | 718 | 25.766% | 38.997% | +13.231 | +0.418 |
| mid urban | 信号/新增噪声比 dB | 8-15 | 293 | 37.201% | 52.560% | +15.358 | +0.000 |
| mid urban | 信号/新增噪声比 dB | <3 | 383 | 18.277% | 26.632% | +8.355 | -0.783 |
| mid urban | 信号/新增噪声比 dB | >=15 | 2206 | 55.213% | 94.560% | +39.347 | +0.045 |
| mid urban | 状态/锁定 | BLOCKED/degraded | 656 | 31.402% | 42.835% | +11.433 | +0.305 |
| mid urban | 状态/锁定 | BLOCKED/unlocked | 678 | 19.322% | 31.268% | +11.947 | -0.295 |
| mid urban | 状态/锁定 | LOS/locked | 1899 | 55.503% | 95.629% | +40.126 | +0.158 |
| mid urban | 状态/锁定 | PARTIAL/degraded | 59 | 50.847% | 76.271% | +25.424 | +0.000 |
| mid urban | 状态/锁定 | PARTIAL/locked | 306 | 52.614% | 87.582% | +34.967 | -0.654 |
| mid urban | 状态/锁定 | PARTIAL/unlocked | 2 | 0.000% | 0.000% | +0.000 | +0.000 |

训练覆盖状态：`VERIFIED_SOURCE_TRAINING_STATE_COVERAGE`；实际 As 观察量 426240，全部 TX×RX×day×scene×state 单元非空：`true`。覆盖重放只验证命名传播状态及实际训练支持，不重放接收锁定、AGC 或噪声波形。

## 实际成本

本次新增 CONT_B01 为 30 个模型 epoch／6660 次更新；复用 CONT_B00 和两个 STAGED 行合计 90 个模型 epoch／19980 次更新。原失败 As 已完成 20 个 epoch／4440 次更新，未用于恢复初始化。

| 范围 | 成功模型 epoch | 失败模型 epoch | 合计模型 epoch | 更新数 |
| --- | --- | --- | --- | --- |
| 主要连续配对（旧 No-As＋新 As＋旧失败） | 60 | 20 | 80 | 17760 |
| 四行完整对比及旧失败 | 120 | 20 | 140 | 31080 |

每行 NN 为实际身份网络处理样本量；“源阶段”含已记录的训练／验证／教师和源域读出，正式预测单独列出。

| 行 | 是否复用 | 源阶段 NN | 正式预测 NN | 合计 NN | 源训练阶段墙钟 | 正式预测墙钟 |
| --- | --- | --- | --- | --- | --- | --- |
| STAGED_B00 | 是 | 7973284 | 1176002 | 9149286 | 2146.190 s | 48.089 s |
| STAGED_B01 | 是 | 7973284 | 1176002 | 9149286 | 2860.909 s | 49.650 s |
| CONT_B00 | 是 | 7919282 | 1176002 | 9095284 | 2132.725 s | 47.367 s |
| CONT_B01 | 否 | 7919282 | 1176002 | 9095284 | 3027.225 s | 48.455 s |

新训练及本次启动 smoke 合计 NN：9096822；复用三行 NN：27393856。旧失败 run 的共享 smoke：1538 NN；P43 历史六行共享 smoke 未分摊到这四行。

失败原 As 已完成步骤的 L 前向下界为 3409920 NN，失败总 NN 为 **UNKNOWN**；未完整记录的失败步骤、U 与评估前向不作估算。四行、旧失败与已分摊 smoke 的已知总 NN 下界为 39902136，总 NN 为 **UNKNOWN**。失败已完成 20 轮墙钟之和为 1795.009 s。

各行墙钟只覆盖各自实际记录范围；并行行的墙钟相加不能代表整批经过时间。预算耗尽也不代表已经收敛。

分析流水线状态：`ANALYZED`；包括等待的流水线墙钟：2271.130 s。本报告收集新增训练更新、NN、IQ 及 checkpoint 读取均为 0。

## 已生成产物与未完成项

实际图表：

- [formal_accuracy.pdf](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/formal_accuracy.pdf>)
- [formal_accuracy.png](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/formal_accuracy.png>)
- [full_training_curves.pdf](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/full_training_curves.pdf>)
- [full_training_curves.png](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/full_training_curves.png>)
- [paired_effects.pdf](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/paired_effects.pdf>)
- [paired_effects.png](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/paired_effects.png>)
- [suburban_gaps.pdf](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/suburban_gaps.pdf>)
- [suburban_gaps.png](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/suburban_gaps.png>)
- [source_lowurban_states.pdf](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/source/source_lowurban_states.pdf>)
- [source_lowurban_states.png](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/source/source_lowurban_states.png>)

实际 CSV：

- [epoch_TX_RX_state.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/coverage/epoch_TX_RX_state.csv>)
- [TX_RX_day_scene_state.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/coverage/TX_RX_day_scene_state.csv>)
- [fixed_CONT_B00_retention.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/fixed_CONT_B00_retention.csv>)
- [fixed_STAGED_B00_retention.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/fixed_STAGED_B00_retention.csv>)
- [overall_by_row.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/overall_by_row.csv>)
- [paired_repairs.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/paired_repairs.csv>)
- [per_row_metrics.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/per_row_metrics.csv>)
- [suburban_gaps.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/suburban_gaps.csv>)
- [trajectory_x_As_interaction.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/trajectory_x_As_interaction.csv>)
- [urban_means.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/urban_means.csv>)
- [source_strata.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/source/source_strata.csv>)
- [training_curves.csv](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/source/training_curves.csv>)

可核查汇总：

- [source_decision.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/source/source_decision.json>)
- [source_summary.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/source/source_summary.json>)
- [failed_original_attempt.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/source/failed_original_attempt.json>)
- [analysis_summary.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/analysis_summary.json>)
- [complete.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/final/complete.json>)
- [complete.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/coverage/complete.json>)
- [complete.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/analysis_publish_r01/retrieved/complete.json>)

当前只有 1 个模型种子，四行不是四个种子。尚未完成跨种子重复验证，不能宣称稳定双提升。连续训练同时改变优化器、EMA、时间缓存及采样器重启，不构成独立学习率因果实验。本轮没有新增 Ag；不据此宣称 Ag×As 机制成立。未采集的新分支／门控轨迹不作事后归因。正式分数不回流调参、晋级、选种子或选择性重跑。
