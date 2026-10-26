# RX 均值差软一致性：完整三种子结果

状态：ANALYZED。两种约束范围各三种子全部完成 E50、源诊断、原始 IQ 的 clean＋六种 practical 正式预测及独立评分。**两种方法都未优于普通 PL 的 clean；本候选未达目标。** 未改变已在本结果读取前固定的 RXMeanMax 候选。

固定原完整 PAMF、单 SIG_TIME、原默认头联合训练。每个 seed 复用合法原 E20，新增 10＋20 轮至 E50/11100，累计 13L＋37LU。保留原 T1/fullL、普通 PL 阈值 0.95/权重 1。三 seed 为 2026092701/02/03；原数据契约/划分 392005 不变。有标签 6300、无标签 56700（TX 隐藏、RX 已知）、源 V 27000。六个已完成 PL/Sup 控制直接复用，不重新训练、预测或评分。

机制只在训练使用原 clean 或其同包 H 派生物，以及训练 U 的五种源 RX 低维响应均值差。随机选择另一源 RX 风格，固定强度 0.1，幅度/相位限制 ±0.5 dB/±0.025 rad。全 U 进行 base→shift 的 detached-soft-target KL，raw/H 各 15 轮；原 PL 不受此损失的接受门控限制。RXMean-U 只优化 U 一致性，RXMean-LU 再优化匹配的 L 对；两组有相同 L 前向量，U 组的 L 项只保留 detached 监测。无 DURA、EMA、新头或推理增参。RX 统计是混合响应，不能解释为纯硬件校正。

正式总体为 cross_rx 126000＋cross_both 42000，按 3:1 汇总，不把两个准确率等权平均。各行全部两人口×七视图预测固定后才独立 truth-last 评分。下表为三 seed 均值±样本标准差，单位 %。

| 方法 | clean | 六景平均 |
| --- | ---: | ---: |
| RXMean-U | 84.532341 ± 0.558877 | 70.364649 ± 1.073717 |
| RXMean-LU | 84.525595 ± 0.629116 | 70.358102 ± 1.097501 |
| SIG1-PL | 84.693849 ± 0.692508 | 70.233234 ± 1.043286 |
| SIG1-Sup | 84.116865 ± 0.887612 | 70.168056 ± 1.035243 |

与同 seed 普通 PL 的配对差（均值±样本标准差，百分点）：

| 方法 | clean 差 | 六景差 |
| --- | ---: | ---: |
| RXMean-U | -0.161508 ± 0.139601 | +0.131415 ± 0.108062 |
| RXMean-LU | -0.168254 ± 0.065806 | +0.124868 ± 0.113989 |

逐 seed 总体配对差，单位百分点：

| 方法 | seed | clean 差 | 六景差 |
| --- | ---: | ---: | ---: |
| RXMean-U | 2026092701 | -0.319643 | +0.246329 |
| RXMean-U | 2026092702 | -0.055357 | +0.116071 |
| RXMean-U | 2026092703 | -0.109524 | +0.031845 |
| RXMean-LU | 2026092701 | -0.242857 | +0.254167 |
| RXMean-LU | 2026092702 | -0.118452 | +0.081548 |
| RXMean-LU | 2026092703 | -0.143452 | +0.038889 |

六景存在小幅收益，不能用来掩盖 clean 损伤。将 L 加入同一一致性目标也没有带来平均优势；此结果只适用于当前响应扰动和权重，不能推导出所有半监督阶段都不应正则 L。两版均未达到原 clean≥85.98174603174603%、六景>72.0961970899471% 的绝对目标，也未满足全部声明分层严格优于 PL/Sup。三种子用于观察波动，不等于统计显著或独立新盲测。

完整日志核验见 [complete_source_audit.json](complete_source_audit.json)：6 行共 39960 条新增 step、180 模型 epoch，全部逐条检查 raw/H 日程、原 PL、fullL、剂量与段间计数，完整 stdout 无 fatal。每模型 56700 个 U 各 30 遍，共 1701000 曝光；raw/H 各 850500；6660 次更新、64L 配对共 426240；20 个 RX 风格对均使用。原生完整训练网络 863730 参数，部署识别 reader 196001 参数，SIG1 池 18976；两者不是同一参数口径。

完整数据对应检查见 [matrix_verification.json](matrix_verification.json)：7056000 次新增 raw 正式判决，12 个候选/引用 score 行、24 项同 seed 物理 ID 顺序检查。详见 [逐景](results/per_scene_mean.csv)、[逐 seed](results/per_seed.csv)、[配对总体](results/paired_summary.csv)、[配对逐景](results/paired_scene_mean.csv) 和 [全部声明分层](results/paired_strata_mean.csv)。每个 source 的实际曝光与接受率见完整源审计；接受率不等于 U 伪标签真实准确率。

源校准共 36 批：共同 alpha 都为 1，444 步 ramp。此前的 [calibration_analysis.json](calibration_analysis.json) 显示辅助/主欧氏梯度范数中位比例 raw 约 3.57% 至 5.64%、H 约 7.18% 至 11.51%，部分批次局部反向；这不证明实际 AdamW 方向或目标损伤因果。保留原 r01 的导入顺序技术失败（0 optimizer）及完成的 U 均值缓存，r02 修复后执行 commit `fa758a113f943f35a9da423fb1ca5cd3ff6a071c`；不存在按目标收益重启或挑 seed。

固定代码入口 `rx_mean_source`、完整日志 `rx_mean_audit`、固定产物读取/汇总 `rx_mean_results`。原始 IQ、权重、逐包 logits/IDs 和私有路径留在原 run/private artifacts，仅提交公开统计。后续 RXMeanMax 三 seed 已在本结果之前固定并排队，待本 run 完成才占 GPU；本负结果不会改变其配方、预算或 seed。原目标仍 active。

补充已存在源预测分析：[同包 raw/H 参照诊断](source_reference_diagnostics/report.md)。两方法×三 seed、648000 份已有 logits/1620 分层，无新 F/训练/target/U 真值。高置信 raw 的 low urban 参照约 38.8% 从 raw 对变为 H 错；H 高置信子集仍约 99.77% 正确。它提示区分参照证据强弱，不能用一致性替代身份纠错证据，也不作为当前 Max 配方修改依据。

已存20项U/RX统计的附加代数检查见[传递表往返/环路诊断](transfer_geometry_report.md)：非自洽对数增益/相位能量0.033164%/0.281932%，投影表RMS变化0.00055448。0新IQ/F/权重/UTX/V/target/训练，不能认定身份保护或性能退步根因。证据不足以优先另开该投影的神经训练，当前PM保持；旧训练和正式结果不改写。
