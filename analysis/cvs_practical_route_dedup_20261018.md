# practical 后续路线去重与 E50 边界

本轮完成实际调用和已完成实验的只读核对，没有登记或启动新训练。保留完整 practical 作为匹配参照；已完成的 E50 证明 urban 预算收益可重复，尚未证明 clean 与星地可重复双提升。不能为已关闭方案追加预算、改种子或换一个相近名称救跑。

## 已完成的 E50 证据

固定种子为 2026092701、2026092702、2026092703。下表均先逐种子配对，再取均值；单位为绝对准确率百分点。所有行均保留同预算 NoAs、全部源 RX、L6300/U56700 隐藏 TX/V27000、默认 cosine 头及 clean＋六种固定 practical。

|配方|As50−own As30 low urban|As50−own As30 mid urban|As50−own As30 clean|As50−匹配 NoAs50 clean|阶段判断|
|---|---:|---:|---:|---:|---|
|原完整 practical|+1.156746|+0.954167|−0.017659|−0.150198|均值 FAIL，2/3 seed 通过|
|固定 As 权重 0.375|+1.199603|+0.995437|−0.171825|−0.672817|均值 FAIL，1/3 seed 通过|

原配方 E50 的 matched clean 逐种子为 +0.365476、+0.436905、−1.252976 pp；固定加权配方为 −0.042857、−0.177976、−1.797619 pp。没有剔除失败种子。两配方 low urban 绝对均值分别为 51.926587% 和 52.673214%，均未超过 60%。

E50 是总预算 50 轮，实际轨迹 20＋10＋20，追加段重置 AdamW/cosine20，不是追加 50 轮或连续 cosine50。原配方保留为有完整星地收益的参照；固定 0.375 配方的双提升候选已关闭。正式结果不用于改变源域资格、重排候选或选择性重跑。

完整逐种子、波动、七视图、宏 F1、最差 RX、错误修复/新增、成本和未完成项见[原配方 E50 报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01/final_analysis.md)及[固定加权 E50 报告](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01/final_analysis.md)。

## 两个不再重开的相近假设

1. 完整 practical 内随机化 nominal reference SNR。实际代码每包只替换新增噪声参考方差，再由同一次 ResidualChannel 重新计算 quality、lock、tracking、AGC。原 IQ 中已有噪声不被删除，reference SNR 不等于真实总接收 SNR。N0prior 已使用等概率 0.5N0/1.5N0，对应约 33.0103/28.2391 dB；它没有额外叠加 AWGN。固定源输入检验失败后，历史又单独登记并完成匹配 E20＋10 训练和七视图评分，按当时预登记识别条件关闭。不得把“源输入实验失败”和“匹配训练完成后的关闭”混为同一 source 判据。

相关训练 run：20261017-phase1-F-crossRX-N0prior-e20plus10-s2026092701-m1-r01，ANALYZED。来源和控制是当时全 5 RX/L6300 的 4L/As128，不是当前 P43 6L/As64；但仅改变分布或幅度，不能构成未检验的新关键假设。见[历史完整训练报告](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-N0prior-e20plus10-s2026092701-m1-r01/report.md)。

2. NoAs20→As10 延迟加入完整 practical。carrierGate 的 AS_BATCH 和 shadowLHS 的 AS_NATIVE 都严格恢复原 fullsource REPEAT NoAs20 学生及 own EMA，仅在后 10 轮加入 As。全部 5 RX、L6300/U56700 隐藏/V27000，4L/As128。匹配源 V1800 的 clean 结果如下。

|历史 run|As 行|匹配 NoAs 行|As clean %|NoAs clean %|配对 pp|
|---|---|---|---:|---:|---:|
|carrierGate E20＋10|AS_BATCH|NOAS_BATCH|97.444444|97.944444|−0.500000|
|shadowLHS E20＋10|AS_NATIVE|NOAS|97.444444|97.888889|−0.444444|

两项登记均 ANALYZED。它们共享 model seed 和 REPEAT20 父来源，不当成两个独立种子。源证据表明已有延迟 As 未提供 clean 保护证据；不外推为所有延迟训练无效。见[carrier 完成报告](../automation_reports/CV-SincNet/20261018-phase1-F-crossRX-carrierGate-e20plus10-s2026092701-m4-r01/final_analysis.md)与[shadow 完成报告](../automation_reports/CV-SincNet/20261018-phase1-F-crossRX-shadowLHS-e20plus10-s2026092701-m4-r01/final_analysis.md)。

当前 P43 精确的 NoAs20→6L/As64×10 未实现，但会同时改变父表征与累计 As 剂量：原 As20→As10 共 426240 个 As 包，新设想只有 142080 个，都是累计 5114880 个 L 样本前向。它不是纯加入时机消融；仅用这点实现差异，不足以重开同类假设。本轮不启动。

## 实际执行边界

核对实际 P43 发布归档 commit 19aedc3604fed9c193237dd92366164178af0983 和 P41 底层调用。尾段确为 native E10、label_epochs=0/pseudo_epochs=10、严格 own student/EMA、reset AdamW/cosine10；ordinary 沿用 E20 args 的 epoch13，SupCon/信道使用累计偏移。仅更改 As 父路径为 NoAs 会被既有来源校验拒绝，本轮未修改或放宽来源规则。

本轮新增训练更新、模型前向、checkpoint/IQ 读取、远端执行和正式预测/评分均为 0；只读已保存的源统计、公开完整报告及发布源码。历史已公开正式数值仅用于如实汇报，不用于设计或晋级新方案。无已登记的新训练假设；不以建议清单代替实验，也不为保持任务运行而制造重复候选。E50 授权继续保留给有合法源证据支持且具有独立关键变化的后续配方。整体性能目标未完成。
