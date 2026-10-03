# CVS读出：完整源V分支归因与TX/RX/日期关联

状态VERIFIED。8个冻结E200模型、32条件、864000个分类决定，始终只有同27000个源V物理样本。全部logit、CE、配对预测和特征关联独立复算；全开准确率及最差RX复现原E200。零训练、零目标访问、不改变源选择。

time_off/behavior_off分别把对应读出恢复为已训练主干下的原skip；both_off同时关闭两者。保留其余训练权重，不能把关闭结果当作从零重训的shallow模型。贡献=全开准确率−干预后准确率。

|结构|条件|准确率/%|最差RX/%|CE|贡献/百分点±seed SD|正贡献seed|
|---|---|---:|---:|---:|---:|---:|
|readout_attention|all_on|98.2046|95.1759|0.094599|+0.0000±0.0000|0/4|
|readout_attention|time_off|96.7880|91.5648|0.146445|+1.4167±0.6454|4/4|
|readout_attention|behavior_off|73.7287|69.0602|1.681607|+24.4759±5.4663|4/4|
|readout_attention|both_off|64.6444|59.0370|2.711579|+33.5602±5.0145|4/4|
|readout_complex_attention|all_on|97.9796|94.9167|0.108483|+0.0000±0.0000|0/4|
|readout_complex_attention|time_off|97.3926|93.2407|0.130688|+0.5870±0.1483|4/4|
|readout_complex_attention|behavior_off|61.7333|51.5880|2.153574|+36.2463±10.5754|4/4|
|readout_complex_attention|both_off|56.5648|48.8889|2.716809|+41.4148±7.3429|4/4|

特征统计覆盖全部源V。每个TX×RX×day单元300条，正交均值散度分解包含TX、RX、day主效应、三项二阶交互、一项三阶交互及cell内残差；每包特征仅保留在地面N607。各项是平衡观测设计上的描述量，不是互信息、因果贡献或物理信道标签，不能推出TX/RX解耦。范数比按每包先计算再汇总，零分母边界见独立分析口径。

四模型seed共享数据划分，SD不是独立域置信区间。干预可能偏离训练分布，收益不可直接相加；准确率2×2交互及全部RX/TX/day单元保存在完整JSON。该源诊断不产生新候选或新的目标测试结果。

[逐seed分类](evidence/per_seed_classification.csv) · [特征因素占比](evidence/per_seed_factor_fractions.csv) · [完整独立复算](evidence/independent_recount.json) · [终态](evidence/final_readback.json)。

## 源证据解释

关闭行为增量使普通注意力与复数混合注意力的源准确率分别下降24.4759和36.2463个百分点；关闭时域增量下降1.4167和0.5870个百分点。同时关闭后只剩64.6444%和56.5648%，远不是独立从零训练shallow的98.4426%。两者权重不同，这不是同模型的训练消融；它说明新增读出与主干/分类头已经形成强依赖，保留skip的接线不能保证训练后仍保留原模型的识别能力。

全源V的行为增量/skip范数比均值为0.5974和0.6271。行为增量的TX主效应散度占比为40.2811%和47.9892%，RX主效应为8.4606%和4.9638%；时域增量的TX×RX交互占比为41.6915%和37.3756%。因此不能简单解释为“新增分支只学了RX”，也不能把较大的TX占比当成泛化保证。源V属于已见RX，raw特征散度不是可解码身份信息或物理因素能量。

上一轮更低训练CE、更高源V CE与更低源分数，结合本次冻结分支依赖，支持继续研究读出如何改变原判别表示以及训练中的共同适应。本次仍不能证明哪个物理失真造成性能下降，不能把移除分支后的结果直接当成新的架构成绩。后续设计依据限定源证据，不能使用历史目标分数调参。

|结构|特征|TX/%|RX/%|TX×RX/%|cell内/%|
|---|---|---:|---:|---:|---:|
|readout_attention|time_skip|18.5113|7.7825|34.3394|31.4382|
|readout_attention|time_delta|10.3672|4.7305|41.6915|37.5877|
|readout_attention|behavior_skip|19.1679|16.2040|24.3462|31.7411|
|readout_attention|behavior_delta|40.2811|8.4606|16.8052|29.1159|
|readout_complex_attention|time_skip|18.4308|7.4423|34.4887|31.8213|
|readout_complex_attention|time_delta|11.0660|4.8650|37.3756|40.6608|
|readout_complex_attention|behavior_skip|19.6309|14.9748|24.4386|32.2496|
|readout_complex_attention|behavior_delta|47.9892|4.9638|16.3182|25.3476|

此表只展示四项，日期及其余交互未合并进这些列；全部8项均值/seed SD见[完整因素汇总](evidence/factor_summary.csv)。[机制判定](evidence/mechanism_verdict.json) · [前轮完整源训练](../20261003-phase1-cvs-neural-readout-identity-manysig-m8-r01/report.md)。本诊断完成，识别性能与Phase1论文目标仍未达成。
