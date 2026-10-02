# CVS相位曲率：冻结源域归因结果

状态VERIFIED。全部8个冻结E200模型完成9条件，72份预测逐文件独立重算，共1944000个预测决定；物理样本始终是同27000条源V，不能当作1944000条独立样本。全开复现原源指标，每次关闭系数后恢复原完整状态。无训练、目标访问或源重选。

贡献=全开准确率−关闭后的准确率，正值表示该修正有帮助，单位百分点。帮助/损害/翻转计数合计四模型，同一物理样本可被不同模型重复计数。

|候选|关闭项|贡献均值±seed SD|全开帮助|全开损害|预测翻转|正贡献seed|
|---|---|---:|---:|---:|---:|---:|
|phase_curvature_lag14|all_on|+0.0000±0.0000|0|0|0|0/4|
|phase_curvature_lag14|curvature_all_off|+0.0046±0.0115|43|38|95|3/4|
|phase_curvature_lag14|curvature_off_0|-0.0037±0.0074|15|19|41|0/4|
|phase_curvature_lag14|curvature_off_1|+0.0009±0.0097|21|20|45|1/4|
|phase_curvature_lag14|curvature_off_2|-0.0028±0.0093|18|21|44|1/4|
|phase_curvature_lag14|curvature_off_3|+0.0065±0.0155|12|5|20|1/4|
|phase_curvature_lag14|curvature_off_4|+0.0000±0.0168|26|26|62|1/4|
|phase_curvature_lag14|curvature_off_5|+0.0037±0.0068|9|5|16|2/4|
|phase_curvature_lag14|input_mix_off|+0.0102±0.0115|43|32|94|3/4|
|phase_curvature_lag24|all_on|+0.0000±0.0000|0|0|0|0/4|
|phase_curvature_lag24|curvature_all_off|+0.0167±0.0178|53|35|113|3/4|
|phase_curvature_lag24|curvature_off_0|-0.0028±0.0106|19|22|51|1/4|
|phase_curvature_lag24|curvature_off_1|+0.0019±0.0111|19|17|46|1/4|
|phase_curvature_lag24|curvature_off_2|+0.0000±0.0068|16|16|42|2/4|
|phase_curvature_lag24|curvature_off_3|+0.0093±0.0239|30|20|63|2/4|
|phase_curvature_lag24|curvature_off_4|+0.0000±0.0228|21|21|49|1/4|
|phase_curvature_lag24|curvature_off_5|-0.0037±0.0052|10|14|29|0/4|
|phase_curvature_lag24|input_mix_off|+0.0028±0.0143|35|32|85|2/4|

这些是固定已训练权重下的局部因果干预，不等于从零重训移除该模块的效果。逐层贡献不可直接相加，层间有交互；源V是已见源RX的验证集，不能据此声称未见RX泛化或唯一TX硬件辨识。诊断不改变原源排名或条件clean矩阵。

[逐seed完整72行](evidence/per_seed_attribution.csv) · [逐文件独立复算](evidence/independent_recount.json) · [进程终态和参数恢复](evidence/final_readback.json)。原始预测保留在N607登记路径。

[完整RX归因](evidence/receiver_attribution.csv)与[全部source混淆矩阵](evidence/source_confusions.csv)已由逐文件预测重算。CE在逐seed表中来自运行时完整V交叉熵记录，不从argmax反推。

[综合诊断](../../../docs/CVS_NETWORK_DIAGNOSIS_20261002.md)和[纯网络结构下一轮](../../../docs/CVS_NEURAL_RESIDUAL_20261002.md)。用户明确只改网络结构，保持单一CE与原训练方案。
