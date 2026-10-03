# CVS补偿响应：完整源V判别几何

状态VERIFIED。12个冻结模型，各27000条源V；主干与响应特征、补偿系数及分类输出均已保留。独立从特征与分类头复算logit，从保存的logit复算CE/间隔；648000个分类决定与上一份源诊断逐样本完全一致。未拟合模型或使用目标数据。

|结构|响应/主干余弦均值|平行响应能量比例|单位特征变化|响应位于分类差异子空间的能量比例|全开CE|关闭CE|
|---|---:|---:|---:|---:|---:|---:|
|response_mean|-0.705676|0.508331|0.469685|0.014400|0.095940|0.078080|
|response_attention|-0.718226|0.524985|0.478018|0.010678|0.098046|0.078341|
|response_order_attention|-0.712295|0.516279|0.472176|0.010393|0.096732|0.077115|

分类差异子空间由归一化类别权重减去类别均值后定义，排除所有logit共同平移。几何比例是逐样本计算再求均值；它不是互信息、可恢复身份信息或新架构收益保证。

|结构|表示|TX均值差异占比|同TX不同RX/day均值差异占比|cell内占比|
|---|---|---:|---:|---:|
|response_mean|base_unit|0.612526|0.194604|0.192869|
|response_mean|response_unit|0.045396|0.266267|0.688337|
|response_mean|joint_unit|0.618102|0.187066|0.194832|
|response_mean|compensation_coefficients|0.027080|0.443222|0.529698|
|response_attention|base_unit|0.610020|0.195091|0.194889|
|response_attention|response_unit|0.037234|0.427118|0.535647|
|response_attention|joint_unit|0.613265|0.189310|0.197425|
|response_attention|compensation_coefficients|0.027360|0.409002|0.563637|
|response_order_attention|base_unit|0.611327|0.194522|0.194151|
|response_order_attention|response_unit|0.044442|0.257832|0.697726|
|response_order_attention|joint_unit|0.615167|0.187875|0.196957|
|response_order_attention|compensation_coefficients|0.024164|0.407319|0.568517|

嵌套散度按TX→TX/RX/day→cell内分解，是给定源数据上的描述量。相同TX在不同RX/day的均值差异不能直接解释为纯信道因素，cell内也可能包含噪声、调制及信道变化。方向统计不能推出可辨识的TX/RX解耦。

[逐seed、正确性分区与CE](evidence/per_seed_geometry.csv) · [完整RX/day单元、预测间隔与独立核对](evidence/geometry_analysis.json) · [四seed汇总](evidence/geometry_summary.json)。每包原始特征保留在N607，不作为新模型训练输入。

## 归一化作用与下一步

源响应与主干平均余弦约−0.71，而不是简单的同向重复。按单包精确分解中心化logit，主干项被||b||/||b+r||放大，另加响应在类别差异方向上的直接项。12个模型的平均放大系数为1.339至1.421，每模型至少99.974%的源包系数大于1，分解最大绝对误差小于9e−6。直接项非零，不能声称全部效果都是温度缩放，也不能把两项范数当作可相加的解释比例。

两个条件都预测错误的源样本，平均CE从约4.71至4.84升为约6.19至6.22。响应在分类差异子空间的能量占比仅约1%至1.4%，但通过公共归一化分母对分类置信度产生了明显影响。消除非判别分量本身也可能帮助抑制干扰，因此这些冻结观察只支持检验融合机制，不证明新约束必然改善性能。

已实现[两种约束融合原型](../../../docs/CVS_RESPONSE_FUSION_DESIGN_20261003.md)：类别差异子空间映射，以及主干独立归一化加有界响应。它们都保持237147个参数、原单一CE和G恒等初始函数。8项结构与梯度测试通过；尚未正式训练或测试，论文和性能目标仍未达成。

[精确归一化分解](evidence/norm_cancellation.json) · [机制结论](evidence/mechanism_verdict.json)。
