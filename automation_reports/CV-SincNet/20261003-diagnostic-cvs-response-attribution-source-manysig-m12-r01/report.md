# CVS显式补偿响应：完整源V冻结归因

状态VERIFIED。12个E200模型完成48条件，共1296000个预测决定；每模型始终使用同27000条源V，不能当作1296000个独立样本。全部物理ID、预测与每包标量重新核对，全开复现E200源V及最差RX。零训练、零目标访问，不改变源选择。

贡献=全开准确率−干预后准确率。auxiliary_off移除整个响应支路；g_identity把G出口置零，并重新计算支路，独立核实恒等归零；uniform_attention用均匀时间权重读出相同值向量；d_off只把D输入通道置零后通过原非线性值编码。它们不是重训消融，贡献不可相加。

|结构|条件|贡献均值±SD/百分点|正贡献seed|帮助|损害|预测变化|
|---|---|---:|---:|---:|---:|---:|
|response_mean|all_on|+0.00000±0.00000|0/4|0|0|0|
|response_mean|auxiliary_off|+0.05741±0.01971|4/4|140|78|315|
|response_mean|g_identity|+0.05741±0.01971|4/4|140|78|315|
|response_attention|all_on|+0.00000±0.00000|0/4|0|0|0|
|response_attention|auxiliary_off|+0.06296±0.05714|4/4|148|80|323|
|response_attention|g_identity|+0.06296±0.05714|4/4|148|80|323|
|response_attention|uniform_attention|+0.03426±0.02310|4/4|50|13|88|
|response_order_attention|all_on|+0.00000±0.00000|0/4|0|0|0|
|response_order_attention|auxiliary_off|+0.07130±0.04479|4/4|167|90|358|
|response_order_attention|g_identity|+0.07130±0.04479|4/4|167|90|358|
|response_order_attention|uniform_attention|+0.01944±0.02794|2/4|52|31|120|
|response_order_attention|d_off|+0.00926±0.00980|3/4|18|8|42|

标量先逐包计算再汇总。相对投影以同包原主干范数为分母；零参照采用已登记数值下限。

|结构|指标|四seed包均值的均值|
|---|---|---:|
|response_mean|attention_entropy|5.5053315|
|response_mean|auxiliary_off_logit_distance|5.6804003|
|response_mean|auxiliary_relative|0.48516246|
|response_mean|g_identity_logit_distance|5.6804003|
|response_mean|g_input_relative|0.10495924|
|response_mean|identity_auxiliary_max_abs|0|
|response_mean|response_feature_relative|0.12693345|
|response_mean|response_pooled_norm|1.2664651|
|response_mean|response_token_norm|7.730821|
|response_attention|attention_entropy|5.4547076|
|response_attention|auxiliary_off_logit_distance|5.8459046|
|response_attention|auxiliary_relative|0.49403622|
|response_attention|g_identity_logit_distance|5.8459046|
|response_attention|g_input_relative|0.11171081|
|response_attention|identity_auxiliary_max_abs|0|
|response_attention|response_feature_relative|0.14826035|
|response_attention|response_pooled_norm|1.3749128|
|response_attention|response_token_norm|9.1738564|
|response_attention|uniform_attention_logit_distance|0.88850842|
|response_attention|uniform_projection_difference_relative|0.078270956|
|response_order_attention|attention_entropy|5.4215748|
|response_order_attention|auxiliary_off_logit_distance|5.7431381|
|response_order_attention|auxiliary_relative|0.48784365|
|response_order_attention|d_feature_relative|0.022369542|
|response_order_attention|d_off_logit_distance|0.60941952|
|response_order_attention|d_off_projection_difference_relative|0.051926008|
|response_order_attention|g_identity_logit_distance|5.7431381|
|response_order_attention|g_input_relative|0.10717976|
|response_order_attention|identity_auxiliary_max_abs|0|
|response_order_attention|response_feature_relative|0.13331054|
|response_order_attention|response_pooled_norm|1.3337043|
|response_order_attention|response_token_norm|8.287522|
|response_order_attention|uniform_attention_logit_distance|1.2270202|
|response_order_attention|uniform_projection_difference_relative|0.10720046|

这是已见源RX上给定冻结权重的依赖证据，移除输入可能偏离训练分布；不能证明信道恢复、TX/RX分离、因果贡献或目标泛化。CE按完整源V记录；独立复算基于argmax，未把它宣称为CE重算。

[逐seed](evidence/per_seed_attribution.csv) · [全部标量](evidence/scalar_summary.csv) · [独立复算与RX混淆矩阵](evidence/independent_recount.json) · [终态](evidence/final_readback.json)。
