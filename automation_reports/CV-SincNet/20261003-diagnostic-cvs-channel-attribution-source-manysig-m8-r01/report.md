# CVS补偿与顺序差：完整源V冻结归因

状态VERIFIED。8个冻结E200模型完成44条件，共1188000个预测决定；每个模型始终使用同27000条源V，不能当作1188000个独立样本。全部预测、物理ID及每包标量已逐文件重新计算，全开复现原源V与最差RX。模型状态不变，零训练、零目标访问、不改变旧选择。

贡献=全开准确率−干预后准确率。正值代表该结构在当前冻结权重下帮助分类；不是从零重训的因果效应。u_off/v_off只移除各自投影，仍保留其他分支；g_identity将补偿设为恒等，保留共享F及原读出权重。

|模型|干预|贡献均值±SD/百分点|正贡献seed|全开帮助|全开损害|预测变化|
|---|---|---:|---:|---:|---:|---:|
|channel_dual|all_on|+0.00000±0.00000|0/4|0|0|0|
|channel_dual|auxiliary_off|+0.14537±0.04851|4/4|360|203|701|
|channel_dual|u_off|+0.04815±0.02512|4/4|153|101|319|
|channel_dual|v_off|+0.04722±0.02745|4/4|150|99|313|
|channel_dual|g_identity|+0.00185±0.00478|2/4|4|2|7|
|channel_order|all_on|+0.00000±0.00000|0/4|0|0|0|
|channel_order|auxiliary_off|+0.16574±0.06977|4/4|368|189|701|
|channel_order|u_off|+0.04259±0.04670|3/4|144|98|318|
|channel_order|v_off|+0.03519±0.04196|3/4|139|101|319|
|channel_order|g_identity|+0.00000±0.00000|0/4|1|1|5|
|channel_order|d_off|-0.00093±0.00185|0/4|0|1|2|

完整V标量为每包分别计算后汇总；下表对四模型的包均值取平均。投影相对量以同包原主干特征范数为分母；D时序相干比例是交叉项先平均后的模长和，除以先取模再平均的和，小值提示平均抵消，不等于身份信息丢失的证明。

|模型|指标|四模型均值|
|---|---|---:|
|channel_dual|auxiliary_relative|0.56120756|
|channel_dual|g_identity_logit_distance|0.033995657|
|channel_dual|g_input_relative|0.0089653745|
|channel_dual|u_projection_relative|0.28128596|
|channel_dual|uv_projection_cosine|0.99992968|
|channel_dual|uv_readout_cosine|0.99998243|
|channel_dual|uv_readout_relative_difference|0.0088408043|
|channel_dual|v_projection_relative|0.27993159|
|channel_order|auxiliary_relative|0.56075557|
|channel_order|d_cross_readout_norm|0.0034769817|
|channel_order|d_cross_temporal_coherence|0.62966394|
|channel_order|d_feature_relative|0.0018012177|
|channel_order|d_off_logit_distance|0.0065390099|
|channel_order|d_projection_relative|0.00016874912|
|channel_order|g_identity_logit_distance|0.049439105|
|channel_order|g_input_relative|0.012009991|
|channel_order|u_projection_relative|0.28114331|
|channel_order|uv_projection_cosine|0.99986761|
|channel_order|uv_readout_cosine|0.99996079|
|channel_order|uv_readout_relative_difference|0.011522483|
|channel_order|v_projection_relative|0.27963123|

这些干预只说明当前冻结网络在已见源RX上的依赖；移除分支可能偏离其训练分布，贡献不可相加。它们不能证明信道恢复、TX/RX分离、未见RX泛化或目标模块消融。D幅度、读出向量及投影分别测量，未使用额外损失或任何训练更新。

[逐seed贡献](evidence/per_seed_attribution.csv) · [全部标量](evidence/scalar_summary.csv) · [独立复算与逐RX混淆矩阵](evidence/independent_recount.json) · [完整终态](evidence/final_readback.json)。原始每包证据保留于N607登记路径。
