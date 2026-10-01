# CVS 性能优先研发：包内注意力统计池化

run_id：`20261001-phase1-cvs-attentive-identity-manysig-m8-r01`。状态 LOCAL_VERIFIED，尚无新 E200 或 clean 结果。两种池化×四 seed，均从零训练，同划分/预算、唯一 CE、无增强/域骨干。源性能最高优先，资源成本次要。

|候选|总参数/有效 CE 参数|Conv/Linear MAC/包|
|---|---:|---:|
|attentive_mean|164417|9716260|
|attentive_moments|164609|9716260|

注意力与波动增益初始化 0，整个模型初始输出与相同随机种子的原残差 CVS 一致。仅改变包内池化，不扩大卷积深度/通道、融合或分类头。评分为包内有界 softmax，原特征值用于加权均值；波动候选合成回原通道数。3 项聚焦检查通过，包括完整模型初始退化、真实 CE 梯度、权重正性/归一化/有界比值、单包独立推理、退化输入和 scratch 防护。CPU profile 仅验证计量，正式 GPU 数据待测。

按 CVS 实验执行与实验登记技能完成本轮矩阵；全部 source E200 后冻结一个候选，默认新确认 run 24 行：4 新预测＋20 冻结控制，只读相同物理 query，独立评分。未选候选不测试，target 不回流研发/源选择/重训。保留原所有产物和负结果；不干预健康进程。见[本地验证](evidence/local_validation.json)、[登记](experiment.json)与[数学/通信/物理/RFFI 分析](../../../docs/CVS_ATTENTIVE_IDENTITY_RESEARCH_20261001.md)。独立 P0/P1 审查与发布后证据随后补充。

独立P0/P1审查PASS，3项检查审查者独立运行全部通过，[审查证据](evidence/independent_review.json)。
