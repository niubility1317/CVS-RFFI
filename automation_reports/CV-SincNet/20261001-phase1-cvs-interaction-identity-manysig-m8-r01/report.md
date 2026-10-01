# CVS性能优先研发：时频交互

run_id：`20261001-phase1-cvs-interaction-identity-manysig-m8-r01`。状态LOCAL_VERIFIED，暂无新E200或clean结果。两个结构假设各4seed，全部从零，同划分/预算，唯一身份CE、无增强/域骨干。性能分数最高优先，参数及计算只作次要比较。

|候选|总参数/有效CE参数|Conv/Linear MAC/包|
|---|---:|---:|
|tf_lowrank32|129089|9672868|
|tf_bilinear|113825|9657476|

两个交互路径均零输出初始化并保留直接路径。rank32候选第一步down梯度0属预期，第二步全分支实测有效；乘积候选按统一逐样本嵌入公式学习。额外融合表达不是简单扩大特征提取器深度/宽度；不声称参数少或source好就自动提高target性能。CPU profile只证明计量可执行，正式GPU结果待测。

唯一来源为合法source全E200曲线和当前结构接口；旧target结果仅留在独立确认报告，训练/选模不消费。每行scratch200x50，source角色与已验证物理契约不变，无继承/教师/目标拟合。两候选完整E200后按0.5源V+0.5最差源RX最高选一个，成本只在性能完全并列后比较，不使用0.2pp成本容差。默认新确认run24行：selected4新预测＋原20冻结控制组只读复用，独立truth-last评分，所有负结果保留。

3项聚焦检查PASS，实际CE两步更新、所有分支与新交互梯度、单包推理、退化输入、scratch/query/noaug防护及更高性能但较大模型优先均核实。[本地验证](evidence/local_validation.json)、[登记](experiment.json)、[设计原理](../../../docs/CVS_INTERACTION_IDENTITY_RESEARCH_20261001.md)。独立P0/P1审查和远端发布证据随后补本run；独占新输出，不改旧所有产物，不干预健康进程。

独立P0/P1审查PASS，3项检查审查者独立运行全部通过，[审查证据](evidence/independent_review.json)。
