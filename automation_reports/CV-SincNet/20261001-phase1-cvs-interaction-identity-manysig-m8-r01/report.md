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

## N607源训练发布已核实

代码`601f6f6d10cb3d7aa3103931c7654faf2b272735`已push并独立核对远端OID。新release传输校验、远端compile与Torch2.1冷进程双候选CE反向检查通过。独立读回dispatcher PID819334的CWD/argv；8行各占一块GPU，进程与resolved/log增长已核实，状态SOURCE_TRAINING。实际6300L/56700Uunused/27000V、每轮50步、纯CE/无增强/域骨干关闭与目标访问关闭一致。发布后源码保持不可变，后续clean按另一个不可覆盖run发布。详见[发布后读回](evidence/source_launch_readback.json)。

## 完整源训练与冻结

8 行均完成 E200/10000 步，完整 1600 epoch、80000 step、CSV 和全部 stdout 已解析，无运行期技术异常；实际源数据角色一致，scratch/noaug/nodomain/CE-only 核实。按登记的性能优先规则冻结 `tf_lowrank32`，源选择独立复算完全一致。源性能完全并列后才考虑参数和计算，不使用测试反馈。确认 run `20261001-phase1-cvs-interaction-clean-manysig-m24-r01` 已预登记，默认完成 4 个新预测和 20 个冻结控制预测的独立评分。见[evidence/source_selection.json](evidence/source_selection.json)和[evidence/source_completion_validation.json](evidence/source_completion_validation.json)。

## 独立 clean 确认完成

选中 `tf_lowrank32` 由独立子 run `20261001-phase1-cvs-interaction-clean-manysig-m24-r01` 完成四 seed clean 测试，原 20 预测只读复用，共 24 行、192 条评分。准确率 78.0741% ± 1.8220%，较上轮残差 CVS -0.3802 个百分点（2/4 seed 提升）。本轮没有进一步提高平均准确率，参数更少不能替代性能目标。源 run 未读取 target，未选候选不测试，测试评分不回流调参或重选。见[完整确认报告](../20261001-phase1-cvs-interaction-clean-manysig-m24-r01/report.md)。


## 2026-10-03完整测试补齐

本批全部8个固定E200模型已纳入[全量clean测试报告](../20261003-phase1-cvs-all-frozen-clean-backfill-manysig-m304-r01/report.md)，每模型168,000个测试样本；此前未晋级且未测试的候选也已补测，已有预测复用。304行预测固定后统一独立truth-last评分，完整分类决定复算通过。测试结果见汇总报告和逐seed/RX/TX表；历史源域选择记录保持原样。
