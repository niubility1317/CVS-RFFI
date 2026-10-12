# Phase1 当前目标与测试范围

用户要求继续实现两份优化报告并达到 clean 三模型种子均值至少84%。最新明确覆盖：每个新模型从头训练30轮；所有分类头使用单一常规cos，供Phase2替换和重新拟合；不用源V性能决定候选是否测试；只测试clean，不追加六practical。源V只健康监控，不选epoch或构成性能晋级门槛。

固定model seeds为2026092701/02/03。保留当前ManySig物理数据角色L/U/V=7/63/30、训练池标签率10%、source RX1/3/4/6/8、source days1/2/3、全部6TX。U TX真值保持隐藏。身份不输入显式CFO标量；纯域泛化训练、测试不拟合query。预测覆盖全部168000 clean物理ID及全部6类，先固定prediction，再由独立scorer连接truth。数据为地面代理且测试基准已暴露，不声称新的独立盲测或在轨验证。

所有候选配方/矩阵/种子/预算在其测试前固定，测试结果仅如实验证实际性能；不按目标错例设置TX专属分支、挑seed、改超参、选epoch或选择性重跑。最终逐seed/RX/day/TX/混淆、Macro-F1、均值/SD与对照差均保留，不能把源成绩或不完整三seed当84%完成。

已完成的源端元反馈/SWA先导保留为历史诊断；旧source资格不再阻止其后续全源三seedclean验证。全源实现按已经披露的五RX扩展：元反馈CE_MATCHED/UNIFORM_SSL/META_SSL共9个新30轮模型；SWA三条30轮轨迹，固定FINAL/SWA两读出。逐row冻结后立即自动clean预测及独立评分。

新报告机制按[追踪表](../analysis/phase1_receiver_invariance_report_20261013_traceability.md)实现：先用已核实的完整源CE30三seed做零梯度判别投影/常规cos方向对照，再分别从头30实现8种物理/局部机制。未完成统计模型或没有物理前提的理论不能改名当损失，数值保护、退化和近似边界实报；不以源V性能筛掉模型。新增机制分开消融，全部只测clean。

每GPU最多3个总实验进程，计入其他owner；不热改、停止或重启健康旧任务。复用VALIDATED_ONCE数据、严格checkpoint实际物理角色/祖先检查、唯一owner及不可覆盖run/release/output、Git代码/config身份、一次最小独立P0/P1、必要负测与真实checkpoint noquerysmoke、单archive比对和compile、postlaunch PID/CWD/argv/GPU/loggrowth。沿用现有最小workflow，不增加签名/hash/审批层。
