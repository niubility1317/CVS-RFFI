# 高权重普通 PL 与完整监督梯度保护

LOCAL_VERIFIED_NOT_LAUNCHED。49 项相关检查及一次独立 P0/P1 PASS；真实原 E20 两臂 no-query smoke 为 launcher 首个空闲 GPU 步骤，通过即重载父权重继续，不添加审批或 smoke token。

固定 HPL_E20_LS6G／HPL_E20_NAMS6G 各 seed2026092701/02/03，原同 seed F21 E20（13L＋7普通PL）独立 own native10＋20 到 E50／11100、13L＋37LU。只加载原 E20，来源／实际 payload／物理数据角色／默认头由原严格 source_parents 检查；不继承原 E30、新候选或按目标挑父权重。数据沿用合法 F0 VALIDATED_ONCE L6300／U56700／V27000；U TX 隐藏，V 不更新任何参数／归一化／EMA／原型，目标仅冻结测试。环境 N607 普通账户 CVS-RFFI、FP32／TF32off，每 worker CPU2；执行 CWD PROJECT/releases/20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01，命令与不可覆盖 run/output/config paths 见 experiment.json。实际代码配置以一个 Git commit 固定，单 archive 本地/远端一次 SHA 对照、远端 compile，独立 PID/CWD/完整命令/GPU/日志增长确认。

源依据：原 E20 完整 V27000/seed、0.95 门控接受后正确率99.9002%／99.8687%／99.9164%，覆盖约93%。预定8门控的无更新源诊断虽按“每 seed 总体≥99.5%”得到0.8，但新接受[0.8,0.95)样本仅约91%正确，故本候选仍0.95。24个固定原E20合法 L/U 源梯度探针（每seed8、全监督L六前向、U真值隐藏、无优化器更新）显示 weight2 U/L 范数中位数0.3178／0.3038／0.3565，14/24方向冲突；这里是裁剪/Adam前欧氏梯度，不是目标或实际优化步骤保证。候选据此固定权重6，避免只放宽门控或无保护放大冲突。详见 source_pseudo_quality.json、source_confidence_frontier_E20.json、source_gradient_diagnostic_E20.json；当前 NAMS 与 E20 新目标分数尚未读取，不从其调参。

原 L 全部六前向目标作为每步 anchor。U保持普通当前学生eval硬伪标签、0.95二值接受、全batch分母、无EMA／原型。LS臂保持一次raw U学生与U CE LS0.01；NAMS臂保持六U前向、ordinary/raw RDrop1、As KL0、LS0.01、浅中层跨源RX0.1、CE归一化1.25、U As exact raw及六前向共享accepted-only同伪类跨源RX统计混合（time_down/t2.pool，p0.5/Beta0.1/eps1e-6，先混合后捕获）。原L信道保持，无U信道、G训练、新推理参数或另训头，默认头随骨干原位联合训练。

将同一 U 标量目标乘6，分别计算完整L与U参数梯度；复用既有受测 protected_loss，按t2／cls_head／other三组（other含浅层time_down）仅投影与L负内积的U方向，再将U范数上限设为L范数，L平方范数≤1e-12的组U为0。显式autograd将完整L＋保护后U接回原生backward，原Adam、学习率、裁剪与decay保持；不误将CE重复计入。保证仅指裁剪/Adam前各组U·L≥0及||U||≤||L||的浮点容差，不承诺实际L损失或目标性能单调。无新增前向，但每步两次autograd求值与保留图可能增加计算和显存，必须报告实际父/增量/累计墙钟，不能称同FLOP或独占GPU小时。首卡真实smoke核实两机制实际参数梯度与显式期望相同、完整L anchor、冲突投影/范数帽、头和浅中层有限梯度、L768、U256/1或6U视图、原L信道64、Uphysical0、临时hook移除；探针丢弃。原E20配对诊断完全相同，核实来源并复用已完成源记录，不重复选门控。

每模型新增30轮/6660步与U1701000，累计U/teacher2097900；LS累计studentU2097900，NAMS10602900/rawAs850500/style39960/sharedplans6660。guard每新增步消费一次U loss，自己的10/20段分别2220/4440条记录与guard状态累加。固定最终学生，不按V微小变化或目标选模。两个相同技术故障停止新增派发，保留健康进程和全部产物；不得依据目标结果停止、重排、调参或选择性重跑。

GPU5/6/7分别固定种子，每卡最多两条本任务、含外部最多三进程、派发前空闲≥10000MiB。guardLS只在同seed旧E20LS的raw/DURA完整预测结束且旧E20NAMS已派发后启动，guardNAMS在旧E20NAMS完整GPU预测结束且新guardLS已派发后启动；旧两臂未CUDA启动PID与新children均预留计数。CPU唯一scorer排旧E20六行终态之后。不会改动旧队列或停止健康任务。

每row训练冻结后自动raw与固定DURA v1.0各七景、168000条/view，六模型共84 cell；所有该row决策冻结后独立truth-last。保留完整各seed/场景/TX/RX/day/RX×day、Macro-F1、raw/恢复配对修复损伤、U接受曝光与实际成本。固定F20/F21参照均完整复用，原raw验收clean同时≥+1pp、六景均值同时严格>+1.5pp；DURA只作独立参考，G不参与训练、其源U权限与三个F seed的SD范围如实保留。不套用旧F50的部署时延。当前历史测试暴露已披露，不称独立盲测；源诊断及数学约束不证明目标达标，goal ACTIVE。预期产物：真实smoke、源诊断复用、segment30/50全部6660条/模型训练记录与guard/style状态、final E50 checkpoint及继承、source/native检查、双七景固定predictions、独立scores及完整同row报告。


## 实际发布与等待状态

VERIFIED：执行 commit `03d563e7cc772f003270597584c587e3952397d4`，单archive传输对照与远端compile通过，远端controller实际PID/CWD/完整命令已独立确认，六行均等待预登记旧E20卡位，尚未占用新GPU或完成真实smoke。旧E20两机制六条GPU worker的PID/CWD/命令/GPU、全部已刷盘原生记录及日志增长同时核实；LS自己的10轮阶段已闭合并继续20轮，NAMS源机制实际执行，每卡两本任务／含外部最多三进程。旧任务保持；新smoke通过后自动独立续训，不能把排队声明为训练或目标成功。
