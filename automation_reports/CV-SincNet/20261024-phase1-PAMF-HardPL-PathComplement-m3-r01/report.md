# 主干互补伪标签监督

LOCAL_VERIFIED_NOT_LAUNCHED。66项相关检查及一次独立P0/P1PASS。真实原E20 source检查是首个释放且匹配seed/card的启动步骤，通过立即重载原父权重并继续。

固定HPL_E20_JOINT_PATH6G，seed2026092701/02/03，原同seed F21 E20独立own native10+20到E50/11100、13L37LU。沿用F0 VALIDATED_ONCE L6300/U56700/V27000；严格核实实际source角色/物理ID契约、scratch继承、cfg/payload及原默认头，不加载原E30或前一候选。U TX隐藏，源V仅无更新诊断，不读target训练/选模。N607普通账户CVS-RFFI、FP32/TF32off、CPU2/worker，CWD PROJECT/releases/20261024-phase1-PAMF-HardPL-PathComplement-m3-r01；完整命令、路径、环境、卡位、stop rule、预期产物见experiment.json。一个实际代码commit、一个archive一次传输比较、远端compile、独立PID/CWD/完整命令/GPU/日志增长确认，保留现有健康任务。

源依据：原Joint run已冻结A/B/C同源U960/V720、共同原E20伪标签/门控、实际部署分量与既有头干预。原E20完整acceptedU CE约.003-.004，去签名约.17-.22，主干单独约.29-.37；原Joint源joint160类间比例与RX散度改善，不能证明弱主干已充分学习。源U只比较共同伪标签，不是U实测正确率。诊断证据早于GuardNAMS目标结果；当前已知GuardNAMS及其他历史目标暴露如实保留，本候选来自此前源C更弱主干路径，并非新盲测或按target挑父。尚未读取Joint/Complement目标分数。

保留Joint6G的.95门控/weight6/LS.01、原监督六U前向ordinary/raw RDrop1、AsKL0、浅中层跨RX.1和actualjoint跨RX.05、六U共享accepted-only同伪类跨源RX style、无U物理增强，以及完整L梯度groupwiseprojection/normcap1、原L/默认头原位联合/nativeCENTER_UNITS AdamW/clip5/decay。

唯一新因素：原ordinary U CE系数.5分为full.25、no-signature.125、core-only.125；同accepted mask/全batch分母、前两ordinaryU pass平均、均LS.01，总CE分母1.25及其他CE/KL/对比系数保持。newU=oldJointU+6*(.125/1.25)*(noSignatureCE+coreCE-2*ordinaryCE)。noSignature为实际joint减同pass sequence_pool已缩放输出；core再减同pass response_projection输出乘response_gain.tanh()，均保留graph。复用既有classify_features，不拟合或另训头；完整签名和参考分量仍参与部署和原完整视图训练，不能称删除网络或物理解耦证明。

每步只增加4次既有头调用，原6U骨干不变。源smoke须证明coreCE对signature-only及response-only参数梯度为零，对真实主干和同一默认头梯度有限非零，完整joint辅助签名梯度仍非零；完整L guard回接参数梯度等于显式期望，L768/U骨干1536/额外headpacket1024/rawAs128/Uphysical0/LPractical64、临时hook/wrapper退出、CE剂量一致。探针丢弃，正式行重载原E20，不据诊断分数改参数。

每模型新增6660步、U1701000/学生U骨干10206000、额外既有head调用26640/packet-headviews6804000；累计U/teacher2097900、学生U骨干10602900。rawAs850500/style39960/sharedplan6660/jointcaptures13320保持；两段实际状态及父/增量/累计墙钟报告，额外head/backward并非免费，共享GPU墙钟不等独占小时。

GPU5/6/7固定seed2701/02/03，每卡最多2本任务、含外部最多3进程，派发前free>=10000MiB。只接替同seed Joint全部GPU预测完成的卡位，并要求同seed Complement已派发；两前序pendingCUDA PID均预留。首个释放且容量足够的匹配seed/card跑其原E20 smoke，smoke_runtime仅临时首row顺序，正式映射/种子/预算不变。CPU唯一scorer接Complement终态后。两个相同技术失败停止新增派发，保留健康任务/产物；无target驱动停训、调参、重排或选择性重跑。

固定最终E50学生，逐row自动raw和固定DURA v1.0各clean+六practical，168000/view，三seed共42cell；整row全部预测固定后独立truth-last。报告逐seed/景/TX/RX/day/RX×day、MacroF1、净纠错/损伤、U接受曝光与实际成本。复用同读出F20/F21，原raw目标clean同时>=+1pp、六景同时严格>+1.5pp，DURA不能替代。G固定R69/266951、FP32/TF32off/reuse+graph/nativeFalse，G原LU无TX标签/F-free约束保持，不参与新F训练，不套旧F50时延。goal ACTIVE。
