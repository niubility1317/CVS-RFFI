# 剂量匹配的互补伪标签监督

LOCAL_VERIFIED_NOT_LAUNCHED。57项相关检查及一次独立P0/P1PASS；真实原E20检查为首个释放且匹配seed/card的source-only步骤，通过即重载原父权重继续，不新增审批。

固定HPL_E20_JOINT_COMP6G，seed2026092701/02/03，原同seed F21 E20独立own native10+20到E50/11100、13L37LU。只用原E20，严格核实实际source角色/物理ID契约、scratch继承、cfg/payload/默认头；不加载原E30或前一候选权重。沿用F0 VALIDATED_ONCE L6300/U56700/V27000，U TX隐藏，V无参数/归一化/EMA/原型/头更新，不读target训练。N607普通账户CVS-RFFI、FP32/TF32off、CPU2/worker，CWD PROJECT/releases/20261024-phase1-PAMF-HardPL-Complement-m3-r01；完整指令、路径、GPU、stop rule及预期产物在experiment.json。一个实际代码commit、一个archive一次传输比较、远端compile、独立PID/CWD/完整命令/GPU/日志增长确认，现有健康任务保持。

源依据和固定设计见原Joint run的source_geometry_trajectory_A/B/C及complementary_U_followup_design.json。固定同源U960/V720及原E20共同伪标签/门控、只读CPU诊断显示浅中层改善不一定传到部署joint；Joint直接对比使源融合类间比例约0.771到0.804、RX/总散度约0.0859到0.0663。无签名视图仍来自同IQ/完整高可信伪标签，但原E20既有头U一致率约94-96%、CE.17-.22，完整视图仅.003-.004；参考响应交换则CE仍低。这里是机制诊断，不是U真实正确率、因果去RX或target成绩；历史测试暴露如实保留。本设计在读取GuardNAMS6G/Joint新target分数前固定。

保持Joint6G的.95门控/weight6/LS.01、六U共享同伪类跨源RX style、无U物理信道、ordinary/raw RDrop1、AsKL0、浅中层CRIC.1、actualjoint CRIC.05、完整L梯度guard/native CENTER_UNITS AdamW/物理clip5/decay和默认头原位联合训练。唯一新因素是将ordinaryU CE系数.5中的.25转给同pass实际joint160减sequence_pool实际输出的CE，普通CE系数变.25、新maskedCE系数.25，总CE分母1.25和全部原KL/对比系数保持：newU=oldJointU+6*(.25/1.25)*(maskedCE-ordinaryCE)。两项CE均LS.01、同accepted mask/全batch分母、前两ordinaryU pass平均。保留完整签名部署和原完整视图训练，不能称删除签名网络。

复用原两ordinaryU已有graph，只增加两次既有classify_features调用，没有新增骨干/teacher前向或新头。临时同pass sequence_pool hook准确退出，h-s的maskedCE对signature-only参数梯度为零、对core/reference及同一默认W有效；joint辅助仍训练signature。整个新U目标仅由完整当前L anchor消费一次，分组投影和范数帽1保持，不重复L/CE梯度，不承诺Adam步/L损失/target单调。首个真实E20 smoke核实这组梯度、实际保护后参数梯度等于显式期望、L768/U1536/额外headpacket512/rawAs128/Uphysical0/LPractical64、共享style/原joint和临时hook退出；探针丢弃后正式重载同seed原E20。

每模型新增6660步/U1701000/学生U骨干10206000，新增既有head调用13320/packet-headviews3402000；累计U/teacher2097900、学生U骨干10602900，额外head视图另计。rawAs850500/style39960/sharedplan6660/jointcapture13320保持。两段实际状态和父/增量/累计墙钟必须报告，额外head/backward并非免费或相同FLOP；共享GPU墙钟不等独占小时。

GPU5/6/7固定seed2701/02/03，每卡最多2本任务/含外部最多3进程/派发前空闲>=10000MiB。新row只接替同seed GuardNAMS全部GPU预测完成的卡位，并要求对应Joint已派发；两侧CUDA前pending PID均预留。首个释放且容量足够的seed/card先用其自己的原E20 smoke，临时smoke_runtime首row对齐该seed，正式三row映射不变。CPU唯一scorer排Joint终态后。两个相同技术失败停止新增派发、保留健康任务和产物，不按target停训、调参、重排或选择性重跑。

固定最终E50学生，逐row自动raw和固定DURA v1.0各clean+六practical、168000/view，三seed共42cell；整row预测固定后独立truth-last。报告每seed/场景/TX/RX/day/RX×day、MacroF1、净纠错/损伤、U接受曝光及实际成本。复用完整同读出F20/F21；原raw目标clean同时>=+1pp、六景同时严格>+1.5pp，DURA参考不能替代。G固定R69/266951、FP32/TF32off/reuse+graph/nativeFalse，G原LU无TX标签/F-free权限保持，不参与新F训练、不套旧F50时延。goal ACTIVE。


## 真实权重检查与三种子实际启动

VERIFIED：执行 commit `7d58c705b65dfd0afa02ee8b7cb079e4e1b8468a`。先释放的 seed 2026092702／GPU6 完成真实原 E20 source smoke PASS；正式 seed／卡映射保持不变。无签名 CE 的实际梯度范数：签名 0、既有头 0.829442375、主干 2.38287784；完整 joint 对比的签名梯度 0.370686237。普通 CE 0.5 分配为完整融合 0.25＋无签名 0.25，总剂量1.25保持；两次既有头调用512条已有特征，U骨干仍1536条，原L768／rawAs128／Uphysical0／LPractical64、完整L guard与显式梯度一致，临时hook／wrapper退出。探针更新丢弃，正式训练重新加载原E20。

三个种子已分别在GPU5/6/7产生实际互补CE／joint／style／guard步骤。进程、CWD、完整命令、GPU与日志增长独立核实，计数见source_activation.json。当前仅是训练前缀；尚无本版目标测试收益，raw＋固定DURA七景及独立truth-last自动续行。前序健康任务保持。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，无U物理生成、EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。新增rawAs850500、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽1及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。

互补CE每模型13320次既有头前向、3402000条已有特征；普通CE系数0.5改为0.25＋无签名CE0.25，总CE归一化1.25和KL保持。无额外骨干前向、新头或推理参数；完整签名仍参与部署与原joint/L损失。


## HPL_E20_JOINT_COMP6G 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.861373 | 92.792593 | 96.181481 | 36 |
| 2026092702 | 99.868683 | 99.854155 | 93.074074 | 96.500000 | 38 |
| 2026092703 | 99.916428 | 99.923003 | 93.066667 | 96.203704 | 20 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_JOINT_COMP6G.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。


## 完整目标测试结果

ANALYZED：各方法三个seed，共3模型、42项raw／固定DURA七景决策均完成独立truth-last评分，六个匹配监督／普通PL对照模型完整复用。逐景／seed／SD／Macro-F1／RX／TX／day、修复损伤及曝光／成本见target_results.md和附属完整表。原raw双门槛尚未满足，研发目标仍ACTIVE。

HPL_E20_JOINT_COMP6G / raw clean：84.958928571% ± 0.050328428%（三个模型 seed 样本 SD）。
HPL_E20_JOINT_COMP6G / DURA-LU clean：85.040079365% ± 0.056180933%（三个模型 seed 样本 SD）。
HPL_E20_JOINT_COMP6G / raw six_mean：70.252777778% ± 0.847748085%（三个模型 seed 样本 SD）。
HPL_E20_JOINT_COMP6G / DURA-LU six_mean：76.446362434% ± 0.500775455%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。


HPL_E20_JOINT_COMP6G 固定匹配比较完成，见 HPL_E20_JOINT_COMP6G_matched_comparison.md；比较不替代原主目标或全六行完成状态，无目标反馈改配方。
