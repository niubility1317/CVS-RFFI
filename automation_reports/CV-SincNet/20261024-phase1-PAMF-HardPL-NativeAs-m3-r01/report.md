# 高权重伪标签的完整原生信道视图对照

LOCAL_VERIFIED_NOT_LAUNCHED：119项相关检查及一次独立P0/P1 PASS；首个匹配释放卡位的真实原E20检查尚PENDING，通过即丢弃探针并重载原权重继续。

三seed2026092701/02/03，各从原同seed F21E20/4440独立执行native10/2220＋20/4440，累计E50/11100＝13L＋37LU，不加载原E30或候选权重。数据复用已有VALIDATED_ONCE F0物理契约L6300/U56700/V27000，实际父配置/payload/角色/物理ID与scratch继承核实。U TX真值隐藏；V仅诊断，无模型/归一化/EMA/原型/头更新。N607普通账户/CVS-RFFI、FP32/TF32off、每worker CPU2；命令、CWD、不可覆盖路径、GPU、停止规则和产物见experiment.json。

固定源依据为原raw PL高精度、原生As视图诊断及Joint完整源记录：U梯度范数平均保留83.24%、生效约L的0.79倍（clip5/AdamW之前）。增强视图当前分类错误不等于原raw伪标签错误。仿真沿用监督L的TX标签保持约定，但不保证每个深衰落实例信息充足、实际U准确率、目标信道真实性或目标收益。源设计7d88684已在RAS/组合/RXE目标分数读取前固定；Joint/Complement/Path及此前目标曝光如实保留，不作新的独立盲测或目标选模。

相对Joint6G仅把原U mixed半批exactraw换成原生sourcePractical As，沿用原交替分区和同监督L生成器；不加增强视图置信/argmax门控或raw回退。原raw argmax伪类、置信度0.95接受集合以及六次学生前向保持，全部接受样本继续参与。保持权重6、LS0.01、ordinary/raw RDrop1、AsKL0、浅中层跨RX0.1/joint160跨RX0.05、六前向共享同伪类跨源RX style、总CE分母1.25、原L/defaulthead联合、完整L guard一次、nativeAdamW/clip5/decay；没有新增损失、teacher、学生前向、头或推理参数。

每模型新6660步/1701000 U曝光，生成As850500、物理学生视图1701000；实际有效物理CE视图为2×rawacceptedhalf，生成/前向成本仍包含被原raw gate拒绝的样本，不能把视图计数等同有效标签数。累计rawteacher/总teacher均2097900，学生U新增10206000/累计10602900，rawAs0；无额外As evalteacher/既有头调用、EMA/原型/Utruth/G训练。逐seed分别记录父/增量/累计时间、接受率与有效视图，不称相同FLOP或独占GPU小时。

GPU5/6/7固定对应三个seed，每卡最多两个本任务、计外部compute进程<3且free≥10000MiB；等待同seed PathReliableAs全部GPU预测结束且ReceiverExtrap已派发，预留两侧pendingCUDA PID。首个匹配释放卡做真实原E20 smoke：L768/U1536/rawteacher256，As生成128/物理学生256/rawAs0，sourcePractical为L64+U128=192；原生As输入/非物理普通输入一致、原接受集合、CE1.25、joint签名梯度、完整L显式参数梯度相同及临时hook退出。探针丢弃，正式重新加载原E20。唯一CPU scorer接RXE终态。两个同类系统性技术错误只停止新派发，保留健康任务和产物；不按目标分数停训、调参、重排或选择性重跑。

最终E50默认学生逐row自动测试raw和固定DURA各clean＋六practical，168000/view，三seed42cell；整row决策固定后独立truth-last。完整逐seed/景/SD/MacroF1/TX/RX/day/RX×day/净纠错/成本，复用F20/F21同读出；固定Joint对照检验添加原生As，固定RAS对照检验增强视图门控/回退及额外teacher成本的整体作用，两者均保留，不称等FLOP门控单因素。raw clean相对两个控制均≥+1个百分点、六景均严格>+1.5个百分点才达到原目标，DURA单列，不能代替raw。G固定DURA v1.0 R69/266951、FP32/TF32off/reuse=True/graph=True/native=False，不用于新F训练、不套用旧F50时延或替换默认部署。目标保持ACTIVE。


## 发布后独立核实

VERIFIED：执行commit `f605efc3fc35617bb007ae25ec6b4822360b95c1` 的控制器存活，CWD及完整命令匹配。三行实际QUEUED，GPU训练worker为0，真实原E20检查PENDING，等待对应seed组合版全部GPU预测完成且RXE已派发。独立核实组合版三个健康worker的PID/CWD/命令/GPU及日志增长，RXE控制器存活且仍三行排队。每卡本任务最多两个，保留全部现有任务；排队不代表训练或目标成功。


真实原 E20 检查及首批启动已独立核实：VERIFIED，执行 commit `f605efc3fc35617bb007ae25ec6b4822360b95c1`，检查 seed 2026092702，L768/U1536/rawteacher256/As128/物理学生256/Practical192，完整 L 显式梯度、joint 签名梯度和临时 hook 清除通过。探针丢弃，正式重新加载原 E20。目前实际 1 个正式 GPU worker 已有正源步骤，并核实 PID/CWD/命令/GPU/日志增长；其余仍排队。完整事实见 first_source_start.json；首个启动不能当作三种子全部启动或测试成功。


原生As真实检查：固定半批生成128，两个原mixed学生实际使用256条物理视图，rawAs回退0；原rawteacher256，无增强视图evalteacher或额外学生前向，原接受mask和CE1.25保持。原生As输入及非物理ordinary输入匹配，生成成本与有效As CE视图2×rawacceptedhalf分别记录；没有用排队或预期计数代替实际训练证据。


## 真实原 E20 检查通过并实际启动三种子

VERIFIED：发布commit `f605efc3fc35617bb007ae25ec6b4822360b95c1` 的真实原E20 GPU source smoke PASS。实际joint160两次捕获512条已有Upacket，跨RX有效anchors=478；实际签名sequence_pool辅助梯度范数=0.370686234且有限非零，原L768／U1536／rawAs0／U As生成128／物理学生256／LPractical64＋U128、共享style计划、完整L guard参数梯度与显式期望相同及临时wrapper退出均符合。探针丢弃，正式行重新加载原E20。

三个seed已按GPU5/6/7启动并实际产生原生joint／guard源步骤；独立核实进程／CWD／完整命令／GPU及日志增长，逐行现有U复用计数见source_activation.json。只按原矩阵接替释放卡位；当前是训练前缀，不代表全部源训练结束或目标收益。训练、raw＋固定DURA七景及truth-last继续自动执行，goal ACTIVE。


## 完整源训练

VERIFIED_SOURCE_FINAL：每个模型严格从原同 seed E20 开始，原生10＋20段各2220／4440步，总计新增30轮／6660步，累计E50／11100；全部19980条闭合记录逐条核对。门控0.95、权重6、U LS0.01、原L／默认头联合训练保持，新增U原生As生成850500／物理学生1701000，无EMA、原型、U真值或G训练。NAMS累计学生U10602900，每model仅在已有U中joint捕获13320批／3402000条、6660次对比。新增原生As850500／物理学生1701000／rawAs0、style39960次／6660共享计划、RDrop／AsKL0／浅中层跨RX及CE归一化1.25均符合预登记。每模型实际6660次完整L anchor的U投影／范数帽1及逐组冲突／帽计数全部核实。父E20／新增30轮／累计经过时间分别保存，不把共享GPU墙钟视为独占小时。

固定最终学生自动完成raw与固定DURA各七景、独立truth-last，源证据不替代目标收益；已经完成的整组三seed先行报告不改变冻结的其余训练，本次源检查不读取目标分数。

原生As版实际有效物理CE学生视图为2×rawacceptedhalf，完整三seed计数在actual_adapter_SSL/U_native_As，生成和学生成本仍包含原raw拒绝样本；无增强视图门控/回退/额外As教师或新增损失，CE1.25、原raw接受集合保持。


## HPL_E20_JOINT_AS6G 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

| seed | E20 接受精度（%） | E50 接受精度（%） | E20 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.900216 | 99.907132 | 92.792593 | 95.714815 | 24 |
| 2026092702 | 99.868683 | 99.922279 | 93.074074 | 95.307407 | 20 |
| 2026092703 | 99.916428 | 99.953413 | 93.066667 | 95.400000 | 12 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL_E20_JOINT_AS6G.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。
