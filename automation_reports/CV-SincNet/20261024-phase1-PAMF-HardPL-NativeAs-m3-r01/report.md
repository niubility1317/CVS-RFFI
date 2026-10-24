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
