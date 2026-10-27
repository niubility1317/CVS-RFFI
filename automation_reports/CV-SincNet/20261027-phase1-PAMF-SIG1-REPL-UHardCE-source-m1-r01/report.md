# REPL：真实L监督与U身份目标分开验证

已登记，尚未发布。固定一个模型seed2026092701，从原同契约单SIG_TIME普通PL scratchE20/4440独立+10+20，累计E50/11100＝13L37LU。原完整PAMF和原6×160默认头联合训练；不读取失败E50初始化、DURA、U真TX或目标。训练只使用合法源clean和直接practical residual/post_sync/noEQ衍生IQ。当前三版HQuality及所有其他健康任务保持原配方。

## 唯一变化

在已完成RB_RXTX语义加权／双直接H／关闭拒绝H互教的配方中，六个raw95接受U的CE都从LS0.01变为普通硬CE。保留真实L的全部原监督与LS0.01、raw95身份和接受集合、Uweight6/cap1、原视图剂量和CE分母、ordinary/raw RDrop、浅中层与joint跨RX约束。没有HQuality门控，没有原型替换头，没有额外前向/教师/推理参数。

计算使用严格恒等式：CE(LS0)−CE(LS0.01)=0.01×（六类logit均值−raw伪身份logit），按原ordinary/mixed/raw系数0.25/0.25/0.125与原1.25分母及同一语义权重加回。这条更正无需额外CE/softmax；原RX_TX重分配仍有其原6CE/3KL计算。实际日志明确区分nativebase LS0.01与effectiveU LS0，不能把配置声明当执行证据。5项聚焦检查含完整hardCE/RDrop scalar及梯度对照。

已有12模型冻结源诊断发现极高置信接受项中LS与硬CE的logit方向常不同，它不证明LS对实际参数或目标有害。当前HQuality首轮虽减轻裁剪，分类头冲突仍在，因此本实验只检验一个独立目标类型假设，不以已有目标分数挑checkpoint或改正在运行的方案。历史普通硬PL与此完整六前向机制不同，已完成控制与源读出复用，无重复监督/PL训练。

## 预算和验收

新增6660步、30轮；源F0 L6300/U56700/V27000及物理ID不变。真实E20联合L/U backward无query检查先行，实际六logit梯度对独立hardCE scalar误差≤2e−6后立即训练。固定finalE50后27000×raw/三H预测，全部固定才做源V评分和RX/day/TX分层。原源门槛保持raw≥26568/27000且H3≥62109/81000；无通过不扩种子或目标，通过也只是允许之后独立登记三种子验证，原clean+1pp/六景>1.5pp及绝对门槛未放宽。

普通账户N607、GPU0–7剩余容量，全owner CUDA+pending<2/free≥10000MiB；固定不可覆盖root、一次归档SHA/远端编译、PID/CWD/cmd/GPU/日志读回。首个继承/角色/finite/预算/执行技术错误停本run新派发，保留所有输出且不自动重跑；科学负结果保留。实际执行commit、环境/CWD、命令、逐行路径、所有seed和完整预期产物见experiment.json/runtime。


## 启动已核实

执行commit `9e2f4faee03025cb93c0653686ca6834b62e6908`，归档传输VERIFIED、远端编译PASS。真实E20联合L/U backward检查PASS：L768/U256/六U前向1536，实际六logit梯度与完整hardCE scalar检查误差4.656612873077393e-10。随后GPU4训练启动，2026-10-27 00:49:36 UTC 独立核实PID/CWD/模块/训练参数/实际GPU与步骤日志增长。步骤日志明确记录nativebase LS0.01、effectiveU LS0及六前向计数，变化已进入实际优化；它不是源或目标性能证据。

当前一个学生源探索，原三版HQuality继续各自健康训练，无目标访问或三种子扩展。详情见[启动证据](launch_evidence.json)。


## 后续正式确认的统一源选择范围

2026-10-27 00:55:01 UTC 已只读核实四版均未完成E50冻结或最终源评分。在目标访问前固定未来确认规则：等待三版HQuality和本UHardCE共四版全部完成；各自原raw≥26568/H3≥62109门槛保持。合格者按raw正确数、H3正确数降序选择，仅完全同分按UHardCE、Fixed、Soft、Gate固定顺序。优先简单目标／常数规则不等于实测时延优势。唯一维护的结构化范围在experiment.json的future_confirmation_source_cohort。

本规则仅决定之后另登记的三种子正式确认，当前健康训练、预算及每run原source_selected生成均不改。若无通过则不登记确认、不访问目标；若有，通过源选择的同一方法复用seed01，仅补自有原E20的seed02/03，六个监督/PL控制全部复用。源通过不是原目标通过，也不据之后目标表现改选另一个源候选、替换seed或选择性重跑。


## E30完整首段核验

新增首10轮／2220条更新和实际E30 checkpoint已完整独立核验：保持单SIG_TIME、原6×160默认头、同seed原E20父模型及finite状态。每一步effectiveU LS0、nativebase LS0.01、原六U前向和L768/步均一致；累计U曝光567000次，默认头继续联合训练，未另拟合头。没有新增模型前向、V预测、U TX或目标读取，最后20轮继续原配方。

分类头的L/U局部冲突比例43.5586%，裁剪比例91.2613%，保留U梯度范数中位比例44.4478%。当前结果证明目标变化已进入全部首段更新，但没有证明取消U标签平滑能降低整体无监督压力或提升目标泛化；其他L/RDrop/跨RX/强H剂量均保持，仍须完成E50冻结评分。

证据：[完整首段与实际权重核验](closed_E30_evidence.json)、[梯度汇总](closed_E30_gradient_summary.csv)。最终分析直接复用本段2220条已验证记录及CP核验，只新增最后4440条和冻结源预测检查。四候选统一源选择规则及原目标门槛保持。
