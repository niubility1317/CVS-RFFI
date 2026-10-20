# R48：35.05万保留优化与27.38万进一步压缩的正式结果

性能优先：retain_refine七景提高1景、下降6景；prune_refine七景提高7景、下降0景。逐景1pp仅是退化上限，不把通过容差、减参或训练提速自动表述为性能提升；逐场景数据优先于均值。指定R46 prune_refine350521继续为唯一直接基准，已揭示正式结果不用于另选父或选择性重跑。

状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

用户在R46结果揭示后指定350521父，exact retain_refine为同预算训练优化控制，273765 prune_refine为GN4重要性进一步剪枝；同原接收链/3prior/6CG，无其他恢复网络。两行均pairedclean IQ+.05三尺度相对功率log1p误差+1→.1冻结父三stageIQ教师，所有coarse/action/prior联合。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。无TX标签模拟配对源域自监督，不称无clean盲学习。

方法commit `86b89ba6bc09b0b3f10fcaff8fc7cd1ecf6d0508`。完整登记、数据与祖先、命令和停止规则见[登记](../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-compact-refine-s2026092701-m2-r48/experiment.json)；全部20轮与TX/RX/day见[结果](../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-compact-refine-s2026092701-m2-r48/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-compact-refine-s2026092701-m2-r48/row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|retain_refine|350521|0.000000%|True|True|66.852976|
|prune_refine|273765|21.897689%|True|True|68.257738|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|retain_refine|clean|84.505357|84.212093|+0.048810|-0.000595|-0.081548|-287|
|retain_refine|practical_high|81.813690|81.288963|-0.638690|-0.604762|+0.494643|+69016|
|retain_refine|practical_mid|80.352381|79.858340|-0.869643|-0.657738|+0.774405|+67932|
|retain_refine|practical_low_suburban|78.857143|78.440328|-0.763095|-0.519048|+0.860714|+66910|
|retain_refine|practical_high_urban|80.707143|80.235571|-0.679762|-0.610119|+0.353571|+68895|
|retain_refine|practical_mid_urban|73.701190|73.556200|-0.745833|-0.525595|+0.016071|+65886|
|retain_refine|practical_low_urban|66.852976|67.122362|-0.657143|+0.080357|-0.363690|+61410|
|prune_refine|clean|84.536310|84.249356|+0.079762|+0.030357|-0.050595|-235|
|prune_refine|practical_high|82.645238|82.236253|+0.192857|+0.226786|+1.326190|+70413|
|prune_refine|practical_mid|81.509524|81.114133|+0.287500|+0.499405|+1.931548|+69876|
|prune_refine|practical_low_suburban|80.002976|79.665539|+0.382738|+0.626786|+2.006548|+68835|
|prune_refine|practical_high_urban|81.564286|81.178715|+0.177381|+0.247024|+1.210714|+70335|
|prune_refine|practical_mid_urban|75.042262|74.871308|+0.595238|+0.815476|+1.357143|+68139|
|prune_refine|practical_low_urban|68.257738|68.419563|+0.747619|+1.485119|+1.041071|+63770|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|retain_refine|136890/101037/112594|1685404|39081004|
|prune_refine|106138/79477/88150|1376476|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|retain_refine|G_only_B1|44.5498/44.5913|45.8523|22.4793|67017728/67017728|
|retain_refine|G_only_B32|57.7966/57.9484|60.1130|549.9111|276216832/276216832|
|retain_refine|G_plus_F_B1|55.1817/55.6139|57.8439|17.9535|67017728/67017728|
|retain_refine|G_plus_F_B32|68.5971/68.0839|70.4111|468.6024|276216832/276216832|
|prune_refine|G_only_B1|41.4050/40.9928|42.5749|24.4258|67017728/1084928|
|prune_refine|G_only_B32|56.9253/51.8359|52.2846|619.6688|276216832/227691520|
|prune_refine|G_plus_F_B1|52.2722/51.4586|53.1965|19.3880|67017728/1201152|
|prune_refine|G_plus_F_B32|65.7682/60.2725|61.1218|530.7903|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。

retain_refine首轮→末轮：loss 0.025517709→0.021891554；iq_nmse 0.024150966→0.021491357；spectral_error 0.0063670598→0.0056597645；teacher_stage_nmse 0.0010483899→0.0011720886；中位每轮174.026s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_refine首轮→末轮：loss 0.13370983→0.023518632；iq_nmse 0.079269997→0.022922496；spectral_error 0.019420517→0.0060034009；teacher_stage_nmse 0.053468811→0.0029596517；中位每轮162.135s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。优化控制和剪枝行同预算、同损失配方，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_compact_refine_r48_20261020/models/<arm>/final.pt；使用experiments.pamf_r46_compact_refine.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。


## 固定方案和历史启动记录

以下为结果揭示前的方法与启动记录，当前训练和正式测试已完成。


状态：**VERIFIED / RUNNING**。

用户明确指定完整 350521 参数的 R46 `prune_refine` 作为后续直接基准，并要求同时关注性能、压缩、时延等实际成本。本轮只在同一接收链恢复器内继续优化和 GN4 通道裁剪，保留父权重、全部历史结果和负结果。用户在七场景结果揭示后指定该父，因此属于暴露工程基准研发，不称新的独立盲测或干净泛化证据。

|固定行|完整参数|context 参数|prior 唯一参数|初始化|
|---|---:|---:|---:|---|
|retain_refine|350521|237927|112594|直接父的精确函数复制，同宽优化对照|
|prune_refine|273765|185615|88150|同一父按每个 GN4 组内的权重重要性进一步裁剪|

新裁剪版相对指定父减参 21.8977%。主分支宽度由 `(12,20,32,44)` 改为 `(12,16,28,40)`，频谱分支由 `(8,12,20,32)` 改为 `(8,12,16,28)`；浅层主宽 12 不变。encoder/decode/skip/GN、action 全局头及三步共享长卷积权重的索引保持一致。三步先验和六次 CG 仍实际执行，因此参数减少不预设同比加速。

两行直接初始化和冻结 IQ 教师均为 `20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46/prune_refine` 的固定源最终 E20/4940，方法 commit `bbb8daf817de050b958d217b70d4c33f4aefe341`。核对 launch、固定方案、完整源训练记录和严格 checkpoint 契约；继承 R44/R38/R17/R07/R04 原 7/63/30 物理源划分，复用 READY/EXACT_MATCH 缓存。无 target 样本参与恢复器训练；历史结果暴露与人工父指定的限制如上披露，不能据新分数再选择性重跑。

两行均固定新增 20 epochs/4940 更新，原 L6300/U56700 和全部 20 轮 received 缓存，L126000/U1134000 次曝光。全部 coarse/action/prior 联合优化，重置 AdamW，LR2e-4，warmup247，cosine 至 2e-5，weight decay1e-4，clip1，batch256/micro64，末批24用真实分母；FP32、TF32关闭。每个组件继承80轮，累计100轮，组件轮数和300；串行训练成本由120增至140轮，不把追加成本隐去。

目标与 R46 相同：最终 paired-clean IQ NMSE，权重0.05的64/128/256多尺度 log1p相对功率频谱误差，以及三阶段 IQ 教师误差的线性1→0.1约束。复数IQ、周期Hann、hop半窗、centerFalse、正交FFT，clean频谱均功率尺度下限1e-4且detach。恢复器训练读取0 TX标签、F权重/输出、V或target；教师和clean不接收梯度。正确称为“TX无标签的模拟配对源自监督”，clean配对实际参与，不称无clean盲恢复。

两行固定最终checkpoint，不源域早停或跨行筛选；每行自动执行源1800×7预测、推理成本测量、独立源评分、正式168000×7预测、独立truth-last评分。正式视图为clean、郊区high/mid/low、urban high/mid/low；每行七视图全部固定后scorer才连接truth。原F50/defaulthead冻结，FFT/PA/time/stats由其原流程计算，不拟合或替换分类头。源域没有提升也不取消正式测试。

比较相对原R17和指定350521父逐场景1个百分点容差（整数正确数最多少1680，含等号）；历史R44描述对照保留。单列lowurban≥70%、其他五个受损场景距rawclean84.6761904762%≤5个百分点的原目标。报告accuracy/Macro-F1、逐TX/RX/day、修复/损伤和波形/FFT；单seed/相关视图限制及TX真实性证据边界保留，参数下降不等于目标完成。

成本使用同一卡、同一源V lowurban的64条IQ，分别测G和G＋原冻结F完整链。batch1每轮预热16次、64次计时、重复3轮；batch32每轮预热4次、16次计时、重复3轮，前向/反向候选顺序交替。报告同步wall-clock median/p90/p95、各轮中位数、吞吐量、驻留及峰值/增量显存，以及完整参数/序列化大小和Conv1d＋Linear MAC。预加载GPU输入的时间不含传输、预处理或数据加载；驻留显存是三G＋一F共享suite，增量峰值对应当前前向，不能称独立部署显存。MAC不含FFT、复数CG、归一化、激活、访存或F。报告恢复器和G＋F两种口径，参数、MAC和实际时延分别解释，不预设加速倍数。

物理GPU0/1各最多一条本任务训练/预测worker，总数最多2；先确认R46对应实际PID退出，再按共享容量free≥18GB和每卡总进程<2获取。CPU2线程、最多一个本任务scorer，共用R41 score.lock；R47完成诊断不构成新启动门槛。普通账户、新不可覆盖run/output/release root、一次归档SHA/远端compile；真实父/F50/2条Ls配对IQ的no-query smoke作为launcher首步，PASS后立即训练。健康其他owner不动，技术失败保留产物且不自动重试；任何新结果不热改健康行。

最小聚焦检查覆盖来源/预算/权限拒绝、共享别名冲突、真值必须晚于全部预测、所有固定行自动正式测试、实际父句柄退出、有限梯度及精确保留对照；一次独立审查只限直接影响下一真实run的P0/P1。精确矩阵、命令、环境/CWD、路径和预期产物写入本run登记，不增加审批或其他gate。


VERIFIED/RUNNING：实际350521父/F50/两Ls配对IQ noquery smoke PASS，两行全部组件有限非零梯度、teacher/F不更新、checkpoint严格重载一致。归档单SHA/compile及controller/两worker实际PID/CWD/argv/空controllerCUDA/各物理GPU0或1和GPU-PID匹配、首轮曝光及日志增长已核实。无本轮正式性能结果；每行E20后自动七正式测试和独立truth-last。
