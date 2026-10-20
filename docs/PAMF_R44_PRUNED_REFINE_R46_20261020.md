# R46：44.28万保留优化与35.05万进一步压缩的正式结果

状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

用户在R44结果揭示后指定442805父，exact retain_refine为同预算训练优化控制，350521 prune_refine为GN4重要性进一步剪枝；同原接收链/3prior/6CG，无其他恢复网络。两行均pairedclean IQ+.05三尺度相对功率log1p误差+1→.1冻结父三stageIQ教师，所有coarse/action/prior联合。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。无TX标签模拟配对源域自监督，不称无clean盲学习。

方法commit `bbb8daf817de050b958d217b70d4c33f4aefe341`。完整登记、数据与祖先、命令和停止规则见[登记](../automation_reports/CV-SincNet/20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46/experiment.json)；全部20轮与TX/RX/day见[结果](../automation_reports/CV-SincNet/20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46/row_metrics.csv)。

|行|完整参数|较442805减参|R17逐景1pp|父442805逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|retain_refine|442805|0.000000%|True|True|67.964881|
|prune_refine|350521|20.840776%|True|True|67.510119|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R38只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较442805父（pp）|较R38（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|retain_refine|clean|84.560119|84.273755|+0.054167|-0.069643|-0.026786|-195|
|retain_refine|practical_high|82.232143|81.825314|-0.186310|+0.084524|+0.913095|+69719|
|retain_refine|practical_mid|80.954762|80.562690|-0.055357|-0.018452|+1.376786|+68944|
|retain_refine|practical_low_suburban|79.498214|79.154995|+0.122024|+0.004762|+1.501786|+67987|
|retain_refine|practical_high_urban|81.225000|80.818665|-0.092262|+0.069643|+0.871429|+69765|
|retain_refine|practical_mid_urban|74.660714|74.422247|+0.433929|-0.115476|+0.975595|+67498|
|retain_refine|practical_low_urban|67.964881|68.032474|+1.192262|-0.304762|+0.748214|+63278|
|prune_refine|clean|84.456548|84.156318|-0.049405|-0.173214|-0.130357|-369|
|prune_refine|practical_high|82.452381|82.040850|+0.033929|+0.304762|+1.133333|+70089|
|prune_refine|practical_mid|81.222024|80.857090|+0.211905|+0.248810|+1.644048|+69393|
|prune_refine|practical_low_suburban|79.620238|79.351692|+0.244048|+0.126786|+1.623810|+68192|
|prune_refine|practical_high_urban|81.386905|81.006498|+0.069643|+0.231548|+1.033333|+70037|
|prune_refine|practical_mid_urban|74.447024|74.361224|+0.220238|-0.329167|+0.761905|+67139|
|prune_refine|practical_low_urban|67.510119|67.825990|+0.737500|-0.759524|+0.293452|+62514|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|同卡父/本版median ms|resident/峰值字节|
|---|---|---:|---:|---|---|
|retain_refine|176962/125509/140334|2053980|49266224|48.1976/48.5497|19108352/336296960|
|prune_refine|136890/101037/112594|1686812|39081004|47.9998/43.5050|18735616/85753344|

同卡16条sourceV、batch1/预热2次，完整context/3prior/6CG；并存负载和顺序影响计时。MAC只含Conv/Linear，不含FFT/复数CG/归一化/激活/访存；不由参数下降推定加速或省峰值显存。

retain_refine首轮→末轮：loss 0.028242456→0.022583989；iq_nmse 0.0263283→0.022103433；spectral_error 0.0069749165→0.0058355203；teacher_stage_nmse 0.0015654107→0.0018877979；中位每轮163.857s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_refine首轮→末轮：loss 0.13009791→0.023733999；iq_nmse 0.078311426→0.023135337；spectral_error 0.017386838→0.0060890483；teacher_stage_nmse 0.050917143→0.0029421006；中位每轮164.083s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承100串行pass+本次20=120；每组件累计80、组件轮次和240，原F50训练另计。优化控制和剪枝行同预算，均重置优化器；与旧R44相比同时改变教师/频谱损失与训练预算，不作单因素机制归因。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r44_pruned_refine_r46_20261020/models/<arm>/final.pt；使用experiments.pamf_r44_pruned_refine.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。


## 固定方案和历史启动记录

以下为结果揭示前的方法与启动记录，当前训练和正式测试已完成。


状态：**VERIFIED / FORMAL_PREDICTING**，两版完整训练结束，正式预测已启动。用户在 R44 完整正式结果揭示后明确要求保留 group_pruned442805 并继续优化、压缩；因此直接父由用户指定，不按 Agent 目标排名选择，不声称新的独立盲测或干净泛化。固定两行同预算比较，已有 44.28 万权重与全部负结果保留，不研发其他恢复网络。

|行|完整参数|上下文参数|较442805减参|主/谱分支宽度|
|---|---:|---:|---:|---|
|retain_refine|442805|302471|0%|12/24/36/48；8/16/24/36|
|prune_refine|350521|237927|20.84%|12/20/32/44；8/12/20/32|

唯一初始化与冻结 IQ 教师为 R44 group_pruned 源域固定最终 E20/4940，方法 `2f5247dbd5fe50f28defa38a0252d70e692934af`。原 R38/R17/R07/R02 祖先、原物理 L6300/U56700/V 与六 TX 不变，恢复器训练始终0TX标签/F权重或输出/V/target。R44源固定训练记录与checkpoint元数据核对后严格加载，不加载其他学生或 F 蒸馏来源；历史目标暴露背景保留。

retain_refine 在训练前逐状态严格保留父函数，作为同预算优化控制。prune_refine 对每个 GN4 组内按原 R44 归一化输出 filter 均方权重分数稳定选通道，使用原已验证一致 encoder/decoder/skip/GN/action global-head 映射；保留主分支最浅12通道，减少更深和谱分支通道。三步 prior 的长卷积权重继续共享，逐步 bias/GN/pointwise 独立，三 prior/六 CG 与完整接收链不变。剪枝只读权重，0IQ/标签/F/目标信息，不声称与父函数等价。

两行分别从同一父独立分叉，新增完整20轮4940更新。使用全部原L6300/U56700及原20个模拟received缓存；每行新增L126000/U1134000曝光，学生context/教师IQ各1260000输入。所有coarse/action/prior联合训练；重置AdamW、LR2e-4/warm247/cos2e-5、WD1e-4、clip1、B256/micro64，尾批24按实际分母，FP32/noTF32、model2026092701/data2026102001。父每组件累计60轮，本轮后80轮，组件轮次和240、串行训练pass累计120，原F50预训练另计。

目标为最终paired-clean IQ NMSE，加0.05的64/128/256三尺度相对功率log1p平方误差均值，再加冻结父三stage IQ NMSE均值。教师系数20轮线性1→0.1。频谱项使用复IQ、periodic Hann、hop=size/2、centerFalse、正交FFT；每条pairedclean平均窗口功率detach/floor1e-4，同时缩放预测与参考。配对clean和教师不反传。这是待验证的训练假设：减弱后期父输出约束，为配对clean的弱谱细节留下优化空间。频谱项不使用身份标签或F知识，不预先声称提升识别或逐细节真实。

术语保持“无 TX 标签的模拟配对源域自监督”，L/U都利用，U_s TX真值隐藏；源clean配对参与，不称不使用clean的盲无监督学习。部署仍只读当前原生float32 `[B,2,256]` IQ并输出同形IQ，原冻结F50及checkpoint默认头统一派生FFT、PA、时域和统计视图，0另训头。

固定最终E20/4940，无早停或按源分数筛行。每行自动source1800×7诊断和同卡原R17/442805父/本版完整链batch1成本，然后原168000物理样本×7正式场景：clean、郊区high/mid/low、urban high/mid/low。每行全部预测固定后独立CPU连接truth，原F50/defaulthead不训练。报告accuracy/Macro-F1、逐TX/RX/day、IQ/FFT/配对特征几何、repair/damage与净修复、相对原R17/指定442805父/R38的逐景差；原R17每景最多下降1pp及指定父1pp分列，70%/其他五景gap≤5pp目标分列，不把均值覆盖单景退化。两行全部正式测试，不根据目标结果调参或选择性重跑。

N607普通账户GPU0/1每卡一own、总最多二own，等待R44对应GPU实际句柄退出、容量free≥18GB/allowner进程<2；其他owner不动。CPU2/max1本任务scorer，复用R41 score.lock。独立不可覆盖run/release/output，方法Git固定；聚焦协议负测、一次限P0/P1审查；launcher首步真实R44/F50/两条Ls配对IQ noquery smoke，PASS立即按卡启动。一次普通预检、归档单SHA/远端compile和实际PID/CWD/argv/GPU/日志读回；技术失败保留受影响产物，无自动重试，不停止或热改健康行。精确commit/命令/环境CWD/路径/预算/矩阵/停止规则与预期产物见run登记。

当前没有本轮源或正式性能结果。减参是确定的结构事实，优化收益、延迟与显存必须实测；参数下降本身不完成用户性能和输入TX细节真实性要求。R45旧正式解析评价独立收尾，不作为本轮训练或预测的门槛。


VERIFIED/RUNNING：实际442805父/F50/两Ls配对IQ noquery smoke PASS，两行全部组件有限非零梯度、teacher/F不更新、checkpoint严格重载一致。归档单SHA/compile及controller/两worker实际PID/CWD/argv/空controllerCUDA/各物理GPU0或1和GPU-PID匹配、首轮曝光及日志增长已核实。无本轮正式性能结果；每行E20后自动七正式测试和独立truth-last。


VERIFIED/SOURCE_TRAINED+FORMAL_PREDICTING：两版完整E20/4940、40轮全部损失/曝光/LR/配方核实，每版L126000/U1134000（U90%），0F训练输入；完整stdout无异常。两版source1800×7预测和独立评分完成，无源选模门槛；168000×7正式预测已实际启动，尚无完整正式评分结果。训练完成不是实验或总目标完成。
