# R48：从 35.05 万版继续优化、压缩并测量推理成本

状态：**VERIFIED / RUNNING**，已发布且两版实际训练。27项聚焦检查、CLIhelp和一次限P0/P1审查通过。

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


方法commit `86b89ba6bc09b0b3f10fcaff8fc7cd1ecf6d0508`。精确命令、环境/CWD、路径、矩阵、预期产物见[登记](experiment.json)。


VERIFIED/RUNNING：实际350521父/F50/两Ls配对IQ noquery smoke PASS，两行全部组件有限非零梯度、teacher/F不更新、checkpoint严格重载一致。归档单SHA/compile及controller/两worker实际PID/CWD/argv/空controllerCUDA/各物理GPU0或1和GPU-PID匹配、首轮曝光及日志增长已核实。无本轮正式性能结果；每行E20后自动七正式测试和独立truth-last。
