# 更早持续的原 F 引导与中间层约束

状态：RUNNING / VERIFIED。用户目标仍为目标集low-urban≥70%、其他五种practical接近原clean（Agent暂按差距≤2个百分点记录）。分别报告恢复IQ与同条clean差距、接入冻结原F/default head后的真实TX识别。源分数、NMSE、检查通过均不能替代完整目标验证。

## 源依据与两个匹配臂

前四个源消融未通过城市增益要求。三teacher后验0.3相对单teacher E10使源low-urban提高1.166667个百分点至87.055556%，但mid-urban只提高0.111111个百分点至93.444444%；后验1.0未改善mid且平均NMSE超过15%源界限。已完成源分段分析也显示平均波形误差略低仍可能增加身份错误。本次检验更早/持续的识别引导和中间特征图约束，不继续增加末端权重，不追认任何失败权重。

直接加载原R02 tf_unet_mixed coarse的源波形规则A选择权重（commit7efdba92932a86bf4b0492a565d35377564a26da,E20/4940），全新Core correction零输出，初始IQ严格等于coarse。仅复用R03在任何优化前、只由8192条源训练观测计算的固定scale0.1169459182403555；该标量0V/0TX/0target/0optimizer，不继承R03 correction20。R03模型只作既有源/目标参考，失败精修或已测目标模型不作初始化。原来源、物理契约、source选择、固定E20与scalar普通元数据已在一次只读预检核实，不增加数据重验或哈希链。

final_guidance与layerwise_guidance均平均三套原F50 CROSS_RX(seed2026092701/02/03)的独立标量loss：末端160维cosine权重1、默认头T=2/T² KL权重0.3、共同IQ NMSE+0.1log power。仅layerwise臂增加0.1×三个teacher各自t1/t2特征图cosine的均值；各图按同一物理记录展平，绝不平均不同teacher的坐标。clean中间参考在no_grad下计算并按observed+clean复用，estimate图保留梯度，hooks在finally移除；对照完全跳过额外clean中间前向。F参数、统计状态和原默认头冻结，不拟合新分类/诊断头，不读L/U TX真值。已有监督F知识使本方案属于识别器引导的自监督恢复，不能称整个系统完全无标签。

## 预算和数据

新增joint30轮/7410步，父coarse20+joint30=串行继承50pass；coarse累计50轮、全新correction30轮，总80组件epoch。原R03 coarse20/correction20+joint10为60组件epoch、10轮识别引导，本次不冒充相同更新/计算或增强历史。两条新臂参数、初始化、预算、物理顺序、优化器严格匹配，仅中间层约束有无不同；额外F前向的实测耗时和显存单独报告。

原R02合法63000条L/U、前20轮配对缓存，新增训练用1..20再1..10，公开重复而不生成新IQ。路由.15clean/.60residual/.25full、每条observed→clean与clean→clean等权；未生成clean槽不当damaged读取。seed2026092701/data seed2026101951、batch256、FP32、reset AdamW/WD1e-4、warmup247、cosine至起始0.1、全模型clip1。父coarse组LR3e-5→3e-6；新correction组采用原scratch correction的2e-4→2e-5，两组实际LR逐轮记录。新增曝光L189000/U1701000、IQ3780000；父20轮曝光L126000/U1134000另记，不隐去继承成本。

## 冻结源规则和自动目标

两臂都固定末轮，无V优化/早停。仍是同一1800条源V/原F1/clean+六practical，所有源预测固定后独立CPU接入V真值。参考改为已完成posterior03的源指标（不继承其权重，不重新晋级它）：low-urban87.055556%、mid-urban93.444444%。每臂需两urban各≥+0.5个百分点、其他四环境最多下降0.5个百分点、相对原clean下降≤0.1个百分点、六环境平均NMSE≤1.15×原feature参考0.028326367493718863，绝不复利。通过者按urban均值最高、NMSE更低、登记顺序冻结选择；均失败不访问新目标。

只对选中的一臂自动运行原不可变168000条clean+六practical，三套原默认头各自argmax，不投票。七视图恢复IQ/物理ID/全部决策固定后，独立无CUDA CPU scorer才接入目标TX/RX/day与评分clean。报告accuracy/Macro-F1、修复/损坏/净修复、逐F/TX/RX/day、距原clean百分点差，以及未作oracle相位/增益/时延对齐的full256/prefix232/tail24 NMSE、logFFTpower均值/中位数/P90。practical推断只见当前received IQ，不读取paired clean/其他视图/目标真值，不更新状态。一个恢复器训练seed、三套原F读出，不称三恢复器种子、新盲测或在轨验证。

## 发布、资源和停止

38项聚焦检查通过，涵盖中间图到IQ梯度/clean隔离、hook正常/异常清理、对照跳过分支、新修正器初始等于coarse、标量权限、E30/7410与缓存循环、两LR/检查点预算、source选择和全部七视图truth-last。一次独立P0/P1审查PASS，无发现。N607普通账号，预登记GPU4 final_guidance/GPU5 layerwise_guidance，每卡一条本run worker，获取时全owner<2且free≥18GB。只读预检时两卡各24124MB空闲、无PID；启动时实际容量再次由既有acquire检查，不停止其他任务。CPU2线程，源scorer最多1个；两源worker完成退出后，仅在所选卡启动一条target worker。

一个code/config commit、一个不可覆盖root、一次预检、一个发布归档SHA比较和远端compile。launcher第一步真实原R02+新correction+三F的两条Ls no-query smoke，证明初始等于coarse、两个模块有限非零梯度、原F全冻结、中间图与默认头到IQ梯度、实际t1/t2形状、新checkpoint往返、0query/0update，PASS立即续训。随后核实实际PID/CWD/完整命令/GPU/日志增长。数据/来源/commit/预算/输出冲突、非有限、OOM或预测失败时保留产物并停止对应技术失败，不自动重试，不热改/停止健康任务。具体命令/环境/路径/矩阵/产物见experiment.json。


## 启动核验

方法 commit：`672a36de61da32f3974472db97448b8977bbae71`，已推送且远端分支 OID 与本地一致。38 项聚焦检查和一次独立 P0/P1 审查 PASS；一次发布归档本地/远端比较及远端 compile/import 通过。真实原 R02 coarse＋全新零输出 correction＋三套原 F 的两条 L_s no-query smoke PASS，初始 IQ 等于 coarse、两个恢复组件均有有限非零梯度、默认头及中间层能向 IQ 反传、F 全冻结、新 checkpoint 往返一致，0V/0TX/0target/0optimizer update，PASS 直接续训。

已从独立后态核实两条实际 worker 的 PID/CWD/完整命令（含原 R02 parent-coarse）、物理 GPU4/5 归属及从首个快照到训练首轮的日志增长。进度快照：

- final_guidance：1/30 轮、247/7410 步，累计训练 109.823 s。
- layerwise_guidance：1/30 轮、247/7410 步，累计训练 118.083 s。

两臂各自完成 63000 条 L/U×三套 F 的源参考缓存，未读取 TX 真值。源 V 评分和新目标测试均尚未发生。完成两臂后按预登记源规则冻结选择；只对通过且选中的恢复器自动完成目标七视图及三套原默认头的独立评分。运行状态和 smoke 不能证明性能改善或用户目标已达成。
