# 冻结R03后的直接残差与确定性流修正

状态：PLANNED。用户目标仍为完整目标low urban≥70%，其余五practical接近原F clean85.113889%（Agent暂采用≤2个百分点口径）；分别报告同物理clean的波形差距和冻结原默认头真实TX识别/净纠错。现有最好65.309127%，未达到目标。

源依据是旧R03生成残差在其AsF/sourceV设置中未通过身份稳定性：residual_flow的三次采样TX切换7.03125%、low urban78.888889%，未访问目标。本候选独立检验无Gaussian、完整R03 anchor与原F指导的确定性流，不追认/加载旧生成或失败精修。借鉴[PMRF](https://arxiv.org/abs/2410.00418)的先预测后运输结构；R03不是已证明MMSE估计，本候选使用当前received条件/无Gaussian/F约束，因此不是PMRF复现或其最优性保证。论文图像质量结论不代表IQ指纹身份能恢复。

|固定行|训练时间/状态|完整推断|GPU|
|---|---|---|---|
|direct_residual|t=0，point=anchor|anchor起始、一步Euler修正|6|
|rectified_flow4|半数t=0、半数U[0,1)，point=anchor+t(clean-anchor)|anchor起始、四步Euler，t=0/.25/.5/.75，dt=.25|7|

原合法R03（method b01b534f52daf9cae9e940ed6ec6f9243bccbdbe）coarse/correction完整state固定、参数和统计不更新。新增相同Core7→2、widths16/32/48/64，210882可训练、2351722总参数；输出投影为0，两行完整初始输出逐点等于原R03。输入为当前received、anchor、point-anchor，均按本帧anchor RMS归一化，再拼一个time通道；输出velocity乘同一RMS，无逐TX/RX路由、场景门控或运行F。推断只接当前单帧IQ，原R03执行一次，newfield执行1或4次，无Gaussian/physicalID噪声、pairedclean、真实信道、TX/RX/场景、其他帧/视图或任何参数/统计更新。状态只在当前帧内积分，不跨query保存。

源训练使用已核实63000 L/U、原R02缓存epochs1..10、.15clean/.60residual/.25full、observed+clean等权与原顺序。time独立seed2026102071、同物理observed/clean配对共享t；flow半t0覆盖实际推断起点，另一半均匀覆盖中间态，FP32舍入不允许采到1。速度监督target=clean-anchor，损失为每帧clean power归一化velocity NMSE＋0.1logpower(anchor+velocity,clean)＋三原F各自final160cosine＋0.3×T²KL（T2）的均值。两个head指导代理均为anchor+velocity，不把clean插值终点作为恢复IQ；完整4步采样才用于源V和目标评分，不能用clean依赖的训练代理冒称推断表现。记录actual calls/IQ、time0数/均值/范围、encoder非零梯度步数和逐损失/LR/路由/曝光。

两新行各E10/2470，L63000/U567000优化曝光、1260000 IQ；原coarse/correction仍各20轮，新field10，累计50组件epoch/串行继承50pass。原F50 CROSS_RX三seed2026092701/02/03、method4df7b94d0fd8ce29056bd8236364e45afa92e07f及原默认头冻结参数/统计，0新头拟合、0L/U TX真值、0F/旧恢复器更新。仅优化新field，AdamW2e-4/warm247/cos至2e-5、WD1e-4、clip1、FP32/noTF32。两行参数、父state、数据/顺序/路由、基础loss与预算匹配；训练时间分布和推断轨迹不同，不称同推断成本，也不把与旧随机流的差异归因于某一个因素。

source V原1800/原F1/七视图仅诊断，无早停、资格、NMSE gate、候选排序或状态更新。两新固定最终E10都自动完整168000物理目标clean+六practical/三原默认头，各自argmax，不投票、不按环境拼接或按分数选候选。各行七视图IQ/ID/预测全部固定后独立无CUDA CPU才连接truth与pairedclean；逐F/TX/RX/day accuracy/MacroF1、修复/损坏/净修复、距原clean差距、full256/prefix232/tail24 NMSE/logFFTpower均值/中位数/P90、训练/预测耗时与显存均保留，不作oracle相位/时延/增益对齐。一个恢复器seed与三读出不冒充三次独立恢复训练、新盲测、Phase2或在轨性能。

40项聚焦检查PASS；唯一一次独立P0/P1审查PASS后发布。launcher首步实际原R03/三F/四Ls no-query smoke，验证相同初始IQ/state、冻结原R03/F、新输出投影梯度、原头到IQ梯度及新checkpoint往返，0V/TX/target/optimizer更新；PASS立即训练。零投影首步encoder梯度可为0，后续实际非零encoder学习步数另行记录。复用同一数据契约与合法父来源，不重复数据验证或增加哈希/签名链。

一次N607普通账号预检VERIFIED：原R03合法、新run/release不存在，GPU6/7各24124MB空闲无PID；现有gradient_balance在GPU4/5实际target PID/CWD/cmd LIVE并保留。每卡一条本owner、计入allowner<2，重叠最多4本owner GPU worker，source≥18GB、target≥12GB，CPU2线程/scorer最多1。新source评分先等前gradient_balance complete/failure，再与本run target共用score.lock。实际source退出后才同卡target，不等另一新行。停止规则：非法输入/权限/代码、输出冲突、非有限/OOM、预算/冻结/执行或预测失败时保留产物并技术失败该行，无自动重试，健康行继续；不因源分数或当前新目标成绩改参数/预算/顺序/选择性重跑，不停止/修改他人任务。

一个实际代码配置commit、不可覆盖run/release、一次归档本地/远端SHA比较及remotecompile/import、真实smoke、PID/CWD/命令/GPU/日志增长随后核实。命令、环境/CWD、矩阵、路径、资源、停止规则与产物见experiment.json。数据/权重/逐样本预测/原日志/私有机器状态不进Git。另一用户对话未被发送指令或修改。
