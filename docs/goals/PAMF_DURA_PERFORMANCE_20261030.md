# DURA 星地性能优化目标（2026-10-30）

2026-10-31 用户新增预算与方向：后续新恢复候选可从零单阶段训练 30 epoch，固定最终轮次；历史 E20/E40 不热改。新增“物理恢复＋有助于原冻结 F 识别的 IQ 特征增强”独立对照，仍限已授权 source F 指导、G 不直接读取 TX 标签/V/target、F/原头不更新。以同一模型的物理输出和增强输出配对测试，分别报告波形保真、真实识别、clean 损伤和成本。1-NMSE 只称能量拟合分数，不称信息恢复比例；clean 识别率不是所有输入处理的普适上界，真实恢复极限仍需注明假设，不能由有限模型的最佳成绩认证。

用户目标：全面分析并优化星地恢复网络，使星地识别性能接近 clean，同时提高训练效率。主分支沿用默认 DURA v1.0 的源数据、权重来源和恢复器 TX 无标签、F-free 训练权限；用户于2026-10-30另行明确允许独立源域F特征指导对照，其信息与成本差异见`PAMF_DURA_F_GUIDANCE_OPTION_20261030.md`及项目协议。本目标授权新的独立研发，不更换现有默认部署版，不改动 Phase2/Phase3 或其他 owner 的健康任务。

## 成功条件

以同一冻结识别器、同一默认头和同一物理记录的 **raw clean 准确率**为参照，逐一报告六种 practical 星地场景的差距。用户于2026-10-30明确确认：六景各自差距不超过 3 个百分点，并保持clean性能。这是正式验收目标，不以六景均值替代逐景达标。原 low urban ≥70%、其他五景差距≤5pp、逐景相对原 R17 最多下降1pp继续单列，达到旧门槛不等于达到本次目标。

每个新恢复网络版本冻结后自动完成原 168000 条记录×七景预测，再由独立 scorer 连接 truth；同时报告 clean 损伤、逐 TX/RX/day、净纠错、完整参数量/存储、同卡时延与显存。clean 相对 raw 的损伤不超过既有 1pp 上限，并争取不下降。目标已暴露，结果属于工程基准，不声称新盲测。仅源 NMSE、减参、提速或 oracle 诊断改善均不能宣告目标完成。

## 已核实起点

固定恢复器为 `experiments/pamf_stage2_inference/frozen_version.json` 指定的 R69 stage2 E20/4940、266951 参数、两 prior/四 CG；已有 graph/reuse 执行提交 `47c264286ca689f060a3f36aafb763fe209a6f06`。下表来自已完成的冻结原 F50 seed2026092701/defaulthead 七景结果；不是当前 Tri-SIG-TIME SSL 的成绩。

| 场景 | DURA v1.0 准确率 |
|---|---:|
| clean |84.677976%|
| high suburban |82.825000%|
| mid suburban |81.560714%|
| low suburban |80.058333%|
| high urban |81.745833%|
| mid urban |75.486310%|
| low urban |68.889881%|

证据：[默认版完整报告](../../automation_reports/CV-SincNet/20261021-phase1-restoration-stage2-graph-s2026092701-r01/report.md)。其 raw clean 正确数由同 row 的 clean 正确数及净纠错还原，约84.6762%；low urban 尚有约15.7863pp差距，达到本次3pp目标需约12.7863pp提升。单卡预加载 B1 G+F 4.531972ms；不把跨卡/共享负载时延与该数直接比值。

历史源诊断已发现弱功率频点和尾部误差突出，但其父模型并非当前 DURA。R41 的物理约束、R22 的复频谱/clean 恒等以及多个剪枝/归一化分支已有正负结果，不重复把旧想法当成新证据。当前 DURA 的教师级联成本和误差仍需单独分析。

## 第一个可证伪实验

当前登记 `20261030-diagnostic-DURA-source-inverse-s2026092701-r02`，只使用原 R07 训练池按角色位置均匀抽取的128 L_s+384 U_s，固定 epoch01/20 的 received 及同份 source 模拟器参数。首先两条真实 L_s 做实际 checkpoint 无 query smoke，随后立即诊断。0 TX 标签、0 F、0 V、0 target、0 优化器更新；不产生部署 prediction。原r01在首步smoke因旧Torch/NumPy2 C桥不兼容失败，保留输出；r02使用项目既有direct-buffer传入和tolist传出，矩阵及科学设置不变。

比较当前 coarse、stage1、stage2、最终 prior，以及固定最后一个 prior 的2/16次CG。另分别替换 source 模拟器真实方差/算子，两者同时替换，并用 clean prior 测软一致性引入的偏差。用固定随机白探针量化算子误差，按场景、state、lock、L/U汇总IQ、尾部和有分母下限的复FFT分组误差。完整源数组统计留存，不能用均值掩盖长尾。

真实算子和 clean prior 只用于源端反事实诊断：两者都不能进入部署、target 或候选权重。这些结果不提供严格的准确率上界；替换算子会使先验与原训练条件不匹配。增加CG会改变版本，诊断输出不视为新版正式测试结果。

根据该源证据区分求解不足、算子失配和先验损伤，再固定下一候选与同预算对照。候选结构/损失/父权重、更新预算和七景矩阵须在该候选目标访问前登记；不根据已暴露目标分数挑父、调参或选择性重测。性能优先，参数与计算成本同时报告，不因更小而默认接受下降。

2026-10-30首个诊断已VERIFIED/ANALYZED_SOURCE_ONLY：512位置×两缓存epoch，CPU2约10.155秒，全部分组统计独立复算。固定prior的16CG使平均IQ误差较原生2CG分别增加9.04%和9.91%；当前最终prior均值误差约0.175，而最终输出约0.020。真实算子/方差替换未稳定改善，说明简单增加求解预算不受本次源证据支持；先验与算子协同补偿是待检验解释。完整结果见[r02报告](../../automation_reports/CV-SincNet/20261030-diagnostic-DURA-source-inverse-s2026092701-r02/report.md)。

已固定下一独立配对run `20261030-phase1-DURA-prior-anchor-s2026092701-m2-r01`：两行从同一DURA v1源final独立续训E20/4940，常规续训对照与增加0.1倍两个内部prior配对clean IQNMSE监督。两行同源数据、相同DURA IQ教师/1→0.1阶段蒸馏、原最终IQ＋0.05谱误差、重置AdamW与相同LR，新增监督为唯一差异；不加参数/阶段/CG，不读取TX/F/V/target训练反馈。每行冻结后自动原F50/source1800×7、target168000×7、同卡graph测速与独立truth-last。该实验尚不代表性能提升，旧部署默认保持。

## 训练效率的新增目标（用户2026-10-30明确要求）

用户指出此前多阶段训练耗时过大、不利于研发迭代。后续将**从随机初始化单阶段联合训练完整恢复器**作为新主研发路线，检验是否能够去掉粗恢复/算子/先验逐级预训练、级联剪枝蒸馏和在线神经教师。质量验收仍为六景各自距rawclean≤3pp并保持clean；不能仅靠缩短epoch、隐藏祖先成本或降低性能宣布训练优化成功。

分别报告依赖链深度、所有祖先与新增优化器步数/源曝光、缓存制备与读取成本、单步/每轮/完整训练墙钟、峰值显存、达到同一源冻结标准的成本及最终正式质量。随机初始化候选的祖先checkpoint加载和学习教师forward应为0，整个网络在一次固定训练日程中联合优化；两阶段推理展开不等于两阶段训练。合法source模拟器算子/方差可在同一次源更新中作为明确披露的辅助监督，不在部署使用，不读取TX/F/V/target训练反馈。

先登记小型 `20261030-diagnostic-DURA-single-stage-cost-s2026092701-r01`：CPU2、256个既有Ls源位置，B32/B256、各2次预热＋6次计时，比较在线教师、同batch缓存教师与从零联合目标。无优化器更新、无新权重或prediction，模型状态前后不变；测forward/objective/backward/clip，明确不含优化器step、IO/传输/初始化和教师缓存生产。这只是CPU成本诊断，不把它当GPU吞吐、训练完成或质量验证。之后以单seed完整E20与七景正式测试检验直接训练候选；当前两行健康续训保留作为旧路线的增量与累计成本参考。

该成本诊断已VERIFIED/ANALYZED_SOURCE_ONLY，B256在线教师/缓存教师/随机联合目标单步中位数为1.535041/1.152319/1.186435秒，随机联合目标较在线教师CPU耗时减少22.71%，模型state未改变。正式直接训练候选登记为`20261030-phase1-DURA-direct-joint-s2026092701-r01`，随机初始化、累计E20/4940、无学习教师/祖先权重；不能将此CPU计时或预算减少当成同质量总成本结论。

旧续训两行均已完成E20并保留final，postfreeze测速因Sinc版本计数失败；已精确复现并修复评估函数构造模式。`20261030-phase1-DURA-prior-anchor-s2026092701-m2-r02`只复用原两份源final补齐同一七景及成本，新增训练0；源checkpoint代码44bb16ed...与新评估代码分别记录。默认部署和原source数据不变，目标继续ACTIVE。

三版首轮完整七景与378个分组已核实：常规续训lowurban67.482738%、先验监督67.857143%、从零单阶段56.041071%，均低于默认68.889881%；从零版clean85.402381%，训练799.978秒/累计20轮，但星地质量未保持，不能称效率目标达成。下一run`20261030-phase1-DURA-direct-fastlr-s2026092701-r01`在读取从零版正式分数前已依据完整source拟合差距固定，保持从零E20和全部配方，只把峰值LR2e-4改1e-3，配对同一初始化/数据/预算。后续仍以冻结后的七景实际收益与成本验收。

后续代码核查确认，从零factory复用了checkpoint-loading skeleton：FIR初值脉冲24，缩窄时重新随机初始化全局头，输出残差也随机；与clean无信道输入的恒等初值不同。独立`20261030-phase1-DURA-direct-identityinit-s2026092701-r01`在fast LR正式结果前固定，保留其全部数据/预算/损失/学习率，只将内部随机滤波器外的残差/action输出头置零、FIR脉冲设0，初始输出保持输入。它不继承任何学习权重，仍一次联合训练E20，实际收益由完整七景判定；不热改当前fast LR。

## 执行边界

默认 N607，按恢复器既有每 GPU 至多一个本 run GPU worker和总容量约定，启动前核实进程与显存；第一诊断仅一条进程。GPU无足够总容量时用CPU2执行同一诊断，不干预其他owner。原 R69/原 R17、现有预测、缓存、日志和默认部署版保持不可覆盖。

只执行项目八项最小流程。通用头脑风暴的额外设计审批/独立计划文件为 `REJECTED_EXTRA_GATE`，不阻断已授权研发。本目标持续 active，直到正式性能验收达标，或有明确科学限制需要用户决定变更范围；不能将诊断完成标为性能目标完成。

可参考展开网络的任务相关参数优化思路：[Model-Driven Deep Learning for MIMO Detection](https://arxiv.org/abs/1907.09439)。该文研究MIMO检测，不直接证明本项目短IQ恢复或TX指纹识别能够达到clean；本项目方向和收益仍以合法源证据及固定正式测试核实。


2026-10-30单阶段fastLR和恒等初始化均完整完成E20/七景/独立126组复算：训练分别约12.85/12.72分钟，lowurban65.641667%/66.442857%，均低于默认68.889881%；六景3pp目标未达。恒等版源IQ误差略高但六个星地准确率更高，进一步表明源均值IQ不足以代理TX身份保真。

源梯度诊断32批全部完成，0模型更新/标签/F/目标读取。方差辅助梯度在全参数25/32批、estimator27/32批与IQ相反；其他辅助项未呈现同等冲突。独立预登记`20261030-phase1-DURA-direct-noVarAux-s2026092701-r01`仅移除该辅助校准损失，保持fastLR普通随机初始化和E20全部其余条件；方差仍由恢复路径学习，不宣称物理噪声校准。该假设来自源证据，正式收益仍未知，自动完成全七景；不按已暴露分数调参或选父。具体证据见对应run报告。


源GPU步骤诊断已完成：统计合并中位131.502→137.562ms，未显示加速，不采用。固定B256 source step profiler显示linear1d插值前/反向是突出GPU event；不将嵌套event相加或外推整轮成本。下一零更新源诊断`20261030-diagnostic-DURA-interpolation-s2026092701-r01`仅对Core的12/15次固定二倍插值作半像素等价展开，5个CPU双精度前向/伴随/边界测试已通过，实际完整GPU参数梯度和速度待测。采用门槛为同卡交替中位步骤至少快10%且三轮均正收益；无权重或目标访问，未替换当前训练。


固定插值源GPU诊断已达到预登记门槛：native/slice同卡交替B256步骤221.756580→189.741313ms，减少14.437%，三轮均正收益，实际完整模型损失和全部裁剪后参数梯度close、state不变、0更新/标签/F/目标。下一`20261030-phase1-DURA-direct-fastinterp-s2026092701-r01`只把该执行替换加入原fastLR完整五项目标和普通从零E20/4940，不与noVar/恒等初始化混合；正式冻结推理同样采用新slice版本，自动七景验证。该步骤收益不代表完整训练或同质量成本收益，默认DURA不变。


noVarAux单阶段E20已完整正式评分并独立复算126组：六星地均高于匹配fastLR，lowurban65.641667→68.470833%，clean84.742262%（raw84.676190%）；但六星地仍低于默认，逐景rawclean差距3.772024至16.205357pp，3pp目标未达。源IQNMSE降低7.331%，训练851.915秒/累计20轮/0祖先教师，跨卡负载不做因果速度比较。该结果不回流fastinterp的已固定目标，默认DURA保持。


两份直接训练源物理诊断已完成：512源位置×2epoch×fastLR/noVarAux，12聚合单元/4176统计独立复算，0更新/标签/F/目标。增加CG没有实质收益；固定prior的oracle A相对learned CG16在fastLR两组均改善约13%，达到预定10%双组门槛，noVar约9.59%/10.84%未双组通过，不放宽门槛。下一`20261030-phase1-DURA-direct-action020-s2026092701-r01`只将原fastLR action监督0.05→0.2，其余原生插值/普通初始化/全五损失/E20/4940全匹配，不混noVar/identity/slice；从零无教师，自动全七景，不能用源oracle当部署或准确率上界。


fastinterp已完成全E20与七景、126组独立复算：lowurban65.819048%，六星地均略高于原生fastLR但低于默认，clean84.632738%，六景3pp未达。完整训练819.008秒，未证明完整训练加速；此前14.437%只属于固定源步骤。单批近似梯度不保证整段AdamW轨迹一致，保留执行分支和全部负结果，不自动推广为默认。


初始化条件诊断已完成64源批/16聚合/1024策略算术核对，0更新/F/TX/目标。identity相对ordinary的noVar梯度中位比0.84269/0.83180，未达到预定双组减半，故不发布该组合。strongA初始梯度范数与base接近，不能把后期裁剪简单归因于初始action主导。强A源E20已完成但IQ误差0.03524高于原fastLR0.02702，仍自动完成七景；F指导选项待用户明确决定，当前未执行，原F-free范围保持。


用户已明确授权新增独立源F指导对照，科学权限在PROJECT_PROTOCOL单列，仅此分支可用冻结F源特征训练G。`20261030-phase1-DURA-source-Fguide-cos010-s2026092701-r01`固定普通从零E20/noVar基础目标＋0.1Fclean特征cos，直接TX标签/V/target训练0、F更新0，Fclean缓存63000一次及新增F(restored)1260000前反向均计成本；明确继承F源有标签知识，不称F-free。原F50/source物理Ls6300/Us56700与G集合匹配已核实，默认R69不变；冻结后自动完整七景。授权不等于已启动或达标，实际状态以run为准。


strongA0.2已完整七景并独立复算126组：lowurban63.557738%（较原fastLR−2.083929pp）、clean85.237500%；六星地较匹配控制3升3降、均低于默认，3pp未达。完整E20源IQ/probe拟合也未改善，保留负结果，不按目标回调权重。新F指导的noVar基础＋cos0.1在本正式结果读取前已经作为获批方案固定，其F教师上游明确入登记，不隐去知识/成本。


已授权F指导分支真实首步smoke PASS并实际RUNNING：原F50特征到IQ非零梯度、F状态/参数不变、G严格重载和原默认头均验证；GPU4/PID3958132独立存活/命令/CWD/显存/日志增长核实，源63000×160D缓存40.32MB在3.782秒完成，至少E3。训练F access=True、直接TX标签0/F更新0，单阶段G从零E20，自动七景，当前尚无新正式结果。真实成本和冻结后性能仍待完整收尾，不将缓存次数节约当整段训练加速。


源F判别/梯度诊断已完成：64批与1280几何标量独立核对，0更新/TX真值/V/目标。Fguide010源F余弦误差约降低26%，但cleanF选择一致率仅+1.5625/+0.1953pp，真实TX准确率未知。当前0.1F/base梯度中位比0.2453/0.2204，未通过预定≤0.2的加到0.3资格，不加权、不放宽门槛。该源现象先于本轮目标分数读取，后续若研究原头判别投影约束须源端独立固定，不能按目标类别/场景调参。当前Fguide010七景保持原计划。


首个已授权F指导cos0.1完成完整E20/4940与七景正式测试，126组独立复算一致。clean84.904167%、lowurban67.827381%，六星地均低于匹配noVarAux，逐景3pp未达、默认不变。新增G完整流程1073.247秒/17.887分钟（缓存3.782秒），F既有50轮/11100更新另计、源有标签知识继承披露；F更新0/状态不变，直接TX/V/target训练0。源诊断先于正式分数读取已否决beta0.3，下一研究限冻结原头可见的源特征方向，不按目标场景或类别定权。


原头可见源特征诊断32批/2560几何标量已独立核对，F/G零更新且状态不变，0直接TX/V/target。冻结原头rank5投影与其中心特征判别误差梯度对齐.9919/.9892，旧cos为.7162/.6126；源校准alpha=.006607891036090233，两epoch预定资格全部通过。下一独立source-headFeature从零单阶段E20/4940，仅把旧.1cos改为该固定原头可见特征项；不加载G，不改F/原头/数据/基础损失/原生执行，自动全七景。该源资格不证明目标提升，3pp目标与默认保持。


source-headFeature已按commit d79a280d843696d680c82bf63b0138905cf1cb5d发布，真实2Ls G/F梯度、F冻结与G严格重载smoke PASS，12聚焦检查及独立P0/P1审查PASS。CPUcontroller PID4025316实际存活/CWD/argv/空CUDA/日志增长已核实，当前仅QUEUED：各卡可见PID≥2，无ownGPUworker/epoch/预测。原控制器按资源约定等待并自动衔接单阶段E20和完整七景，不重发或更改其他owner。性能与3pp验收未完成。


默认R69源分阶段F特征定位完成：512源×2cached视图/10路径/32批，CPU13.895秒、7560统计独立复算、F/G零更新且状态不变。stage2相对stage1与cleanF选择一致率+4.8828/+4.6875pp，独立lastprior仅63.09/64.45%；保留两个有效推理步骤，继续优化一次联合训练。CG4/8/16没有teacher保持收益、F余弦误差反而增大，全部未通过预定逐源场景保护/双视图条件，不发布增加CG版本。真实A/noise仅源反事实，不能作目标上界。headFeature当前GPU资源队列保持，目标3pp与正式七景验收未完成。


研发成本盘点已核对七个单阶段版本完整140条epoch和882个既有独立正式分组：训练阶段12.72至18.20分钟、发布到完整分析17.27至26.20分钟，目标七景预测占16.67%至28.63%、评分0.63%至0.93%。七版均未达3pp、均低于默认六星地，不能用减少阶段声称同质量更快。针对逐批IQ转Python list的CPU开销，新DLPack后owned FP32 copy已通过真实2Ls G/F共1356标量逐位一致/F-G不变检查，4聚焦测试及独立审查PASS；三种合成CPU形状9轮交替均达到中位至少减半，完整GPU预测/训练加速尚未测得。仅后续release采用，当前headFeature d79归档/队列保持不变。


用户在旧headFeature队列等待约40分钟后明确允许资源调整：仅GPU4/6/7选一张、启动前free≥12000MiB且最多已有2个可见计算进程，本任务新增1个后总数≤3；own每卡1/goal最多2不变。只在旧r01仍无GPU row时锁内精确停止本run CPU排队控制器，保留全部smoke/日志并独立登记r02，避免重复；其他owner/科学模型/数据/E20预算保持。若r01已开始GPU训练则拒绝切换并保留健康旧任务。新r02沿用既定source headFeature系数和全七景，含已逐位验证的ownedFP32输出转换；真实启动/旧进程后态以本run产物读回为准，不把授权或登记当成启动成功。


资源授权已实际落地并独立读回VERIFIED：旧r01 PID4025316仅CPU队列、无GPU row，在capacity锁内精确停止后/proc确认不再运行，旧smoke/日志保留，登记REPLACED不恢复。新r02按commit81dfc80a2d4687f9b9ab7bc9c28ede6b092f7c4a在GPU4/PID4060125实际RUNNING，source2Ls真实梯度/F冻结/G严格重载smoke PASS，完整argv/CWD/CUDA/显存/日志增长核实；该卡总3进程、own约2020MiB/free20965MiB，未改其他owner。源63000×160缓存3.450秒完成，至少E1/247，首轮82.045秒，仍从零单阶段累计E20/4940并自动全七景；当前无正式成绩，3pp目标及默认保持。


headFeature资源r02完整完成E20/4940和七景正式测试，126组独立复算一致。clean84.902381%（较raw+380条正确），lowurban67.876786%，六星地均低于匹配noVar及默认；相对cos0.1仅lowurban+0.049405pp，其余五星地下降，逐景3pp未达。G新增流程1623.216秒/27.054分钟，发布到完整分析1999.843秒/33.331分钟，target七景337.460秒；共享三进程和不同模型/时段不做跨运行因果速度比。F末状态不变/更新0、G从零祖先0/直接TX-V-target训练0，F上游50轮知识/成本另计。全部负结果保留，默认DURA不变；下一研究应从当前新G的源stage/prior/求解特征定位开始，不按目标类别/场景调参。


noVar/headFeature源阶段对照完成：64批/15120统计独立复算，0训练/TX/V/target，默认同源baseline复用。headFeature source native教师一致数+8/+5（512/视图），但prior比native少11/3条、特征误差更大，prior-only双视图门槛均失败，不发布删最后求解版本。补充冻结后V truth-side独立复算315组：headFeature六星地sourceV均高于noVar，lowurban83.00→85.7222%/净+49，五个源RX均净增益；目标在共同三采集日期仍六景负差，不能只归因新day0。该现象支持优先检验采集风格迁移鲁棒性，但RX/日期/样本分布因果未隔离，不能按目标日期调参或以子集代替全七景。下一候选先查历史pair_aug_control/observable_response实际机制避免重复；默认与3pp/效率目标保持。


源采集响应扰动诊断完成：CPU2/8.427秒/0GPU，512×2源位置、1080统计独立复算，原分支重现既有 noVar 对照。功率保持最大相对误差2.98e-7、同A噪声重放误差4.04e-14以内；MSE仅增6.5297%/7.1732%，未达到预定双视图10%，因此不据此启动扰动训练，也不事后加强扰动或放宽阈值。F/G状态不变，0更新/直接TX/V/target，默认与3pp及训练效率目标保持。


完整源20轮曲线显示 noVar 最后五轮IQ误差比前五轮低约11.52%，未据目标结果选epoch/父模型。独立预登记从零单阶段E40/9880对照，固定noVar损失、20份原源模拟缓存循环两遍，不加载E20权重、不加F指导或未达门槛的采集扰动。14项聚焦测试/唯一P0/P1审查/真实source smoke通过；执行d0ac66573，N607 GPU7 worker296374，首轮36.317秒，完整七景与成本待完成。E40与E20及DURA累计120组件轮次预算分别披露，不承诺等质量加速，默认与3pp目标保持。


从零单阶段E40完整完成9880更新和全七景正式评分，126组独立复算一致。clean 84.814881%，lowurban 67.800595%，六景3pp=False、clean保持=True、R17逐景1pp容忍=False。训练1400.325秒、全流程1602.930秒；40轮/祖先0/无F教师，预算与原E20及默认历史120组件轮次分别披露。全部结果保留，不据目标改轮次或选择性重跑，默认保持。


本轮12行正式对照（默认、两个历史续训、八个E20从零分支及一个E40）已汇总于E40 run的research_summary.md/json；不新增目标输入/真值读取。E40源V面板1800条五景改善、high郊区少2条，目标六景全部较E20下降；本次源收敛未转为迁移收益。目标126组完整聚合已公开，default/3pp门槛不变，无本目标GPU训练进程继续运行，目标仍ACTIVE。


源PSD重加权诊断完成：63000源L/U/15RX-day均值，9bin平滑、.001相对floor（实际0触发），经验增益0.3434至2.9121；不是因果RX传函。固定noVar两视图IQ误差+32.3115%/+25.3545%，功率与原A/噪声配对通过；独立57600系数/780统计复算，CPU2/3.597秒/0更新，F仅2Ls smoke，0F指导/TX/V/target。允许另预登记从零E20/p.5/单G前向频谱对照，原手工2tap失败候选不扩大/不训练；资格不保证TX或目标收益，完整七景仍必须执行。


源频谱配对增强E20对照已发布：b1ed34788，N607 GPU4 worker699710，当前3/20/741更新151.651秒。21项新旧相关测试/唯一P0/P1审查PASS；真实2Ls都经增强、G梯度/重载/F默认头smoke通过，训练重置独立pair RNG。仍从零联合20轮、1次G前向/p.5、0G/F教师或直接TX/V/target训练，自动完整七景；source统计bank仅训练/smoke读取，推理不需要它。原DURA与3pp目标保持。


源频谱配对增强E20已完整完成20/4940、全七景及126组独立复算。实际增强629996/1260000（49.9997%），随机调用重放一致、跨RX格子正确；G流程992.782秒、全流程1303.957秒，无G/F教师/直接TX-V-target训练。clean84.848810%，lowurban66.245238%；六景3pp=False、clean保持=True、R17逐景1pp=False。全部结果保留，默认保持。


保留clean参考的源信道重配诊断完成：两个缓存视图512/512，匹配scene/state/lock，同RX/day控制与跨RX比较；独立donor/RNG/1716统计复算。cross/within IQ比1.001705/0.958650，算子误差比1.016785/0.994420，未过预定双10%，不启动该重配训练、不放宽阈值。CPU2/8.123秒/0更新，无TX/V/target/F指标读取；代码中的RX/day绑定未在本源匹配反事实表现强效，不据此认定为迁移主因。下一项可检验同冻结F原头后验的连续源约束与特征距离差异，仍需独立F-guided登记/从零E20/原头不更新/无硬伪TX真值/自动全七景；本条未启动。


连续后验F-guided对照已发布：c3b76d2bf，N607 GPU4 worker765216，实际source2Ls后验IQ梯度/严格重载/F冻结smoke通过，15项测试/唯一P0/P1审查PASS；63000源160D特征缓存2.966秒，当前E5/1235更新。G仍从零单阶段E20/4940，原pairedclean不变；F原头将特征确定性映射为soft6D后验，T1/KL(clean||restored)/0.1，无硬伪TX或真实TX读取。F上游50轮知识成本明确复用，不能称F-free；自动全七景及3pp/clean/效率目标保持。


连续后验F-guided E20完成4940更新与完整七景，126组独立复算，F参数/状态不变，0直接TX/V/target/硬伪TX/G祖先。clean84.381548%，lowurban67.943452%；六景3pp=False/clean保持=False/R17逐景1pp=False。G流程1116.279秒，全流程1420.082秒；F既有50轮知识成本另计，非F-free，默认保持。


源精确相位配对E20对照已发布：028be06be，N607 GPU7 worker797202，当前E1/247更新。原缓存实际生成器含15% clean，取消重复新增clean混合思路，未实现或启动。新对照只旋转clean/received并令真实action image乘R²，保留噪声/功率/方差；8项聚焦检查和唯一P0/P1审查PASS，真实2Ls的G梯度/重载/F默认头smoke与配对误差检查通过。仍从零单阶段20轮/4940，一次G前向，0F/直接TX/V/target/恢复器祖先训练，冻结自动全七景，默认与3pp/clean/效率目标保持。


源精确相位配对E20完成4940更新、七景和126组独立复算。1260000条旋转/4940调用完整重放一致，原缓存clean实际14.974365%，0F/直接TX/V/target/恢复器祖先训练。clean83.858929%，lowurban43.226786%；六景3pp=False/clean保持=False/R17逐景1pp=False。G流程1019.345秒，全流程1329.381秒；默认保持。


同F50源相位敏感性检查完成：source512×8/4096特征输入，F状态不变，CPU2/计算3.605564秒，0G读取/训练/TX/V/target；独立重算88组792统计。7个非零角度的教师选择一致率85.742188%至91.210938%，最差源RX69.607843%；7角度资格全部失败，不能忽略输出全局相位，不启动phasequotient损失训练，不放宽阈值。完整报告20261030-diagnostic-DURA-source-Fphase-s2026092701-r01，默认与3pp/clean/效率目标保持。

历史R52已核对：旧350521 G的固定clean与坐标传递clean参考控制，不是解析相位校正；不能据其误差宣称等变应为恢复目标，也不改标当前DURA。本次F相位诊断补充原F50整体敏感性。下一源端问题可检验公开L-STF相关器所给clean相位在RX/day间是否稳定；源均值不能直接替代目标clean相位，尚未实现/启动校正器。


公开L-STF统一相位参考检查完成：source63k公共增益，62488拟合/512统计留出互斥；µ-23.297473°、全局rho.970012726<.99，4/15sessionrho<.95。留出强制规范化后F原选择488/512=95.3125%、最差RX90/103=87.378641%，Fcos.013402436/KL.102256606，双条件FAIL。独立复算141统计/1536F指标，CPU2/计算1.442474秒，0G/训练/TX/V/target；不发布或训练统一µ/RX专属fallback。后续只可先检验已有G输出的source逐条相位oracle效应，不能把oracle当部署或准确率上界；默认和3pp/clean/效率目标保持。


source phase-only oracle闭合：固定noVar与default/source512×clean/缓存01/20，6cells/132groups/5412统计独立复算，G/F不变，CPU2/计算16.605771秒/0更新/TX/V/target。noVar整体相位仅移除10.161418%/9.109140%IQ误差，F教师选择+0/+2，部分scene-1；clean保持512。两个缓存视图均未过原资格，不启动phaseweight训练。default相位可移除9.652549%/9.279142%、F选择各+1，原G平均相位差约1.7°至1.8°。关闭本轮整体相位优先路线，残差尚含其他形状/幅度/滤波/时变相位/噪声，不能全称TX损伤；oracle不部署/不作准确率上界，默认与3pp/clean/效率目标保持。


同phase-oracle六源输出事后频谱统计：每包按cleanFFT能量分4×64bins，Q1能量88.4778%、Q4仅.109379%；noVarQ4相对NMSE.44至.46，F选择变化组整体NMSE也更高，不能全归因弱频点/TX损伤。Parseval和分区校验通过，0新G/F前向/训练/TX/V/target。R02粗恢复只IQ+.1log功率谱；R01确有解析RF损失但主身份资格全失败。下一当前结构RF约束必须独立匹配、先source梯度/成本固定权重，不把历史负结果当成功依据；完整3pp/clean/效率目标保持。


源解析RF输出梯度诊断闭合：source512×缓存01/20，CPU2/1.232532秒，0训练/TX/V/target。五解析支路之和，唯一alpha=0.0095279442618058027；两个view压力.258124/.237195、平均cos.282386/.283367、负cos均0/16，通过预定可行性。完整双梯度1048576值与IQ闭式公式独立复算一致。仅支持独立scratchE20 matched RF辅助损失试验，不作G参数方向、TX身份或目标性能成功结论；默认DURA和逐景3pp/clean/效率目标未达成。


源解析RF辅助E20完成4940更新、七景和126组独立复算。固定alpha=0.009527944261805803，解析变换2.52m输入/参数不变，单G前向，0训练好F/直接TX/V/target/恢复器祖先训练。clean84.661310%，lowurban66.466071%；六景3pp=False/clean保持=False/R17逐景1pp=False。G流程1093.996秒，全流程1428.170秒；其他任务后续新增造成同卡4PID，未追加本任务，默认保持。


同原F50 source RX几何：复用63000缓存，fit62488/holdout512，0G读取/神经更新/TX/V/target。连续后验坐标条件均值，不硬伪TX；headrank5保护后共享RX占条件RX均值差能量32.271767%，留一坐标cos.475377/30条全正，fit-probecos.976573，留一RX捕获均值66.073853%/5RX全>10%。完整10.08m特征独立QR/lstsq复算57统计通过，CPU2/0.755323秒，仅几何资格PASS。头选择保持系代数构造，不是G/身份/目标效果。下一先固定.25规范特征的source梯度/实质变化校准，不直接开训练，默认和3pp/clean/效率目标保持。


头保护RX特征梯度校准完成：同noVar源512×缓存01/20，G/F不变，0更新/直接TX/V/target；三类1572864梯度值独立重算一致，CPU2/4.062794秒。新旧Fcos梯度差中位42.3231%/47.8118%，IQcos.045729/.042799、负批次均0、压力.239969/.266674，两视图PASS；唯一lossweight=0.019528750294704578，特征收缩固定.25，不加大。只支持一个独立scratchE20匹配F-guided新参考试验，不能称参数方向或目标成功；默认/3pp/clean/训练效率目标保持。


头保护RX特征参考单阶段试验已发布：fb795cebeb21d3cbd97aad05aecfa1afff441018，CPUcontroller956189/worker957861/GPU7，真实F-G梯度/冻结/重载/原头与source参考smoke通过；准入已有2PID/free22382MiB，后态共4PID，仅own1，不改别人。G仍scratchE20/4940、原IQ缓存/.25feature收缩/源校准weight.019528750294704578，自动全七景；F源50轮知识继承，非F-free，未声称正式收益，默认保持。


头保护RX特征参考E20完成4940更新和全七景/126组独立复算，固定.25收缩/weight.019528750294704578，1.26m源参考访问/F不变，G祖先0/直接TX-V-target训练0，非F-free。clean84.866667%，lowurban68.417857%；六景3pp=False/clean保持=True/R17逐景1pp=True。G流程1156.041秒，全流程1580.748秒；参考及权重同时改变，不作孤立RX因果结论，默认保持。


用户E30/IQ特征增强方向已落实代码489f339757b7b3fb9214b3341d780ec98bb14e89：scratch单阶段30/7410，新增1030参数受限shape修正、F梯度仅增强/独立clip，真实2Ls smoke与9项检查/一次审查PASS，controller1018738，状态RUNNING，默认不变。并行既有IQ全量分析已完成28单元/77.055秒，0新G/F；default lowurban NMSE6.9580%/energyfit93.0420%与识别68.8899%/clean保留81.3568%分开，尾部24点占该景误差19.56%（高郊60.92%）。不能从均值/幅相oracle认证恢复极限，完整分析见20261031-analysis-DURA-recoverability-s2026092701-r01/analysis.md。新E30性能与效率、六景3pp/clean验收仍未完成。


已知source action＋独立Gaussian先验参照完成：fit62488排除probe512/缓存01与20，CPU18.357643秒，G0/F1026/更新0/直接TX-V-target0；288统计独立复核。相对default源NMSE约降9.75%/16.87%，F-clean选择净−6/0（纠9损15、纠8损8），lowurban小组−2/+1；不是目标TX效果或恢复上界。算子和先验同时改变，AGC条件噪声/非Gaussian假设不满足，不能孤立归因或认证极限；本参照不部署/不触发新训练，健康E30保持。见20261031-diagnostic-DURA-known-action-Gaussian-s2026092701-r01/analysis.md。


E30单阶段IQ增强已完整30/7410及双读出七景，252组与全量成对IQ约束独立复算。physical/enhanced lowurban66.707143/66.039881%，clean84.842262/84.455952%，增强七景全负；low净−1121、clean净−649，clean类别1−7.6536pp。两个读出0/6达3pp，默认保持。G流程1611.311秒/26.855分钟，全流程2058.587秒/34.310分钟；G+F_B1默认4.6481ms、增强5.3925ms（共享负载）。实际930新增/267881总参数纠正原手算多100；未热改旧489程序/权重，未来本地配置已改且计数测试9PASS。该headfactor2受限增强未成功，不据目标再调幅度/权重；下一来源核查可研究原R17合规IQ教师如何帮助从零单阶段收敛，F来源权限保持。见20261031-phase1-DURA-IQenhance-E30-s2026092701-r01/analysis.md。


原R17 IQ教师来源与路径已核实：原cd977/source7/63/30，coarse/action/prior各20历史组件轮次，无TX/F/target优化，sourceV标签选择先于target。CPU13.514224秒/512×2×4路径/208组2288统计复核。coarse IQ NMSE2.5020/2.3306%、cleanF一致467/487，比noVar coarse13.2109/12.7773%和257/272好；R17 stage1/2却不统一优于coarse，默认DURA中间IQ也不宜强认为clean。下一候选固定从零单阶段30轮，coarse/action/prior始终联合，前10轮原R17coarseIQ提示权重1→.1、后20轮0，基础noVar4项保持，无F/TX/V/target训练、无权重复制；计划teacherIQ630k缓存一次/完整成本后实测，仍待实现与实际发布。所有新版本正式七景保持，默认和逐景3pp目标ACTIVE。见20261031-diagnostic-DURA-R17-source-IQpaths-s2026092701-r01/analysis.md。


早期coarse IQ教师单阶段E30已实际发布：代码64b8d453e4ead7b8242163bef67dc3db964d5a1f，controller1087077/worker1087169/GPU4存活、CWD/argv/CUDA/GPU/日志核实。7checks/ONEP0P1及source2Ls冻结R17/学生梯度/strictreload/F仅smoke PASS。准入已有1PID/free22883MiB，后态2PID，无其他owner修改。630k教师IQ缓存132.756142秒/1.29024GB numeric，使用coarse1305958参数/历史20轮，R17包4665073参数/60组件轮次；教师更新0且不变。G无权重复制，已E1/30，训练F/TX/V/target0，前10轮hint1→.1再0、30轮全联合、原266951图不变；自动七景与完整实际成本待结束。默认/逐景3pp/clean目标ACTIVE。


取消 priorIQ 辅助拟合单项 E30 对照 VERIFIED/RUNNING：eb46cab8ef9c253c193a10db1724ed9835ed2cf9，controller1104507/worker1104599/GPU6，实际2Ls smoke PASS，CWD/argv/CUDA/GPU/loggrowth核实；准入1PID/free19245MiB，后态2PID，本任务1worker、他人不变。G从零30/7410，保留266951图/2prior4CG27FFT，仅prioraux.1→0，方差aux0延续noVar，IQ/STFT/action不变，F/TX/教师/V/target训练0。源阶段差异非因果证明；固定final30全七景，无性能停止。旧coarsehint已结束；评分按真实旧句柄退出串行，默认与3pp/clean目标保持ACTIVE。


早期coarseIQhint E30完整结案：7410更新/L189000U1701000/教师缓存630k且不变/学生不继承权重，训练F/TX/V/target0。七景固定后truth-last，126组独立复算，clean84.958333/lowurban65.366667；全部六景低于默认与E30physical，R17逐景1pp也未通过，3pp目标未完成。缓存132.756s/循环989.817s/G流程1123.106s/全流程1303.066s，266951图不变/B1G+F4.633238ms，完整30曲线与成本已归档。单阶段效率可控但质量未达标，默认保持；不依据负结果热改正在运行的noPriorAux。


E30IQ增强既有14组冻结logits的事后概率分析完成（2b051a0c3），新增模型前向/更新0，2项算术检查通过/配对计数与原独立评分一致。七景maxP上升而trueP/accuracy下降，NLL/Brier/ECE全变差；lowurbanmaxP88.4598→88.8727%、trueP65.1644→64.6837%，32017条增强后错但更有把握（包含旧错误，不是新增数），实际净−1121。均值true-margin亦上升，不能替代真身份验收；不推定对抗扰动或任何增强无效。探索性评分后分析不回流训练或调参，详见原run概率表与JSON；恢复极限仍未知，默认/3pp目标保持。


源频率oracle fb1427f3c 完成，CPU14.557697s/F14848+2smoke/0G-update-TX-V-target/F不变，728组16744标量复算。弱Q4complex：noVar478→479/496→495，default486→484/498→498，不稳定；Q4phase/magnitude同样无稳定收益。强Q1complex：noVar494/507、default501/506（相对native+16/+11/+15/+8）。不因Q4相对误差大直接加权，不把选择一致当TXaccuracy；下一源检查分离Q1/Q2幅/相，非部署/非认证极限/不自动训练，默认与完整3pp/clean/效率目标保持。


noPriorAux E30 eb46cab8e完整结束：7410步/L189000U1701000/0G祖先教师-F-TX-V-target训练，priorloss0而网络/4CG保留；全30loss重算及七景独立126组一致。clean84.676786（净+1raw）、lowurban66.020238，七景均低于E30physical、六景均低于默认，3pp/R17每景1pp未达。循环1001.619769s/全流程1183.545899s，实际前序进程检查.055396s，无他人干预。source/model准备未单独计时，不填0。默认保持，目标ACTIVE；源跨模型prior误差观察不足以证明删该辅助项有效。


强频点幅相source诊断779f1e91e完成，CPU7.239624s/F8704+2/0G-update-TX-V-target/F不变，416组9568标量复算。noVar Q1phase选择478→488/496→502、移除约30%IQ误差，缓存01的mid场景−1；Q1幅度+13/+3，defaultQ1phase+6/+3且一景−1。不是所有场景保证，不冒充旧globalphase资格。可独立检验Q1精确相位MSE分量权重1→2，保留IQ/幅度/先验/action与固定E30/正式七景；部署无clean/mask，无目标反馈，当前oracle不能部署/不是真实TXaccuracy或极限。默认与目标ACTIVE。


strongPhase E30已实际发布84c5f08b3f334781cad4cb96f9e22bd33d9000eb，controller1134232/worker1134322/GPU4存活，真实2Ls梯度/strictreload/F仅smoke PASS，CWD/argv/CUDA/GPU/loggrowth核实；8checks/ONEP0P1/preflight/archiveSHA/compile PASS，准入1PID/free22883MiB/后态2PID，本任务1worker。他人不变。Gscratch30/7410，noVar四loss+Q1精确phasecomponent1，使该项1→2；图266951/2prior4CG27FFT不变/训练额外2FFT/部署额外0，F/TX/教师/V/target优化0，F-informed源机制选择披露。已E1/30，phase项非零/加权loss一致，固定final30全七景，不按源误差或目标反馈停训/重跑；默认与目标ACTIVE。


strongPhase E30 84c5f08b3完整结案：7410步/30loss复算/原168000x7及独立126组；clean84.436905、lowurban67.311310。较同预算physical midU+0.347619pp(+584)/lowU+0.604167pp(+1015)，其余四景及clean下降，rawclean净−402；六景仍全部低于默认，3pp/clean未通过，不晋级。训练1002.686482s/全流程1187.861899s，266951图/2prior4CG27FFT，F-TX-teacher-V-target优化0。保留混合结果不按目标调mask/权重；宽coarse容量对照在本成绩读取前按旧source拟合/参数盘点固定，不混本phase损失。默认与目标ACTIVE。


wideCoarse E30实际发布a55a568fc06cc3047647e8201d7fe5ca16f872b2：controller1152012/worker1152105/GPU6，source2Ls新coarse非零梯度/1466771严格重载/原F noquery PASS；11checks/ONEP0P1/preflight/archiveSHA/compile与PID-CWD-argv-CUDA-GPU-loggrowth核实，准入1PID/free19245MiB/后态2PID，他人保持。完整coarse1305958+原compactaction79477/prior81336，总1466771/ConvLinearMAC116098088/2prior4CG27FFT；较原R17少68.558%参数、比default大449.453%，实际时延/全训练成本待测。原64b8教师缓存只读复用，当前加载/前向0/handle核对0.010655s，历史630k生产132.756142s/1.29GB/coarse20-bundle60另列。当前E3/30，IQhint访问189000次/加权loss一致，F/TX/V/target优化0、无教师权重继承；同旧hint配方只改容量，不混phase。固定30后全七景，默认/3pp/clean/效率目标ACTIVE。


原R17最终native IQ source缓存预制已发布6848d46e95741082e2783a525e6d0d376e8ff7c6：controller1174803/worker1174895/GPU4，7checks/ONEP0P1/preflight/archiveSHA/compile与真实2Ls原G noquery冻结state PASS；PID/CWD/argv/CUDA/GPU/loggrowth独立核实，准入1PID/free22883MiB/后态2，与wideCoarseGPU6合计2本任务worker。他人保持。原teacher选在target前/source7-63-30/coarse-action-prior各20历史轮次；当前生产0训练/F/TX/V/target，20x63000最终第三stage IQ/2.58048GB，已2视图/126000输入，完成与全成本待实测。后续若试验已固定紧凑scratchE30/noVar4base+非clean源native教师IQMSE权重1，sourceclean教师项0/mask仅训练，先另登记不自动由producer启动、不按wide目标结果调权重。默认和完整3pp/clean/效率目标ACTIVE。
