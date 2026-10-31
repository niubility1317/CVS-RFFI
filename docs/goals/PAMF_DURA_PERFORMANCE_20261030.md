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


宽coarse E30完整结案：a55a568，1466771参数/116098088MAC，30/7410/L189000U1701000/源IQ教师复用630k/新teacher加载前向0。全部七景预测后truth-last，独立126组一致；clean84.863690%，lowurban66.073810%，六景均低于默认与E30physical，0/6通过3pp，原R17逐景1pp未通过。源coarse NMSE .02399570/final .01776716低于匹配紧凑hint仍未改善身份。训练2091.901s/G流程2092.824s/全流程2451.928s，B1G+F默认4.671210ms/本版5.894090ms，历史缓存132.756s/60组件轮次另计；默认保持、目标ACTIVE。最终IQ教师候选的非clean权重1/clean0在本版正式分数读取前已以945114bf4固定，不由这些负结果调权重。


原R17最终IQ源缓存生产完整完成：code6848，20×63000/1260000 GPU输入+CPU smoke2/2.58048GB，完整20数组finite/shape/dtype/ID-role独立核实、teacher不变。新增训练/F/TX/V/target/updates0，循环972.193282s（16.203min）/全制备976.382917s/peakallocated75658240bytes。历史R17coarse/action/prior各20、合计60组件轮次保留；源缓存可复用但不当免费知识或新样本。状态ANALYZED仅SOURCE_CACHE_ONLY，不是新G正式收益，默认保持/目标ACTIVE。


最终IQ教师学生已独立登记并启动：20261031-phase1-DURA-nativeIQhint-E30-s2026092701-r01，执行728129ece6855743fc4939193ab2bc3b7693b9aa，controller1192588/worker1192680/GPU6，准入已有1PID/free19245MiB/后态2PID，真实2Ls目标梯度/严格重载/原F noquerysmoke PASS，8checks与唯一P0P1 PASS。紧凑G266951 scratch单阶段E30/7410；noVar4base+非clean最终原R17IQ MSE权重1/clean教师0，已在945114bf4且wide正式分数读取前固定。当前E3、teacherref189000/active160614，实时teacher/F/TX优化0，source缓存复用句柄.011453秒而历史972.193282秒/1.26m/2.58048GB和60组件轮次另计。后续固定final30自动七景/truth-last；目前不是正式收益，默认保持/目标ACTIVE。


源模块梯度诊断结案：20261031-diagnostic-DURA-source-module-gradients-s2026092701-r01/7f7cfd68，旧noVarcf40源final20/4940/ancestor0，512源×缓存01/20/32B32/1024G+2Ls，CPU2/0GPU，0F/TX/V/target/updates，model全state不变/p.gradNone。384组norm-dot-cos-ratio/全32批汇总独立复算一致；coarseclean与finalIQ在coarse18/32负cos、均值-.038594/范数比中位12.068895，actionprobe2/32负cos、均值.364921；均未达到事先24/32判据。不放宽/不启动梯度分离训练，局部sourcegeometry不当身份或完整轨迹证据。循环28.349202s，4tests/唯一P0P1/实际PID-CWD-argv-CUDA空-loggrowth和自然结束已核实，默认保持/目标ACTIVE。


最终IQ教师学生E30完整结案：20261031-phase1-DURA-nativeIQhint-E30-s2026092701-r01/728129ec，7410更新/L189000U1701000/teacherref1890000/active1606874/新teacher前向加载0，Gscratch/ancestor0/训练F-TX-V-target0；完整30日志和逐cache mask独立复算。全7×168000预测后truth-last/126组独立一致，clean84.823810%/lowurban67.375595%，3pp0/6/clean保持True/R17逐景1ppFalse。训练1056.551s/G流程1057.366s/全流程1241.367s；266951参数/2prior4CG27FFT，历史缓存972.193s/1.26m/2.58048GB+60组件轮次另计。默认保持，完整目标ACTIVE；不回流调权重、epoch或场景。


30份源信道抽样对照已发布：20261031-phase1-DURA-freshViews-E30-s2026092701-r01，执行32a32998d3c76105b017a0905300e8edbce64e16，紧凑G266951 scratch单阶段E30/7410/noVar四基础项与model/data/probe RNG不变。已知原缓存只有20份、预算30使后10轮重复01..10，故仅新增原native namespace21..30，同AUG2026101912/RX2026101911/六practical+15%clean，原20只读链接，物理ID/roles/clean/Vpanel不变。新10共630000模拟访问/4.1958GB数值数据，源制备成本实际单列、无神经teacher/F/TX/V/target优化。4checks/唯一P0P1 PASS、真实2Ls G梯度/严格重载/原F noquerysmoke PASS、一次preflight/archive比较/compile后实际controller1218619/CWD-argv-CUDA空/8CPU子进程/loggrowth核实。当前SOURCE_DATA_PREPARATION，已4/10视图、GPU训练尚未开始；最近52.276848s/63k/419580000bytes/replay2.416e-16。全10完成才GPU6 own1/max3PID/free12GB/goal2，固定final30后自动全七景168000/truth-last/profile，其他owner保持。默认不变/完整目标ACTIVE。


SOURCE_COMPARISON_CORRECTION：N607 Torch2.1纯CPU工厂trace发现旧IQenhance physical虽643个G初state与纯G完全一致，其enhanced factory torch.manual_seed2026103107随后只恢复CPU RNG，额外重设CUDA探针；compact/fresh实际CUDAprobe2026092701。旧489代码与修复前现代码仅参数计数元数据不同，该事实适用于历史已执行旧增强对照。0CUDAinit/G-Fforward/checkpoint/updates/TX-V-target。旧七景真实指标和预算比较保留，但所有历史vs该physical比较不再称严格同探针单机制配对；不回流改旧权重、预测或选epoch。当前代码修复CPU head初始化用default_generator避免CUDA副作用，头和G初值保持。新独立repeatViews E30纯G20+repeat10作为fresh30的主配对对照，目标访问前补登记，网络/损失/model-CUDA-npseed/LR预算匹配，仅source21..30抽样不同；新GPU4+fresh6总2，评价实际等fresh worker/scorer退出，完整自动七景。fresh健康32a进程保持，默认与目标ACTIVE。


新源抽样/严格纯G配对均已实际GPU训练：fresh32a worker1224405/GPU6 E27/30、sourceview27；repeat8cf worker1232279/GPU4 E6/30，均真实2Ls梯度/strictreload/Fsmoke PASS、实际CWD-argv-CUDA-GPUPID/loggrowth/合法准入核实，合计2ownworker/每卡1。fresh10新视图630000模拟访问/4.1958GB、CPU8制备530.740140s，旧20只读复用/物理ID-roles-Vpanel保持，真实读view21..30。两组scratch30/7410/noVar4基础项、model/CUDAprobe2026092701/npshuffle2026103004，训练F/TX/neuralteacher/initcheckpoint0；fresh与repeat各自都固定final30后全七景truth-last。N607修复后CPU工厂trace再验证CUDA调用只余2026092701/643G初state一致/0CUDAinit-GFforward-checkpoint-TX-V-target，旧archive不改。数值轨迹不承诺跨GPU逐位相同，旧physical比较限制已全局注明；默认保持/完整目标ACTIVE。


Fresh30本行正式结案：32a/30轮7410/L189000U1701000，实际sourceview1..30而非重复10，G训练F/TX/neuralteacher/initcheckpoint0。全7×168000固定后truth-last、126组独立一致，clean84.461905（raw−360条/−.214286pp）、lowurban66.973810%，七景均低于默认、0/6通过3pp；原R17逐景最多下降1pp的历史容忍通过，但当前优化目标未通过。额外源制备530.740140s/4.1958GB、G训练976.251282s、全分析1691.498856s；缓存可复用但成本不隐藏。独立同CUDAprobe纯G repeatViews8cf对照仍健康训练，当前matched_control_delta=null，不以旧physical探针混杂比较提前归因；两行不因本行target结果调参或重跑，默认保持/目标ACTIVE。


纯G源抽样配对矩阵完整结案：repeat8cf E30/7410/source01..20+repeat10，clean84.630357/lowurban67.372619，30完整日志/126单行记录复算；fresh32a全30独立抽样同预算同G初值-CUDAprobe-npshuffle。全14预测核对后truth-last，126配对净纠错独立一致：fresh在5/6星地提高（high+1.317857/mid+1.099405/lowsub+.572024/highurban+1.132143/midurban+.043452），lowurban−.398810/−670、clean−.168452/−283；两组0/6通过3pp且clean低于raw，不替换默认。学生循环fresh976.251s/repeat981.684s，source新增530.740s/4.1958GB另计，全分析1691.499s/1162.601s；source第一20轮曲线IQ最大绝对差0.002244012，跨GPU不承诺逐位轨迹，单pair不称seed均值或统计显著。旧physical CUDAseed混杂与修复已披露，所有旧产物保持。当前两worker均自然结束，默认保持/目标ACTIVE。


完整两阶段源context诊断已结束：20261031-diagnostic-DURA-source-context-oracles-s2026092701-r01/97da8a29，主对照纯repeat源E30/7410、512源128L384U×clean/01/20，CPU36.665秒/0GPU/0优化TX-V-target。真实算子替换使主对照NMSE降低约11%，但Fclean选择−2/−5、clean−1；rank32真实算子表示误差很小而预测平滑后IQ略差、F选择+1/0。两项均未过事前固定源门槛，不据此启动新G或放宽。default换clean粗恢复反而恶化，提示潜在表示共同适配；该反事实不是可部署方案、真实TX准确率或恢复上限。六份完整NPZ/480分组case独立复算，G/F全state不变、所有计数闭合。见automation_reports/CV-SincNet/20261031-diagnostic-DURA-source-context-oracles-s2026092701-r01/analysis.md；默认保持/目标ACTIVE。


sourceclean局部敏感度诊断20261031-diagnostic-DURA-source-cleanTangent-s2026092701-r01/ff6926fa已完成：固定repeatE30/source512×01/20，head-visible5D Jclean二次型与exact特征IQ梯度mean cos.643863/.661080，普通IQ约.071；493/484条非负、五RX均正，全部源门槛通过。alpha按预定.25/maxmedian normratio固定0.0033130025035573727。局部线性近似误差中位仍约.53，不称globallyexact或目标收益；仅支持另登记scratchsingleE30，不自动训练。CPU5.796s/cache3.691s，F1536+6smoke/0G新前向/0TX-V-target-updates、Fstate不变，2048条几何及IQ-J算术独立复算。全源缓存估计645.12MB/30轮读19.3536GB，实耗待真实训练；原F50知识/成本披露，默认保持/目标ACTIVE。


独立新G 20261031-phase1-DURA-cleanTangent-E30-s2026092701-r01 已真实RUNNING：代码7dd799ebb294ffe8711f530872e4922f100ed498/release20261031-DURA-cleanTangent-E30-r01，sourcequalified固定alpha.0033130025035573727，原noVar四loss+clean局部身份敏感度二次型，266951参数/两prior四CG，scratchsinglejoint30/7410，不直接读TX/V/target。源Fclean63000×5VJP缓存645.12MB实际30.008929秒，完整独立readback finite/IDs/basis通过，Fstate不变后释放，学生训练0Fforward；F50知识/成本明确继承。7tests/唯一P0P1/真实2Ls G-F梯度与strictreload PASS，一次archiveSHA/compile、controller1285659/worker1285750/GPU4/CWD-argv-CUDA-loggrowth已核实，至少E2并实际执行新loss。全30后自动source1800×7/target168000×7/allviews固定后truth-last；当前无正式收益结论，默认保持/完整目标ACTIVE。登记event已成功写入，之后index刷新遇另一owner短暂build锁；核实event唯一且锁自然消失后仅重建managed索引，没有重复record、重复launch或操作其他进程。


clean局部敏感度G 20261031-phase1-DURA-cleanTangent-E30-s2026092701-r01/7dd799eb 已完整结案：scratchsingle30/7410/L189000U1701000、F50 sourceJ63000/315000VJP/645.12MB缓存30.008929s，学生训练F0/直接TX-V-target0，完整30曲线/全部loss与缓存计数核对通过。全7×168000固定后truth-last，126正式及126pureG配对分组独立一致，worker1285750/controller1285659自然结束。相对pureG四星地+0.526786至+1.577976pp，中/低城区−0.446429/−0.747619pp、clean−.043452pp；七景均低于默认，clean84.586905/lowurban66.625，3pp0/6/clean严格保持False/R17逐景1ppFalse/旧lowurban70False。源V低城区也−2.5pp，不能仅归因目标RX；局部梯度代理不等于完整身份保真。训练1022.804362s/含准备G1053.873159s/全分析1235.243895s，266951参数/25.068072m ConvLinearMAC/2prior4CG27FFT，F上游50轮成本披露；同质量效率未达。默认保持/目标ACTIVE，不回调alpha/选epoch/重测。


源特征度量压力诊断20261031-diagnostic-DURA-source-boundedTangent-s2026092701-r01/aa284ea1完整完成：全63000 sourcecleanJ/纯repeatE30 source512×01/20，CPU7.343s/因子4.855s645.12MB/新增F1024+2/VJP1024/0G-TX-V-target-updates/F不变。98.55%源因子的最强相对方向谱值>1，不是实际98.55%梯度超界；当前实际guide/IQ p95为1.610/1.349、中位.25。固定谱cap1满足输出loss/梯度上界，但cos.643863→.575031/.661082→.587405，下降超过预定.02，两资格FAIL，不放宽/不训练该G。全63000谱/trace/finite/ID和2048条几何独立复算一致；强度与方向需同时处理，不把上界当身份收益，后续如研究方向保持的梯度平衡须独立固定验证。默认保持/完整目标ACTIVE。


下一独立候选的数学原型：保留原方向的逐样本步内梯度平衡，代码 `experiments/pamf_dura_efficiency/tangent_balance.py`。原alpha不变，令s=min(1,||g_IQ||/||g_feature||)，每步把s视为固定权重（detach），附加项为alpha*s*原tangentLoss。这样在当前IQ输出点保留原特征梯度方向，未过强样本完全不变，并由Cauchy不等式同时得到附加loss≤IQNMSE与附加IQ梯度范数≤IQ梯度范数。它是动态重加权的步内surrogate，不宣称是对随IQ重算权重的全局固定标量能量求完整导数；也不保证G参数梯度范数。4项针对任意误差、方向/参考隔离、未限幅恒等、零/非法边界的测试通过。

该原型不是刚才失败的静态谱裁剪方案，不放宽其已失败判据。目前没有新模型训练、正式预测或性能结论。后续若进入实验，单独登记source方向/压力校验、scratch单阶段30/7410及完整七景truth-last；原F50的既有sourceclean J缓存可只读复用，学生训练不需要F前反向，F源知识和历史缓存成本继续披露。目标仍为逐六景距rawclean≤3pp、clean保持及同等质量效率，默认DURA保持。


保留方向的动态平衡G 20261031-phase1-DURA-balancedTangent-E30-s2026092701-r01 已真实RUNNING，执行75da21407e7a6790f45e4a8ac4bcc2ea5b6d120e/release20261031-DURA-balancedTangent-E30-r01。Gscratchsingle30/7410/266951/两prior四CG，原alpha不变，步内detach scalar缩放过强IQ附加梯度，非全局固定能量完整导数/非参数梯度上界。rawsourceJ63000/645.12MB只读复用，核对.081749s，新F前向/VJP/cache存储0，F源50轮及旧30.009s缓存成本披露；未读producerG/target。7tests/唯一P0P1/实际2LsG三模块梯度-实际IQ界-F冻结-严格重载通过，actualcontroller1326970/worker1327059/GPU4/CWD-argv-CUDA-loggrowth核实，至少E2全前缀检查通过/压力max1.000000119。source构造保留旧方向不是泛化保证，不放宽失败静态谱门槛；自动满30并七景全部预测后truth-last。默认不变/六景3pp-clean-同质量效率目标ACTIVE。


## 逐条平衡局部身份约束：完整 E30 与七景

20261031-phase1-DURA-balancedTangent-E30-s2026092701-r01/75da2140。VERIFIED / ANALYZED。单阶段 scratch E30/7410，L189000/U1701000；30 轮完整 loss/曝光/缓存视图/逐批次压力上界核对通过。复用 source J，缓存复用与 G 训练期间新增 F 前向/VJP/缓存均为 0（真实模型检查另有 2 条 F 输入，冻结后预测仍正常执行 F）；继承 F50 监督知识，仍为 F-guided。七景×168000 全部预测固定后 truth-last，126 正式与252 双对照配对分组独立复算。clean 84.814286%，low urban 67.031548%；低城区相对未平衡版 +0.406548、纯 G -0.341071、默认 DURA -1.858333 个百分点。六景3pp通过 0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练 998.836772 秒，含源准备 999.795406 秒，全量分析 1180.638900 秒；266951 参数/25.068072m ConvLinear MAC/2prior4CG27FFT，默认保持。 当前完整目标继续 ACTIVE；单 seed 已暴露工程基准，不称新盲测或物理恢复极限。完整源/正式/TX/RX/day/净纠错与成本见该 run 的 analysis.md/results.json。没有采用失败的静态谱因子，不依据目标结果回调 alpha 或调类别权重。


源噪声稳定性诊断 20261031-diagnostic-DURA-source-noiseStability-s2026092701-r01/7475398c 完整结束：固定pureG repeatE30/defaultR69、source512×cache01/20，按缓存标量方差作同一±各向同性扰动（不是原生噪声重抽），clean不扰动；两缓存增益比1.111890/1.005703均低于事先1.25，不能明确定位为单阶段特异噪声放大，不放宽、不启动该训练。CPU20.558967秒/G-F各6144+4smoke/0TX-V-target-updates，22528逐条指标和108分组独立复算，PID1349938自然结束；默认与完整目标ACTIVE。


源四相位IQ合成诊断 20261031-diagnostic-DURA-source-quarterOrbit-s2026092701-r01/d2f2dc59 完整完成：default主对象/pureG描述，source512×clean/01/20，精确四直角旋转后G输出逆旋转再IQ平均，native重放0/C4代数等变误差1.192e-7，非符号/顺序实现错误。默认缓存NMSE约增至21.935/20.624倍、clean-F486→352/498→367、fullclean512→509，pureG同样失败；原生G在腐蚀输入的旋转路由转回后仍严重不齐，直接平均抵消。固定门槛FAIL不放宽/不改主/不发布runtime，不取消已登记正式矩阵。CPU28.622022s/12288G+36smoke/6144F+8smoke/0TX-V-target-updates，27648逐条指标和132分组独立复算，PID1357440自然结束。4倍G成本/8prior16CG108FFT，未测GPU时延不称加速；默认与完整目标ACTIVE。


源软坐标跨RX参考 20261031-diagnostic-DURA-source-softRXreference-s2026092701-r01/210e2380 几何资格PASS：sourceF63000缓存，fit62488（L6172/U56316）排除512；按RX/day软6坐标均值，其他4RX×3day等权、beta.25混原unitfeature。90单元最小有效N684.292、512/512原头判断保持，head可见RX关联离散度比.523619、软坐标间离散度比1.014648。不同于旧headnull RXcanonical；softF输出不是真TX/硬伪标签，无另训头。CPU统计.212823s/新F骨干2/head-only63512+2smoke/0G-TX-V-target-updates，378000软概率/90cell/30reference/512probe/11groups独立复算。只支持另登记IQ梯度/可达性校准，未启动新G或正式预测；F历史50/11100及40.32MB缓存3.782s披露。实际PID1374175/CWD/env/log与launchargv/commit/完整产物核对，直接proc cmdline短暂为空保留UNKNOWN，未重复启动；默认和完整目标ACTIVE。


软参考源IQ校准 20261031-diagnostic-DURA-source-softReferenceIQ-s2026092701-r01/cf018bed 全部PASS：beta.25/alpha.0033130025035573727/原rawJ不改；source512 minnorm位移IQNMSEmean.003394%/P95.010540%，FP32实际线性残差max6.492e-6，真实F相对偏移误差median.143853/p95.495730，512clean选择保持。纯G8cf源01/20 prototype/exact meancos.656546/.669879（IQ.0474/.0488，无偏移metric对同prototype精确梯度.4670/.4670），非负501/491、每RX正、实际梯度压力max1.000000358。BLOCKED/unlocked方向仍弱、线性误差中位约.5；guide值超IQ13/17，不能复用旧值上界。CPU3.156706s/F1536+2/inputVJP1024+2/0G-新完整J-TX-V-target-updates，512可达性/1024几何/2097152闭式分量/52分组独立复算；PID1392918自然结束。只支持另登记scratchjoint30/7410+完整七景，未自动训练/新正式收益，默认与完整目标ACTIVE。


## 跨RX软参考30轮版本已启动

20261031-phase1-DURA-softReference-E30-s2026092701-r01。VERIFIED / RUNNING。执行 b6136c53106c9e8e6273a606262d5152a4e08d23 已提交、推送并独立核对远端OID；一次N607 preflight/单archive传输SHA/remotecompile通过。真实2Ls G三模块梯度、实际附加IQ梯度cap、F冻结/原默认头/严格重载通过。实际controller 1410473/worker 1410565/GPU 4 的CWD、完整命令、CUDA、GPU和日志增长核实，旧own句柄全部消失，其他owner保持。至少E1完整前缀预算/加权loss/曝光/逐步cap核对通过，最大压力1.000000119。

新增63000×5 FP32参考偏移缓存1.26MB，生产0.566940秒；独立复算63000条/315000分量、30个otherRX软中心与512校准条，最大偏移差0.0。新F前向/VJP0；原rawJ645.12MB只读复用核对0.075864秒，历史F50/11100、rawJ30.008929秒、F1603.782270秒、参考几何0.212823秒及IQ校准3.156706秒单列。训练中不加载F。当前正式成绩待产出，不将源域校准当作性能验收；满30后自动完成全七景预测/独立truth-last。默认DURA保持，目标ACTIVE。


## 跨RX软参考：完整E30与七景

20261031-phase1-DURA-softReference-E30-s2026092701-r01/b6136c531。VERIFIED / ANALYZED。G从零单阶段30轮/7410步，L189000/U1701000；30轮完整loss/曝光/缓存视图/逐批次压力检查通过。复用原source J和F160，新增固定soft跨RX/day参考偏移1.26MB/0.566940秒，新F前向/VJP0、训练F加载0；继承F50监督知识，属于F-guided。七景×168000全部预测固定后truth-last，126正式、105源V与252双对照配对分组独立复算，14个源/正式IQ场景误差重新核对。clean 84.888095%，low urban 68.655357%；低城区较实例特征约束 +1.623810、纯G +1.282738、默认DURA -0.234524 个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=True，旧低城区70%=False。G训练990.811749秒，含worker源准备991.736709秒，controller至全量分析1172.872900秒。266951参数/25.068072m ConvLinear MAC/2prior4CG27FFT，默认DURA保持；完整目标仍ACTIVE。 当前证据限单seed、已暴露工程基准，不称新盲测或恢复极限。完整源/正式/TX/RX/day/净纠错与全成本见本run的analysis.md/results.json；源校准通过不等于身份性能通过，不因目标结果改变固定参考或调类别权重。


## 保留源稳定正交特征的10维校准

20261031-diagnostic-DURA-source-softReference10IQ-s2026092701-r01/516ea8e3。VERIFIED / ANALYZED，仅SOURCE_ONLY，科学判定FAIL、技术执行通过。源512原head5/J5保持，追加fit62488/90RX-day软均值的秩5正交方向；参考位移覆盖19.4824%→46.6467%、RX离散度比0.527648、软坐标间离散度比1.019489。FP32线性残差max1.124e-5、512clean判决保持、IQ位移均值0.012443%均通过；真实F偏移误差P95=1.147242>1，新增5维独立cos=0.398134/0.382603<0.5，未通过固定资格。CPU2/PID1465120/CWD-argv-env-loggrowth/真实2Ls检查核实，循环8.452086秒自然结束（新增512×5J 3.908956秒）；F2048+2smoke/VJP4608+2smoke、0G/TX-V-target/更新。512最小IQ位移、combined/aux共2048方向、2621440闭式分量/52分组独立复算。没有新G训练、完整源J或正式预测，不搜索维数/beta/alpha/阈值，默认和完整目标保持。 原5维软参考E30七景正收益保持其原身份；本次只是新source-only诊断，没有取消任何已登记的新G正式矩阵。后续优先研究精确非线性约束的有效性和实际计算成本，不通过放宽失败资格推进。当前六景3pp、clean保持、同质量训练效率目标继续ACTIVE。


## 精确非线性源域参考可达性

20261031-diagnostic-DURA-source-nonlinearReference-s2026092701-r01/b621e3d5。VERIFIED / ANALYZED，仅SOURCE_ONLY。固定原source10参考/beta.25/alpha不变，512源128L384U、8次GaussNewton重线性化/半步0.5。实际F相对偏移误差median 0.010535806/P95 0.205998719，IQ位移NMSEmean 0.013526%/P95 0.027196%，512原clean判决保持，固定可达性PASS。真实2Ls随机G/原F精确guide梯度注入、IQ cap、三模块梯度及F/G状态检查通过；G0优化/0保存权重，未使用失败clean-J或旧G误差。CPU2/PID1501100/CWD-argv-env-loggrowth已核实，主循环54.974718秒自然结束；F4608/VJP40960，加smokeF4/VJP2/G2。4096闭式IQ更新/2097152分量/4608非线性观测几何及最终判定独立复算，0TX/V/target/完整源J/新G训练。只支持另登记GPU步开销测量，不自动训练、不称目标识别收益或硬件细节恢复；旧10维局部近似FAIL保持，默认和完整目标保持。 后续另登记GPU步开销测量，若可行再登记Gscratchsinglejoint30/7410和全七景truth-last；本次未启动这些阶段。六景逐个距rawclean≤3pp、clean保持、同质量训练效率目标继续ACTIVE。


## 精确非线性源约束GPU步成本

20261031-diagnostic-DURA-exactReference-stepCost-s2026092701-r01/6dd3eabf。VERIFIED / ANALYZED，仅SOURCE_ONLY。GPU4原cached5→exact10，cache01中位110.329→138.148ms、cache20中位115.832→144.347ms，中位比1.252144/1.246179、配对中位比1.244635/1.238350均≤1.5；P95比1.576407/1.310425完整保留。新峰值allocated1.292GiB，F共同驻留非独立部署显存。48计时/24配对/末4组加权loss独立复算；G/F状态不变、0优化/TX-V-target/新训练，GPU G15360/F7680/VJP7680、CPUsmoke另G2/F4/VJP2。controller1530960/worker1531466/CWD-argv-CUDA-GPU-loggrowth及自然结束核实；默认保持。只支持另登记scratchjoint30/全7正式，不声称整流程或同质量效率成功。 后续保护F加载的Torch/CUDA随机状态，再另登记GscratchsingleE30/7410与七景truth-last。当前六景3pp/clean/同质量效率目标继续ACTIVE。


## 精确非线性source10恢复器E30已启动

20261031-phase1-DURA-exactReference-E30-s2026092701-r01。VERIFIED / RUNNING。执行cc6440f3be59239e54cf4e6c900d75d3c8314ce8已提交推送并独立核对远端OID；一次N607 preflight/单归档传输SHA/remotecompile通过。实际controller 1570379/worker 1570547/GPU 4 的CWD、完整命令、CUDA、GPU和日志增长核实，旧own句柄全部消失。启动前同卡1个计算进程、free 22883MiB，新增本任务1个，其他owner保持。真实2Ls精确IQ梯度cap/三个G模块梯度/原F默认头冻结/严格epoch0重载通过；F加载后CPU/CUDA RNG恢复已实际核实，F路径也隔离随机流。至少完成2轮，完整前缀预算/加权loss/源视图/Fforward-VJP-targetrefs/逐批次cap核对通过，已验证F输入126000/输入VJP126000，最大压力1.000000119。

新增source target cache63000×10 FP32/2.52MB，生产0.509850秒；独立复算630000分量、basis10与512资格记录，最大差0.0。生产新F/VJP0，不读J或GN修改IQ；训练在线F完整预算1890000前向+1890000输入反向，原F50及缓存/源资格成本保留，是F-guided。当前正式结果待产出；固定最终30轮后自动profile/source1800×7/target168000×7/全部预测后独立truth-last。默认DURA保持，目标ACTIVE。


## 精确非线性特征参考：完整E30与七景

20261031-phase1-DURA-exactReference-E30-s2026092701-r01/cc6440f3。VERIFIED / ANALYZED。G从零单阶段30轮/7410步，L189000/U1701000；完整30轮loss/曝光/缓存视图/逐批次压力检查通过。source10跨RX/day软参考目标2.52MB/0.509850秒，生产新F/VJP0、训练在线F1890000前向+1890000输入VJP，训练不读旧J；F最终状态不变、原默认头保持。继承F50监督知识，属于F-guided；约束5→10维与导数方式同时改变，不作单因素因果归因。七景×168000全部预测固定后truth-last，126正式、105源V与252双对照配对分组独立复算，14个源/正式IQ场景误差重新核对。clean 84.970238%，low urban 68.219048%；低城区较缓存5维 -0.436310、纯G +0.846429、默认DURA -0.670833 个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练1261.405657秒，含worker源准备1262.390721秒，controller至全量分析1441.549878秒。266951参数/25.068072m ConvLinear MAC/2prior4CG27FFT，默认DURA保持；完整目标仍ACTIVE。 当前证据限单seed、已暴露工程基准，不称新盲测或恢复极限。完整源/正式/TX/RX/day/净纠错与成本见本run的analysis.md/results.json；同质量实际效率仍未证明，目标继续ACTIVE。


## 原头5维与新增5维源梯度分解

20261031-diagnostic-DURA-source-headAux-s2026092701-r01/31d61413。VERIFIED / ANALYZED，SOURCE_ONLY，固定科学判定FAIL。source512（128L/384U）×01/20、CPU2、主循环5.400632秒自然结束。head/aux负cos比例19.53125%/20.3125%，两者均未达事前25%；joint/head cos中位0.820375/0.815878。相同IQ修正预算下，head相对joint与clean F判断一致数+3/−1，未达双视图至少+4，第二缓存还有源RX损伤。不据此启动删aux5的E30、不改阈值或步长。主F3072/输入VJP2048、smokeF4/VJP6，0新G/优化/TX/V/target，F状态不变；1024条几何、2048步及22分组独立复算。默认DURA与完整3pp/clean/同质量效率目标保持。 本次局部等步长检查不能隔离完整G训练轨迹或证明目标因果，不按结果放宽门槛。详见本run分析与结果。


## 先验内部相位坐标恢复器已启动

20261031-phase1-DURA-phasePrior-E30-s2026092701-r01。VERIFIED / RUNNING。执行12c7421528c51ee6a0fac4b5bffdbe5057432ff0已提交推送并独立核对远端OID；一次N607 preflight/单归档传输SHA/remotecompile通过。实际controller1643978/worker1644071/GPU4的CWD、完整命令、CUDA、nvidia PID与日志增长已核实；旧own句柄均消失，其他owner保持。启动前同卡1个计算进程、free22883MiB，新增本任务1个。3项聚焦检查、实际factory/预算/入口及唯一P0P1通过。真实2Ls三G模块梯度/strictreload/native-optimized一致/两个prior共同相位对应关系/原F默认头通过；smokeG8、prior-only8、F2。至少完成1轮/247步，完整前缀loss/曝光/源视图/无F-TX-teacher-init计数核对通过，首轮34.008975秒。新G仍从零单阶段30/7410，266951参数与初始权重匹配pureG；prior内部坐标操作不增加参数或Core调用，但实际训练/推理成本待测，不预称加速或泛化收益。不读取F/教师/TX/V/target训练，研发曾用源F诊断明确披露。固定最终30后自动profile/source1800×7/target168000×7，逐role全预测固定后独立truth-last；源表现较差也不取消。默认DURA保持、完整目标ACTIVE。


## 先验相位坐标：完整E30与七景

20261031-phase1-DURA-phasePrior-E30-s2026092701-r01/12c74215。VERIFIED / ANALYZED。G从零单阶段30轮/7410步，L189000/U1701000；完整30条loss/曝光/源视图/clip/F-TX-teacher-init计数核对通过。仅包裹两个prior的相位坐标，不增参数/Core调用、保留共享别名与初值；训练F/教师/TX/V/target0。研发曾使用源F诊断披露，先验相位性质不冒称整体G或F不变性。七景×168000全部预测固定后truth-last，126正式、105源V、126pureG配对分组与14个源/正式IQ误差独立复算。clean 84.792857%，low urban 65.272024%；低城区较纯G -2.100595、默认DURA -3.617857个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。训练及保存1013.246842秒、控制器至自动全量分析1195.450898秒；266951参数/25.068072m ConvLinear MAC/两prior四CG27FFT，坐标操作另有非卷积成本，实测时延完整保留。默认DURA保持，同质量实际效率仍未证明，完整目标ACTIVE。 仅单seed、已暴露工程基准，不称新的盲测或恢复极限。完整TX/RX/day、净纠错、source/formal与成本见本run分析。


## 恢复后原F四分支源损伤诊断

20261031-diagnostic-DURA-source-branchFeatures-s2026092701-r01/ac309dd9。VERIFIED / ANALYZED，SOURCE_ONLY。固定defaultDURA主对象/pureG描述、source512（128L/384U）×缓存01/20；CPU2主循环2.426344秒自然结束。主PA替换clean分量后与clean F选择一致数净增10/3，第二视图不足事前双视图各+6且RX3净−1，主科学判定FAIL，不放宽、不改主。描述性的时序签名替换净增21/11，五源RX均正；它仅定位源特征损伤，不自动晋级或直接启动新G。F主2560/显式原头12800，smokeF4/head4，0新G/权重加载/优化/TX/V/target；F状态不变。2048恢复观测、12800原头特征行及104分组独立复算。默认DURA与完整目标保持。 只读native源IQ与实际F分量，时序签名描述结果不能替换原PA主判定或证明IQ可实现性；后续须独立源验证，不以目标分组调参。详见本run analysis.md/results.json。


## 时序签名的有限 IQ 可达性源验证

20261031-diagnostic-DURA-source-signatureIQ-s2026092701-r01/d8e47399。VERIFIED / ANALYZED，SOURCE_ONLY。defaultDURA原源512（128L/384U）×缓存01/20，CPU2主循环5.614998秒自然结束。1% nativeIQ RMS球内固定四步SIG修正使cleanF选择一致数486→501、498→504，净+15/+6；同球最近clean波形仅+3/+1。SIG平均相对误差降63.674%/66.793%，IQNMSE降0.620%/0.603%，五源RX均不负。缓存01clean84→83，故总判定FAIL，不放宽或自动训练。clean组SIG误差反而增约10.88/10.54倍，四步损失震荡支持固定归一化步长在小误差样本过冲的解释。不是目标TX效果/部署算法/恢复极限。完整F2048/前缀4096/VJP4096/原头3072，0G/模型更新/TX-V-target，F状态不变；1024源观测/3072原头输入/52组独立复算。默认与完整目标保持。 归一化固定步幅的clean过冲是新的源证据，后续仍须独立注册，不改本次FAIL或直接训练。完整分析见本run analysis.md/results.json。


## 时序签名 IQ 损失缩步

20261031-diagnostic-DURA-source-signatureLossStep-s2026092701-r01/3cd8781b。VERIFIED / ANALYZED，SOURCE_ONLY。CPU2/5.812552秒自然结束，原source512×01/20/defaultDURA。损失缩步IQ修正仍净增15/6条cleanF选择一致，clean84/81均保持，clean组SIG误差分别降低82.143%/77.978%，修复旧固定步幅的clean过冲。缓存01全部条件通过；缓存20 RX8净−1，其他条件通过，总判定FAIL，不更改旧FAIL/本次阈值，不直接部署或自动训练。总体SIG误差降64.598%/67.777%，IQNMSE降0.625%/0.605%，同球最近波形对照+3/+1；不是目标TX收益/物理极限。完整F2048/前缀4096/VJP4096/原头3072，smoke另F2/前缀2/VJP4/原头2；0G/模型更新/TX-V-target、F不变；1024观测/3072头输入/52组独立复算。 默认与完整目标ACTIVE，详细分析见本run analysis.md/results.json。


## 实际时序签名指导：完整E30与七景

20261031-phase1-DURA-signatureGuide-E30-s2026092701-r01/47591cdf。VERIFIED / ANALYZED。G从零单阶段30轮/7410步，L189000/U1701000；完整30条物理+SIGloss/曝光/源视图/clip/F-TX-init计数与实际逐batchcap0.5核对通过。新增实际加权TripleSIG源缓存40.32MB/0.847077秒/63000前缀输入；训练1890000前缀+1890000输入VJP、完整F后续路径0，F全state不变、加载和路径RNG隔离。F50源监督知识继承、属于F-guided；原源oracle资格FAIL保持，不称其晋级或物理上限。七景×168000全部预测固定后truth-last，126正式、105源V、126pureG配对分组与14个源/正式IQ误差独立复算；512缓存签名与完整F误差8.047e-7。clean 84.769048%，low urban 68.835119%；低城区较纯G +1.462500、默认DURA -0.054762个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=True，旧低城区70%=False。G训练及保存1112.173977秒，含worker准备1113.944462秒，控制器至自动全量分析1298.002890秒；266951参数/25.068072m ConvLinear MAC/两prior四CG27FFT。默认DURA保持，同质量实际效率须结合正式质量判断，完整目标ACTIVE。 单seed、已暴露工程基准，完整TX/RX/day、净纠错、source/formal与成本见本run分析。


## PA＋SIG源互补诊断

20261031-diagnostic-DURA-source-PA-SIG-s2026092701-r01/0cffd8b3：VERIFIED SOURCE_ONLY，联合相对SIG额外+5/0，第二缓存纠1损1（RX1+1/RX8−1），未达双视图各+2，主FAIL；其他保护通过，不自动增加PA训练版。主完整F0/head5120、smoke2Ls/F2/head2、0G/优化/TX-V-target；CPU0.053669秒，1024观测/52分组独立复算。潜在特征oracle不等于可实现IQ、真实TX或物理上限；默认与完整目标ACTIVE。详见本run analysis.md/results.json。


## 中层签名辅助监督：完整E30与七景

20261031-phase1-DURA-midSIG-E30-s2026092701-r02/5464a942。VERIFIED / ANALYZED。输出端SIG＋中层连续特征辅助G从零单阶段30轮/7410步，L189000/U1701000。完整287591参数（原主体266951＋训练专用投影20640），权重0.01；投影推理不调用但参数/存储完整计入。原四项物理损失与输出端SIG保持；全部30条损失加和/曝光/源视图/计数和online输入空间cap0.5核对通过，aux普通参数梯度不受该输出空间cap约束，全G仍clip1。只读复用40.32MB源clean签名，新增缓存F/复制0；历史63000F前缀/0.847077秒与F50源监督知识披露，旧G不是初始化祖先。训练1890000前缀＋1890000输入VJP、完整F后续路径0，中层目标1890000；F全state不变。F-guided，不称F-free。每role全部七景预测固定后独立truth-last，126正式/105源V/252双对照配对分组/14源正式IQ误差复算；同一缓存原独立512完整F比对证据复用，本轮不重复NN检查。六星地较原SIG提高0/6、较纯G提高6/6、较默认DURA提高0/6。clean 84.819643%，low urban 67.922619%；低城区较原SIG -0.912500、纯G +0.550000、默认DURA -0.967262个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练及保存1060.800148秒，含worker准备1061.919420秒，控制器至自动全量分析1246.813894秒；25.068072m ConvLinear MAC/两prior四CG27FFT。共享GPU单次时间不证明因果加速或同质量效率。r01首步错误检查已保留FAILED且训练/正式0；r02自然完成，未按目标分数改层位/权重/轮次或选择性重跑。默认DURA保持，完整目标ACTIVE，有限模型不认证物理极限。 单seed、已暴露工程基准，完整TX/RX/day、纠错、source/formal与成本见本run分析。


## 源SIG接收机均值方向：完整判定

20261031-diagnostic-DURA-source-SIG-RX-subspace-s2026092701-r01/67073970。VERIFIED / ANALYZED_SOURCE_ONLY，科学判定FAIL。固定512源记录全部排除后，以62488源clean SIG和公共RX拟合rank4均值span；该4/160维承载31.347992%拟合源签名中心化能量。两个缓存的完整SIG修正与clean F选择一致数507/509，正交补修正502/505，分别净−5/−4；接收机与场景保护也未通过，clean及整体特征误差条件通过。不把RX均值span视为已证实噪声，不放宽条件或改选描述分支，不启动相应G训练。CPU2主循环0.157431秒；主完整F0/原头4096，真实2Ls检查另F2/原头2，0G加载/前向/模型优化/TX/V/target。独立62488统计/谱投影、1024源观察、4096原头特征行及52分组复算通过（3.768744秒，0新增NN）；最大logit差7.53699822731e-06。源clean F选择代理不是真实TX准确率，latent混合特征不保证IQ可达或目标泛化。默认DURA及完整3pp/clean/同质量效率目标保持ACTIVE。 详见本run analysis.md/results.json。


## IQ与签名共同步进：完整源判定

20261031-diagnostic-DURA-source-commonSIG-step-s2026092701-r01/88029b4a。VERIFIED / ANALYZED_SOURCE_ONLY，科学判定FAIL。固定defaultDURA/source512（128L/384U）×01/20，四步共同IQ-SIG方向与旧SIG-only对照。一致数501/504与旧SIG-only501/504相同，IQ误差相对旧控制下降6.417819%/6.112685%，但源RX保护失败、第一缓存场景保护也失败。IQ输出空间负余弦仅0.927734%/0.634766%，cap0.5合成方向对SIG的一阶预测增加比例均为0，未支持此处大规模方向冲突这一假设。所有逐包IQ相对anchor保护通过，非线性SIG误差没有自动同等改善；不启动该G、不改阈值。CPU2主循环5.335702秒，独立复算0.491907秒，完整4096step-record/1024观察/4096原头特征行/52组通过；独立最大IQ步进差1.19456782244e-07。主完整F1024/前缀4096/VJP4096/原头4096，真实2Ls另F2/前缀2/VJP2/原头2；0G加载/前向/模型更新/TX/V/target。源clean F选择代理不是TX准确率，输出空间结论不是G参数或全轨迹结论。默认DURA/完整目标ACTIVE保持。 完整结果见run analysis.md/results.json。


## 固定尾24点效用：完整源判定

20261031-diagnostic-DURA-source-tail24-utility-s2026092701-r01/28628057。VERIFIED / ANALYZED_SOURCE_ONLY，科学判定FAIL。固定defaultDURA/source512（128L/384U）×01/20，以源inverse尾24点单位误差3.20倍为研究依据。尾24点sourceclean修复将clean F选择一致数486/498变成494/504，净+8/+6；完全相同逐条L2修正能量的最近全窗口clean IQ为506/508，净+20/+10。尾部相对全窗口净−12/−4，额外收益、源RX和场景保护均失败；较native收益、clean保护和完整特征误差条件通过，不加对应边界网络或调整片段/阈值。尾24点占该两组总IQ误差约26.72%/24.88%，单位采样误差约3.52/3.20倍；高误差位置并不自动成为最有利的身份修复位置。CPU2主循环1.461744秒，独立检查0.369215秒；1024源观察/4096IQ和原头特征行/52组、精确尾部支持域/等能量全窗口公式复算通过。主完整F2048/原头4096，真实2Ls另F2/原头2，0G加载/前向/模型更新/TX/V/target。源clean F一致代理不是真实TX，两个配对clean oracle不可部署或称物理上限。默认DURA及完整3pp/clean/同质量效率目标ACTIVE保持。 详见本run analysis.md/results.json。


## 源残差符号覆盖G：完整E30与七景

20261031-phase1-DURA-antitheticNoise-E30-s2026092701-r01/9925a6c6。VERIFIED / ANALYZED。原SIG-only主体266951参数，G从零单阶段30轮/7410步，L189000/U1701000，无G祖先/新可训练分支/推理增强。源残差符号请求944502、有效802880、请求clean跳过141622；全30轮独立重放专用NumPy2026103107和shuffle2026103004，计数一致，重复视图有效符号变化267569，不冒充新独立物理数据或完整模拟器重抽。源receivedRMS与cA/c²variance一致，额外Aclean1.89m/调整后Aprobe3.78m，原四物理项及SIG alpha1/cap.5保持；旧F成对梯度方差FAIL保留。clean SIG40.32MB只读复用/新缓存F0，历史63000前缀/0.847077秒与F50/11100监督知识披露，F-guided；online前缀/VJP各1.89m、完整F后续路径0、F状态不变。每role全七景预测固定后truth-last，126正式/105源V/252双对照配对分组/14IQ误差独立复算；原SIGcache证明复用，未重复NN检查。六星地较原SIG提高0/6、纯G提高2/6、默认DURA提高0/6。clean 84.866667%，low urban 67.717262%；低城区较原SIG -1.117857、纯G +0.344643、默认DURA -1.172619个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练及保存1080.461290秒，含worker准备1081.536184秒，控制器至自动全量分析1263.754893秒；25.068072m ConvLinearMAC/2prior4CG27FFT。共享GPU单次时间不是因果速度比或同质量效率证明；默认DURA保持，完整目标ACTIVE，有限模型不认证恢复极限。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异和成本见run分析。


## response＋SIG源互补诊断

20261031-diagnostic-DURA-source-response-SIG-s2026092701-r01/a7164335：VERIFIED SOURCE_ONLY，联合相对SIG额外+2/0，第二缓存纠2损2（RX1−1/RX3+1，低城区+2/中城区−2），未达双视图各+2，主FAIL；其他原保护条件通过，不增加response训练版。主F0/head5120、smoke2Ls/F2/head2、0G/优化/TX-V-target；CPU0.067106秒，1024观测/52分组独立复算。潜在分量oracle不是可实现IQ、真实TX或物理上限；默认与完整目标ACTIVE。详见本run analysis.md/results.json。


## 接收上下文FiLM：完整E30与七景

20261031-phase1-DURA-contextFiLM-E30-s2026092701-r01/d32e98f9。VERIFIED / ANALYZED。接收上下文FiLM G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；281415参数（新增14464），两prior四encoder层27D同record输入统计调制。原SIG-only物理4项/alpha1-cap0.5、原G初值及IQ、共享权重和数据/RNG/LR配对保持；不读取TX/RX/day/scene/state标签作为G输入，无新增增强/类别头/目标训练。最终8个新增参数张量全部发生变化=True，严格E30加载通过，不单独当作正确状态分工或硬件提取因果证明。源clean SIG40.32MB复用、新缓存F0；历史63000prefix/0.847077秒与F50/11100监督知识及成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变。每role全七景预测固定后truth-last，126正式/105源V/252双对照配对分组/14IQ误差独立复算，全30轮loss/曝光/视图/cap核对。六星地较SIG提高0/6、纯G提高6/6、默认DURA提高0/6。clean 84.945238%，低城区 68.210119%；低城区较SIG -0.625000、纯G +0.837500、默认DURA -0.679762个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=True，旧低城区70%=False。G训练及保存1141.237876秒、含worker准备1142.484089秒、控制器至自动分析1325.757887秒；ConvLinearMAC25082088/2prior4CG27FFT，统计/逐元素开销由实测体现。同卡G+F B1默认4.594755ms、本版4.695388ms。共享GPU单次墙钟不是因果加速证明，默认DURA保持，有限模型实测不认证物理恢复极限，完整目标仍按全部要求判断。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异和实际成本见run分析。


## 接收机场景均衡源梯度诊断

20261031-diagnostic-DURA-source-RX-gradients-s2026092701-r01/495aedf6。VERIFIED / ANALYZED，SOURCE_ONLY。固定原SIG-only G E30/source47591与原F50，CPU2主循环5.791993秒自然结束。5RX×7景×12×缓存01/20，共840次观测/477条物理记录/两视图重合363，L86/U754。整体G梯度中cos<−0.05的RX对为3/10、1/10；5RX与未归一化等RX均值梯度全部同向，最小cos分别0.162204、0.286014，两个视图均无达到阈值的均值冲突。固定双视图主条件FAIL，不从action/prior分项换主，不自动增加接收机重加权训练。G/F全state保持、param.grad为空，0优化/TX/V/target；主G840/Fprefix840/VJP840/fullF0，smoke真实2Ls/G2/fullF4/prefix2/VJP2。完整选样与10梯度向量的8组几何/80对夹角/40均值夹角独立复算，0额外NN。仅是最终冻结点、有限源样本、固定cap的局部训练代理梯度，不代表AdamW/全轨迹/真实TX/硬件或恢复极限；默认和完整目标保持。 详见run analysis.md/results.json。


## 物理与签名参数梯度分解

20261031-diagnostic-DURA-source-SIG-parameter-s2026092701-r01/81c0a94f。VERIFIED / ANALYZED，SOURCE_ONLY。固定原SIG-only G/source47591 E30与原F50，复用840次源观测/477物理/L86U754，CPU2主循环9.601164秒。完整G的SIG/四物理项梯度范数比，等RX汇总为0.133067、0.126745，各RX仅1/5、0/5低于0.1；固定双视图主假设FAIL。SIG与物理汇总梯度cos为0.262249、0.229477，总方向与物理方向cos为0.992390、0.992891，指导已进入参数空间，不用范数直接判断任务重要性。分量和重现旧总梯度relativeL2误差1.883058e−6、1.520253e−6；20分量向量/48比值/96夹角/16跨RX保留率独立复算。G/F状态保持，0优化/TX/V/target；主G840/prefix840/FVJP840/fullF0/GparamVJP样本数1680，smokeG2/fullF4/prefix2/FVJP2/GparamVJP4。不据此新增参数归一化或放大指导训练，不从action/prior分项换主；原SIG真实收益与旧RX等FAIL保持。这是局部固定cap的raw批均值参数梯度，不是AdamW实际更新、TX重要性、硬件或恢复极限；默认与完整目标保持。 详见run analysis.md/results.json。


## 跨源RX签名关系：完整E30与七景

20261031-phase1-DURA-relationSIG-E30-s2026092701-r01/53aec992。VERIFIED / ANALYZED。跨源RX余弦关系SIG指导G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；完整G266951参数、推理路径和物理4损失保持。只改变F指导，源RX仅用于batch训练损失、不进入G。SUM关系VJP含两端，逐record IQ梯度cap0.5后detached注入，是局部预条件代理导数；实例SIG误差只记录。源SIG40.32MB只读复用、逐batch161列临时上下文，新完整数组复制/缓存F0；历史63000prefix/0.847077秒与F50/11100源监督知识成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变，0直接TX/V/target/classloss/新头。每role全七景固定后truth-last，126正式/105源V/252双对照配对分组/14IQ独立复算。全30轮loss/曝光/视图/cap核对，按实际shuffle独立重放跨RX有向配对数与active记录数一致。六星地较SIG提高0/6、纯G提高3/6、默认DURA提高0/6。clean 84.842857%，低城区 68.009524%；低城区较SIG -0.825595、纯G +0.636905、默认DURA -0.880357个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练及保存1026.200308秒、含worker准备1027.364699秒、控制器至自动分析1196.162898秒。ConvLinearMAC25068072/2prior4CG27FFT，同卡G+F B1默认4.592008ms、本版4.585469ms。共享GPU单次墙钟不证明因果加速，特征关系不保证纯TX/硬件不变性，有限模型成绩不认证物理恢复极限。默认DURA保持，完整目标ACTIVE。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异及成本见run分析。


## 小权重粗恢复监督：完整E30与七景

20261031-phase1-DURA-coarseAnchor-E30-s2026092701-r01/28555c9c。VERIFIED / ANALYZED。小权重粗恢复监督G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；完整G266951参数、推理图、原四物理项与SIG alpha1/IQcap.5保持。只增加全30轮alpha×coarse-clean逐包NMSE，alpha=.02071440650605139=.25/历史源32批梯度中位比12.068895139571893；该统计是旧noVarE20/coarseclean对finalIQ，不称当前SIG训练实际比例或cap。旧梯度分离主FAIL18/32未达24保持，本版不detach/预训练/复制G或用R17IQ教师；aux直接只更新coarse，原联合路径保持，无新G/F前向、训练参数VJP或推理模块。源SIG40.32MB只读复用、新缓存F0，历史63000prefix/0.847077秒与F50/11100源监督知识成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变，0直接TX/V/target/classloss/新头/sourceRX输入。每role全七景固定后truth-last，126正式/105源V/252双对照配对分组/14IQ独立复算；30轮loss/alpha/曝光/视图/cap核对。六星地较SIG提高0/6、纯G提高2/6、默认DURA提高0/6。clean 84.612500%，低城区 67.280357%；低城区较SIG -1.554762、纯G -0.092262、默认DURA -1.609524个百分点。六景3pp通过0/6，clean严格保持=False，R17逐景1pp=False，旧低城区70%=False。G训练及保存1020.983031秒、含worker准备1022.027162秒、控制器至自动分析1194.541898秒。ConvLinearMAC25068072/2prior4CG27FFT，同卡G+F B1默认4.591921ms、本版4.587787ms。共享GPU单次墙钟不证明因果加速，coarse更似clean不等于真实身份增强，有限模型成绩不认证物理恢复极限。默认DURA保持，完整目标ACTIVE。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异及成本见run分析。


## 原SIG训练期执行优化：完整E30与七景

20261031-phase1-DURA-SIGfastTrain-E30-s2026092701-r01/84d9d483。VERIFIED / ANALYZED。原SIG训练期插值执行优化G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；完整G266951、原四物理项与SIG alpha1/IQcap.5保持，无新增loss/增强/参数。仅训练四Core12/15固定二倍插值采用既有半像素slice（另3次native），train factory try/finally恢复、冻结加载逐次原native Core断言，profile/源V/正式预测均为原native。既有源B256完整前向/梯度容差和同卡步骤14.437%收益证据复用，不重做CUPTI，不当作当前整轮速度保证或7410步逐位轨迹等价。源SIG40.32MB只读复用、新缓存F0，历史63000prefix/0.847077秒与F50/11100源监督知识成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变，0直接TX/V/target/classloss/新头/sourceRX输入。每role全七景固定后truth-last，126正式/105源V/252双对照配对分组/14IQ独立复算；30轮原loss/曝光/视图/cap核对，CPU scorer和原coarseAnchor同锁互斥且输出独立。六星地较SIG提高2/6、纯G提高6/6、默认DURA提高1/6。clean 84.867262%，低城区 69.287500%；低城区较SIG +0.452381、纯G +1.914881、默认DURA +0.397619个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=True，旧低城区70%=False。G训练及保存1021.596186秒、含worker准备1022.589685秒、控制器至自动分析1211.514897秒。ConvLinearMAC25068072/2prior4CG27FFT，同卡G+F B1默认4.684524ms、本版4.678386ms。共享GPU单次墙钟不证明因果加速，数值局部等价不能替代正式识别质量，有限模型成绩不认证物理恢复极限。默认DURA保持，完整目标ACTIVE。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异及成本见run分析。


## 周期坐标残差效用：固定源判定

20261031-diagnostic-DURA-source-period20-utility-s2026092701-r01/58da3da0。VERIFIED / ANALYZED_SOURCE_ONLY，主假设FAIL。固定defaultDURA/source512（128L/384U）×缓存01/20、1024次观测/512物理记录，0G加载/前向、0优化/TX/V/target。固定窗口80:160/四个20点周期，将pairedclean−native误差分为周期均值P和窗口残差R；主R与同修正L2的全窗最近clean控制比较。R的clean F选择一致数485/498，相对native486/498净−1/0、相对同能量全窗496/504净−11/−6；两view额外效用失败，第一view源RX保护也失败，其他clean/scene/Fcos条件通过。periodic修正492/501净+6/+3仅描述，修正能量不同且没有其单独同能量控制，不换主/窗口/period/阈值或据此加训练loss。clean窗口周期间残差能量约6.03%，窗口恢复误差中残差占44.14%/56.69%，高相对误差不自动意味着更高身份效用。CPU2主循环3.079785秒、F完整输入3072+真实2非cleanLs clean/native缓存logits smoke4，F全state保持、grad为空。独立dense80×80投影/全部1024×3IQ与同能量/窗口外实际bits/原cache配对、36组144case统计和判定复算，最大IQ公式差1.338257e−7、指标差0、0新增NN。复核中FP32先减再升精度与FP64先升再减的标量分别重算并保留原2e−6约束，无模型重跑或判据变化。F50/source4df7/50轮11100步知识及旧defaultG/缓存历史成本披露；这是源训练池Fclean选择代理，非真TX、RX噪声/硬件因果、可部署修复或物理上限。默认DURA及完整目标ACTIVE保持。 详见run分析和独立复算。


## prior 发射机特征参与解码及匹配辅助对照：完整 E30 七景

20261031-phase1-DURA-SIGcontextPrior-E30-s2026092701-r01 与 20261031-phase1-DURA-SIGpriorAux-E30-s2026092701-r01，执行 4452e09285b650536ab5e9dcc2acbfd9268c00e8。

VERIFIED / ANALYZED。发射机特征参与 prior 解码，从零单阶段30轮/7410步，L189000/U1701000、无G祖先。保留四物理+输出SIG alpha1/cap.5，新增 .01×两个prior中层SIG相对MSE均值，辅助梯度位于输出cap之外、全G裁剪1；主版预测参与三个解码层有界调制，对照推理跳过预测头。完整G344295参数/ConvLinearMAC25144872/两prior四CG27FFT，训练与冻结推理均native。源F-onlySIG40.32MB复用、新缓存F0，F50/source4df7/50轮11100步及历史缓存成本披露，F-guided；在线prefix/VJP各1.89m、辅助预测3.78m、完整F训练0，F不变，0TX/V/target训练/新分类头。每角色全七景预测后truth-last，126正式/105源/378三对照配对/14IQ独立复算，全30轮损失、辅助均值和权重、曝光、视图及原cap核对。六星地较匹配新版提高6/6、较原SIG提高6/6、较纯G提高6/6、较默认提高2/6。clean 84.799405%、低城区 69.658333%，3pp通过1/6、clean严格保持=True、R17逐景1pp=True、旧低城区70%=False。G训练保存1104.903494秒、含准备1105.953533秒、控制器至自动分析1292.594890秒；同卡G+F B1默认5.036128ms、本版5.133852ms。共享GPU墙钟不作因果加速或等质量速度比，有限模型不认证恢复极限，实际F表征不等于纯硬件特质。旧midSIG/FiLM/period20等负结果保留；默认DURA保持，完整目标ACTIVE。

VERIFIED / ANALYZED。prior 特征仅训练约束，从零单阶段30轮/7410步，L189000/U1701000、无G祖先。保留四物理+输出SIG alpha1/cap.5，新增 .01×两个prior中层SIG相对MSE均值，辅助梯度位于输出cap之外、全G裁剪1；主版预测参与三个解码层有界调制，对照推理跳过预测头。完整G308231参数/ConvLinearMAC25068072/两prior四CG27FFT，训练与冻结推理均native。源F-onlySIG40.32MB复用、新缓存F0，F50/source4df7/50轮11100步及历史缓存成本披露，F-guided；在线prefix/VJP各1.89m、辅助预测3.78m、完整F训练0，F不变，0TX/V/target训练/新分类头。每角色全七景预测后truth-last，126正式/105源/378三对照配对/14IQ独立复算，全30轮损失、辅助均值和权重、曝光、视图及原cap核对。六星地较匹配新版提高0/6、较原SIG提高6/6、较纯G提高6/6、较默认提高2/6。clean 84.966071%、低城区 69.476786%，3pp通过1/6、clean严格保持=True、R17逐景1pp=True、旧低城区70%=False。G训练保存1044.993129秒、含准备1046.173709秒、控制器至自动分析1216.978897秒；同卡G+F B1默认4.593904ms、本版4.594088ms。共享GPU墙钟不作因果加速或等质量速度比，有限模型不认证恢复极限，实际F表征不等于纯硬件特质。旧midSIG/FiLM/period20等负结果保留；默认DURA保持，完整目标ACTIVE。
