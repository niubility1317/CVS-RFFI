# 可选对照：源域 F 特征指导的单阶段恢复

状态：**AUTHORIZED_FOR_INDEPENDENT_SOURCE_F_GUIDED_CONTRAST**。用户已明确选择“允许新增独立 F 指导对照”。本文件固定该独立分支的权限和矩阵；实际发布、训练、评分状态按独立run登记及产物核实，不把授权当成已经运行或完成。现有 F-free 实验继续原矩阵。

## 为什么需要单独决定

会话和当前工作区 AGENTS.md 的 2026-10-20 约定明确要求：“不读 TX 真值、训练好的 F 权重/输出或 target 来优化恢复器；F 仅在冻结后的评测与无 query smoke 使用。”用户本次明确为独立对照放开其中的**源端训练读取冻结 F 权重/特征**，不继承为 F-free 方法，不修改 DURA 默认。科学权限与声明差异同步记录于`docs/PROJECT_PROTOCOL.md`。

现有结果尚未满足六个星地场景各距同一 raw clean ≤3pp。源 IQ 均值改善不能充分代理身份特征保持；这支持检验任务约束的价值，并不证明 F-free 路线已经达到不可突破的上限。本方案来自这一目标与损失缺项，不依据目标分数选择 checkpoint 或中间轮次。

## 具体候选与对照

|项目|固定方案|
|---|---|
|恢复器|完整266951参数、两prior/四CG、原生插值；普通随机初始化|
|源基础目标|既有noVarAux：最终IQ NMSE＋0.1内部prior NMSE＋0.05多尺度谱误差＋0.05源模拟器action probe NMSE；不训练logvariance辅助校准|
|源基础选择依据|已完成源梯度诊断中的方差辅助冲突与源IQ误差改善；不从目标分数排序选父；不加载任何已训练G|
|唯一新增项|0.1×逐例余弦特征误差：`1−cos(F.features(G(received)), stopgrad(F.features(clean)))`|
|F|同一原F50 seed2026092701/defaulthead，原source-final50/11100；全参数冻结、eval模式，不重训骨干或分类头|
|训练数据|原R07合法63000源IQ：6300 L_s＋56700 U_s，原7/63/30物理划分及epoch01..20模拟配对|
|直接TX标签读取|0；但通过F继承源有标签知识，必须明确披露，不能称为原F-free或完全无TX监督|
|预算|从零单阶段E20/4940、B256、相同seed/批序/两probe/LR峰1e-3/末2e-5/热身247、AdamW/裁剪/FP32/TF32off|
|对照|复用已完成且数据/初始化/源基础目标/预算匹配的noVarAux原七景；不同权限与不同运行时间明确标注|
|默认版本|保持DURA v1.0，不自动替换|

这里的F指`20261016-phase1-F-supcon-e50-clean-manysig-m9-r02/CROSS_RX-s2026092701`的原冻结识别器，不挪用当前Tri-SIG-TIME SSL身份或分数。来源按已有`load_reader`严格检查，必须与原源物理角色匹配；原数据VALIDATED_ONCE复用，不建立新的数据验证或签名链。

## 训练效率与实现边界

G只有一次联合训练，没有coarse/action/prior逐级预训练，也不从旧G蒸馏。可将每条源clean的冻结160维F特征计算一次：63000×160×FP32为40.32MB特征数值，另计索引及metadata；只在授权后生成，不将这些特征用于Phase2/3。

每次G更新还需一次`F(G(received))`，梯度只穿过F到G输入，F参数与状态保持不变。额外F前反向和缓存成本要单独计时；不能宣称与F-free同成本，也不能把F的既有50轮有标签训练成本隐藏为零。报告新增G成本、复用F来源/既有预算和整体配对语义。

特征约束实现位于`experiments/pamf_dura_efficiency/proposed_feature_loss.py`；它只接受冻结eval reader和160维参考，目标特征detach，保留IQ梯度。原型开发阶段只用随机线性ToyReader测试梯度隔离；真实F兼容性、来源与首步smoke在独立run的授权后最小发布流程中核实，实际状态以该run产物为准。

## 测试、停止与声明

训练仍不读V或target。固定最后E20，不按目标分数选模型、调权重、早停或选择性重跑。冻结后自动同F50/source1800×7、target168000×7，所有七景prediction固定后独立truth-last，保留全部正负结果。

逐景报告accuracy/MacroF1、correct、repair/damage、TX/RX/day、相对rawclean与DURA/R17差距、clean损伤和完整训练/推理成本。成功条件不变：六星地各距raw clean≤3pp且clean保持。任务特征保持不等于每个TX物理细节真实重建，不能仅凭F一致性或正式准确率宣称波形真实性。

普通账户N607；本goal最多2条GPU worker，每卡最多1条own，启动时free≥12GB且可见compute PID<2，保留其他owner健康进程。输出root/代码提交/矩阵在启动前独立登记；权限未获明确同意不得发布本候选。合规/来源/非有限/输出碰撞/预测覆盖失败时保留产物并按既有技术停止规则处理，不自动回退到其他权重或改配方。

用户已明确允许该独立F指导对照；原F-free研究和既有权重/结果保留。新增真实F调用和训练按本方案及最小发布流程执行，不扩大到目标端学习。

实际独立run已完成训练和七景正式评分；真实F/G梯度、冻结检查及独立126组复算通过，但星地目标未达，详见[该run报告](../../automation_reports/CV-SincNet/20261030-phase1-DURA-source-Fguide-cos010-s2026092701-r01/report.md)。

## 源证据驱动的后续特征约束诊断

首个cos0.1对照已经完整训练/正式七景评分，实际新增G流程1073.247秒、clean84.904167%、lowurban67.827381%，未达逐景3pp，全部结果保留。此前先于其目标分数读取完成的source-task-geometry显示，整体160维余弦误差约降低26%，与clean F判别选择的一致率改善较小；加到0.3的预定源条件失败，明确不增加同一项系数。

下一独立源诊断`20261030-diagnostic-DURA-head-feature-geometry-s2026092701-r01`仍处于用户已授权的源F权重/特征范围：只从同一冻结原头的归一化中心权重取SVD行空间，在至多5维方向约束归一化源特征差，不拟合投影或更改分类头，不读直接TX/V/target。该方向由上述源现象提出，不按已暴露目标场景/类别挑方向或权重。

诊断在固定旧noVarAux源E20上复用同512源位置×cached01/20，CPU2/0GPU/0更新；新约束的系数只按全32批projected/base梯度范数中位数确定`alpha=0.25/median_ratio`，无网格搜索。两缓存视图分别要求校准后中位比在[0.125,0.5]、与基础目标平均余弦非负且负余弦批数≤8/16、与原头中心特征差的梯度对齐高于旧cos且为正、源teacher选择仍有差异。条件全部成立才另登记从零单阶段E20独立投影对照；否则不发布该训练。原头中心差只作源几何诊断，不作为真实TX标签或准确率。源资格不能代替正式七景性能；新训练一旦发布，仍固定finalE20且自动完整七景，无默认升级。


上述源投影诊断已完整完成并独立复算2560个几何标量，两缓存视图资格均PASS，唯一源系数0.006607891036090233。新独立候选`20261030-phase1-DURA-source-headFeature-s2026092701-r01`仅以该项替换首个候选的0.1余弦项，其余普通从零/noVar基础/原生插值/E20/4940及七景矩阵保持。直接TX/V/target训练0、F/原头更新0，G既有祖先0；来源、真实启动/成本/正式成绩以新run报告为准，不因源对齐通过自动认定成功。

另登记零更新源定位`20261030-diagnostic-DURA-source-stage-fidelity-s2026092701-r01`，在用户已固定的默认R69各恢复阶段及固定prior的CG4/8/16计算同一源F特征/教师选择，补充早先仅看IQ的物理求解诊断。仍只用512源位置×cached01/20、CPU2/0GPU，不改F/G权重，不读直接TX/V/target。source真实算子/噪声仅作反事实，不能部署或称准确率上界。增加CG只有在两视图总体教师一致至少多6/512条、cos误差≤原生0.9倍、逐源场景教师一致正确数均不减少时才另登记正式运行版；多项通过只选最小CG，否则不由本诊断发布求解版本。当前headFeature训练/七景队列保持，不因该源诊断取消已发布版本的正式测试。


用户另行明确允许headFeature资源切换r02：仅GPU4/6/7中的一张、启动后总可见计算进程最多3，新增本任务1个且free≥12GB；只替换仍未启动GPU的旧CPU队列，其他owner保持。此项仅资源/执行版本改变，科学sourceF指导系数、G从零单阶段E20、F冻结/直接TX-V-target隐藏与完整七景保持。新run `20261030-phase1-DURA-source-headFeature-s2026092701-r02`，禁止将资源授权解释为目标端学习或默认版本替换。


headFeature r02完整七景仍未达3pp后，另登记源阶段对照`20261030-diagnostic-DURA-source-stage-compare-s2026092701-m2-r01`：固定noVar与headFeature source-final、同512源位置×cached01/20、CPU2/0GPU/0更新，复用同F/同队列的默认R69历史源阶段统计。只在headFeature最后prior的两视图teacher一致数均比其native多≥6/512、Fcos误差≤0.9倍、逐source scene一致数均不减少时，才另登记该冻结G的prior-only运行版并完成正式七景；否则不发布该运行版。noVar/default仅对照，不能择其为父或用oracle部署；该源判断不是真实TX准确率或目标上界，不改变任何已完成权重/负结果。


## 连续后验特征约束独立对照

独立登记 `20261030-phase1-DURA-source-posteriorKL010-s2026092701-r01`：同一冻结F的源clean/restored特征由原联合分类头确定性映射成6维连续后验，KL(clean||restored)、T1、系数0.1。没有硬伪TX标签、真实TX读取、头训练或F更新，不扩大到target。G从零单阶段E20/4940，原noVar骨架/数据/日程不变，source特征缓存一次，restoredF每visit一次；仍明确F-guided/继承F上游50轮知识与成本。

该项来源于已完成source-task-geometry的特征距离与选择保持不等价现象；0.1仅与旧cos同数字，不称梯度强度匹配或最优参数。固定finalE20后自动source1800×7/target168000×7、独立truth-last，逐景3pp与clean要求保持。旧cos、headFeature、F-free结果及默认DURA不改变，信道重配失败候选不追加训练。


独立源域F相位诊断：20261030-diagnostic-DURA-source-Fphase-s2026092701-r01，固定同F50/原头/source512×8角度，CPU2/0更新/G读取0/逐样本TX-V-target读取0。只用于判断全局相位商空间损失是否值得另立研究，全部角度/RX阈值预登记；F-informed而非F-free设计，不改变默认或当前恢复器，完整3pp/clean/效率目标保持。


公开L-STF全局相位参考诊断：20261030-diagnostic-DURA-source-phase-anchor-s2026092701-r01。原63k源中62488只拟合一个公共圆均值，512统计留出验证同F完整特征/原头保持；15RX/day逐一判据。0G/神经更新/逐样本TX/V/target，只属源F-informed设计检查，不继承全局相位不变性假设，不授权RX专属或目标相位拟合，默认保持。


独立source phase-only oracle诊断：20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01。固定noVar/defaultR69与同F50/source512×3views，原clean仅用于counterfactual单位相位对齐；0更新/TX/V/target，oracle不可部署或作为准确率上界。sourceF信息只用于phaseweight研究资格；只noVar按预定双视图/全部scene/clean条件晋级，不改默认或目标权限。


sourceclean局部敏感度诊断20261031-diagnostic-DURA-source-cleanTangent-s2026092701-r01/ff6926fa已完成：固定repeatE30/source512×01/20，head-visible5D Jclean二次型与exact特征IQ梯度mean cos.643863/.661080，普通IQ约.071；493/484条非负、五RX均正，全部源门槛通过。alpha按预定.25/maxmedian normratio固定0.0033130025035573727。局部线性近似误差中位仍约.53，不称globallyexact或目标收益；仅支持另登记scratchsingleE30，不自动训练。CPU5.796s/cache3.691s，F1536+6smoke/0G新前向/0TX-V-target-updates、Fstate不变，2048条几何及IQ-J算术独立复算。全源缓存估计645.12MB/30轮读19.3536GB，实耗待真实训练；原F50知识/成本披露，默认保持/目标ACTIVE。


独立新G 20261031-phase1-DURA-cleanTangent-E30-s2026092701-r01 已真实RUNNING：代码7dd799ebb294ffe8711f530872e4922f100ed498/release20261031-DURA-cleanTangent-E30-r01，sourcequalified固定alpha.0033130025035573727，原noVar四loss+clean局部身份敏感度二次型，266951参数/两prior四CG，scratchsinglejoint30/7410，不直接读TX/V/target。源Fclean63000×5VJP缓存645.12MB实际30.008929秒，完整独立readback finite/IDs/basis通过，Fstate不变后释放，学生训练0Fforward；F50知识/成本明确继承。7tests/唯一P0P1/真实2Ls G-F梯度与strictreload PASS，一次archiveSHA/compile、controller1285659/worker1285750/GPU4/CWD-argv-CUDA-loggrowth已核实，至少E2并实际执行新loss。全30后自动source1800×7/target168000×7/allviews固定后truth-last；当前无正式收益结论，默认保持/完整目标ACTIVE。登记event已成功写入，之后index刷新遇另一owner短暂build锁；核实event唯一且锁自然消失后仅重建managed索引，没有重复record、重复launch或操作其他进程。


clean局部敏感度G 20261031-phase1-DURA-cleanTangent-E30-s2026092701-r01/7dd799eb 已完整结案：scratchsingle30/7410/L189000U1701000、F50 sourceJ63000/315000VJP/645.12MB缓存30.008929s，学生训练F0/直接TX-V-target0，完整30曲线/全部loss与缓存计数核对通过。全7×168000固定后truth-last，126正式及126pureG配对分组独立一致，worker1285750/controller1285659自然结束。相对pureG四星地+0.526786至+1.577976pp，中/低城区−0.446429/−0.747619pp、clean−.043452pp；七景均低于默认，clean84.586905/lowurban66.625，3pp0/6/clean严格保持False/R17逐景1ppFalse/旧lowurban70False。源V低城区也−2.5pp，不能仅归因目标RX；局部梯度代理不等于完整身份保真。训练1022.804362s/含准备G1053.873159s/全分析1235.243895s，266951参数/25.068072m ConvLinearMAC/2prior4CG27FFT，F上游50轮成本披露；同质量效率未达。默认保持/目标ACTIVE，不回调alpha/选epoch/重测。


源特征度量压力诊断20261031-diagnostic-DURA-source-boundedTangent-s2026092701-r01/aa284ea1完整完成：全63000 sourcecleanJ/纯repeatE30 source512×01/20，CPU7.343s/因子4.855s645.12MB/新增F1024+2/VJP1024/0G-TX-V-target-updates/F不变。98.55%源因子的最强相对方向谱值>1，不是实际98.55%梯度超界；当前实际guide/IQ p95为1.610/1.349、中位.25。固定谱cap1满足输出loss/梯度上界，但cos.643863→.575031/.661082→.587405，下降超过预定.02，两资格FAIL，不放宽/不训练该G。全63000谱/trace/finite/ID和2048条几何独立复算一致；强度与方向需同时处理，不把上界当身份收益，后续如研究方向保持的梯度平衡须独立固定验证。默认保持/完整目标ACTIVE。


下一独立候选的数学原型：保留原方向的逐样本步内梯度平衡，代码 `experiments/pamf_dura_efficiency/tangent_balance.py`。原alpha不变，令s=min(1,||g_IQ||/||g_feature||)，每步把s视为固定权重（detach），附加项为alpha*s*原tangentLoss。这样在当前IQ输出点保留原特征梯度方向，未过强样本完全不变，并由Cauchy不等式同时得到附加loss≤IQNMSE与附加IQ梯度范数≤IQ梯度范数。它是动态重加权的步内surrogate，不宣称是对随IQ重算权重的全局固定标量能量求完整导数；也不保证G参数梯度范数。4项针对任意误差、方向/参考隔离、未限幅恒等、零/非法边界的测试通过。

该原型不是刚才失败的静态谱裁剪方案，不放宽其已失败判据。目前没有新模型训练、正式预测或性能结论。后续若进入实验，单独登记source方向/压力校验、scratch单阶段30/7410及完整七景truth-last；原F50的既有sourceclean J缓存可只读复用，学生训练不需要F前反向，F源知识和历史缓存成本继续披露。目标仍为逐六景距rawclean≤3pp、clean保持及同等质量效率，默认DURA保持。


保留方向的动态平衡G 20261031-phase1-DURA-balancedTangent-E30-s2026092701-r01 已真实RUNNING，执行75da21407e7a6790f45e4a8ac4bcc2ea5b6d120e/release20261031-DURA-balancedTangent-E30-r01。Gscratchsingle30/7410/266951/两prior四CG，原alpha不变，步内detach scalar缩放过强IQ附加梯度，非全局固定能量完整导数/非参数梯度上界。rawsourceJ63000/645.12MB只读复用，核对.081749s，新F前向/VJP/cache存储0，F源50轮及旧30.009s缓存成本披露；未读producerG/target。7tests/唯一P0P1/实际2LsG三模块梯度-实际IQ界-F冻结-严格重载通过，actualcontroller1326970/worker1327059/GPU4/CWD-argv-CUDA-loggrowth核实，至少E2全前缀检查通过/压力max1.000000119。source构造保留旧方向不是泛化保证，不放宽失败静态谱门槛；自动满30并七景全部预测后truth-last。默认不变/六景3pp-clean-同质量效率目标ACTIVE。


## 逐条平衡局部身份约束：完整 E30 与七景

20261031-phase1-DURA-balancedTangent-E30-s2026092701-r01/75da2140。VERIFIED / ANALYZED。单阶段 scratch E30/7410，L189000/U1701000；30 轮完整 loss/曝光/缓存视图/逐批次压力上界核对通过。复用 source J，无新增 F 前向/VJP/缓存；继承 F50 监督知识，仍为 F-guided。七景×168000 全部预测固定后 truth-last，126 正式与252 双对照配对分组独立复算。clean 84.814286%，low urban 67.031548%；低城区相对未平衡版 +0.406548、纯 G -0.341071、默认 DURA -1.858333 个百分点。六景3pp通过 0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练 998.836772 秒，含源准备 999.795406 秒，全量分析 1180.638900 秒；266951 参数/25.068072m ConvLinear MAC/2prior4CG27FFT，默认保持。 当前完整目标继续 ACTIVE；单 seed 已暴露工程基准，不称新盲测或物理恢复极限。完整源/正式/TX/RX/day/净纠错与成本见该 run 的 analysis.md/results.json。没有采用失败的静态谱因子，不依据目标结果回调 alpha 或调类别权重。
