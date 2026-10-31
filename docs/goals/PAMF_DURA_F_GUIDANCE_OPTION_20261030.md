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


## 源时序签名与接收机均值方向

20261031-diagnostic-DURA-source-SIG-RX-subspace-s2026092701-r01：独立CPU源诊断，排除固定512后用其余62488源clean SIG和公共RX估计五均值rank4 span。只比较源latent完整SIG与正交补修正，不将均值span视为已证实噪声、不使用TX/V/target、不加载或训练G。双view各较完整SIG至少+2、逐RX/clean保护、非clean每景最多−1及特征误差条件预先固定；PASS仅支持另登记源IQ可达性。原F50知识/缓存成本、旧FAIL、默认与完整目标保持，实际结果见run。


## 源SIG接收机均值方向：完整判定

20261031-diagnostic-DURA-source-SIG-RX-subspace-s2026092701-r01/67073970。VERIFIED / ANALYZED_SOURCE_ONLY，科学判定FAIL。固定512源记录全部排除后，以62488源clean SIG和公共RX拟合rank4均值span；该4/160维承载31.347992%拟合源签名中心化能量。两个缓存的完整SIG修正与clean F选择一致数507/509，正交补修正502/505，分别净−5/−4；接收机与场景保护也未通过，clean及整体特征误差条件通过。不把RX均值span视为已证实噪声，不放宽条件或改选描述分支，不启动相应G训练。CPU2主循环0.157431秒；主完整F0/原头4096，真实2Ls检查另F2/原头2，0G加载/前向/模型优化/TX/V/target。独立62488统计/谱投影、1024源观察、4096原头特征行及52分组复算通过（3.768744秒，0新增NN）；最大logit差7.53699822731e-06。源clean F选择代理不是真实TX准确率，latent混合特征不保证IQ可达或目标泛化。默认DURA及完整3pp/clean/同质量效率目标保持ACTIVE。 详见本run analysis.md/results.json。


## IQ与时序签名共同步进源检验

20261031-diagnostic-DURA-source-commonSIG-step-s2026092701-r01：固定defaultDURA/source512×01/20与原SIG-only损失缩步控制，四步单位IQ/SIG导数角平分方向，精确IQ线最小与SIG局部缩步、每步0.25%/总1%原RMS球。只用源配对clean/冻结F，0G模型训练/TX/V/target；有限IQ界不是非线性SIG或G参数保证。双view相对SIG-only至少+2一致、RX/clean保护、非clean每景最多−1、SIG≤.9native与逐包IQ保护预先固定。PASS仅支持另登记E30，默认与全部旧负结果保持。


## IQ与签名共同步进：完整源判定

20261031-diagnostic-DURA-source-commonSIG-step-s2026092701-r01/88029b4a。VERIFIED / ANALYZED_SOURCE_ONLY，科学判定FAIL。固定defaultDURA/source512（128L/384U）×01/20，四步共同IQ-SIG方向与旧SIG-only对照。一致数501/504与旧SIG-only501/504相同，IQ误差相对旧控制下降6.417819%/6.112685%，但源RX保护失败、第一缓存场景保护也失败。IQ输出空间负余弦仅0.927734%/0.634766%，cap0.5合成方向对SIG的一阶预测增加比例均为0，未支持此处大规模方向冲突这一假设。所有逐包IQ相对anchor保护通过，非线性SIG误差没有自动同等改善；不启动该G、不改阈值。CPU2主循环5.335702秒，独立复算0.491907秒，完整4096step-record/1024观察/4096原头特征行/52组通过；独立最大IQ步进差1.19456782244e-07。主完整F1024/前缀4096/VJP4096/原头4096，真实2Ls另F2/前缀2/VJP2/原头2；0G加载/前向/模型更新/TX/V/target。源clean F选择代理不是TX准确率，输出空间结论不是G参数或全轨迹结论。默认DURA/完整目标ACTIVE保持。 完整结果见run analysis.md/results.json。


## 固定尾24点身份效用

20261031-diagnostic-DURA-source-tail24-utility-s2026092701-r01：由已有源inverse尾部单位采样误差3.20倍提出，固定defaultDURA/source512×01/20；sourceclean尾24替换对比同L2修正能量的最近全窗口clean IQ，接原F默认头，无G/TX/V/target。两view尾部较native至少+4/512、较全窗口+2，RX/clean及场景保护、F特征误差条件预先固定；PASS仅支持另登记边界结构研究，不自动训练。配对clean oracle不可部署，不是真实TX或物理上限，默认及全部旧负结果保持。


## 固定尾24点效用：完整源判定

20261031-diagnostic-DURA-source-tail24-utility-s2026092701-r01/28628057。VERIFIED / ANALYZED_SOURCE_ONLY，科学判定FAIL。固定defaultDURA/source512（128L/384U）×01/20，以源inverse尾24点单位误差3.20倍为研究依据。尾24点sourceclean修复将clean F选择一致数486/498变成494/504，净+8/+6；完全相同逐条L2修正能量的最近全窗口clean IQ为506/508，净+20/+10。尾部相对全窗口净−12/−4，额外收益、源RX和场景保护均失败；较native收益、clean保护和完整特征误差条件通过，不加对应边界网络或调整片段/阈值。尾24点占该两组总IQ误差约26.72%/24.88%，单位采样误差约3.52/3.20倍；高误差位置并不自动成为最有利的身份修复位置。CPU2主循环1.461744秒，独立检查0.369215秒；1024源观察/4096IQ和原头特征行/52组、精确尾部支持域/等能量全窗口公式复算通过。主完整F2048/原头4096，真实2Ls另F2/原头2，0G加载/前向/模型更新/TX/V/target。源clean F一致代理不是真实TX，两个配对clean oracle不可部署或称物理上限。默认DURA及完整3pp/clean/同质量效率目标ACTIVE保持。 详见本run analysis.md/results.json。


## 源残差符号数据覆盖对照

20261031-phase1-DURA-antitheticNoise-E30-s2026092701-r01：独立G从零单阶段E30/7410，原SIG-only主体/四物理项/alpha1-cap0.5不变。专用源Bernoulli0.5残差符号增强，重归一化原接收RMS并同步A/variance，数值clean跳过；0TX/V/target，不声称完整模拟重抽或条件AGC分布等价。只读复用原F-only SIG40.32MB、F50知识/成本披露，原F成对方差FAIL保持。主对照SIG47591与次对照pureG8cf，全部自动七景truth-last，默认与完整目标保持。


## 源残差符号覆盖G实际启动

20261031-phase1-DURA-antitheticNoise-E30-s2026092701-r01。VERIFIED RUNNING。执行9925a6c6c4e73e3fa1f76f9b2c4105bfbfa91801，controller1774712/worker1774802/GPU4实际PID/CWD/完整命令、空CUDA控制器与CUDA4 worker、计算PID和日志增长已核实。启动前free22883MiB/已有1计算PID，新增1个符合总数≤3。首步真实2非cleanLs强制正负两路，实际RMS/Action-probe/variance/原F和SIGcache/guidecap/三G模块梯度/严格E0/native-optimized全部PASS。已完成2/30轮/494步；本轮requested=0.498095、effective=0.423016、功率比=1.000000000，数值clean跳过。SIGcache40.32MB只读复用，句柄核对与mmap打开0.054573秒（不是完整数值读完耗时），新F缓存/存储0；F50上游知识/原缓存成本披露。固定30后自动全部七景与独立truth-last，当前没有正式性能结论，默认保持。


## 源残差符号覆盖G：完整E30与七景

20261031-phase1-DURA-antitheticNoise-E30-s2026092701-r01/9925a6c6。VERIFIED / ANALYZED。原SIG-only主体266951参数，G从零单阶段30轮/7410步，L189000/U1701000，无G祖先/新可训练分支/推理增强。源残差符号请求944502、有效802880、请求clean跳过141622；全30轮独立重放专用NumPy2026103107和shuffle2026103004，计数一致，重复视图有效符号变化267569，不冒充新独立物理数据或完整模拟器重抽。源receivedRMS与cA/c²variance一致，额外Aclean1.89m/调整后Aprobe3.78m，原四物理项及SIG alpha1/cap.5保持；旧F成对梯度方差FAIL保留。clean SIG40.32MB只读复用/新缓存F0，历史63000前缀/0.847077秒与F50/11100监督知识披露，F-guided；online前缀/VJP各1.89m、完整F后续路径0、F状态不变。每role全七景预测固定后truth-last，126正式/105源V/252双对照配对分组/14IQ误差独立复算；原SIGcache证明复用，未重复NN检查。六星地较原SIG提高0/6、纯G提高2/6、默认DURA提高0/6。clean 84.866667%，low urban 67.717262%；低城区较原SIG -1.117857、纯G +0.344643、默认DURA -1.172619个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练及保存1080.461290秒，含worker准备1081.536184秒，控制器至自动全量分析1263.754893秒；25.068072m ConvLinearMAC/2prior4CG27FFT。共享GPU单次时间不是因果速度比或同质量效率证明；默认DURA保持，完整目标ACTIVE，有限模型不认证恢复极限。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异和成本见run分析。


## response＋SIG源互补诊断

20261031-diagnostic-DURA-source-response-SIG-s2026092701-r01/a7164335：VERIFIED SOURCE_ONLY，联合相对SIG额外+2/0，第二缓存纠2损2（RX1−1/RX3+1，低城区+2/中城区−2），未达双视图各+2，主FAIL；其他原保护条件通过，不增加response训练版。主F0/head5120、smoke2Ls/F2/head2、0G/优化/TX-V-target；CPU0.067106秒，1024观测/52分组独立复算。潜在分量oracle不是可实现IQ、真实TX或物理上限；默认与完整目标ACTIVE。详见本run analysis.md/results.json。


## 接收上下文调制多尺度先验对照

20261031-phase1-DURA-contextFiLM-E30-s2026092701-r01：G从零单阶段30轮/7410，保留原SIG-only四物理项/alpha1-cap0.5与配对数据、初始化、随机数、LR。新增两prior各27D同record输入统计→32→192、四层有界FiLM，末层零初值；281415参数（+14464），不读RX/scene/state标签或F作G输入，0TX/V/target。复用原F-only SIG40.32MB，F50知识/成本披露；SIG47591主对照、pureG8cf次对照，自动原七景truth-last、实际训练与推理成本，默认及完整目标保持。


## 接收上下文FiLM：完整E30与七景

20261031-phase1-DURA-contextFiLM-E30-s2026092701-r01/d32e98f9。VERIFIED / ANALYZED。接收上下文FiLM G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；281415参数（新增14464），两prior四encoder层27D同record输入统计调制。原SIG-only物理4项/alpha1-cap0.5、原G初值及IQ、共享权重和数据/RNG/LR配对保持；不读取TX/RX/day/scene/state标签作为G输入，无新增增强/类别头/目标训练。最终8个新增参数张量全部发生变化=True，严格E30加载通过，不单独当作正确状态分工或硬件提取因果证明。源clean SIG40.32MB复用、新缓存F0；历史63000prefix/0.847077秒与F50/11100监督知识及成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变。每role全七景预测固定后truth-last，126正式/105源V/252双对照配对分组/14IQ误差独立复算，全30轮loss/曝光/视图/cap核对。六星地较SIG提高0/6、纯G提高6/6、默认DURA提高0/6。clean 84.945238%，低城区 68.210119%；低城区较SIG -0.625000、纯G +0.837500、默认DURA -0.679762个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=True，旧低城区70%=False。G训练及保存1141.237876秒、含worker准备1142.484089秒、控制器至自动分析1325.757887秒；ConvLinearMAC25082088/2prior4CG27FFT，统计/逐元素开销由实测体现。同卡G+F B1默认4.594755ms、本版4.695388ms。共享GPU单次墙钟不是因果加速证明，默认DURA保持，有限模型实测不认证物理恢复极限，完整目标仍按全部要求判断。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异和实际成本见run分析。


## 接收机场景均衡源梯度诊断

20261031-diagnostic-DURA-source-RX-gradients-s2026092701-r01/495aedf6。VERIFIED / ANALYZED，SOURCE_ONLY。固定原SIG-only G E30/source47591与原F50，CPU2主循环5.791993秒自然结束。5RX×7景×12×缓存01/20，共840次观测/477条物理记录/两视图重合363，L86/U754。整体G梯度中cos<−0.05的RX对为3/10、1/10；5RX与未归一化等RX均值梯度全部同向，最小cos分别0.162204、0.286014，两个视图均无达到阈值的均值冲突。固定双视图主条件FAIL，不从action/prior分项换主，不自动增加接收机重加权训练。G/F全state保持、param.grad为空，0优化/TX/V/target；主G840/Fprefix840/VJP840/fullF0，smoke真实2Ls/G2/fullF4/prefix2/VJP2。完整选样与10梯度向量的8组几何/80对夹角/40均值夹角独立复算，0额外NN。仅是最终冻结点、有限源样本、固定cap的局部训练代理梯度，不代表AdamW/全轨迹/真实TX/硬件或恢复极限；默认和完整目标保持。 详见run analysis.md/results.json。


## 物理与签名参数梯度分解

20261031-diagnostic-DURA-source-SIG-parameter-s2026092701-r01/81c0a94f。VERIFIED / ANALYZED，SOURCE_ONLY。固定原SIG-only G/source47591 E30与原F50，复用840次源观测/477物理/L86U754，CPU2主循环9.601164秒。完整G的SIG/四物理项梯度范数比，等RX汇总为0.133067、0.126745，各RX仅1/5、0/5低于0.1；固定双视图主假设FAIL。SIG与物理汇总梯度cos为0.262249、0.229477，总方向与物理方向cos为0.992390、0.992891，指导已进入参数空间，不用范数直接判断任务重要性。分量和重现旧总梯度relativeL2误差1.883058e−6、1.520253e−6；20分量向量/48比值/96夹角/16跨RX保留率独立复算。G/F状态保持，0优化/TX/V/target；主G840/prefix840/FVJP840/fullF0/GparamVJP样本数1680，smokeG2/fullF4/prefix2/FVJP2/GparamVJP4。不据此新增参数归一化或放大指导训练，不从action/prior分项换主；原SIG真实收益与旧RX等FAIL保持。这是局部固定cap的raw批均值参数梯度，不是AdamW实际更新、TX重要性、硬件或恢复极限；默认与完整目标保持。 详见run analysis.md/results.json。


## 跨RX签名关系指导对照

20261031-phase1-DURA-relationSIG-E30-s2026092701-r01：G原266951从零单阶段30轮/7410，原物理4项/初始化/RNG/LR/数据保持，指导改为源cross-RX相对余弦距离关系，实例误差仅日志。all-pair SUM inputVJP包含两端，再逐IQ detached cap0.5，1prefix/VJP每访问；F-only sourceSIG40.32MB与源RX仅训练loss，0新持久缓存/额外Fpass/新G参数/推理关系输入，0TX/V/target。原F50知识和缓存成本披露，原G非祖先。关系共同旋转自由度由物理锚点约束，真实TX收益只凭完整原七景truth-last验收；默认与旧FAIL保持。


## 跨源RX签名关系：完整E30与七景

20261031-phase1-DURA-relationSIG-E30-s2026092701-r01/53aec992。VERIFIED / ANALYZED。跨源RX余弦关系SIG指导G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；完整G266951参数、推理路径和物理4损失保持。只改变F指导，源RX仅用于batch训练损失、不进入G。SUM关系VJP含两端，逐record IQ梯度cap0.5后detached注入，是局部预条件代理导数；实例SIG误差只记录。源SIG40.32MB只读复用、逐batch161列临时上下文，新完整数组复制/缓存F0；历史63000prefix/0.847077秒与F50/11100源监督知识成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变，0直接TX/V/target/classloss/新头。每role全七景固定后truth-last，126正式/105源V/252双对照配对分组/14IQ独立复算。全30轮loss/曝光/视图/cap核对，按实际shuffle独立重放跨RX有向配对数与active记录数一致。六星地较SIG提高0/6、纯G提高3/6、默认DURA提高0/6。clean 84.842857%，低城区 68.009524%；低城区较SIG -0.825595、纯G +0.636905、默认DURA -0.880357个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=False，旧低城区70%=False。G训练及保存1026.200308秒、含worker准备1027.364699秒、控制器至自动分析1196.162898秒。ConvLinearMAC25068072/2prior4CG27FFT，同卡G+F B1默认4.592008ms、本版4.585469ms。共享GPU单次墙钟不证明因果加速，特征关系不保证纯TX/硬件不变性，有限模型成绩不认证物理恢复极限。默认DURA保持，完整目标ACTIVE。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异及成本见run分析。


## 弱粗恢复clean监督：固定单E30对照

20261031-phase1-DURA-coarseAnchor-E30-s2026092701-r01。原SIG-only主体/四物理项/alpha1-cap.5保持，仅加全30轮coarse-cleanNMSE固定alpha=0.02071440650605139（历史源32批coarse/finalIQ梯度中位比12.068895139571893，固定压力.25；不是当前动态cap）。旧梯度分离FAIL保持，不detach/预训练/复制G，不用R17IQ教师；原F-onlySIG40.32MB复用、新F缓存0，0新推理模块/参数/额外训练G-Fforward或VJP，0TX/V/target。全部原七景自动truth-last，无源或目标成绩选模，默认与完整目标保持。


## 原SIG训练期执行优化：单E30独立对照

20261031-phase1-DURA-SIGfastTrain-E30-s2026092701-r01。原SIG四物理+特征alpha1/cap.5、266951、初值/RNG/单30轮预算不变，只训练四Core的12/15插值采用已有源等价/同卡步骤14.437%证据支持的slice实现。训练工厂try/finally还原，冻结加载逐次native类断言，profile/全部原七景仍native。无新增loss/参数/教师IQ/增强，旧G非祖先，F-onlySIG40.32MB复用和F50知识成本披露；0TX/V/target。与coarseAnchor各自匹配原SIG，GPU单任务总≤2、每卡own1，CPUscore共用当前coarseAnchor锁而输出独立。完整目标ACTIVE、默认保持，不承诺整轮提速或位级长轨迹一致。


## 小权重粗恢复监督：完整E30与七景

20261031-phase1-DURA-coarseAnchor-E30-s2026092701-r01/28555c9c。VERIFIED / ANALYZED。小权重粗恢复监督G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；完整G266951参数、推理图、原四物理项与SIG alpha1/IQcap.5保持。只增加全30轮alpha×coarse-clean逐包NMSE，alpha=.02071440650605139=.25/历史源32批梯度中位比12.068895139571893；该统计是旧noVarE20/coarseclean对finalIQ，不称当前SIG训练实际比例或cap。旧梯度分离主FAIL18/32未达24保持，本版不detach/预训练/复制G或用R17IQ教师；aux直接只更新coarse，原联合路径保持，无新G/F前向、训练参数VJP或推理模块。源SIG40.32MB只读复用、新缓存F0，历史63000prefix/0.847077秒与F50/11100源监督知识成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变，0直接TX/V/target/classloss/新头/sourceRX输入。每role全七景固定后truth-last，126正式/105源V/252双对照配对分组/14IQ独立复算；30轮loss/alpha/曝光/视图/cap核对。六星地较SIG提高0/6、纯G提高2/6、默认DURA提高0/6。clean 84.612500%，低城区 67.280357%；低城区较SIG -1.554762、纯G -0.092262、默认DURA -1.609524个百分点。六景3pp通过0/6，clean严格保持=False，R17逐景1pp=False，旧低城区70%=False。G训练及保存1020.983031秒、含worker准备1022.027162秒、控制器至自动分析1194.541898秒。ConvLinearMAC25068072/2prior4CG27FFT，同卡G+F B1默认4.591921ms、本版4.587787ms。共享GPU单次墙钟不证明因果加速，coarse更似clean不等于真实身份增强，有限模型成绩不认证物理恢复极限。默认DURA保持，完整目标ACTIVE。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异及成本见run分析。


## 固定20点坐标的误差效用

20261031-diagnostic-DURA-source-period20-utility-s2026092701-r01。CPU2、复用defaultDURA/source512×01/20缓存，0G/TX/V/target/更新；仅F3072+4smoke。固定[80,160)/4×20对clean-native误差正交分解P/R，主R与同修正L2全窗最近clean控制比较。双view各较native+4/512、较等能量+2、5RX/clean保持/非clean最多−1及Fcos不高；P仅描述、不换主或坐标。旧repeatShrink保护与本恢复误差oracle不同，旧FAIL保持；源F选择代理非真TX，oracle非部署或恢复上限，PASS不自动训练，默认与完整目标保持。


## 原SIG训练期执行优化：完整E30与七景

20261031-phase1-DURA-SIGfastTrain-E30-s2026092701-r01/84d9d483。VERIFIED / ANALYZED。原SIG训练期插值执行优化G从零单阶段30轮/7410步，L189000/U1701000，无G祖先；完整G266951、原四物理项与SIG alpha1/IQcap.5保持，无新增loss/增强/参数。仅训练四Core12/15固定二倍插值采用既有半像素slice（另3次native），train factory try/finally恢复、冻结加载逐次原native Core断言，profile/源V/正式预测均为原native。既有源B256完整前向/梯度容差和同卡步骤14.437%收益证据复用，不重做CUPTI，不当作当前整轮速度保证或7410步逐位轨迹等价。源SIG40.32MB只读复用、新缓存F0，历史63000prefix/0.847077秒与F50/11100源监督知识成本保留，F-guided。训练prefix/VJP各1.89m、完整F后续0，F不变，0直接TX/V/target/classloss/新头/sourceRX输入。每role全七景固定后truth-last，126正式/105源V/252双对照配对分组/14IQ独立复算；30轮原loss/曝光/视图/cap核对，CPU scorer和原coarseAnchor同锁互斥且输出独立。六星地较SIG提高2/6、纯G提高6/6、默认DURA提高1/6。clean 84.867262%，低城区 69.287500%；低城区较SIG +0.452381、纯G +1.914881、默认DURA +0.397619个百分点。六景3pp通过0/6，clean严格保持=True，R17逐景1pp=True，旧低城区70%=False。G训练及保存1021.596186秒、含worker准备1022.589685秒、控制器至自动分析1211.514897秒。ConvLinearMAC25068072/2prior4CG27FFT，同卡G+F B1默认4.684524ms、本版4.678386ms。共享GPU单次墙钟不证明因果加速，数值局部等价不能替代正式识别质量，有限模型成绩不认证物理恢复极限。默认DURA保持，完整目标ACTIVE。 单seed/已暴露工程基准，完整TX/RX/day、净纠错、源与正式差异及成本见run分析。


## 周期坐标残差效用：固定源判定

20261031-diagnostic-DURA-source-period20-utility-s2026092701-r01/58da3da0。VERIFIED / ANALYZED_SOURCE_ONLY，主假设FAIL。固定defaultDURA/source512（128L/384U）×缓存01/20、1024次观测/512物理记录，0G加载/前向、0优化/TX/V/target。固定窗口80:160/四个20点周期，将pairedclean−native误差分为周期均值P和窗口残差R；主R与同修正L2的全窗最近clean控制比较。R的clean F选择一致数485/498，相对native486/498净−1/0、相对同能量全窗496/504净−11/−6；两view额外效用失败，第一view源RX保护也失败，其他clean/scene/Fcos条件通过。periodic修正492/501净+6/+3仅描述，修正能量不同且没有其单独同能量控制，不换主/窗口/period/阈值或据此加训练loss。clean窗口周期间残差能量约6.03%，窗口恢复误差中残差占44.14%/56.69%，高相对误差不自动意味着更高身份效用。CPU2主循环3.079785秒、F完整输入3072+真实2非cleanLs clean/native缓存logits smoke4，F全state保持、grad为空。独立dense80×80投影/全部1024×3IQ与同能量/窗口外实际bits/原cache配对、36组144case统计和判定复算，最大IQ公式差1.338257e−7、指标差0、0新增NN。复核中FP32先减再升精度与FP64先升再减的标量分别重算并保留原2e−6约束，无模型重跑或判据变化。F50/source4df7/50轮11100步知识及旧defaultG/缓存历史成本披露；这是源训练池Fclean选择代理，非真TX、RX噪声/硬件因果、可部署修复或物理上限。默认DURA及完整目标ACTIVE保持。 详见run分析和独立复算。


## 两 prior 的发射机特征预测与解码条件：匹配单阶段 E30

20261031-phase1-DURA-SIGcontextPrior-E30-s2026092701-r01 与 20261031-phase1-DURA-SIGpriorAux-E30-s2026092701-r01。共同保留原四物理+输出 SIG alpha1/cap.5，新增 .01×两个 prior 中层 SIG 相对 MSE 均值；主版预测参与三个解码层调制，对照只训练约束、推理跳过预测头。主版 344295 参数、对照 308231；同源 F-only 缓存、同初值与随机流、原生执行、从零单 30 轮/7410，全七景自动正式测试，不源筛取消。两新版互为调制消融，共同主基线为原 SIG47591。一次 P0/P1 每候选 PASS；旧中层/FiLM/period20 等负结果保留，无资格或收益预判。F-guided、0 TX/V/target 训练、默认保持，完整目标 ACTIVE。


## prior 发射机特征参与解码及匹配辅助对照：完整 E30 七景

20261031-phase1-DURA-SIGcontextPrior-E30-s2026092701-r01 与 20261031-phase1-DURA-SIGpriorAux-E30-s2026092701-r01，执行 4452e09285b650536ab5e9dcc2acbfd9268c00e8。

VERIFIED / ANALYZED。发射机特征参与 prior 解码，从零单阶段30轮/7410步，L189000/U1701000、无G祖先。保留四物理+输出SIG alpha1/cap.5，新增 .01×两个prior中层SIG相对MSE均值，辅助梯度位于输出cap之外、全G裁剪1；主版预测参与三个解码层有界调制，对照推理跳过预测头。完整G344295参数/ConvLinearMAC25144872/两prior四CG27FFT，训练与冻结推理均native。源F-onlySIG40.32MB复用、新缓存F0，F50/source4df7/50轮11100步及历史缓存成本披露，F-guided；在线prefix/VJP各1.89m、辅助预测3.78m、完整F训练0，F不变，0TX/V/target训练/新分类头。每角色全七景预测后truth-last，126正式/105源/378三对照配对/14IQ独立复算，全30轮损失、辅助均值和权重、曝光、视图及原cap核对。六星地较匹配新版提高6/6、较原SIG提高6/6、较纯G提高6/6、较默认提高2/6。clean 84.799405%、低城区 69.658333%，3pp通过1/6、clean严格保持=True、R17逐景1pp=True、旧低城区70%=False。G训练保存1104.903494秒、含准备1105.953533秒、控制器至自动分析1292.594890秒；同卡G+F B1默认5.036128ms、本版5.133852ms。共享GPU墙钟不作因果加速或等质量速度比，有限模型不认证恢复极限，实际F表征不等于纯硬件特质。旧midSIG/FiLM/period20等负结果保留；默认DURA保持，完整目标ACTIVE。

VERIFIED / ANALYZED。prior 特征仅训练约束，从零单阶段30轮/7410步，L189000/U1701000、无G祖先。保留四物理+输出SIG alpha1/cap.5，新增 .01×两个prior中层SIG相对MSE均值，辅助梯度位于输出cap之外、全G裁剪1；主版预测参与三个解码层有界调制，对照推理跳过预测头。完整G308231参数/ConvLinearMAC25068072/两prior四CG27FFT，训练与冻结推理均native。源F-onlySIG40.32MB复用、新缓存F0，F50/source4df7/50轮11100步及历史缓存成本披露，F-guided；在线prefix/VJP各1.89m、辅助预测3.78m、完整F训练0，F不变，0TX/V/target训练/新分类头。每角色全七景预测后truth-last，126正式/105源/378三对照配对/14IQ独立复算，全30轮损失、辅助均值和权重、曝光、视图及原cap核对。六星地较匹配新版提高0/6、较原SIG提高6/6、较纯G提高6/6、较默认提高2/6。clean 84.966071%、低城区 69.476786%，3pp通过1/6、clean严格保持=True、R17逐景1pp=True、旧低城区70%=False。G训练保存1044.993129秒、含准备1046.173709秒、控制器至自动分析1216.978897秒；同卡G+F B1默认4.593904ms、本版4.594088ms。共享GPU墙钟不作因果加速或等质量速度比，有限模型不认证恢复极限，实际F表征不等于纯硬件特质。旧midSIG/FiLM/period20等负结果保留；默认DURA保持，完整目标ACTIVE。


## 固定权重的特征条件源检验：配对特征用途通过

20261031-diagnostic-DURA-source-priorContext-oracle-s2026092701-r01 / 9d266929d4e0eea27057eebeda8239fd24f0b6d9。固定 main SIGcontextPrior final30 的 source512×01/20 四路径干预已闭合，主条件PASS。正常与clean F选择一致487/500、关闭481/497、本条clean SIG498/505、循环移位480/498；同权重关闭净−6/−3、理想配对特征净+11/+5，5RX与全部场景保护通过。144组case/实际线性/输入置换独立复算，0新增NN；主CPU2 14.107593秒，G4096/F4096、smoke G12/F4，0TX/V/target/训练。源特权参照不可部署、不是真实TX准确率或恢复极限，固定权重可能有分布共同适配边界。支持另行登记更强内部特征预测假设，不自动加训；默认DURA和完整3pp/clean/同质量效率目标保持。 详见该run完整分析。


## 更深特征改善SIG预测：固定线性与非线性单E30对照

20261031-phase1-DURA-SIGdeepLinear-E30-s2026092701-r01 与 20261031-phase1-DURA-SIGdeepMLP-E30-s2026092701-r01。源固定权重条件oracle两视图主条件PASS后，独立登记两个从零单阶段30轮假设；读取已算出的encoder2pool4+encoder3pool2=192维，原浅层预测加深层零初始化残差，线性192→160与192→160-SiLU-160→160比较。406055/457575参数，原四物理+输出SIGalpha1cap.5+.01mean2aux及conditionerbound.1不改；初始main参数/SIG/IQ/RNG保留，G祖先0、F-only缓存复用，0TX/V/target训练。固定final30与全七景自动truth-last，新增两版互比并匹配main4452；默认保持、完整目标ACTIVE，不用源代理替代正式TX收益。


## 深层线性与非线性SIG预测：完整单E30七景

20261031-phase1-DURA-SIGdeepLinear-E30-s2026092701-r01 与 20261031-phase1-DURA-SIGdeepMLP-E30-s2026092701-r01；执行df7fe90d16910f7373d79ee93291f265506a8434。

VERIFIED / ANALYZED。深层线性SIG预测，单阶段从零30轮/7410，L189000/U1701000，G祖先0；原main344295架构增加从encoder2pool4+encoder3pool2形成的192维特征残差预测，原128→160浅预测保留。线性192→160零初始化、MLP192→160-SiLU-160→160末层零初始化；初始main权重/别名/SIG/IQ/随机流保持，MLP首层初始梯度0预期，真实新增末层辅助梯度非零。原四物理+输出SIGalpha1/cap.5+.01mean2priorSIG辅助、conditionerbound.1不改。完整G406055/ConvLinearMAC25206312/两prior四CG27FFT，native全部路径。源F-onlySIG40.32MB复用/newFcache0，F50/source4df7/50-11100及历史缓存成本披露，F-guided；在线prefix/VJP各1.89m，完整F训练0，0TX/V/target训练或分类头改动。源条件oracle两视图主条件PASS仅为学习假设证据，不自动继承为本版资格/目标收益/恢复极限；本版固定final30，每角色全七景预测后truth-last，126正式105源V378三对照配对14IQ独立复算，全30轮loss/均值权重/曝光/视图/cap/deep配置和来源核对。六星地较原特征条件版提高0/6、较另一深层版提高6/6、较原SIG提高4/6、较默认DURA提高1/6。clean84.833929%、低城区68.905952%，3pp通过0/6、clean严格保持=True、R17逐景1pp=True、旧lowU70%=False。G训练保存1065.143283秒、含准备1066.391048秒、控制器至自动分析1238.358895秒；同卡G+F B1默认4.643276ms、本版4.709474ms。共享GPU墙钟不作因果提速或等质量速度比，源代理和更深预测不能代替真实TX净纠错。默认DURA和完整3pp/clean/同质量效率目标保持ACTIVE。

VERIFIED / ANALYZED。深层MLP SIG预测，单阶段从零30轮/7410，L189000/U1701000，G祖先0；原main344295架构增加从encoder2pool4+encoder3pool2形成的192维特征残差预测，原128→160浅预测保留。线性192→160零初始化、MLP192→160-SiLU-160→160末层零初始化；初始main权重/别名/SIG/IQ/随机流保持，MLP首层初始梯度0预期，真实新增末层辅助梯度非零。原四物理+输出SIGalpha1/cap.5+.01mean2priorSIG辅助、conditionerbound.1不改。完整G457575/ConvLinearMAC25257512/两prior四CG27FFT，native全部路径。源F-onlySIG40.32MB复用/newFcache0，F50/source4df7/50-11100及历史缓存成本披露，F-guided；在线prefix/VJP各1.89m，完整F训练0，0TX/V/target训练或分类头改动。源条件oracle两视图主条件PASS仅为学习假设证据，不自动继承为本版资格/目标收益/恢复极限；本版固定final30，每角色全七景预测后truth-last，126正式105源V378三对照配对14IQ独立复算，全30轮loss/均值权重/曝光/视图/cap/deep配置和来源核对。六星地较原特征条件版提高0/6、较另一深层版提高0/6、较原SIG提高0/6、较默认DURA提高0/6。clean84.927381%、低城区68.705357%，3pp通过0/6、clean严格保持=True、R17逐景1pp=True、旧lowU70%=False。G训练保存1114.365791秒、含准备1115.550746秒、控制器至自动分析1301.024889秒；同卡G+F B1默认4.619074ms、本版4.708815ms。共享GPU墙钟不作因果提速或等质量速度比，源代理和更深预测不能代替真实TX净纠错。默认DURA和完整3pp/clean/同质量效率目标保持ACTIVE。
