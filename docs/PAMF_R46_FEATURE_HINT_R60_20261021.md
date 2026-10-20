# R60：既有压缩图训练期中间特征保留的正式结果

VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。context_hint 273765参数，七景较父提高7/下降0，lowurban 68.232143%（较父+0.722024pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版47.6136/47.1164ms，G+F B1父/本版59.5221/58.5740ms；all_hint 273765参数，七景较父提高1/下降6，lowurban 67.719643%（较父+0.209524pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.9480/45.2744ms，G+F B1父/本版57.7602/56.0171ms。context_hint相对预登记同图同预算无hint控制R53prune_decay七景提高3/下降4，lowurban差-0.295238pp；all_hint相对预登记同图同预算无hint控制R53prune_decay七景提高0/下降7，lowurban差-0.807738pp。相同.02context权重，all_hint比context_hint额外.02prior：七景提高0/下降7，lowurban差-0.512500pp。原参数80+20=100轮，context/prior各84模块的训练期特征约束；0额外G前向/校准IQ/新增参数，全部hooks保存前移除，额外训练开销已列账。原指定350521直接父保持，正负结果保留；不把容差、减参或中间特征拟合当作整体提升或逐细节TX真实性。原完整目标ACTIVE。

**VERIFIED / ANALYZED**。两固定行完整E20/4940、source1800×7和正式168000×7；逐行七预测固定后独立truth-last、所属controller/worker实际退出。全部正负结果保留，不反馈调参或选择性重跑。

性能优先。两行保持同273765参数普通GN/U-Net/3prior6CG图、weight-only通道图和指定R46 prune_refine350521直接父，不继承R53/R55/R57/R58训练后的学生。均使用原pairedclean IQ+.05谱误差+线性1→.1三阶段IQ教师，coarse/action/prior联合优化；context_hint加.02context保留通道中间特征NMSE，all_hint另加.02prior。context/prior各84个GN的post-affine/pre-SiLU输出，全部保留通道及时间位置，每IQ每module除以对应教师能量floor1e-4，分别module等权。相同context系数，prior是额外项，不把两部分平均后称为相同系数。

教师是同一冻结350521恢复器，特征detach、0教师更新，源训练读取TX标签/F权重/F输出/V/target均为0。使用合法source模拟clean配对和U_s，属于TX无标签源自监督，不能称无clean盲学习。特征来自已有父/学生前向；额外G前向、校准IQ、新可学习参数、推理buffer/分支均为0。hooks已在保存前移除；训练激活和归约开销实际列账。核心原型及系数在读取R58源验证/正式分数前固定。

方法commit `04357684a7c069308e6c89a64d3bb51d54376fdf`。完整登记、数据与祖先、命令和停止规则见[登记](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-feature-hint-s2026092701-m2-r60/experiment.json)；全部20轮与TX/RX/day见[结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-feature-hint-s2026092701-m2-r60/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-feature-hint-s2026092701-m2-r60/row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|context_hint|273765|21.897689%|True|True|68.232143|
|all_hint|273765|21.897689%|True|True|67.719643|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|context_hint|clean|84.464881|84.167214|+0.008333|-0.041071|-0.122024|-355|
|context_hint|practical_high|82.590476|82.177510|+0.138095|+0.172024|+1.271429|+70321|
|context_hint|practical_mid|81.388095|80.990628|+0.166071|+0.377976|+1.810119|+69672|
|context_hint|practical_low_suburban|79.847619|79.506400|+0.227381|+0.471429|+1.851190|+68574|
|context_hint|practical_high_urban|81.552976|81.160419|+0.166071|+0.235714|+1.199405|+70316|
|context_hint|practical_mid_urban|74.934524|74.760164|+0.487500|+0.707738|+1.249405|+67958|
|context_hint|practical_low_urban|68.232143|68.429644|+0.722024|+1.459524|+1.015476|+63727|
|all_hint|clean|84.380952|84.082116|-0.075595|-0.125000|-0.205952|-496|
|all_hint|practical_high|81.850000|81.379149|-0.602381|-0.568452|+0.530952|+69077|
|all_hint|practical_mid|80.648810|80.182269|-0.573214|-0.361310|+1.070833|+68430|
|all_hint|practical_low_suburban|79.056548|78.657570|-0.563690|-0.319643|+1.060119|+67245|
|all_hint|practical_high_urban|80.876190|80.417467|-0.510714|-0.441071|+0.522619|+69179|
|all_hint|practical_mid_urban|74.345238|74.059649|-0.101786|+0.118452|+0.660119|+66968|
|all_hint|practical_low_urban|67.719643|67.703894|+0.209524|+0.947024|+0.502976|+62866|

预登记同宽控制：R53 prune_decay与新行具有相同273765图、weight-only初始化、E20/4940、micro256及IQ教师日程，只读复用不重复训练；新增特征损失与训练期开销分别披露，不据正式分数重新选择父或学生。

|特征约束行|无特征损失控制|场景|准确率变化（pp）|MacroF1变化（pp）|
|---|---|---|---:|---:|
|context_hint|R53/prune_decay|clean|-0.063690|-0.076697|
|context_hint|R53/prune_decay|practical_high|+0.181548|+0.206014|
|context_hint|R53/prune_decay|practical_mid|+0.039286|+0.061610|
|context_hint|R53/prune_decay|practical_low_suburban|-0.010714|+0.012396|
|context_hint|R53/prune_decay|practical_high_urban|+0.141071|+0.168849|
|context_hint|R53/prune_decay|practical_mid_urban|-0.126786|-0.071243|
|context_hint|R53/prune_decay|practical_low_urban|-0.295238|-0.142759|
|all_hint|R53/prune_decay|clean|-0.147619|-0.161796|
|all_hint|R53/prune_decay|practical_high|-0.558929|-0.592347|
|all_hint|R53/prune_decay|practical_mid|-0.700000|-0.746748|
|all_hint|R53/prune_decay|practical_low_suburban|-0.801786|-0.836433|
|all_hint|R53/prune_decay|practical_high_urban|-0.535714|-0.574104|
|all_hint|R53/prune_decay|practical_mid_urban|-0.716071|-0.771758|
|all_hint|R53/prune_decay|practical_low_urban|-0.807738|-0.868509|

同一context系数下，all_hint相对context_hint的额外prior约束差值：

|场景|准确率差（pp）|MacroF1差（pp）|
|---|---:|---:|
|clean|-0.083929|-0.085098|
|practical_high|-0.740476|-0.798361|
|practical_mid|-0.739286|-0.808359|
|practical_low_suburban|-0.791071|-0.848830|
|practical_high_urban|-0.676786|-0.742953|
|practical_mid_urban|-0.589286|-0.700516|
|practical_low_urban|-0.512500|-0.725750|

训练成本（不同GPU和时段仅作描述，不据绝对值排名）：

|行|每轮中位秒|完整训练秒|训练进程peak allocated字节|context/prior特征向量数|
|---|---:|---:|---:|---|
|context_hint|74.8928|1499.1237|1810064384|105840000/0|
|all_hint|75.5263|1518.9311|2382277632|105840000/105840000|

特征向量是每IQ每GN的一次误差归约，context共105840000，prior为0或105840000；不是额外独立IQ或G前向。训练peak来自本进程完整训练，包含模型/教师/快照/优化器/激活，不称独立推理显存。


|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|context_hint|106138/79477/88150|1389212|30761512|
|all_hint|106138/79477/88150|1389212|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|context_hint|G_only_B1|47.6136/47.1164|51.8190|21.0701|67194880/1084928|
|context_hint|G_only_B32|80.6490/74.1270|88.2660|422.9428|276216832/227691520|
|context_hint|G_plus_F_B1|59.5221/58.5740|63.2654|16.9495|67194880/1201152|
|context_hint|G_plus_F_B32|90.8885/81.5730|91.4951|388.8751|276216832/227691520|
|all_hint|G_only_B1|45.9480/45.2744|49.5937|22.0338|67194880/1084928|
|all_hint|G_only_B32|57.9943/52.1443|53.4461|613.1209|276216832/227691520|
|all_hint|G_plus_F_B1|57.7602/56.0171|72.6215|17.1782|67194880/1201152|
|all_hint|G_plus_F_B32|102.6253/89.2920|105.6899|354.7944|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。

context_hint首轮→末轮：loss 0.13894124→0.024618948；iq_nmse 0.078929561→0.022826246；spectral_error 0.019337447→0.0059842263；teacher_stage_nmse 0.053116139→0.0027394729；teacher_context_feature_nmse 0.29643346→0.060977155；teacher_prior_feature_nmse 0→0；中位每轮74.893s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
all_hint首轮→末轮：loss 0.14591514→0.02639591；iq_nmse 0.078442476→0.022919142；spectral_error 0.019240697→0.0059924539；teacher_stage_nmse 0.052686813→0.0024632725；teacher_context_feature_nmse 0.29558136→0.059684519；teacher_prior_feature_nmse 0.39560947→0.086856384；中位每轮75.526s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。两行同预算、同全局batch256/micro256/directbuffer、同1→.1 IQ教师；训练时分别+.02context或+.02context+.02prior，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_feature_hint_r60_20261021/models/<arm>/final.pt；使用experiments.pamf_r46_feature_hint.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。


## 固定方案与历史发布记录

# R60 原型：训练时保留压缩图的中间响应

状态：**VERIFIED / WAITING_R58_ACTUAL_EXIT**，已发布、未训练；唯一一次独立 P0/P1 审查通过；完整 launcher 已实现，47 项聚焦检查和 CLI help 通过，无源验证或正式性能结论。方案及核心原型在读取 R58 源验证和正式分数前固定。直接使用指定 R46 prune_refine 350521 父，不继承 R53/R55/R57/R58 训练后的学生。

只优化现有压缩网络的训练：保持原weight-only剪枝的273765参数/context185615、普通GN4、U-Net与3prior6CG，与R53 prune_decay相同数据自由通道图和初始化。没有新增推理参数、buffer、分支或更换恢复网络。新增训练时中间响应约束，教师仍为同一冻结350521恢复器，0TX标签/F权重/F输出/V/target训练。

固定两行context_hint/all_hint。两行均保留pairedclean IQ NMSE＋0.05谱误差＋线性1→0.1三阶段IQ教师，在此基础上加0.02×context特征NMSE；all_hint再加0.02×prior特征NMSE。相同context系数，比较prior额外约束，不能称两个位置平均后的系数相同。168个GN的post-affine/pre-SiLU对应保留通道全部时间位置参与；context和prior各84个模块，分别module等权。每IQ每module误差除以所选教师通道能量（下限1e-4），不跨样本计算。教师特征detach、教师无梯度；hooks仅训练期间存在，完成后移除，不写入推理checkpoint。

中间特征来自原本就执行的教师和学生前向，额外G前向和源校准输入为0；会增加训练激活、归约及内存开销，必须实测，不能称训练免费。原R07固定6300L_s＋56700U_s模拟received/clean配对和顺序不变；每行E20/4940、global256/micro256、重置AdamW，全部coarse/action/prior联合训练。原参数继承80＋新增20＝100轮，无新可学习参数；源曝光L126000/U1134000、teacher/student各1260000，原串行140pass、组件轮次和300与F50成本分别报告。

主要既有控制为预登记R53 prune_decay：相同273765图、weight-only初始化、E20/4940、IQ教师日程、global256/micro256及源曝光，差别为中间特征损失与训练开销。该旧控制只读复用，不按正式分数重排或改变父、宽度、系数。新两行训练后无论源分数如何，自动source1800×7、原冻结F50/defaulthead正式168000×7及独立truth-last，另实测同卡G/G+F的B1/B32时延、吞吐和显存。

仅GPU3/2各一条本任务worker，须等待R58同卡所属训练/预测worker完成并实际退出，总计最多两条；CPU2/共享R41 score.lock，R59身份诊断按原矩阵完成。错误祖先/数据/物理ID、输出碰撞、非有限损失或不完整前向保留产物并停止，无自动重试；不触碰健康任务。完整pipeline已接好，真实checkpoint/2条L_s noquery smoke作为launcher第一步一次执行，PASS后自动继续；完整方法commit、唯一P0/P1审查和发布登记按原最小流程完成，不增加设计审批或重复检查。

性能优先。原70%/5pp与输入TX细节保真仍未完成；中间响应贴近教师本身不是正式识别收益或逐细节TX真实性证明。有clean模拟配对的TX无标签源自监督不冒称无clean盲学习。47项检查只验证接线、梯度、训练权限、严格重载、完整源预算和实际前序进程退出保护，不将合成IQ结果当作有效性。

REJECTED_EXTRA_GATE：头脑风暴 skill 的另行设计审批、书面计划审批不属于本项目八项最小发布流程；按用户明确约定，将方法、矩阵、停止规则保留在本报告，继续执行既有授权，不新增审批门槛。


VERIFIED/WAITING_R58_ACTUAL_EXIT：归档单SHA/远端compile、真实350521父/F50/2Ls noquerysmoke通过，两scope全组件有限非零梯度、父及F不更新、训练hooks移除后独立权重严格重载一致。controller与两条CPU等待worker的实际PID/CWD/argv/CPU2/日志增长核实；均未建立GPU context、0epoch，等待R58同卡row完成并实际PID退出后才分配GPU3/2，跨run最多两条GPU worker。保持273765普通GN图和指定350521直接父，0新增推理参数/校准IQ/额外G前向，训练期开销待实测；尚无正式性能结论。每行固定E20/4940后自动source七景/成本/正式168000×7/独立truth-last，全部正负结果保留，原完整目标ACTIVE。
