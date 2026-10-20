# R60：既有压缩图训练期中间特征保留的正式结果

VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。context_hint 273765参数，七景较父提高7/下降0，lowurban 68.232143%（较父+0.722024pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版47.6136/47.1164ms，G+F B1父/本版59.5221/58.5740ms；all_hint 273765参数，七景较父提高1/下降6，lowurban 67.719643%（较父+0.209524pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.9480/45.2744ms，G+F B1父/本版57.7602/56.0171ms。context_hint相对预登记同图同预算无hint控制R53prune_decay七景提高3/下降4，lowurban差-0.295238pp；all_hint相对预登记同图同预算无hint控制R53prune_decay七景提高0/下降7，lowurban差-0.807738pp。相同.02context权重，all_hint比context_hint额外.02prior：七景提高0/下降7，lowurban差-0.512500pp。原参数80+20=100轮，context/prior各84模块的训练期特征约束；0额外G前向/校准IQ/新增参数，全部hooks保存前移除，额外训练开销已列账。原指定350521直接父保持，正负结果保留；不把容差、减参或中间特征拟合当作整体提升或逐细节TX真实性。原完整目标ACTIVE。

**VERIFIED / ANALYZED**。两固定行完整E20/4940、source1800×7和正式168000×7；逐行七预测固定后独立truth-last、所属controller/worker实际退出。全部正负结果保留，不反馈调参或选择性重跑。

性能优先。两行保持同273765参数普通GN/U-Net/3prior6CG图、weight-only通道图和指定R46 prune_refine350521直接父，不继承R53/R55/R57/R58训练后的学生。均使用原pairedclean IQ+.05谱误差+线性1→.1三阶段IQ教师，coarse/action/prior联合优化；context_hint加.02context保留通道中间特征NMSE，all_hint另加.02prior。context/prior各84个GN的post-affine/pre-SiLU输出，全部保留通道及时间位置，每IQ每module除以对应教师能量floor1e-4，分别module等权。相同context系数，prior是额外项，不把两部分平均后称为相同系数。

教师是同一冻结350521恢复器，特征detach、0教师更新，源训练读取TX标签/F权重/F输出/V/target均为0。使用合法source模拟clean配对和U_s，属于TX无标签源自监督，不能称无clean盲学习。特征来自已有父/学生前向；额外G前向、校准IQ、新可学习参数、推理buffer/分支均为0。hooks已在保存前移除；训练激活和归约开销实际列账。核心原型及系数在读取R58源验证/正式分数前固定。

方法commit `04357684a7c069308e6c89a64d3bb51d54376fdf`。完整登记、数据与祖先、命令和停止规则见[登记](experiment.json)；全部20轮与TX/RX/day见[结果](results.json)及[逐组CSV](row_metrics.csv)。

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
