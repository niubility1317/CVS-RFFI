# R53：持续IQ教师约束的正式性能与推理成本

VERIFIED/ANALYZED：三行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及三套独立权重CPU严格重载均已核实。prune_decay 273765参数，七景较父提高6/下降1，lowurban 68.527381%（较父+1.017262pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版44.1931/43.3225ms，G+F B1父/本版56.2200/54.1512ms；retain_anchor 350521参数，七景较父提高2/下降5，lowurban 67.636310%（较父+0.126190pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版44.2953/44.1067ms，G+F B1父/本版55.4548/54.3477ms；prune_anchor 273765参数，七景较父提高4/下降3，lowurban 68.347619%（较父+0.837500pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版43.4629/42.8090ms，G+F B1父/本版54.6030/54.1270ms。retain_anchor相对同宽R49/retain_refine七景提高7/下降0，lowurban差+0.814286pp；prune_anchor相对同宽R53/prune_decay七景提高0/下降7，lowurban差-0.179762pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。

状态：VERIFIED / ANALYZED。三固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

用户指定350521父保持唯一直接来源；现有350521/273765两宽度，同原接收链/3prior/6CG。retain_anchor/prune_anchor的冻结父三stage IQ教师系数持续为1，prune_decay控制仍1→.1；其余pairedclean IQ+.05三尺度相对功率STFT、全部coarse/action/prior联合、E20/global256micro256均保持。350521衰减控制复用R49 retain_refine，不重复训练；273765衰减控制在R53新训，R48micro64仅历史对照。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。属于有clean模拟配对的TX无标签源域自监督。

方法commit `df36c3d3da4e026b747776b65b58d4921dc1a71e`。完整登记、数据与祖先、命令和停止规则见[登记](experiment.json)；全部20轮与TX/RX/day见[结果](results.json)及[逐组CSV](row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|prune_decay|273765|21.897689%|True|True|68.527381|
|retain_anchor|350521|0.000000%|True|True|67.636310|
|prune_anchor|273765|21.897689%|True|True|68.347619|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|prune_decay|clean|84.528571|84.243911|+0.072024|+0.022619|-0.058333|-248|
|prune_decay|practical_high|82.408929|81.971496|-0.043452|-0.009524|+1.089881|+70016|
|prune_decay|practical_mid|81.348810|80.929018|+0.126786|+0.338690|+1.770833|+69606|
|prune_decay|practical_low_suburban|79.858333|79.494004|+0.238095|+0.482143|+1.861905|+68592|
|prune_decay|practical_high_urban|81.411905|80.991570|+0.025000|+0.094643|+1.058333|+70079|
|prune_decay|practical_mid_urban|75.061310|74.831407|+0.614286|+0.834524|+1.376190|+68171|
|prune_decay|practical_low_urban|68.527381|68.572403|+1.017262|+1.754762|+1.310714|+64223|
|retain_anchor|clean|84.543452|84.243584|+0.086905|+0.037500|-0.043452|-223|
|retain_anchor|practical_high|82.301190|81.851466|-0.151190|-0.117262|+0.982143|+69835|
|retain_anchor|practical_mid|80.898214|80.485980|-0.323810|-0.111905|+1.320238|+68849|
|retain_anchor|practical_low_suburban|79.405952|79.073483|-0.214286|+0.029762|+1.409524|+67832|
|retain_anchor|practical_high_urban|81.206548|80.795877|-0.180357|-0.110714|+0.852976|+69734|
|retain_anchor|practical_mid_urban|74.374405|74.274720|-0.072619|+0.147619|+0.689286|+67017|
|retain_anchor|practical_low_urban|67.636310|67.936572|+0.126190|+0.863690|+0.419643|+62726|
|prune_anchor|clean|84.490476|84.194383|+0.033929|-0.015476|-0.096429|-312|
|prune_anchor|practical_high|82.341071|81.911280|-0.111310|-0.077381|+1.022024|+69902|
|prune_anchor|practical_mid|81.207143|80.808851|-0.014881|+0.197024|+1.629167|+69368|
|prune_anchor|practical_low_suburban|79.764286|79.463331|+0.144048|+0.388095|+1.767857|+68434|
|prune_anchor|practical_high_urban|81.269643|80.863416|-0.117262|-0.047619|+0.916071|+69840|
|prune_anchor|practical_mid_urban|75.030952|74.898483|+0.583929|+0.804167|+1.345833|+68120|
|prune_anchor|practical_low_urban|68.347619|68.578503|+0.837500|+1.575000|+1.130952|+63921|

同宽度教师系数消融：以下只比较预登记对应控制，保留七景全部差值；不按正式分数重新选择父或学生。350521控制来自R49，273765控制来自本轮prune_decay。

|持续约束行|衰减控制|场景|准确率变化（pp）|MacroF1变化（pp）|
|---|---|---|---:|---:|
|retain_anchor|R49/retain_refine|clean|+0.102381|+0.099986|
|retain_anchor|R49/retain_refine|practical_high|+0.419643|+0.484957|
|retain_anchor|R49/retain_refine|practical_mid|+0.472024|+0.533882|
|retain_anchor|R49/retain_refine|practical_low_suburban|+0.469048|+0.523330|
|retain_anchor|R49/retain_refine|practical_high_urban|+0.444643|+0.498154|
|retain_anchor|R49/retain_refine|practical_mid_urban|+0.643452|+0.663369|
|retain_anchor|R49/retain_refine|practical_low_urban|+0.814286|+0.802679|
|prune_anchor|R53/prune_decay|clean|-0.038095|-0.049528|
|prune_anchor|R53/prune_decay|practical_high|-0.067857|-0.060216|
|prune_anchor|R53/prune_decay|practical_mid|-0.141667|-0.120166|
|prune_anchor|R53/prune_decay|practical_low_suburban|-0.094048|-0.030672|
|prune_anchor|R53/prune_decay|practical_high_urban|-0.142262|-0.128155|
|prune_anchor|R53/prune_decay|practical_mid_urban|-0.030357|+0.067076|
|prune_anchor|R53/prune_decay|practical_low_urban|-0.179762|+0.006100|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|prune_decay|106138/79477/88150|1378332|30761512|
|retain_anchor|136890/101037/112594|1687260|39081004|
|prune_anchor|106138/79477/88150|1378332|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|prune_decay|G_only_B1|44.1931/43.3225|65.2033|20.2011|67194880/1084928|
|prune_decay|G_only_B32|94.5927/89.7459|98.4629|385.7593|276216832/227691520|
|prune_decay|G_plus_F_B1|56.2200/54.1512|75.3409|16.9726|67194880/1201152|
|prune_decay|G_plus_F_B32|105.8081/93.7408|99.0755|385.0358|276216832/227691520|
|retain_anchor|G_only_B1|44.2953/44.1067|45.4732|22.6591|67194880/67194880|
|retain_anchor|G_only_B32|58.3138/58.2915|59.2950|548.8869|276216832/276216832|
|retain_anchor|G_plus_F_B1|55.4548/54.3477|57.0154|18.2470|67194880/67194880|
|retain_anchor|G_plus_F_B32|67.9548/67.4781|69.9772|472.5757|276216832/276216832|
|prune_anchor|G_only_B1|43.4629/42.8090|43.7215|23.3329|67194880/1084928|
|prune_anchor|G_only_B32|57.9682/52.2611|53.6952|611.0934|276216832/227691520|
|prune_anchor|G_plus_F_B1|54.6030/54.1270|55.5921|18.4777|67194880/1201152|
|prune_anchor|G_plus_F_B32|67.7278/60.9471|62.6041|522.1784|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。

prune_decay首轮→末轮：loss 0.13370979→0.023484957；iq_nmse 0.079269947→0.02289074；spectral_error 0.019420509→0.0060005；teacher_stage_nmse 0.053468816→0.0029419191；中位每轮50.954s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
retain_anchor首轮→末轮：loss 0.025517714→0.022345617；iq_nmse 0.02415097→0.021504429；spectral_error 0.0063670601→0.0056991055；teacher_stage_nmse 0.0010483912→0.00055623289；中位每轮56.365s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_anchor首轮→末轮：loss 0.13370979→0.025510478；iq_nmse 0.079269968→0.02297493；spectral_error 0.019420515→0.0060431003；teacher_stage_nmse 0.053468798→0.0022333932；中位每轮49.658s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。三行同预算、同全局batch256/micro256/directbuffer，教师系数按固定行分别为恒1或1→.1，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_anchor_refine_r53_20261021/models/<arm>/final.pt；使用experiments.pamf_r46_anchor_refine.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。
