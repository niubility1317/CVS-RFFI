# R54：同一压缩拓扑的源通道筛选正式结果

VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。activation_energy 273765参数，七景较父提高1/下降6，lowurban 66.435714%（较父-1.074405pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版41.6582/40.9480ms，G+F B1父/本版53.0625/52.8909ms；projection_sensitivity 273765参数，七景较父提高1/下降6，lowurban 66.919048%（较父-0.591071pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.7906/44.9914ms，G+F B1父/本版58.4112/57.2981ms。activation_energy相对同宽权重筛选R53/prune_decay七景提高1/下降6，lowurban差-2.091667pp；projection_sensitivity相对同宽权重筛选R53/prune_decay七景提高1/下降6，lowurban差-1.608333pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。

状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

唯一直接初始化与冻结IQ教师仍为用户指定R46 prune_refine350521。两行均沿用完整273765拓扑，区别是source激活能量与随机输出投影gain敏感度选择通道；原3prior/6CG、GN4组配额与共同prior map、全部coarse/action/prior联合、pairedclean IQ+.05谱误差+1→.1IQ教师、E20/global256micro256均保持。固定同宽weight-only控制复用R53 prune_decay，不重复训练。校准共用一次128源样本/256冻结G输入、0optimizer更新，另计计算与曝光；不称data-free初始化。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。属于有clean模拟配对的TX无标签源域自监督。

方法commit `554c52baf75ab8da2618a5480851b74271018103`。完整登记、数据与祖先、命令和停止规则见[登记](experiment.json)；全部20轮与TX/RX/day见[结果](results.json)及[逐组CSV](row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|activation_energy|273765|21.897689%|True|False|66.435714|
|projection_sensitivity|273765|21.897689%|True|True|66.919048|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|activation_energy|clean|84.455357|84.153089|-0.001190|-0.050595|-0.131548|-371|
|activation_energy|practical_high|82.469048|82.061644|+0.016667|+0.050595|+1.150000|+70117|
|activation_energy|practical_mid|80.848810|80.435107|-0.373214|-0.161310|+1.270833|+68766|
|activation_energy|practical_low_suburban|78.913690|78.541988|-0.706548|-0.462500|+0.917262|+67005|
|activation_energy|practical_high_urban|81.212500|80.826914|-0.174405|-0.104762|+0.858929|+69744|
|activation_energy|practical_mid_urban|73.793452|73.635062|-0.653571|-0.433333|+0.108333|+66041|
|activation_energy|practical_low_urban|66.435714|66.645823|-1.074405|-0.336905|-0.780952|+60709|
|projection_sensitivity|clean|84.617262|84.331613|+0.160714|+0.111310|+0.030357|-99|
|projection_sensitivity|practical_high|81.789286|81.336497|-0.663095|-0.629167|+0.470238|+68975|
|projection_sensitivity|practical_mid|80.459524|80.061037|-0.762500|-0.550595|+0.881548|+68112|
|projection_sensitivity|practical_low_suburban|78.921429|78.659155|-0.698810|-0.454762|+0.925000|+67018|
|projection_sensitivity|practical_high_urban|80.740476|80.327364|-0.646429|-0.576786|+0.386905|+68951|
|projection_sensitivity|practical_mid_urban|73.769643|73.786442|-0.677381|-0.457143|+0.084524|+66001|
|projection_sensitivity|practical_low_urban|66.919048|67.437611|-0.591071|+0.146429|-0.297619|+61521|

同宽度通道筛选对照：以下只比较预登记R53 prune_decay权重筛选控制，保留全部七景差值；额外共用源校准256次冻结G输入另计，不以正式结果重新选择父、宽度或筛选方式。

|源通道筛选行|权重筛选控制|场景|准确率变化（pp）|MacroF1变化（pp）|
|---|---|---|---:|---:|
|activation_energy|R53/prune_decay|clean|-0.073214|-0.090822|
|activation_energy|R53/prune_decay|practical_high|+0.060119|+0.090149|
|activation_energy|R53/prune_decay|practical_mid|-0.500000|-0.493910|
|activation_energy|R53/prune_decay|practical_low_suburban|-0.944643|-0.952016|
|activation_energy|R53/prune_decay|practical_high_urban|-0.199405|-0.164657|
|activation_energy|R53/prune_decay|practical_mid_urban|-1.267857|-1.196345|
|activation_energy|R53/prune_decay|practical_low_urban|-2.091667|-1.926580|
|projection_sensitivity|R53/prune_decay|clean|+0.088690|+0.087702|
|projection_sensitivity|R53/prune_decay|practical_high|-0.619643|-0.634999|
|projection_sensitivity|R53/prune_decay|practical_mid|-0.889286|-0.867981|
|projection_sensitivity|R53/prune_decay|practical_low_suburban|-0.936905|-0.834849|
|projection_sensitivity|R53/prune_decay|practical_high_urban|-0.671429|-0.664207|
|projection_sensitivity|R53/prune_decay|practical_mid_urban|-1.291667|-1.044965|
|projection_sensitivity|R53/prune_decay|practical_low_urban|-1.608333|-1.134793|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|activation_energy|106138/79477/88150|1388124|30761512|
|projection_sensitivity|106138/79477/88150|1388124|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|activation_energy|G_only_B1|41.6582/40.9480|42.6066|24.1912|67194880/1084928|
|activation_energy|G_only_B32|56.7120/50.9963|52.4088|625.4821|276216832/227691520|
|activation_energy|G_plus_F_B1|53.0625/52.8909|60.4331|18.5866|67194880/1201152|
|activation_energy|G_plus_F_B32|66.2411/60.4560|64.2429|520.8585|276216832/227691520|
|projection_sensitivity|G_only_B1|45.7906/44.9914|49.7962|22.0389|67194880/1084928|
|projection_sensitivity|G_only_B32|81.1216/71.6087|80.6873|442.4472|276216832/227691520|
|projection_sensitivity|G_plus_F_B1|58.4112/57.2981|60.8298|17.3375|67194880/1201152|
|projection_sensitivity|G_plus_F_B32|91.1957/81.1561|86.0988|395.5609|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。


共用源校准：128条物理样本（64L_s/64U_s）、256次冻结G输入、耗时9.5506s；source received01/20、两次Rademacher投影；0TX/F/V/target/optimizer更新、父状态不变。具体分数/map/行位置保持私有，不是运行时输入。

activation_energy首轮→末轮：loss 0.14409604→0.023421186；iq_nmse 0.084582491→0.022831636；spectral_error 0.020308574→0.0059982311；teacher_stage_nmse 0.058498123→0.0028963785；中位每轮49.705s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
projection_sensitivity首轮→末轮：loss 0.1344389→0.023329311；iq_nmse 0.079862335→0.022750676；spectral_error 0.018631943→0.0059851801；teacher_stage_nmse 0.053644969→0.0027937648；中位每轮52.969s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。两行同E20训练预算、同batch256/micro256/directbuffer、同1→.1教师；额外共用256次冻结G源校准、0optimizer更新，不重复计算为两份或隐藏为0数据初始化，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_source_prune_r54_20261021/models/<arm>/final.pt；使用experiments.pamf_r46_source_prune.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。
