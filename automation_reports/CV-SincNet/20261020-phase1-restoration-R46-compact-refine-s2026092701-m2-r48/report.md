# R48：35.05万保留优化与27.38万进一步压缩的正式结果

性能优先：retain_refine七景提高1景、下降6景；prune_refine七景提高7景、下降0景。逐景1pp仅是退化上限，不把通过容差、减参或训练提速自动表述为性能提升；逐场景数据优先于均值。指定R46 prune_refine350521继续为唯一直接基准，已揭示正式结果不用于另选父或选择性重跑。

状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

用户在R46结果揭示后指定350521父，exact retain_refine为同预算训练优化控制，273765 prune_refine为GN4重要性进一步剪枝；同原接收链/3prior/6CG，无其他恢复网络。两行均pairedclean IQ+.05三尺度相对功率log1p误差+1→.1冻结父三stageIQ教师，所有coarse/action/prior联合。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。无TX标签模拟配对源域自监督，不称无clean盲学习。

方法commit `86b89ba6bc09b0b3f10fcaff8fc7cd1ecf6d0508`。完整登记、数据与祖先、命令和停止规则见[登记](experiment.json)；全部20轮与TX/RX/day见[结果](results.json)及[逐组CSV](row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|retain_refine|350521|0.000000%|True|True|66.852976|
|prune_refine|273765|21.897689%|True|True|68.257738|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|retain_refine|clean|84.505357|84.212093|+0.048810|-0.000595|-0.081548|-287|
|retain_refine|practical_high|81.813690|81.288963|-0.638690|-0.604762|+0.494643|+69016|
|retain_refine|practical_mid|80.352381|79.858340|-0.869643|-0.657738|+0.774405|+67932|
|retain_refine|practical_low_suburban|78.857143|78.440328|-0.763095|-0.519048|+0.860714|+66910|
|retain_refine|practical_high_urban|80.707143|80.235571|-0.679762|-0.610119|+0.353571|+68895|
|retain_refine|practical_mid_urban|73.701190|73.556200|-0.745833|-0.525595|+0.016071|+65886|
|retain_refine|practical_low_urban|66.852976|67.122362|-0.657143|+0.080357|-0.363690|+61410|
|prune_refine|clean|84.536310|84.249356|+0.079762|+0.030357|-0.050595|-235|
|prune_refine|practical_high|82.645238|82.236253|+0.192857|+0.226786|+1.326190|+70413|
|prune_refine|practical_mid|81.509524|81.114133|+0.287500|+0.499405|+1.931548|+69876|
|prune_refine|practical_low_suburban|80.002976|79.665539|+0.382738|+0.626786|+2.006548|+68835|
|prune_refine|practical_high_urban|81.564286|81.178715|+0.177381|+0.247024|+1.210714|+70335|
|prune_refine|practical_mid_urban|75.042262|74.871308|+0.595238|+0.815476|+1.357143|+68139|
|prune_refine|practical_low_urban|68.257738|68.419563|+0.747619|+1.485119|+1.041071|+63770|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|retain_refine|136890/101037/112594|1685404|39081004|
|prune_refine|106138/79477/88150|1376476|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|retain_refine|G_only_B1|44.5498/44.5913|45.8523|22.4793|67017728/67017728|
|retain_refine|G_only_B32|57.7966/57.9484|60.1130|549.9111|276216832/276216832|
|retain_refine|G_plus_F_B1|55.1817/55.6139|57.8439|17.9535|67017728/67017728|
|retain_refine|G_plus_F_B32|68.5971/68.0839|70.4111|468.6024|276216832/276216832|
|prune_refine|G_only_B1|41.4050/40.9928|42.5749|24.4258|67017728/1084928|
|prune_refine|G_only_B32|56.9253/51.8359|52.2846|619.6688|276216832/227691520|
|prune_refine|G_plus_F_B1|52.2722/51.4586|53.1965|19.3880|67017728/1201152|
|prune_refine|G_plus_F_B32|65.7682/60.2725|61.1218|530.7903|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。

retain_refine首轮→末轮：loss 0.025517709→0.021891554；iq_nmse 0.024150966→0.021491357；spectral_error 0.0063670598→0.0056597645；teacher_stage_nmse 0.0010483899→0.0011720886；中位每轮174.026s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_refine首轮→末轮：loss 0.13370983→0.023518632；iq_nmse 0.079269997→0.022922496；spectral_error 0.019420517→0.0060034009；teacher_stage_nmse 0.053468811→0.0029596517；中位每轮162.135s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。优化控制和剪枝行同预算、同损失配方，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_compact_refine_r48_20261020/models/<arm>/final.pt；使用experiments.pamf_r46_compact_refine.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。
