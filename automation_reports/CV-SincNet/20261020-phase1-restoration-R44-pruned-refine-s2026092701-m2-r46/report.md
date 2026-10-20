# R46：44.28万保留优化与35.05万进一步压缩的正式结果

状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

用户在R44结果揭示后指定442805父，exact retain_refine为同预算训练优化控制，350521 prune_refine为GN4重要性进一步剪枝；同原接收链/3prior/6CG，无其他恢复网络。两行均pairedclean IQ+.05三尺度相对功率log1p误差+1→.1冻结父三stageIQ教师，所有coarse/action/prior联合。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。无TX标签模拟配对源域自监督，不称无clean盲学习。

方法commit `bbb8daf817de050b958d217b70d4c33f4aefe341`。完整登记、数据与祖先、命令和停止规则见[登记](experiment.json)；全部20轮与TX/RX/day见[结果](results.json)及[逐组CSV](row_metrics.csv)。

|行|完整参数|较442805减参|R17逐景1pp|父442805逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|retain_refine|442805|0.000000%|True|True|67.964881|
|prune_refine|350521|20.840776%|True|True|67.510119|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R38只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较442805父（pp）|较R38（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|retain_refine|clean|84.560119|84.273755|+0.054167|-0.069643|-0.026786|-195|
|retain_refine|practical_high|82.232143|81.825314|-0.186310|+0.084524|+0.913095|+69719|
|retain_refine|practical_mid|80.954762|80.562690|-0.055357|-0.018452|+1.376786|+68944|
|retain_refine|practical_low_suburban|79.498214|79.154995|+0.122024|+0.004762|+1.501786|+67987|
|retain_refine|practical_high_urban|81.225000|80.818665|-0.092262|+0.069643|+0.871429|+69765|
|retain_refine|practical_mid_urban|74.660714|74.422247|+0.433929|-0.115476|+0.975595|+67498|
|retain_refine|practical_low_urban|67.964881|68.032474|+1.192262|-0.304762|+0.748214|+63278|
|prune_refine|clean|84.456548|84.156318|-0.049405|-0.173214|-0.130357|-369|
|prune_refine|practical_high|82.452381|82.040850|+0.033929|+0.304762|+1.133333|+70089|
|prune_refine|practical_mid|81.222024|80.857090|+0.211905|+0.248810|+1.644048|+69393|
|prune_refine|practical_low_suburban|79.620238|79.351692|+0.244048|+0.126786|+1.623810|+68192|
|prune_refine|practical_high_urban|81.386905|81.006498|+0.069643|+0.231548|+1.033333|+70037|
|prune_refine|practical_mid_urban|74.447024|74.361224|+0.220238|-0.329167|+0.761905|+67139|
|prune_refine|practical_low_urban|67.510119|67.825990|+0.737500|-0.759524|+0.293452|+62514|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|同卡父/本版median ms|resident/峰值字节|
|---|---|---:|---:|---|---|
|retain_refine|176962/125509/140334|2053980|49266224|48.1976/48.5497|19108352/336296960|
|prune_refine|136890/101037/112594|1686812|39081004|47.9998/43.5050|18735616/85753344|

同卡16条sourceV、batch1/预热2次，完整context/3prior/6CG；并存负载和顺序影响计时。MAC只含Conv/Linear，不含FFT/复数CG/归一化/激活/访存；不由参数下降推定加速或省峰值显存。

retain_refine首轮→末轮：loss 0.028242456→0.022583989；iq_nmse 0.0263283→0.022103433；spectral_error 0.0069749165→0.0058355203；teacher_stage_nmse 0.0015654107→0.0018877979；中位每轮163.857s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_refine首轮→末轮：loss 0.13009791→0.023733999；iq_nmse 0.078311426→0.023135337；spectral_error 0.017386838→0.0060890483；teacher_stage_nmse 0.050917143→0.0029421006；中位每轮164.083s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承100串行pass+本次20=120；每组件累计80、组件轮次和240，原F50训练另计。优化控制和剪枝行同预算，均重置优化器；与旧R44相比同时改变教师/频谱损失与训练预算，不作单因素机制归因。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r44_pruned_refine_r46_20261020/models/<arm>/final.pt；使用experiments.pamf_r44_pruned_refine.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。
