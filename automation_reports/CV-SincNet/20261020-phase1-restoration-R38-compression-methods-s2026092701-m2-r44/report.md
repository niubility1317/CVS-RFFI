# R44：低秩卷积与重要性剪枝的正式结果

状态：VERIFIED / ANALYZED。两方法均完成固定E20/4940、source1800×7及target168000×7；每行全部预测固定后独立CPU truth-last，所属controller/worker均已退出。全部结果保留，测试不回流训练或选择性重跑。

唯一直接父/冻结IQ教师为用户指定R38 dense_half794529；low_rank保留通道、长卷积半秩SVD并保持共享因子，group_pruned按GN4组内encoder权重重要性选通道。固定两方法各新增20轮，paired-clean IQNMSE+10到1三stageIQ教师约束。全部coarse/action/prior联合，L126000/U1134000曝光、0TX/F/V/target优化；原F50/defaulthead仅冻结评测。不是无clean盲学习。

方法commit：`2f5247dbd5fe50f28defa38a0252d70e692934af`。完整矩阵、命令、数据契约、祖先来源、预算和停止规则见[登记](experiment.json)；统计、完整20轮曲线和逐TX/RX/day见[结果JSON](results.json)及[逐组CSV](row_metrics.csv)。

|方法|完整参数|较79.45万减参|相对R17逐景1pp通过|相对R38逐景1pp通过|low urban（%）|
|---|---:|---:|---|---|---:|
|low_rank|530161|33.2735%|False|False|65.771429|
|group_pruned|442805|44.2682%|True|False|66.772619|

1pp是逐场景退化容差，不等于性能优化成功；另列原lowurban≥70%、其他五景距rawclean84.6761904762%≤5pp。单seed、用户指定的已暴露工程基准，不称新独立盲测或干净泛化证明。

|方法|场景|准确率（%）|Macro-F1（%）|相对R38（pp）|相对R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|
|low_rank|clean|84.551786|84.259375|-0.077976|-0.035119|-209|
|low_rank|practical_high|81.379167|80.927807|-0.768452|+0.060119|+68286|
|low_rank|practical_mid|79.918452|79.455084|-1.054762|+0.340476|+67203|
|low_rank|practical_low_suburban|78.248810|77.805332|-1.244643|+0.252381|+65888|
|low_rank|practical_high_urban|80.151190|79.704297|-1.004167|-0.202381|+67961|
|low_rank|practical_mid_urban|72.865476|72.522713|-1.910714|-0.819643|+64482|
|low_rank|practical_low_urban|65.771429|65.693770|-2.498214|-1.445238|+59593|
|group_pruned|clean|84.505952|84.216444|-0.123810|-0.080952|-286|
|group_pruned|practical_high|82.418452|82.021788|+0.270833|+1.099405|+70032|
|group_pruned|practical_mid|81.010119|80.608682|+0.036905|+1.432143|+69037|
|group_pruned|practical_low_suburban|79.376190|79.028723|-0.117262|+1.379762|+67782|
|group_pruned|practical_high_urban|81.317262|80.922358|+0.161905|+0.963690|+69920|
|group_pruned|practical_mid_urban|74.226786|73.977900|-0.549405|+0.541667|+66769|
|group_pruned|practical_low_urban|66.772619|66.819605|-1.497024|-0.444048|+61275|

|方法|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|同卡R38/本版median ms|resident/峰值字节|
|---|---|---:|---:|---:|---|
|low_rank|213902/144869/171390|2508604|56799808|41.8769/51.9555|19421696/185215488|
|group_pruned|176962/125509/140334|2056092|49266224|45.1313/50.3344|19108352/336296960|

延迟为同卡16条sourceV、batch1、2次预热、完整context/三prior/六CG；顺序及并存负载影响计时。MAC仅Conv/Linear，不含FFT/复数CG/归一化/激活/访存。参数减少不预先推定加速或峰值显存降低。

low_rank完整20轮首轮→末轮：loss 2.0783148→0.028578833；iq_nmse 0.21699548→0.025384925；teacher_stage_nmse 0.18613193→0.0031939085。中位每轮183.045s。lowurban70通过=False；其他五景gap5pp通过=False。日志发现：[]

group_pruned完整20轮首轮→末轮：loss 1.7560273→0.028025675；iq_nmse 0.18583916→0.024819454；teacher_stage_nmse 0.15701881→0.003206221。中位每轮163.245s。lowurban70通过=False；其他五景gap5pp通过=False。日志发现：[]

全帧/主体232/尾24的paired-clean NMSE、FFT log-power差距、配对clean身份特征几何及全部TX/RX/day保留在JSON。识别改善或这些几何指标均不能证明每个新增细节属于输入TX。恢复器0TX训练与评分时独立打开标签分别说明。

祖先60串行pass+R38joint20+新增joint20=100；每组件累计60，组件轮次和180，原F50预训练另计。与R13E10/constant teacher1/LR1e-4并非同预算单因素比较。

两份独立恢复器已取回并CPU严格加载、有限synthetic IQ输出。路径：local_artifacts/pamf_restoration/r38_compression_methods_r44_20261020/models/<arm>/final.pt。使用experiments.pamf_r38_compression_methods.runtime.load_checkpoint，原生float32[B,2,256]输入/同形输出；接原F50/defaulthead。权重不入Git。长期总目标继续active，不由本轮产物完整代替总目标达成。
