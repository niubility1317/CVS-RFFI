# 有限 As bank 的质量分层风险：E30 四条件结果

20261018-phase1-FcrossRX-qualityRisk-e30-s2026092701-m2-r01；模型 seed 2026092701。两个新行均由同一个原 P43 B01 ownAs20 学生与 ownEMA 追加 10 轮；原 NoAs30/As30 的完整冻结读出复用。

源三条件 PASS=False，固定 stage1 关闭=True。hard<8 dB 的 S−I 等场景变化 -0.526661 pp（要求≥+.5），easy≥8 dB +0.319663 pp（要求≥−.2），fullV clean S−原As -0.007407 pp（要求≥−.1）。源判定先于正式读取，失败关闭/noM3/noE50；通过仅有另行登记M3资格，分析不自动启动。

正式 S clean−原NoAs +0.000595 个绝对百分点，0.1 pp 阶段约束=True。low/mid urban−NoAs +21.141667/+27.683929 pp。该正式阶段判定与源资格不同；n=1，不能宣称可重复双提升。

I01/BANK_IID30 与 S01/BANK_STRAT30 使用同一 L-only bank（6300×40=252000 个完整原 practical 观测）。I每步从有限bank有放回均匀抽64、权重1；S每bin有放回抽16，权重4p_g，p_g=N_g/N_bank。四bin依据实际 signal_to_added_noise_db：<3、[3,8)、[8,15)、≥15。它不是总SNR，不是白输入 quality_snr_db，也不以locked/degraded排除BLOCKED或unlocked。

两行对同一有限bank的名义风险有相同期望，不能冒称与在线无限practical风险相同。原ordinary128/ORD128对比、raw128、6L/768NN、总L尺度1.25、As .25份额、LS.01和sameIQ两次RDrop CE/KL保留。mixed原64槽被bank IQ及其真实标签/domain替换，CE和对称KL均按包权重乘后除128，不对selected权重或组内样本重新归一化。全部bank合法L仍可监督；没有额外部署模块、置信门或目标物理标签。

|视图|NoAs30 %|原As30 %|bank IID %|bank stratified %|S−I pp|I−原As pp|S−原As pp|
|---|---:|---:|---:|---:|---:|---:|---:|
|clean|85.247619|85.341071|85.432143|85.248214|-0.183929|0.091071|-0.092857|
|practical_high|42.685119|81.405357|81.320833|81.450000|0.129167|-0.084524|0.044643|
|practical_mid|40.995238|78.661310|78.621429|78.680952|0.059524|-0.039881|0.019643|
|practical_low_suburban|39.472619|75.861905|75.874405|75.887500|0.013095|0.012500|0.025595|
|practical_high_urban|41.439881|77.767262|77.752976|77.812500|0.059524|-0.014286|0.045238|
|practical_mid_urban|35.132738|62.806548|62.812500|62.816667|0.004167|0.005952|0.010119|
|practical_low_urban|30.139881|51.244048|51.320833|51.281548|-0.039286|0.076786|0.037500|
|urban_mean|35.570833|63.939286|63.962103|63.970238|0.008135|0.022817|0.030952|

|条件|视图|accuracy %|macro-F1 %|最差RX %|
|---|---|---:|---:|---:|
|NOAS30|clean|85.247619|84.979636|78.450000|
|NOAS30|practical_high|42.685119|39.311416|34.304167|
|NOAS30|practical_mid|40.995238|37.954813|33.520833|
|NOAS30|practical_low_suburban|39.472619|36.758731|32.054167|
|NOAS30|practical_high_urban|41.439881|38.481030|34.000000|
|NOAS30|practical_mid_urban|35.132738|33.446603|29.495833|
|NOAS30|practical_low_urban|30.139881|29.184589|25.750000|
|AS30|clean|85.341071|85.044474|78.362500|
|AS30|practical_high|81.405357|81.084230|71.908333|
|AS30|practical_mid|78.661310|78.359706|70.212500|
|AS30|practical_low_suburban|75.861905|75.572741|67.741667|
|AS30|practical_high_urban|77.767262|77.423208|69.108333|
|AS30|practical_mid_urban|62.806548|62.450463|56.458333|
|AS30|practical_low_urban|51.244048|50.887035|46.379167|
|BANK_IID30|clean|85.432143|85.145291|78.529167|
|BANK_IID30|practical_high|81.320833|81.002350|71.479167|
|BANK_IID30|practical_mid|78.621429|78.316602|69.858333|
|BANK_IID30|practical_low_suburban|75.874405|75.580435|67.637500|
|BANK_IID30|practical_high_urban|77.752976|77.406787|68.745833|
|BANK_IID30|practical_mid_urban|62.812500|62.417201|56.337500|
|BANK_IID30|practical_low_urban|51.320833|50.902397|46.500000|
|BANK_STRAT30|clean|85.248214|84.947711|78.437500|
|BANK_STRAT30|practical_high|81.450000|81.125515|72.070833|
|BANK_STRAT30|practical_mid|78.680952|78.373109|70.412500|
|BANK_STRAT30|practical_low_suburban|75.887500|75.598441|67.883333|
|BANK_STRAT30|practical_high_urban|77.812500|77.460487|69.170833|
|BANK_STRAT30|practical_mid_urban|62.816667|62.440774|56.466667|
|BANK_STRAT30|practical_low_urban|51.281548|50.903967|46.541667|

|比较|视图|修复|新增错误|净修复|
|---|---|---:|---:|---:|
|BANK_STRAT30−BANK_IID30|clean|345|654|-309|
|BANK_STRAT30−BANK_IID30|practical_high|950|733|217|
|BANK_STRAT30−BANK_IID30|practical_mid|1007|907|100|
|BANK_STRAT30−BANK_IID30|practical_low_suburban|1074|1052|22|
|BANK_STRAT30−BANK_IID30|practical_high_urban|967|867|100|
|BANK_STRAT30−BANK_IID30|practical_mid_urban|1362|1355|7|
|BANK_STRAT30−BANK_IID30|practical_low_urban|1612|1678|-66|
|BANK_IID30−AS30|clean|536|383|153|
|BANK_IID30−AS30|practical_high|817|959|-142|
|BANK_IID30−AS30|practical_mid|969|1036|-67|
|BANK_IID30−AS30|practical_low_suburban|1076|1055|21|
|BANK_IID30−AS30|practical_high_urban|990|1014|-24|
|BANK_IID30−AS30|practical_mid_urban|1441|1431|10|
|BANK_IID30−AS30|practical_low_urban|1849|1720|129|
|BANK_STRAT30−AS30|clean|408|564|-156|
|BANK_STRAT30−AS30|practical_high|862|787|75|
|BANK_STRAT30−AS30|practical_mid|977|944|33|
|BANK_STRAT30−AS30|practical_low_suburban|1111|1068|43|
|BANK_STRAT30−AS30|practical_high_urban|943|867|76|
|BANK_STRAT30−AS30|practical_mid_urban|1459|1442|17|
|BANK_STRAT30−AS30|practical_low_urban|1825|1762|63|
|BANK_IID30−NOAS30|clean|3891|3581|310|
|BANK_IID30−NOAS30|practical_high|68425|3517|64908|
|BANK_IID30−NOAS30|practical_mid|67255|4043|63212|
|BANK_IID30−NOAS30|practical_low_suburban|65685|4530|61155|
|BANK_IID30−NOAS30|practical_high_urban|65227|4221|61006|
|BANK_IID30−NOAS30|practical_mid_urban|53379|6877|46502|
|BANK_IID30−NOAS30|practical_low_urban|44498|8914|35584|
|BANK_STRAT30−NOAS30|clean|3689|3688|1|
|BANK_STRAT30−NOAS30|practical_high|68624|3499|65125|
|BANK_STRAT30−NOAS30|practical_mid|67364|4052|63312|
|BANK_STRAT30−NOAS30|practical_low_suburban|65673|4496|61177|
|BANK_STRAT30−NOAS30|practical_high_urban|65288|4182|61106|
|BANK_STRAT30−NOAS30|practical_mid_urban|53357|6848|46509|
|BANK_STRAT30−NOAS30|practical_low_urban|44388|8870|35518|
|AS30−NOAS30|clean|3868|3711|157|
|AS30−NOAS30|practical_high|68469|3419|65050|
|AS30−NOAS30|practical_mid|67279|4000|63279|
|AS30−NOAS30|practical_low_suburban|65618|4484|61134|
|AS30−NOAS30|practical_high_urban|65141|4111|61030|
|AS30−NOAS30|practical_mid_urban|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|44297|8842|35455|

S−I是分层加权相对同bank IID的增量；I−原liveAs30体现有限bank替代的近似变化；不能将S−原As全部归为分层收益。全部六配对和RX/day/TX错误转移在formal_summary保留；无虚构target质量/遮挡标签。

|条件|fullV clean %|V1800 clean %|两draw mid/low %|
|---|---:|---:|---:|
|NOAS30|98.359259|98.444444|39.583333|
|AS30|98.255556|98.333333|65.944444|
|BANK_IID30|98.237037|98.222222|65.666667|
|BANK_STRAT30|98.248148|98.333333|65.625000|

|条件|源场景|分组|n(两draw)|accuracy %|macro-F1 %|真实类margin/30|
|---|---|---|---:|---:|---:|---:|
|NOAS30|practical_mid_urban|hard_lt8|1101|22.706630|22.278561|-0.051483728|
|NOAS30|practical_mid_urban|easy_ge8|2499|52.741096|49.374568|0.012323139|
|NOAS30|practical_low_urban|hard_lt8|1921|22.175950|22.050073|-0.052597720|
|NOAS30|practical_low_urban|easy_ge8|1679|50.982728|47.695601|0.010658140|
|AS30|practical_mid_urban|hard_lt8|1101|34.695731|34.088824|-0.008645286|
|AS30|practical_mid_urban|easy_ge8|2499|89.595838|89.460210|0.146054789|
|AS30|practical_low_urban|hard_lt8|1921|32.431026|31.970710|-0.012098534|
|AS30|practical_low_urban|easy_ge8|1679|89.577129|89.325532|0.136811882|
|BANK_IID30|practical_mid_urban|hard_lt8|1101|34.241599|33.530617|-0.008315248|
|BANK_IID30|practical_mid_urban|easy_ge8|2499|89.195678|89.059461|0.146458685|
|BANK_IID30|practical_low_urban|hard_lt8|1921|32.691307|32.151305|-0.011826244|
|BANK_IID30|practical_low_urban|easy_ge8|1679|88.981537|88.709147|0.137098804|
|BANK_STRAT30|practical_mid_urban|hard_lt8|1101|33.969119|33.509835|-0.008510356|
|BANK_STRAT30|practical_mid_urban|easy_ge8|2499|89.715886|89.605397|0.146517843|
|BANK_STRAT30|practical_low_urban|hard_lt8|1921|31.910463|31.451772|-0.012003325|
|BANK_STRAT30|practical_low_urban|easy_ge8|1679|89.100655|88.847370|0.137055233|

hard/easy在每个mid/low场景先合并两个固定draw，再两场景等权；不按组样本量跨场景加权。source_metrics含各draw与pooled、四bin、全部state×lock及RX/day/TX、count/F1和已有logits算出的margin；这些描述不是新增门槛。fullV27000只有原生accuracy/RX，不虚构逐包F1。V1800两draw不是两个模型seed，源无highurban；既有panel2.45GHz与训练/正式2.462GHz差异保留。

|条件|suburban视图|绝对accuracy %|自身clean−suburban pp|
|---|---|---:|---:|
|NOAS30|practical_high|42.685119|42.562500|
|NOAS30|practical_mid|40.995238|44.252381|
|NOAS30|practical_low_suburban|39.472619|45.775000|
|AS30|practical_high|81.405357|3.935714|
|AS30|practical_mid|78.661310|6.679762|
|AS30|practical_low_suburban|75.861905|9.479167|
|BANK_IID30|practical_high|81.320833|4.111310|
|BANK_IID30|practical_mid|78.621429|6.810714|
|BANK_IID30|practical_low_suburban|75.874405|9.557738|
|BANK_STRAT30|practical_high|81.450000|3.798214|
|BANK_STRAT30|practical_mid|78.680952|6.567262|
|BANK_STRAT30|practical_low_suburban|75.887500|9.360714|

clean下降也会缩小suburban差距，须同时看绝对准确率；不将缩差自动当保护。全5源RX/L6300/U56700真值隐藏/V27000/split392005/Q256/defaultcos联合训练权限不变，无RX留出。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source NN|formal NN|native秒|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|NOAS30|6078780|918004|0|963900|7960684|12600|1176002|2146.190|
|AS30|6078780|918004|0|963900|7960684|12600|1176002|2860.909|
|BANK_IID30|2271960|324002|0|567000|3162962|12600|1176002|929.600|
|BANK_STRAT30|2271960|324002|0|567000|3162962|12600|1176002|941.813|

新增两尾段共20modelEpoch/4440更新，实际训练及读出8703128NN，共享smoke770NN，科学训练/读出及共享smoke合计8703898NN；另有R01技术失败258NN（0更新/0真实IQ/0传播模拟），整个候选实际8704156NN（科学参考8703898，含已知R01失败参考8704156，以实计为准）。同一原As20父只实际复用一次，各候选累计轨迹分别披露；旧两E30及历史读出不重计。bank构造252000完整物理模拟另计，无模型NN，wall/storage以bank summary实际记录为准。

bank生成计时状态=UNAVAILABLE_R02_FINALIZER_EXCEPTION，缺失计时字段=['wall_seconds', 'raw_dataset_cache_cpu_wall_seconds', 'simulation_adapter_cpu_wall_seconds', 'IQ_cache_write_cpu_wall_seconds', 'metadata_write_cpu_wall_seconds']；R02最终manifest异常导致未落盘的计时保持null，不用文件时间边界或R03 finalize计时冒充原模拟kernel/缓存写耗时；既有252000次模拟只计一次。

bank采样/缓存复制/加权CE-KL约简的CPUwall与实际调用数在training_logs和costs.risk_accounting中；无新NN不等于零计算，也不把CPUwall当CUDAevent或把包含关系的timer相加。source预测wall未单列即缺失，不用离线分析wall替代。

全部两新行4440步/20轮及L/U流完整解析，旧审计不回扫；4×7固定预测及独立scorer完成后才truth join，重核504CM/84分区。源先判、正式不反馈候选/参数/seed或救跑。单seed不足可重复clean正收益，目标未由本轮自动完成。分析NN/CP/IQ/updates均0。

四个逻辑条件不是 2×2 因子实验，不报告交互项；IID 是同 bank 名义风险对照。源资格只用于预先固定的后续决定，源分层 accuracy 变化不能证明梯度方差降低，也不能预测正式 0.1 pp 约束。

完成状态 ANALYZED；汇总文件缺项：0（11 个文件齐备）。source 预测 wall 未单列：NOAS30, AS30, BANK_IID30, BANK_STRAT30。本次整理 NN/CP/IQ/更新/预测/评分/remote 均为 0。
