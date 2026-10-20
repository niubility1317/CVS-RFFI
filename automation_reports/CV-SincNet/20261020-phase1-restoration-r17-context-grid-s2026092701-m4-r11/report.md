# R17 压缩与源 clean 联合优化上下文矩阵 R11

状态：ANALYZED。同一源 clean 纠正目标下，固定两种 prior，补齐 full/quarter context；R10 已登记 half context 作为匹配对照。性能对照保留完整 context，更小两组检验继续压缩。验收以同一正式目标七场景为准：各场景相对原 R17 下降不超过 1 个百分点；另报实际提分、low urban≥70%、其他场景距 clean 和配对 clean 恢复差距。容差通过不等于提分。

|固定行|完整/可训练参数|context参数|context保留比例|比原4665073减少|GPU|
|---|---:|---:|---:|---:|---:|
|full_context_quarter|2582905|2157547|1.0|44.6331%|2|
|full_context_eighth_pointwise|2352389|2157547|1.0|49.5744%|3|
|quarter_context_quarter|1212813|787455|0.25|74.0023%|6|
|quarter_context_eighth_pointwise|982297|787455|0.25|78.9436%|7|

只原 source-selected R17/cd977448/E20/4940 CPU 初始化，无已测学生继承。完整 context 逐位复制原权重；quarter context 的84个长卷积做无IQ CPU平衡SVD；prior固定quarter共享或eighth并共享原1×1，保留各步bias/GN。三步×2CG不变，coarse/action/prior全部联合优化。原R17对象在建立optimizer前删除，正式训练0R17 teacher/context前向。

固定损失与R10完全相同：最终恢复与paired source-clean的IQ NMSE（clean能量detach、下限1e-4）+T1 KL(Fclean||Frestored)+0.5×单位F160距离平方，系数1/1/0.5，无R17波形模仿。原F50/默认联合训练头冻结，0头拟合。源数据为原63000=L6300/U56700及原20个received缓存、15%clean/85%residual；0新TX、V或target训练标签。使用源配对波形及既有监督F知识，不称严格无监督或F-free。推理只当前received IQ，0clean/Fteacher/TX/RX/场景/cache/状态更新。

E20/4940、B256、model2026092701/data2026102001、AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32。每行L126000/U1134000/source1260000、studentcontext1260000、Fclean/student各1260000、R17训练前向0。继承coarse20/action20/R17prior20加一次joint20为80串行训练pass；各组件累计40，component-epoch sum120。F50预训练50/11100及smoke/profile成本另计。

固定最终E20，无V选模、早停或source性能门槛。每行自动source1800×7诊断/profile后执行正式168000×7预测；同row七视图IQ/logit/features/physical IDs全部固定后，独立CPU才读truth/clean。报告准确率/MacroF1/TX/RX/day/净纠错、完整参数和文件、full256/prefix232/tail24/FFT/配对clean特征差、全部20轮日志和batch1 profile（排除GPU payload/optimizer）。精确1pp按100×(原正确数−新正确数)≤N验收，不能把相对百分比当百分点。所有固定行及负结果保留。

full/half/quarter context成本在R08目标暴露前已做0IQ静态审计，本轮沿用固定两种prior和R10目标，补齐上下文层级。已看到R08/R09的benchmark结果，如实披露，不称新独立盲测；不据目标改秩、损失、预算、父权重、排序或选择性重跑。

42项合成/协议检查和唯一独立P0/P1审查PASS（独立复跑42项）。真实原R17/F50/2Ls noquery smoke是launcher首步，检查全部组件和F输入有限非零梯度、原父/F/head0更新、严格复载逐位输出，0V/target/TX/optimizer更新，PASS立即继续。

R09根终态且其实际GPU句柄退出后接续GPU2/3/6/7，每卡一own worker；R10健康4/5保持，共最多6路。所有R17 scorer共享R06 score.lock，最多1个own CPU scorer。新不可覆盖run/release/output；一次普通N607预检、单archive SHA比较、远端compile/help、实际PID/CWD/cmd/commit/GPU/loggrowth核实。技术错误停止受影响行并保留，健康行继续；任一失败则完成健康行后根状态FAILED，不自动重试或回退。精确命令、环境、路径和预期产物见experiment.json。

发布前预检 VERIFIED：普通N607账户、R09旧实际GPU句柄0、新run/release不存在，GPU2/3/6/7各24124MiB空闲。R10在GPU4/5继续；复用合法source/R17/F50与formal句柄，无新增数据重验。

实际发布 VERIFIED：方法commit `5f8c563e96766fee9f4601fe64bf0da4395892ae` 已推送并独立核实远端OID；单归档SHA比较、远端compile/help通过。真实原R17/F50/2Ls noquery smoke PASS，四行coarse/action/各prior与F输入有限非零梯度，原初始化父/F/head0更新、严格复载逐位输出。R09实际句柄退出后，在GPU2/3/6/7各一新worker；实际PID/CWD/完整命令/commit/GPU/日志增长及E1均核实。R10保持原固定方案完成，不以其新结果改R11设计、预算或顺序。24项完整汇总检查PASS，检查动态context量/比例、全部预算和七视图/整数1pp/真实涨分/成本；汇总工具不修改运行中方法commit。当前尚无R11目标结果。


## 完整目标结果

四个固定行完成E20/4940、source1800×7及同168000×七目标场景。每行七视图预测全部固定后独立CPU truth-last；实际controller/worker已退出。七场景原R17逐项1pp验收通过：quarter_context_eighth_pointwise。原R17主验收保持，另报已验收R10同1592461参数版与同prior/half-context匹配对照。所有固定行、失败场景及权重保留；未将结果回流当前训练、超参、父权重、排序或选择性重跑。

|版本|完整参数|较原R17减少|最大下降pp|low urban|达到70%|1pp验收|
|---|---:|---:|---:|---:|---|---|
|full_context_quarter|2582905|44.6331%|1.419643|67.599405%|False|False|
|full_context_eighth_pointwise|2352389|49.5744%|2.030357|67.310714%|False|False|
|quarter_context_quarter|1212813|74.0023%|2.436310|65.776190%|False|False|
|quarter_context_eighth_pointwise|982297|78.9436%|0.060714|67.312500%|False|True|

容差通过与性能改善分开报告；各场景升降不以均值替代。low urban≥70%独立判定，其他环境距clean不自设阈值。单seed/单原F50seed/已暴露benchmark，不称全局最简、新独立盲测、多seed总体保证、Phase2或真实在轨验证。

|版本|场景|准确率|相对原R17pp|相对R10同prior half-context pp|相对已验收R10 1592461版pp|距本版clean pp|
|---|---|---:|---:|---:|---:|---:|
|full_context_quarter|clean|84.494048|-0.092857|-0.742262|-0.258929|0.000000|
|full_context_quarter|practical_high|79.899405|-1.419643|+0.526190|-0.652976|4.594643|
|full_context_quarter|practical_mid|78.551786|-1.026190|+0.623214|-0.641071|5.942262|
|full_context_quarter|practical_low_suburban|77.105952|-0.890476|+0.664881|-0.509524|7.388095|
|full_context_quarter|practical_high_urban|79.000000|-1.353571|+0.651190|-0.555952|5.494048|
|full_context_quarter|practical_mid_urban|73.373214|-0.311905|+0.564286|-0.340476|11.120833|
|full_context_quarter|practical_low_urban|67.599405|+0.382738|+0.951786|-0.001190|16.894643|
|full_context_eighth_pointwise|clean|85.622024|+1.035119|+0.869048|+0.869048|0.000000|
|full_context_eighth_pointwise|practical_high|79.288690|-2.030357|-1.263690|-1.263690|6.333333|
|full_context_eighth_pointwise|practical_mid|78.046429|-1.531548|-1.146429|-1.146429|7.575595|
|full_context_eighth_pointwise|practical_low_suburban|76.689286|-1.307143|-0.926190|-0.926190|8.932738|
|full_context_eighth_pointwise|practical_high_urban|78.355952|-1.997619|-1.200000|-1.200000|7.266071|
|full_context_eighth_pointwise|practical_mid_urban|72.970238|-0.714881|-0.743452|-0.743452|12.651786|
|full_context_eighth_pointwise|practical_low_urban|67.310714|+0.094048|-0.289881|-0.289881|18.311310|
|quarter_context_quarter|clean|84.668452|+0.081548|-0.567857|-0.084524|0.000000|
|quarter_context_quarter|practical_high|78.978571|-2.340476|-0.394643|-1.573810|5.689881|
|quarter_context_quarter|practical_mid|77.423214|-2.154762|-0.505357|-1.769643|7.245238|
|quarter_context_quarter|practical_low_suburban|76.074405|-1.922024|-0.366667|-1.541071|8.594048|
|quarter_context_quarter|practical_high_urban|77.917262|-2.436310|-0.431548|-1.638690|6.751190|
|quarter_context_quarter|practical_mid_urban|71.922619|-1.762500|-0.886310|-1.791071|12.745833|
|quarter_context_quarter|practical_low_urban|65.776190|-1.440476|-0.871429|-1.824405|18.892262|
|quarter_context_eighth_pointwise|clean|84.526190|-0.060714|-0.226786|-0.226786|0.000000|
|quarter_context_eighth_pointwise|practical_high|81.741071|+0.422024|+1.188690|+1.188690|2.785119|
|quarter_context_eighth_pointwise|practical_mid|80.271429|+0.693452|+1.078571|+1.078571|4.254762|
|quarter_context_eighth_pointwise|practical_low_suburban|78.733333|+0.736905|+1.117857|+1.117857|5.792857|
|quarter_context_eighth_pointwise|practical_high_urban|80.439881|+0.086310|+0.883929|+0.883929|4.086310|
|quarter_context_eighth_pointwise|practical_mid_urban|74.025000|+0.339881|+0.311310|+0.311310|10.501190|
|quarter_context_eighth_pointwise|practical_low_urban|67.312500|+0.095833|-0.288095|-0.288095|17.213690|

上下文匹配消融复用完整冻结R10汇总，0新模型前向/truth访问。原始父初始化、source数据/顺序/seed、直接clean波形及F目标、优化器和E20预算保持；只比较full/.25与既定half context，参数及实际算量有变化。不能将单seed汇总差当样本配对显著性或稳定因果结论。

### full_context_quarter 的配对clean差距与源诊断

|场景|source准确率|source相对原R17pp|full256 NMSE|prefix232 NMSE|tail24 NMSE|配对clean特征余弦距离|raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.388889|-0.055556|0.00234212051|0.00243501877|0.00150481693|0.000186050471|297/603/-306|
|practical_high|98.055556|+0.222222|0.0217870213|0.00826490391|0.156314626|0.0146050034|70035/4235/+65800|
|practical_mid|97.666667|+0.888889|0.0288097132|0.0157708284|0.158594564|0.0189119391|69563/4656/+64907|
|practical_low_suburban|97.000000|+1.055556|0.036354389|0.0237517487|0.161798596|0.0235181879|69095/5127/+63968|
|practical_high_urban|98.222222|+1.222222|0.0296318177|0.0164758693|0.160350397|0.0182328802|70706/4679/+66027|
|practical_mid_urban|93.555556|+2.111111|0.0644118115|0.0527416579|0.179966077|0.036057584|71902/6567/+65335|
|practical_low_urban|88.333333|+1.500000|0.0960985497|0.0852438435|0.202804506|0.0527584925|71159/8495/+62664|

完整20轮采用固定最终E20，未按最小loss或V/target择优。loss首轮→末轮：loss 0.28245651→0.12712995；iq_nmse 0.052751702→0.03972561；task_KL 0.19908283→0.073498333；task_feature_distance 0.030621984→0.013906008。训练2421.508秒，中位每轮121.101秒。

### full_context_eighth_pointwise 的配对clean差距与源诊断

|场景|source准确率|source相对原R17pp|full256 NMSE|prefix232 NMSE|tail24 NMSE|配对clean特征余弦距离|raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.388889|-0.055556|0.00195770524|0.00194652926|0.00206479174|0.00106173102|2338/749/+1589|
|practical_high|98.111111|+0.277778|0.0241574068|0.0107637402|0.155384541|0.0159501284|68372/3598/+64774|
|practical_mid|97.555556|+0.777778|0.0310807005|0.0180487987|0.158758819|0.0198375192|68118/4060/+64058|
|practical_low_suburban|96.944444|+1.000000|0.0386421084|0.0259506218|0.162949264|0.0241188686|67965/4697/+63268|
|practical_high_urban|98.000000|+1.000000|0.0318280086|0.0187538173|0.159817874|0.0193488039|69044/4099/+64945|
|practical_mid_urban|93.500000|+2.055556|0.066597499|0.0548664555|0.181099698|0.03592439|70949/6291/+64658|
|practical_low_urban|88.500000|+1.666667|0.0984735414|0.0875103325|0.204842746|0.051962465|70399/8220/+62179|

完整20轮采用固定最终E20，未按最小loss或V/target择优。loss首轮→末轮：loss 0.2659444→0.1325598；iq_nmse 0.048340001→0.041801291；task_KL 0.18843879→0.076141007；task_feature_distance 0.029165615→0.014617507。训练2408.263秒，中位每轮120.287秒。

### quarter_context_quarter 的配对clean差距与源诊断

|场景|source准确率|source相对原R17pp|full256 NMSE|prefix232 NMSE|tail24 NMSE|配对clean特征余弦距离|raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.500000|+0.055556|0.00373269478|0.00380877499|0.00324893184|0.000448937819|747/760/-13|
|practical_high|98.111111|+0.277778|0.026851641|0.013043833|0.161494538|0.017734101|68112/3859/+64253|
|practical_mid|97.333333|+0.555556|0.0373424329|0.0240772031|0.16681996|0.0223798826|67386/4375/+63011|
|practical_low_suburban|96.277778|+0.333333|0.0481237695|0.0353060029|0.173261538|0.0274463035|67127/4892/+62235|
|practical_high_urban|97.388889|+0.388889|0.0393807366|0.0258224346|0.171361253|0.0215635505|68557/4349/+64208|
|practical_mid_urban|91.555556|+0.111111|0.0924259946|0.0799268484|0.213048801|0.0401832908|69431/6533/+62898|
|practical_low_urban|84.944444|-1.888889|0.139122277|0.12715131|0.252757847|0.0581430495|68131/8530/+59601|

完整20轮采用固定最终E20，未按最小loss或V/target择优。loss首轮→末轮：loss 1.2012136→0.18365804；iq_nmse 0.5339319→0.065756983；task_KL 0.58965111→0.098320116；task_feature_distance 0.077630532→0.01958094。训练2190.845秒，中位每轮109.638秒。

### quarter_context_eighth_pointwise 的配对clean差距与源诊断

|场景|source准确率|source相对原R17pp|full256 NMSE|prefix232 NMSE|tail24 NMSE|配对clean特征余弦距离|raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.555556|+0.111111|0.00371295353|0.00377613679|0.00319331395|0.00111137086|985/1237/-252|
|practical_high|98.000000|+0.166667|0.0287818406|0.0145636499|0.167764142|0.0185529571|72507/3613/+68894|
|practical_mid|96.944444|+0.166667|0.0398188345|0.0260956325|0.174024478|0.0233700946|71846/4050/+67796|
|practical_low_suburban|96.000000|+0.055556|0.0514902286|0.0382672101|0.180852532|0.0288738459|71185/4483/+66702|
|practical_high_urban|97.111111|+0.111111|0.0418119319|0.0278387312|0.178244308|0.0229221396|72675/4229/+68446|
|practical_mid_urban|89.944444|-1.500000|0.0975223482|0.0849005431|0.220244482|0.0432957448|72665/6235/+66430|
|practical_low_urban|83.944444|-2.888889|0.147130236|0.135767654|0.256596953|0.062208984|70538/8356/+62182|

完整20轮采用固定最终E20，未按最小loss或V/target择优。loss首轮→末轮：loss 1.2401803→0.20415267；iq_nmse 0.55800302→0.068163471；task_KL 0.60446679→0.11469172；task_feature_distance 0.07771049→0.02129747。训练2181.027秒，中位每轮109.223秒。

成本来自固定16条source V、batch1/2warmup，完整context/CG纳入，GPU payload/optimizer排除；共存负载和顺序影响时间，不将其当部署加速证明。

|版本|checkpoint字节|原R17中位ms|本版中位ms|延迟比|原R17 resident|本版resident|原增量peak|本版增量peak|目标预测秒|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|full_context_quarter|10658788|44.226|63.692|1.4401|35870720|27540992|1158656|575215104|993.793|
|full_context_eighth_pointwise|9744196|46.544|52.752|1.1334|35870720|26617856|1158656|304672256|991.187|
|quarter_context_quarter|5238140|46.481|78.353|1.6857|35870720|22073344|1158656|575215104|1002.978|
|quarter_context_eighth_pointwise|4313308|49.658|67.316|1.3556|35870720|21150208|1158656|575186944|1010.473|

各行新增E20/4940/L126000/U1134000，source/studentcontext/Fclean/Fstudent各1260000，正式R17 teacher/context前向0，完整coarse/action/prior各4940更新，0F/head/原R17更新。继承coarse20/action20/R17prior20与joint20共80串行pass，各组件40、component sum120；F50预训练50/11100与smoke/profile另计。完整恢复器已取回local_artifacts/pamf_restoration/r17_context_grid_20261020_r11/models/<arm>/final.pt，大小/来源/共享alias/严格加载与合成零IQ输出检查VERIFIED；0实际IQ重新访问，权重不入Git。加载experiments.pamf_r17_context_grid.runtime.load_checkpoint，原生尺度float32 [B,2,256]输入与同形恢复IQ输出，I/Q顺序，接原F50/defaulthead。
