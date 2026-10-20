# R17 八分之一上下文压缩与源 clean 联合优化 R12

状态：ANALYZED。补齐此前0IQ静态容量审计的eighth context层级，沿用与R10/R11相同的两种固定prior、直接source-clean纠正损失和E20预算。R10提供half context，R11提供full/quarter context匹配对照。以正式目标七场景逐项相对原R17下降不超过1个百分点为底线，另报真实提分、low urban≥70%、其他环境距clean和paired-clean恢复差距。容差通过不等于性能改善。

|固定行|完整/可训练参数|context参数|context比例|较原4665073减少|GPU|
|---|---:|---:|---:|---:|---:|
|eighth_context_quarter|907657|482299|0.125|80.5436%|4|
|eighth_context_eighth_pointwise|677141|482299|0.125|85.4849%|5|

只原source-selected R17/cd977448/E20/4940的CPU初始化，84个context长卷积做.125平衡SVD、0IQ校准；prior固定quarter共享或eighth并共享原1×1，保留各步bias/GN。全部coarse/action/prior联合优化，三步×2CG不变；原R17对象建立optimizer前删除，0正式R17 teacher/context前向。原F50/defaulthead冻结，0头拟合；不继承任何已测学生权重。

固定损失为最终恢复与paired source-clean的IQ NMSE（clean能量detach/floor1e-4）+T1 KL(Fclean||Frestored)+0.5单位F160距离平方，系数1/1/0.5，无R17波形模仿。源数据原63000=L6300/U56700、原20个received缓存和15%clean/85%residual；0新TX/V/target训练标签。使用配对波形及原监督F知识，不称严格无监督或F-free。推理仅当前received IQ，0clean/Fteacher/TX/RX/scene/cache/状态更新。

每行E20/4940，B256/model2026092701/data2026102001，AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32。L126000/U1134000/source1260000，studentcontext/Fclean/Fstudent各1260000，R17训练forward0，coarse/action/prior各4940更新，0F/head更新。继承coarse20/action20/R17prior20加一次joint20为80串行pass，各组件累计40/component sum120；F50预训练50/11100、初始化/smoke/profile另计。

固定最终E20，无V选模、早停或源性能门槛。每行自动source1800×7诊断/profile后正式168000×7全部预测，所有七视图IQ/logit/features/physical IDs固定后独立CPU才读truth/clean。精确1pp按100×(原正确数−新正确数)≤N判断；报告全部20轮曲线、准确率/MacroF1/TX/RX/day/raw净纠错、full256/prefix232/tail24/FFT/paired-clean特征差和完整参数/文件/延迟/显存。profile包含完整context/CG，GPU payload/optimizer排除。单seed/共存负载不称稳定总体提升或部署加速，全部固定行和负结果保留。

结构成本来自早于R08目标的0IQ审计，本轮补齐既定context层级；不依据已暴露目标分数选秩/系数/预算/父权重或顺序，不重跑负结果。R10 benchmark已暴露，R11尚无目标结果，如实披露，非新独立盲测，不称全局最简/Phase2/真实在轨验证。

40项合成/协议检查PASS，唯一独立P0/P1审查PASS，独立复跑40项。真实原R17/F50/2Ls noquery smoke为launcher首步：全部coarse/action/prior及F输入有限非零梯度、原父/F/head0更新、严格复载输出一致、0TX/V/target/optimizer更新，PASS立即继续。R10根终态且实际GPU句柄退出后接续GPU4/5，每卡一own worker，R11健康2/3/6/7保持/max6ownGPU。所有scorer共享R06 score.lock，CPU2线程/max1own scorer。

一次普通N607预检、新不可覆盖run/release/output、单归档SHA比较、远端compile/help，以及实际PID/CWD/完整命令/commit/GPU/日志增长/E1核实。技术错误停止受影响行并保留，健康行继续，完成后若任一失败则根FAILED，无自动重试/回退。命令、环境、CWD、路径及预期产物见experiment.json。

普通N607预检VERIFIED：R10实际GPU句柄0，GPU4/5各24124MiB空闲，R11健康四卡继续。新run/release不存在，复用合法source/R17/F50/formal句柄，不新增数据重验。40项聚焦检查与唯一独立P0/P1审查PASS，审查0真实IQ/checkpoint/远端访问。

实际发布 VERIFIED：方法commit `152f11283ba81b81f1a9e58b5b094f656f1c308b` 已推送并独立核实远端OID；单归档SHA比较、远端compile/help通过。真实原R17/F50/2Ls noquery smoke PASS，两行coarse/action/各prior与F输入有限非零梯度，原初始化父/F/head0更新、严格复载逐位输出。R10实际句柄退出后，在GPU4/5各一新worker；实际PID/CWD/完整命令/commit/GPU/日志增长及E1均核实。R10保持原固定方案完成，不以其新结果改R12设计、预算或顺序。24项完整汇总检查PASS，检查动态context量/比例、全部预算和七视图/整数1pp/真实涨分/成本；汇总工具不修改运行中方法commit。当前尚无R12目标结果。


## 完整目标结果

两个固定行完成E20/4940、source1800×7及同168000×七目标场景。每行七视图预测全部固定后独立CPU truth-last；实际controller/worker已退出。七场景原R17逐项1pp验收通过：eighth_context_quarter。原R17主验收保持，另报已验收R10同1592461参数版与同prior/half-context匹配对照。所有固定行、失败场景及权重保留；未将结果回流当前训练、超参、父权重、排序或选择性重跑。

|版本|完整参数|较原R17减少|最大下降pp|low urban|达到70%|1pp验收|
|---|---:|---:|---:|---:|---|---|
|eighth_context_quarter|907657|80.5436%|0.723214|66.493452%|False|True|
|eighth_context_eighth_pointwise|677141|85.4849%|2.232738|64.983929%|False|False|

容差通过与性能改善分开报告；各场景升降不以均值替代。low urban≥70%独立判定，其他环境距clean不自设阈值。单seed/单原F50seed/已暴露benchmark，不称全局最简、新独立盲测、多seed总体保证、Phase2或真实在轨验证。

|版本|场景|准确率|相对原R17pp|相对R10同prior half-context pp|相对已验收R10 1592461版pp|距本版clean pp|
|---|---|---:|---:|---:|---:|---:|
|eighth_context_quarter|clean|85.005952|+0.419048|-0.230357|+0.252976|0.000000|
|eighth_context_quarter|practical_high|81.097024|-0.222024|+1.723810|+0.544643|3.908929|
|eighth_context_quarter|practical_mid|79.701786|+0.123810|+1.773214|+0.508929|5.304167|
|eighth_context_quarter|practical_low_suburban|78.160119|+0.163690|+1.719048|+0.544643|6.845833|
|eighth_context_quarter|practical_high_urban|80.047619|-0.305952|+1.698810|+0.491667|4.958333|
|eighth_context_quarter|practical_mid_urban|73.754762|+0.069643|+0.945833|+0.041071|11.251190|
|eighth_context_quarter|practical_low_urban|66.493452|-0.723214|-0.154167|-1.107143|18.512500|
|eighth_context_eighth_pointwise|clean|85.108929|+0.522024|+0.355952|+0.355952|0.000000|
|eighth_context_eighth_pointwise|practical_high|81.086905|-0.232143|+0.534524|+0.534524|4.022024|
|eighth_context_eighth_pointwise|practical_mid|79.928571|+0.350595|+0.735714|+0.735714|5.180357|
|eighth_context_eighth_pointwise|practical_low_suburban|78.436905|+0.440476|+0.821429|+0.821429|6.672024|
|eighth_context_eighth_pointwise|practical_high_urban|79.743452|-0.610119|+0.187500|+0.187500|5.365476|
|eighth_context_eighth_pointwise|practical_mid_urban|72.569048|-1.116071|-1.144643|-1.144643|12.539881|
|eighth_context_eighth_pointwise|practical_low_urban|64.983929|-2.232738|-2.616667|-2.616667|20.125000|

上下文匹配消融复用完整冻结R10汇总，0新模型前向/truth访问。原始父初始化、source数据/顺序/seed、直接clean波形及F目标、优化器和E20预算保持；只比较.125与既定half context，参数及实际算量有变化。不能将单seed汇总差当样本配对显著性或稳定因果结论。

### eighth_context_quarter 的配对clean差距与源诊断

|场景|source准确率|source相对原R17pp|full256 NMSE|prefix232 NMSE|tail24 NMSE|配对clean特征余弦距离|raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.555556|+0.111111|0.00176617247|0.00177187927|0.00168690039|0.000694151619|1174/620/+554|
|practical_high|97.944444|+0.111111|0.0339471698|0.0177129172|0.193134442|0.0194295887|71303/3491/+67812|
|practical_mid|96.166667|-0.611111|0.0474637672|0.0316791199|0.202212587|0.024396183|70857/4018/+66839|
|practical_low_suburban|95.222222|-0.722222|0.0617530458|0.0463923216|0.212236747|0.0301237088|70347/4608/+65739|
|practical_high_urban|96.500000|-0.500000|0.0510923304|0.0349409468|0.208870903|0.0241488945|71808/4021/+67787|
|practical_mid_urban|88.000000|-3.444444|0.123209044|0.108000048|0.269925654|0.0464216322|72107/6131/+65976|
|practical_low_urban|79.611111|-7.222222|0.18784979|0.174057126|0.31981495|0.0675501674|69194/8388/+60806|

完整20轮采用固定最终E20，未按最小loss或V/target择优。loss首轮→末轮：loss 1.9962499→0.25265841；iq_nmse 1.1009404→0.084949472；task_KL 0.79350295→0.14268668；task_feature_distance 0.10180653→0.025022255。训练2135.509秒，中位每轮106.803秒。

### eighth_context_eighth_pointwise 的配对clean差距与源诊断

|场景|source准确率|source相对原R17pp|full256 NMSE|prefix232 NMSE|tail24 NMSE|配对clean特征余弦距离|raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.500000|+0.055556|0.000822211849|0.000809829275|0.000927234534|0.000779816939|1538/811/+727|
|practical_high|97.777778|-0.055556|0.0358176902|0.019364493|0.198266506|0.0196822099|71343/3548/+67795|
|practical_mid|95.833333|-0.944444|0.0499420166|0.0338675268|0.208868533|0.0247394964|70992/3772/+67220|
|practical_low_suburban|95.000000|-0.944444|0.0639982596|0.0483146831|0.219179526|0.0301574469|70427/4223/+66204|
|practical_high_urban|96.111111|-0.888889|0.0561669059|0.0397815853|0.217700377|0.0251905136|71317/4041/+67276|
|practical_mid_urban|86.166667|-5.277778|0.137282565|0.121583685|0.291715831|0.0496412814|70249/6265/+63984|
|practical_low_urban|77.111111|-9.722222|0.206422776|0.192049637|0.347717375|0.0721905902|66857/8587/+58270|

完整20轮采用固定最终E20，未按最小loss或V/target择优。loss首轮→末轮：loss 2.0237559→0.29122438；iq_nmse 1.1148209→0.093323914；task_KL 0.80596373→0.17044876；task_feature_distance 0.10297123→0.027451696。训练2106.918秒，中位每轮105.305秒。

成本来自固定16条source V、batch1/2warmup，完整context/CG纳入，GPU payload/optimizer排除；共存负载和顺序影响时间，不将其当部署加速证明。

|版本|checkpoint字节|原R17中位ms|本版中位ms|延迟比|原R17 resident|本版resident|原增量peak|本版增量peak|目标预测秒|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|eighth_context_quarter|4016380|46.743|69.325|1.4831|35870720|20852736|1158656|575215104|992.230|
|eighth_context_eighth_pointwise|3091548|44.153|55.327|1.2531|35870720|19929600|1158656|304672256|974.784|

各行新增E20/4940/L126000/U1134000，source/studentcontext/Fclean/Fstudent各1260000，正式R17 teacher/context前向0，完整coarse/action/prior各4940更新，0F/head/原R17更新。继承coarse20/action20/R17prior20与joint20共80串行pass，各组件40、component sum120；F50预训练50/11100与smoke/profile另计。完整恢复器已取回local_artifacts/pamf_restoration/r17_eighth_context_20261020_r12/models/<arm>/final.pt，大小/来源/共享alias/严格加载与合成零IQ输出检查VERIFIED；0实际IQ重新访问，权重不入Git。加载experiments.pamf_r17_eighth_context.runtime.load_checkpoint，原生尺度float32 [B,2,256]输入与同形恢复IQ输出，I/Q顺序，接原F50/defaulthead。
