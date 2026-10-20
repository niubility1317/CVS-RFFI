# R17 压缩与恢复识别联合优化 R09

状态：ANALYZED。用户最新明确“压缩的同时也要优化”。本轮不把仅维持1pp容差当成识别优化成功，固定两种压缩架构×两种训练目标：共同联合优化coarse/action/prior全部参数，比较原教师行为保真与源域clean恢复/身份纠正。原F50/defaulthead冻结，性能以正式目标测试为准，逐场景原R17−1pp是底线，low urban≥70%和其余环境接近clean的实际差距独立报告，“接近”不自设阈值。

|固定行|完整/可训练参数|比原4665073减少|F教师参考|额外直接clean IQNMSE|GPU|
|---|---:|---:|---|---:|---:|
|joint_teacher_quarter|1822977|60.9229%|original R17 restored|0.0|2|
|joint_clean_quarter|1822977|60.9229%|source paired clean|1.0|3|
|joint_teacher_eighth_pointwise|1592461|65.8642%|original R17 restored|0.0|6|
|joint_clean_eighth_pointwise|1592461|65.8642%|source paired clean|1.0|7|

只从原source-selected R17/cd977448/E20/4940初始化与蒸馏，无R04至R08学生继承。CPU数据无关半秩SVD压84个context长卷积，prior为quarter共享或eighth并共享原1x1，保留逐步bias/GN，架构/初始权重构造与R08对应行相同。区别是本轮全部coarse/action/prior参数联合优化，补偿压缩误差并检验恢复/识别收益；原R17教师和原F50默认头完全冻结。新机制由用户要求及source clean契约/冻结代码路径审计推导，不由目标分数调秩、系数、预算、父权重、候选重排或重跑；已暴露基准，不称新盲测。

joint_teacher损失为原固定10×三步学生/原R17 IQNMSE均值 + KL_T1(F原R17恢复||F学生恢复) + .5×unitF160距离平方。joint_clean仍保留10×三步原R17 IQNMSE，再加1×最终source paired-clean IQNMSE，并将KL/feature教师参考改为同物理源样本的clean IQ；后两项系数1。clean objective是直接波形恢复和clean身份特征目标的组合，不单独归因某一项。clean仅来自合法source L/U缓存，0新TX标签、0V/target训练数据；F来自原source监督训练知识，不能称F-free无监督。恢复器推理仅当前received IQ，0clean/Fteacher/TX/RX/场景/cache或目标状态更新；原F只在下游同默认头识别。

联合梯度实现：student直接调用压缩coarse和action estimator，绕开旧ActionRestorer.estimate对coarse的no_grad；teacher helper只在原teacher分支no_grad，student上下文保留图，不在调用端整体包no_grad。optimizer包含全部压缩模型参数，clip1；coarse/action/prior必须获得有限非零梯度并实际更新，原teacher/F/defaulthead保持不变。context参数1397619在本轮可训练，不能沿用R08“冻结context”说法。

数据、顺序和预算固定：原R07缓存63000=L6300/U56700、epochs1..20、15%clean/85%residual，batch256/model2701/data2026102001；AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32，E20/4940。每行L126000/U1134000，原teacher context/student context各1260000，原R17三步teacher1260000，Fteacher/Fstudent各1260000，teacher F参考因目标而异；F/student输入反传和全恢复器反传成本另计。继承coarse20/action20/R17prior20后一次联合E20，共80串行训练pass，各组件累计40（component-epoch sum120，不冒充120串行轮）；原F50预训练50/11100单列，0新头拟合。

source/V只诊断，无早停/性能资格门槛；固定最终E20后每行自动七源视图诊断/profile/独立truth-last，再同168000×clean和六practical全目标预测，七视图IQ/logit/feature/IDs固定后独立CPU才读truth/clean。按每场景最多损失1680条正确数判1pp，另报真实正确数增益、零下降、MacroF1、TX/RX/day/confusion、raw修复/损伤/净修复、full256/prefix232/tail24/FFT/pairedclean特征差距和完整成本。不能用参数减少或源分数替代目标收益。所有固定行含负结果完整报告。

42项合成/协议检查和唯一独立P0/P1审查PASS；独立复跑42项，未访问真实IQ/权重/远端。覆盖四模型完整成本/全参数可训练、coarse/action/prior梯度和实际更新、原teacher/F冻结、teacher原上下文与student可导上下文隔离、两种loss参考与IQ项、strictreload、数据权限/预算、truth-last和失败隔离。launcher首步真实原R17/F50/2Ls CPU noquery smoke，验证上述梯度/冻结/teacher原阶段逐位等价/strictreload，0TX/V/target/optimizer更新，PASS立即排队。

GPU2/3/6/7等待R07根终态及全部实际ownedGPU句柄EXITED，不读其评分；随后每卡一个own，R08健康4/5继续，最多六路own GPU。全部R17共享R06score.lock，CPU2线程/最多一个own scorer。一次普通N607预检、新不可覆盖root/release、单archive SHA比较/远端compile、实际PID/CWD/完整命令/GPU/loggrowth核实。技术错误停止失败行并保留，健康行继续，无自动重试/回退；性能负结果不作重启理由。profile排除GPU payload/optimizer，报告resident和增量peak，并保留coowner/顺序限制。具体命令、环境/CWD、路径和产物见experiment.json。

发布前证据：42项合成/协议检查与唯一独立P0/P1审查PASS，独立复跑42项，审查未访问真实IQ/权重/远端。已检查全部coarse/action/prior联合梯度/更新、原teacher与F冻结、两种损失参照、严格复载、R07实际进程等待与独立truth-last自动测试。普通N607预检VERIFIED，R07实际GPU句柄全部退出，GPU2/3/6/7空闲；R08健康4/5继续。新root/release不存在，数据句柄复用，不重复重验。

实际发布 VERIFIED：方法commit `63a8b45b502351ba72aaf80f83b377036ce255fe` 已推送且远端OID独立比较一致；单归档SHA比较/远端compile/help通过。实际原R17/F50/2Ls noquery smoke PASS，原teacher三步输出逐位等价，四行student coarse/action/prior和原F输入梯度均有限非零，原teacher/F/head0更新，严格复载输出逐位一致。先核实R07完整测试/评分与实际controller/四个GPU句柄退出，再在GPU2/3/6/7各启动一个新worker。实际PID/CWD/完整命令/commit/GPU/loggrowth与首轮完成均核实，R08健康4/5继续。没有读取R07结果改变R09已固定设计/训练/矩阵。

完整汇总工具16项合成检查PASS，覆盖联合组件预算、全模型optimizer参数、teacher/clean损失参照及clean IQ权重、80串行pass与120组件epoch区别、四行全七视图/分组、整数1pp重算与actual gains/70%状态分开、payload-free推理profile。该工具未改运行中方法commit；当前尚无R09目标准确率。

分析完善：17项聚焦合成检查PASS。同压缩规模teacher/clean目标比较逐源/目标七场景准确率、原始正确数、MacroF1、分组及配对clean波形差；clean减teacher固定方向。只比较完成后的冻结汇总，不从汇总推算逐样本配对检验，不单独归因IQ或F参照，0新模型forward/训练/target访问。方法commit及运行中配方不变。


## 完整目标结果

四个固定候选均完成E20/4940、source七视图诊断和同168000物理样本×七目标场景；全七预测固定后独立CPU truth-last，实际controller/worker句柄全部退出。逐场景相对原R17下降<=1个百分点，以整数正确数验收；补报相对已验收3118001参数版本差值。完整分组见results.json/row_metrics.csv。

|版本|场景|目标准确率|原R17准确率|相对原R17（百分点）|相对3118001版（百分点）|1pp验收|
|---|---|---:|---:|---:|---:|---|
|joint_teacher_quarter|clean|84.789286|84.586905|+0.202381|+0.202976|通过|
|joint_teacher_quarter|practical_high|80.686310|81.319048|-0.632738|-0.268452|通过|
|joint_teacher_quarter|practical_mid|79.443452|79.577976|-0.134524|+0.123810|通过|
|joint_teacher_quarter|practical_low_suburban|78.061905|77.996429|+0.065476|+0.425595|通过|
|joint_teacher_quarter|practical_high_urban|79.673214|80.353571|-0.680357|-0.316071|通过|
|joint_teacher_quarter|practical_mid_urban|73.268452|73.685119|-0.416667|-0.446429|通过|
|joint_teacher_quarter|practical_low_urban|66.433929|67.216667|-0.782738|-0.922619|通过|
|joint_clean_quarter|clean|85.267857|84.586905|+0.680952|+0.681548|通过|
|joint_clean_quarter|practical_high|80.064881|81.319048|-1.254167|-0.889881|未通过|
|joint_clean_quarter|practical_mid|78.713690|79.577976|-0.864286|-0.605952|通过|
|joint_clean_quarter|practical_low_suburban|77.263690|77.996429|-0.732738|-0.372619|通过|
|joint_clean_quarter|practical_high_urban|79.115476|80.353571|-1.238095|-0.873810|未通过|
|joint_clean_quarter|practical_mid_urban|73.297619|73.685119|-0.387500|-0.417262|通过|
|joint_clean_quarter|practical_low_urban|67.082143|67.216667|-0.134524|-0.274405|通过|
|joint_teacher_eighth_pointwise|clean|84.434524|84.586905|-0.152381|-0.151786|通过|
|joint_teacher_eighth_pointwise|practical_high|80.585714|81.319048|-0.733333|-0.369048|通过|
|joint_teacher_eighth_pointwise|practical_mid|79.296429|79.577976|-0.281548|-0.023214|通过|
|joint_teacher_eighth_pointwise|practical_low_suburban|77.776786|77.996429|-0.219643|+0.140476|通过|
|joint_teacher_eighth_pointwise|practical_high_urban|79.446429|80.353571|-0.907143|-0.542857|通过|
|joint_teacher_eighth_pointwise|practical_mid_urban|72.639286|73.685119|-1.045833|-1.075595|未通过|
|joint_teacher_eighth_pointwise|practical_low_urban|65.663095|67.216667|-1.553571|-1.693452|未通过|
|joint_clean_eighth_pointwise|clean|84.833333|84.586905|+0.246429|+0.247024|通过|
|joint_clean_eighth_pointwise|practical_high|81.450000|81.319048|+0.130952|+0.495238|通过|
|joint_clean_eighth_pointwise|practical_mid|79.872619|79.577976|+0.294643|+0.552976|通过|
|joint_clean_eighth_pointwise|practical_low_suburban|78.294643|77.996429|+0.298214|+0.658333|通过|
|joint_clean_eighth_pointwise|practical_high_urban|80.279762|80.353571|-0.073810|+0.290476|通过|
|joint_clean_eighth_pointwise|practical_mid_urban|73.789881|73.685119|+0.104762|+0.075000|通过|
|joint_clean_eighth_pointwise|practical_low_urban|67.107143|67.216667|-0.109524|-0.249405|通过|

本轮结论：**符合七场景1个百分点容差：joint_teacher_quarter, joint_clean_eighth_pointwise**。本轮固定已测试候选中，达标最少完整参数为 1592461（joint_clean_eighth_pointwise），比原R17减少65.8642%，比已验收3118001参数版再减少48.9269%。这不是全局最简证明。单训练seed/单原F50seed/已暴露基准，不称新盲测或多seed总体保证；所有固定候选都已报告，目标结果未回流训练、超参、父权重、候选重排或选择性重跑。

|版本|完整参数|参数较原R17减少|相对原R17最大下降（百分点）|相对3118001版最大下降（百分点）|零下降诊断|low urban准确率|达到70%|
|---|---:|---:|---:|---:|---|---:|---|
|joint_teacher_quarter|1822977|60.9229%|0.782738|0.922619|False|66.433929|否|
|joint_clean_quarter|1822977|60.9229%|1.254167|0.889881|False|67.082143|否|
|joint_teacher_eighth_pointwise|1592461|65.8642%|1.553571|1.693452|False|65.663095|否|
|joint_clean_eighth_pointwise|1592461|65.8642%|0.109524|0.249405|False|67.107143|否|

压缩容差通过不等于优化成功；下面逐行报告实际涨分和退化，不用均值替代逐场景验收。单训练seed/已暴露基准不能证明稳定总体提升，lowurban是否达到70%依实际结果；其他环境接近clean未自设阈值。

|版本|上涨场景|不变场景|下降场景|七场景均不下降|lowurban达到70%|
|---|---|---|---|---|---|
|joint_teacher_quarter|clean, practical_low_suburban||practical_high, practical_mid, practical_high_urban, practical_mid_urban, practical_low_urban|False|False|
|joint_clean_quarter|clean||practical_high, practical_mid, practical_low_suburban, practical_high_urban, practical_mid_urban, practical_low_urban|False|False|
|joint_teacher_eighth_pointwise|||clean, practical_high, practical_mid, practical_low_suburban, practical_high_urban, practical_mid_urban, practical_low_urban|False|False|
|joint_clean_eighth_pointwise|clean, practical_high, practical_mid, practical_low_suburban, practical_mid_urban||practical_high_urban, practical_low_urban|False|False|

两种架构各自teacher/clean纠正目标为预登记匹配比较，两者皆联合优化coarse/action/prior；clean纠正同时加入配对clean波形和clean F参照，不能单独归因某项。两种结构也保留对应R08冻结context对照的独立结果，不将跨预算/目标差混称单一因素。

同规模目标对照以clean纠正减teacher保真表示，正准确率差代表纠正更高，负NMSE差代表纠正更接近配对clean；它是汇总差值，不据此推算逐样本配对显著性。

|规模|场景|准确率差（百分点）|正确数差|MacroF1差（百分点）|全帧NMSE差|主体232 NMSE差|尾24 NMSE差|
|---|---|---:|---:|---:|---:|---:|---:|
|quarter|clean|+0.478571|+804|+0.411015|+0.000145792961|+0.000145785045|+0.000145077007|
|quarter|practical_high|-0.621429|-1044|-0.588915|-2.25007534e-06|+3.07559967e-05|-0.000453472137|
|quarter|practical_mid|-0.729762|-1226|-0.717535|+4.3772161e-06|+3.33506614e-05|-0.000306174159|
|quarter|practical_low_suburban|-0.798214|-1341|-0.834625|-2.09920108e-05|-1.27032399e-05|-2.36928463e-05|
|quarter|practical_high_urban|-0.557738|-937|-0.502821|+6.58221543e-05|+8.11992213e-05|-0.000189848244|
|quarter|practical_mid_urban|+0.029167|+49|+0.180815|+0.000339768827|+0.000277105719|+0.000888541341|
|quarter|practical_low_urban|+0.648214|+1089|+0.785130|+0.000831544399|+0.000620521605|+0.00259651244|
|eighth_pointwise|clean|+0.398810|+670|+0.374455|-0.000370299909|-0.000382449361|-0.000259669498|
|eighth_pointwise|practical_high|+0.864286|+1452|+1.138909|+0.00012197718|-2.943445e-05|+0.00211475044|
|eighth_pointwise|practical_mid|+0.576190|+968|+0.858100|+0.00016756542|+3.47942114e-06|+0.00234419107|
|eighth_pointwise|practical_low_suburban|+0.517857|+870|+0.786664|+0.000165147707|-2.57845968e-05|+0.00262667239|
|eighth_pointwise|practical_high_urban|+0.833333|+1400|+1.126144|+0.000202108175|+4.02368605e-05|+0.00226987898|
|eighth_pointwise|practical_mid_urban|+1.150595|+1933|+1.597498|+0.000726092607|+0.000492665917|+0.00339622796|
|eighth_pointwise|practical_low_urban|+1.444048|+2426|+1.958506|+0.001459077|+0.00109533966|+0.0052216053|

相对已验收2352389参数版的补充对照如下；不改变固定原R17逐场景1pp主验收，也不以补充结果回流训练或重排。每场景原始正确数/波形/分组差值保存在results.json的relative_to_accepted_2352389。最大下降为有符号差，若全部提高则可为负。

|版本|完整参数|相对2352389再减参数|相对2352389最大下降（百分点）|lowurban准确率|lowurban相对2352389（百分点）|
|---|---:|---:|---:|---:|---:|
|joint_teacher_quarter|1822977|22.505291%|0.927976|66.433929|-0.927976|
|joint_clean_quarter|1822977|22.505291%|1.038095|67.082143|-0.279762|
|joint_teacher_eighth_pointwise|1592461|32.304521%|1.698810|65.663095|-1.698810|
|joint_clean_eighth_pointwise|1592461|32.304521%|0.254762|67.107143|-0.254762|

### joint_teacher_quarter 的配对clean差距与源诊断

|场景|source准确率|source相对原R17（百分点）|目标全帧NMSE|主体232 NMSE|尾24 NMSE|配对clean特征余弦距离|目标raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.388889|-0.055556|0.00136962871|0.00137402222|0.00130231038|0.00179589842|1585/1395/+190|
|practical_high|97.944444|+0.111111|0.0177042745|0.00725405011|0.119568974|0.013718266|71071/3949/+67122|
|practical_mid|96.944444|+0.166667|0.023352541|0.013168836|0.12240047|0.0178037975|71022/4617/+66405|
|practical_low_suburban|96.166667|+0.222222|0.029531328|0.0195767153|0.126084849|0.0223494973|70861/5287/+65574|
|practical_high_urban|97.333333|+0.333333|0.0234318059|0.0131265931|0.123746701|0.0169050135|71561/4403/+67158|
|practical_mid_urban|91.222222|-0.222222|0.0505858324|0.0410022996|0.143262863|0.0337065011|72264/7105/+65159|
|practical_low_urban|84.777778|-2.055556|0.0759527087|0.066875428|0.162974507|0.0507620163|70360/9654/+60706|

完整20轮日志已分析，采用固定最终E20，未按最小训练loss或V/target择优。各项loss首轮→末轮：loss 1.5172993→0.13586355；iq_nmse 0.13915347→0.030526751；teacher_stage_nmse 0.11366514→0.0084029734；task_KL 0.33477073→0.043557552；task_feature_distance 0.045877151→0.008276261。训练耗时3038.324秒，中位每轮152.045秒。

### joint_clean_quarter 的配对clean差距与源诊断

|场景|source准确率|source相对原R17（百分点）|目标全帧NMSE|主体232 NMSE|尾24 NMSE|配对clean特征余弦距离|目标raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.388889|-0.055556|0.00151542167|0.00151980727|0.00144738739|0.00111141591|1929/935/+994|
|practical_high|98.277778|+0.444444|0.0177020244|0.00728480611|0.119115502|0.0132505782|69929/3851/+66078|
|practical_mid|97.388889|+0.611111|0.0233569182|0.0132021867|0.122094296|0.0175058208|69589/4410/+65179|
|practical_low_suburban|97.000000|+1.055556|0.029510336|0.0195640121|0.126061156|0.0221844893|69348/5115/+64233|
|practical_high_urban|97.888889|+0.888889|0.023497628|0.0132077923|0.123556852|0.0166753456|70437/4216/+66221|
|practical_mid_urban|93.166667|+1.722222|0.0509256013|0.0412794054|0.144151405|0.0339255631|71704/6496/+65208|
|practical_low_urban|87.444444|+0.611111|0.0767842531|0.0674959496|0.165571019|0.0505600758|70415/8620/+61795|

完整20轮日志已分析，采用固定最终E20，未按最小训练loss或V/target择优。各项loss首轮→末轮：loss 1.682317→0.21039386；iq_nmse 0.13697916→0.030756033；teacher_stage_nmse 0.11174526→0.0089130112；task_KL 0.37702007→0.076632538；task_feature_distance 0.050865129→0.01387518。训练耗时3047.510秒，中位每轮152.391秒。

### joint_teacher_eighth_pointwise 的配对clean差距与源诊断

|场景|source准确率|source相对原R17（百分点）|目标全帧NMSE|主体232 NMSE|尾24 NMSE|配对clean特征余弦距离|目标raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.500000|+0.055556|0.00138210272|0.00138308038|0.00134465948|0.0013533351|1195/1601/-406|
|practical_high|97.888889|+0.055556|0.0181595478|0.00739671476|0.122464545|0.0133249937|70863/3910/+66953|
|practical_mid|96.888889|+0.111111|0.0239525866|0.0135069638|0.125082374|0.0177266691|70690/4532/+66158|
|practical_low_suburban|96.000000|+0.055556|0.0302791949|0.0201280266|0.128444418|0.0225506276|70391/5296/+65095|
|practical_high_urban|97.277778|+0.277778|0.023900399|0.0133183431|0.126342595|0.0168000367|71159/4382/+66777|
|practical_mid_urban|90.500000|-0.944444|0.0510359406|0.0413700826|0.144177869|0.0345824622|71360/7258/+64102|
|practical_low_urban|83.944444|-2.888889|0.0763110444|0.0673449859|0.162068263|0.0520765781|69321/9910/+59411|

完整20轮日志已分析，采用固定最终E20，未按最小训练loss或V/target择优。各项loss首轮→末轮：loss 1.4964613→0.14278948；iq_nmse 0.13695115→0.030885071；teacher_stage_nmse 0.11236983→0.0087274602；task_KL 0.32805941→0.046915681；task_feature_distance 0.044703546→0.0085991952。训练耗时3046.106秒，中位每轮151.895秒。

### joint_clean_eighth_pointwise 的配对clean差距与源诊断

|场景|source准确率|source相对原R17（百分点）|目标全帧NMSE|主体232 NMSE|尾24 NMSE|配对clean特征余弦距离|目标raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.388889|-0.055556|0.00101180281|0.00100063102|0.00108498998|0.00105104677|1093/829/+264|
|practical_high|98.222222|+0.388889|0.018281525|0.00736728031|0.124579296|0.0132366698|71810/3405/+68405|
|practical_mid|97.277778|+0.500000|0.024120152|0.0135104433|0.127426565|0.017577067|71172/4046/+67126|
|practical_low_suburban|97.111111|+1.166667|0.0304443426|0.020102242|0.131071091|0.022352431|70702/4737/+65965|
|practical_high_urban|98.000000|+1.000000|0.0241025072|0.0133585799|0.128612474|0.016635254|72058/3881/+68177|
|practical_mid_urban|93.111111|+1.666667|0.0517620333|0.0418627486|0.147574097|0.0339567177|72373/6338/+66035|
|practical_low_urban|87.222222|+0.388889|0.0777701214|0.0684403256|0.167289868|0.0508097038|70525/8688/+61837|

完整20轮日志已分析，采用固定最终E20，未按最小训练loss或V/target择优。各项loss首轮→末轮：loss 1.6855178→0.22025272；iq_nmse 0.13632456→0.03115918；teacher_stage_nmse 0.11199075→0.0093044485；task_KL 0.37910622→0.081892554；task_feature_distance 0.050179541→0.014156497。训练耗时3029.044秒，中位每轮151.022秒。

成本测量只用固定16条source V、batch1/2warmup，包含完整coarse/action/三步CG；CPU加载后删除checkpointpayload，再放入GPU，无GPU optimizer/F/teacher。各版本测量顺序及其他owner仍影响时间，不视为部署加速证明。

|版本|checkpoint字节|原R17中位ms|精简版中位ms|延迟比|原R17resident字节|精简resident字节|原R17增量peak字节|精简增量peak字节|目标预测秒（含F及写盘）|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|joint_teacher_quarter|7678140|47.012|91.948|1.9558|35870720|24508416|1158656|1116272640|1003.227|
|joint_clean_quarter|7678140|47.709|92.716|1.9434|35870720|24508416|1158656|1116272640|1016.579|
|joint_teacher_eighth_pointwise|6753244|47.602|81.389|1.7098|35870720|23585280|1158656|1116272640|993.815|
|joint_clean_eighth_pointwise|6753244|46.274|80.122|1.7315|35870720|23585280|1158656|1116272640|987.364|

每行新增20轮/4940更新/L126000/U1134000；原R17 teacher、原teacher context/student context、原Fteacher/Fstudent各1260000输入前向；student的context/prior及F输入反传，coarse/action/prior均4940更新，0F/head/teacher更新。teacher F参照按固定行使用原R17恢复或source paired clean，clean IQ直接损失权重0/1。0校准IQ；CPU均值SVD初始化时间单列。继承coarse20+action20+R17prior20与本轮joint20共80串行pass，每组件累计40，component-epoch总和120，不把120称串行轮；原F50预训练50/11100和教师成本单列。

四份完整恢复器已取回本地 models/<arm>/final.pt，大小与远端一致、来源/最终预算/共享alias严格加载通过，单条合成零IQ输出有限；0实际IQ/target访问。无optimizer/F/teacher权重，包含完整联合优化后的coarse/action/prior，不提交权重到Git。加载使用experiments.pamf_r17_joint_optimize.runtime.load_checkpoint，输入原生尺度float32 [B,2,256]、I/Q顺序，输出同形状恢复IQ，下游仍接原F/defaulthead。
