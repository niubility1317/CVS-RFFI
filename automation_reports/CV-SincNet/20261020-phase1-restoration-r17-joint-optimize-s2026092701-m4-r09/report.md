# R17 压缩与恢复识别联合优化 R09

状态：RUNNING。用户最新明确“压缩的同时也要优化”。本轮不把仅维持1pp容差当成识别优化成功，固定两种压缩架构×两种训练目标：共同联合优化coarse/action/prior全部参数，比较原教师行为保真与源域clean恢复/身份纠正。原F50/defaulthead冻结，性能以正式目标测试为准，逐场景原R17−1pp是底线，low urban≥70%和其余环境接近clean的实际差距独立报告，“接近”不自设阈值。

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
