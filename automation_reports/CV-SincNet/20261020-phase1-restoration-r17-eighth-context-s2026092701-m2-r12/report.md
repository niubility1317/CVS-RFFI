# R17 八分之一上下文压缩与源 clean 联合优化 R12

状态：PLANNED。补齐此前0IQ静态容量审计的eighth context层级，沿用与R10/R11相同的两种固定prior、直接source-clean纠正损失和E20预算。R10提供half context，R11提供full/quarter context匹配对照。以正式目标七场景逐项相对原R17下降不超过1个百分点为底线，另报真实提分、low urban≥70%、其他环境距clean和paired-clean恢复差距。容差通过不等于性能改善。

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
