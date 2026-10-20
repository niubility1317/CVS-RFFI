# 冻结原 F 知识约束的 R17 精简 R04

状态：RUNNING。仅优化原 R17 精简版，最大程度精简且不降低目标性能。原 R17 为 source-selected-before-target 的 unroll_unshared/cd977448/E20/4940，原 F50 seed2701/4df7/默认联合训练头不变。最终按同 168000 物理样本×clean及六practical，逐场景正确数≥原R17，容许下降0个百分点；平均值不能掩盖单场景下降。low urban≥70%及其余环境接近clean的长期目标未完成；不自设“接近”的数值阈值。基准已暴露，不称新盲测。

|固定源候选|完整参数|新 prior 参数|相对原 4665073|
|---|---:|---:|---:|
|step_norm_task|3118001|960454|减少33.1629%|
|activation_task|3775105|1617558|减少19.0773%|

两臂只从原R17初始化，不加载R01/R02/R03失败学生。step_norm_task共享原三步长卷积均值权重，保留每步bias/GN/1×1和其他状态；activation_task沿用源激活二阶矩半秩初始化，固定R07epoch01/seed2026102002排列前512/每IQ每层8位置/ridge=max(1e-8,.001meanDiag)/SVD(WcholC)/逆白化平衡因子，0TX/F/V/target或优化更新。stepnorm无校准额外访问，activation额外512教师前向和实际L/U计数单列。

依据R02已保存源预测配对分析：low urban有49/69次预测翻转，损失20/31条原正确样本；48/64次翻转位于教师置信度<.9。该证据仅支持检验F任务保真约束，不证明目标改善。纯数学原型在R03结果前已提交；R03保持原配方。新候选不读取任何新压缩模型目标成绩来设计或选模。

新损失固定为10×三步学生/原R17教师IQNMSE均值 + KL_T1(F原R17恢复输出 || F学生恢复输出) + .5×单位F160特征距离平方，后两项系数均1。教师参照是同一receivedIQ的原R17恢复，不是clean F输出。source clean仅detached能量分母floor1e-4/日志诊断。原F50及原默认头冻结，student路径仅对输入反传；只更新精简prior，0新TX标签/0F参数更新/0头拟合。明确使用源监督训练好的F知识，不能称为F-free波形无监督方法。

数据、顺序和优化预算与原匹配控制保持：R07完整L6300/U56700/63000缓存epochs1..20、15%clean/85%residual，无额外clean拼接；E20/4940、batch256、model2701/data2026102001、AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32。每臂L126000/U1134000，原R17教师1260000训练输入；新增Fteacher1260000+Fstudent1260000前向并对student输入反传，0F优化。prior继承20+新20=40，frozencoarse20/action20共80继承串行pass；原F50预训练50/11100及新增教师/校准成本分别披露。

本轮源筛查是健康筛查：每个1800条V视图accuracy≥原R17−1个百分点，mean6NMSE≤1.05×原R17且cleanNMSE≤.02。V面板较小，此规则避免把少量源样本翻转等同最终目标性能验收。该规则仅用于本轮，不能追溯重判R01至R03或补测旧模型；不同源规则的“合格”不能直接比较。所有固定源候选完成后，按完整参数最少再名称选一个，自动全目标确认或0target；无落选补测、目标反馈、选择性重跑。目标最终门槛始终逐场景0下降。

每行七视图预测全部固定后独立CPU truth-last评分；报告逐场景accuracy/MacroF1/TX/RX/day/confusion/repair/damage，full256/prefix232/tail24波形与配对clean特征差距。原F/defaulthead始终同一权重，不拟合新头。R02stepnorm与R03half仅作已有匹配源域控制，不新增其目标测试。推理只含当前IQ、原coarse/action/3步×2CG及精简prior；恢复checkpoint不含F/教师/校准cache，原F仅为下游识别器。

19项聚焦检查及唯一独立P0/P1审查PASS。launcher第一步加载实际原R17/原F50/2Ls无query smoke，验证任务输入梯度有限非零、combined prior梯度、F/context冻结和严格复载逐位输出；0TX/V/target/optimizer更新，PASS立即排队。尚未运行实际smoke，不能以合成检查证明真实F兼容或目标性能。

N607普通账户，GPU4/5等待R03根终态且实际ownedGPU进程退出；每卡一own，source≥18GB/target≥12GB/allowner<2，保留其他owner。CPU2线程/max1ownscorer，R03终态后ownscore.lock。不可覆盖新root，一次预检/一次归档SHA比较/远端新modulecompile；实际PID/CWD/cmd/GPU/日志增长核实。技术失败保留、健康行继续、pendingtarget暂停、无自动重试；性能下降不是技术重启理由。具体命令/CWD/路径/环境/产物见experiment.json。

发布前预检 VERIFIED：原R17完整预算及源选择先于目标、原R07/source缓存、原F50基线句柄匹配；R03实际GPU进程已退出，GPU4/5各24124 MB空闲。复用已有数据验证；实际F输入梯度兼容性仍由launcher第一步验证。

实际发布 VERIFIED：方法commit `a310b3b5ff30e23584e7bc950e8a0ff96ac685fb` 已推送/OID一致，一次归档SHA比较、新module远端compile/help通过。实际原R17/原F50/2Ls无query smoke PASS，原F输入梯度有限非零，各prior梯度有效、F/teacher/context冻结、严格复载输出逐位一致；0TX/V/target/optimizer更新。R03根终态且实际GPU handles退出后，两source实际PID/CWD/完整命令/commit/GPU/日志增长及首轮已核实。

|候选|校准IQ|L_s曝光|U_s曝光|初始化秒|
|---|---:|---:|---:|---:|
|step_norm_task|0|0|0|0.087326|
|activation_task|512|55|457|3.982702|

任务约束实际参与训练，training_F_access=True，原F及默认头0参数更新。真实smoke和loss不证明目标性能，本轮尚无完整源健康筛查或目标结果。

同row补充分析已实现，7项合成/权限负检查PASS：整轮结束后，比较原R17与两source候选、以及仅已选目标候选的保存预测，报告lost/gained/decision flips、教师置信度及TX/RX/day分组。先验证全部既有七视图和原独立评分，再读取对应truth；不新增模型推理、原始IQ、权重、参数拟合或落选目标访问。该补充不影响科学门槛、运行方法commit或自动测试，不增加发布阻塞项；尚未读取实际R04结果。
