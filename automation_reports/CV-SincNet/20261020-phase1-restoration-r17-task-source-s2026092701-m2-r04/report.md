# 冻结原 F 知识约束的 R17 精简 R04

最新用户验收（2026-10-20）：**以目标域测试为准，各场景准确率较原R17下降不超过1个百分点**，例如80%至79%，含边界；不是相对下降1%。每场景168000条对应最多减少1680条正确样本，逐场景按整数正确数判断，其他场景提升不能抵消超限。该要求覆盖下文原预登记的目标0下降门槛；历史0下降字段仅作额外诊断。更新时R04两臂仍在源训练，0新候选目标访问。健康训练、候选、源筛查和自动目标范围保持冻结，独立结果分析按最新要求验收；low urban≥70%与接近clean长期目标仍保留。

状态：ANALYZED。仅优化原 R17 精简版，最大程度精简且不降低目标性能。原 R17 为 source-selected-before-target 的 unroll_unshared/cd977448/E20/4940，原 F50 seed2701/4df7/默认联合训练头不变。最终按同 168000 物理样本×clean及六practical，逐场景正确数≥原R17，容许下降0个百分点；平均值不能掩盖单场景下降。low urban≥70%及其余环境接近clean的长期目标未完成；不自设“接近”的数值阈值。基准已暴露，不称新盲测。

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


## 完整结果

两个候选均完成固定E20/4940、L126000/U1134000与全部20条训练记录解析；教师/context/F参数更新、TX/V/target标签训练读取均为0；明确使用冻结原F知识，每臂Fteacher/student各1260000前向及输入反传，0头拟合。全部source七预测固定后独立CPU评分，controller与所有source/target真实owned进程已退出。该收集仅读取完整汇总与日志，不新增训练/预测或读取IQ、TX标签。

源域准确率（%，1800物理样本；不作为目标准确率）：

|场景|原R17|分步状态共享|激活校准半秩|
|---|---:|---:|---:|
|clean|98.444444|98.444444|98.444444|
|practical_high|97.833333|98.000000|97.666667|
|practical_mid|96.777778|96.944444|96.777778|
|practical_low_suburban|95.944444|95.777778|96.222222|
|practical_high_urban|97.000000|97.000000|97.166667|
|practical_mid_urban|91.444444|91.388889|91.611111|
|practical_low_urban|86.833333|86.055556|86.722222|

源健康筛查（允许1个百分点）：{"step_norm_task": false, "activation_task": false}。固定源选择结果：`None`。各场景准确率/波形条件与完整分层、混淆矩阵及净修复算术见[完整汇总](results.json)。

本轮结论：**两候选均未通过预登记源健康筛查，0新目标预测；未交付能保持目标性能的结构压缩版本**。完整训练和源域负结果保留，原R17继续作为部署对照。没有在本轮结果后改变阈值、补测或选择性重跑，不能把源域结果写成目标退化/保持结论。


实际成本（完整恢复器含冻结coarse/action与CG，FP32；模型文件不含优化器）：

|候选|完整参数|参数减少|文件字节|优化阶段秒|原R17 batch1中位ms|压缩版中位ms|延迟比|PyTorch峰值allocated比|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|step_norm_task|3118001|33.1629%|12777244|2108.753|44.124|42.118|0.9545|0.8329|
|activation_task|3775105|19.0773%|15475140|2100.415|43.922|84.642|1.9271|31.0189|

优化时间包含原R17教师、Fteacher/student前向及输入反传和20轮cache/优化，不含初始化SVD或排队。延迟对照为各行同一GPU/同16个sourceV IQ、batch1、2次warmup、顺序测量，含完整恢复图、不含F识别器/磁盘写入；热缓存、顺序及其他owner干扰可能影响结果，不声称独占GPU或统计保证。共享参数不等于减少三步前向。

校准实际成本（每行额外训练IQ教师前向，不计为优化更新）：

|候选|IQ|L_s|U_s|每层patch数|校准与初始化秒|
|---|---:|---:|---:|---:|---:|
|step_norm_task|0|0|0|0|0.087326|
|activation_task|512|55|457|4096|3.982702|

仅activation臂额外512校准，stepnorm校准0；样本、固定位置和ridge在目标前固定；层加权误差只作初始化诊断，不作身份收益替代。

全日志损失摘要（固定finalE20，训练最小值不作选模）：

|候选|损失项|首轮|末轮|20轮均值|每轮线性斜率|
|---|---|---:|---:|---:|---:|
|step_norm_task|loss|0.158338763|0.0504897538|0.0730761086|-0.00367772844|
|step_norm_task|iq_nmse|0.0301646762|0.0255763918|0.0265487316|-0.000145138393|
|step_norm_task|teacher_stage_nmse|0.00537576364|0.0025750892|0.00312191933|-8.84850849e-05|
|step_norm_task|task_KL|0.0918377028|0.0210454352|0.0363893382|-0.00249603691|
|step_norm_task|task_feature_distance|0.0127434233|0.00369342683|0.00546757704|-0.000296840671|
|activation_task|loss|0.10375872|0.0307285751|0.0492530802|-0.00283267798|
|activation_task|iq_nmse|0.0275554591|0.0231037662|0.0243274523|-0.000166151195|
|activation_task|teacher_stage_nmse|0.00333914418|0.00133981257|0.00184513542|-7.75722893e-05|
|activation_task|task_KL|0.0624261839|0.0149194302|0.026952669|-0.00184060641|
|activation_task|task_feature_distance|0.00794109425|0.00241101931|0.00384905699|-0.000216348681|

独立重算315组混淆矩阵、accuracy/Macro-F1、repair−damage/net与分组样本数；TX/RX/day分区分别回合至各view总矩阵。全部结果是单一训练seed及固定测试集点估计，七视图相关，不增加独立物理样本量；不称总体/多seed零退化、在轨效果或数学全局最简。实际方法commit`a310b3b5ff30e23584e7bc950e8a0ff96ac685fb`保持；新增分析代码只读取完整聚合，不修改健康run。

与原R17逐条预测对照（仅复用已冻结预测，全部七视图先于truth；相对原R17的lost/gained区别于前述相对raw的repair/damage）：

|范围|候选|场景|原正确变错|原错误修复|净正确数差|决策翻转|
|---|---|---|---:|---:|---:|---:|
|source|step_norm_task|clean|0|0|+0|0|
|source|step_norm_task|practical_high|1|4|+3|6|
|source|step_norm_task|practical_mid|6|9|+3|19|
|source|step_norm_task|practical_low_suburban|13|10|-3|28|
|source|step_norm_task|practical_high_urban|10|10|+0|22|
|source|step_norm_task|practical_mid_urban|32|31|-1|75|
|source|step_norm_task|practical_low_urban|40|26|-14|89|
|source|activation_task|clean|0|0|+0|0|
|source|activation_task|practical_high|4|1|-3|5|
|source|activation_task|practical_mid|7|7|+0|18|
|source|activation_task|practical_low_suburban|4|9|+5|19|
|source|activation_task|practical_high_urban|6|9|+3|15|
|source|activation_task|practical_mid_urban|17|20|+3|44|
|source|activation_task|practical_low_urban|30|28|-2|80|

完整教师置信度/波形与特征偏离及TX/RX/day分组见[保存预测配对汇总](saved_prediction_fidelity.json)。这项同row事后分析不改变选模、模型或目标准确率门槛，不作因果归因。
