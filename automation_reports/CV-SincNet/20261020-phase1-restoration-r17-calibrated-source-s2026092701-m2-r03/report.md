# 源激活校准的 R17 低秩精简 R03

状态：RUNNING。用户要求专门精简R17，最大程度精简且不能降低目标性能。固定原source-selected R17 unroll_unshared/cd977448/E20/4940，原F50 seed2701/原默认头，同168000物理样本×clean及六practical准确率逐项不降，容许下降0个百分点。原low urban67.216667%；70%和其他环境接近clean长期目标未完成，接近clean不自行新增数值阈值。基准已暴露，不称新盲测。

本方案在R02仍运行、尚无新压缩模型目标结果时固定。仅依据R01完整源域负结果和原R17权重只读分析：统一半秩权重相对Frobenius误差.52557，要求逐核保留99%权重能量时没有正参数节省；权重误差不等于输入分布下的输出误差，因此检验激活加权初始化。它不是对R02中间loss或目标分数的响应，不使用R01/R02学生权重。

|固定源候选|完整参数|新prior参数|相对原4665073|
|---|---:|---:|---:|
|activation_half|3775105|1617558|减少19.0773%|
|activation_two_thirds|4241809|2084262|减少9.0730%|

两臂从原R17三独立先验出发，仅对分解后权重更少的长卷积分解。校准固定取R07/cache/epochs/01/received.npy的63000训练IQ，seed2026102002排列前512条；只读L_s/U_s角色以计数，0TX标签。每层每IQ固定8个均匀输出位置，以真实卷积stride/padding/dilation提取输入patch，流式未中心化二阶矩C=E[xxᵀ]。ridge=max(1e-8,.001×mean(diagC))，对W·Cholesky(C+ridgeI)做截断SVD，左因子逆白化，平衡分配sqrt singular values；原bias/GN/1×1及其他状态保持。正则和样本/位置/seed/秩在目标前固定，不以V选择校准超参。新增512源IQ教师前向和实际L/U曝光、patch数、耗时及层误差单列，0优化更新；校准统计不进入推理。

原coarse/action/native variance/SCALE²、3步×2CG保持；只优化10×三步学生/冻结原R17输出IQNMSE均值，源clean仅detached能量分母floor1e-4与日志诊断。R07完整L6300/U56700/63000缓存epochs1..20、15%clean/85%residual，无额外clean拼接。E20/4940、batch256、model2701/data2026102001、AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32保持。优化曝光L126000/U1134000，教师优化阶段1260000输入，另加512校准；prior继承20+新20=40，coarse20/action20冻结，共80继承串行训练pass，额外教师/校准与父成本分别报告。0TX/F/V/target训练或校准，0头拟合。

每臂source V1800七视图预测全部固定后独立CPU truth-last评分；每viewaccuracy>=原R17、mean6NMSE<=1.05×原R17、cleanNMSE<=.02。合格者按最少完整参数再名称固定选一个自动全目标确认；无合格则0target，失败后不补测落选者、不回流调参/重排/重跑。原F50/defaulthead只在冻结后识别；整链原分类器含源TX监督，不称全链无监督。目标七预测固定后独立truth-last，报告逐场景/逐TX/RX/day accuracy/MacroF1/confusion/repair/damage、全帧/主体232/尾24波形与配对clean特征差距。目标准确率按原始正确数逐项验收，不以均值或舍入掩盖下降；既有原R17完整对照复用。

推理仅当前IQ，单一恢复IQ供原F所有分支，0teacher/F/clean/校准cache/身份/RX/场景/其他帧/状态更新。每行同GPU、同16个sourceV IQ、batch1/2warmup实测完整链延迟和PyTorch峰值allocated，含coarse/action/CG，不含F和写盘；参数减少不预先承诺加速。checkpoint无优化器，原始IQ/权重/逐样本预测/日志/机器状态不提交。

18项聚焦检查PASS，唯一独立P0/P1审查独立复跑18项检查PASS，未发现确定P0/P1。launcher第一步原R17/原F50/2Ls实际checkpoint无querysmoke，校准仅该2Ls，验证各学生梯度、冻结教师/context、严格复载逐位输出、教师stage函数；0TX/V/target/optimizer，PASS立即排队。数学/梯度/源资格不是目标性能保持证据。

N607普通账户，GPU4/5等待R02根终态且所有实际owned GPU handles退出；一own/card、source≥18GB/target≥12GB/allowner<2，保留其他owner。CPU2线程/max1ownscorer，等R02终态后ownscore.lock。一次新root/release预检、一次归档SHA比较/远端compile、实际PID/CWD/cmd/GPU/日志增长核实。技术失败保留产物/健康行继续/pendingtarget暂停/无自动重试；科学下降不作技术重启理由。命令、CWD、路径、资源与预期产物见experiment.json。


发布前普通账户预检 VERIFIED：原R17源选择先于目标、完整E20/4940及0TX/F/target训练、同F50完整基线句柄、原R07source63000/V1800缓存均匹配；不重新验证已有数据。R02实际owned GPU handles仍有2个，新控制器等其根终态和真实handles退出，不改当前任务。源码/方案在任何新压缩目标结果前固定，实际checkpoint无querysmoke由launcher第一步执行。


实际发布 VERIFIED：方法commit `bbccd16408a0d9fc356b48f3579044e2284112dd` 已push且远端OID独立比较一致；一次归档SHA比较、新模块远端compile与help均通过。原R17/原F50/2Ls实际无querysmoke PASS，各步prior梯度有限非零、teacher/context冻结、严格checkpoint复载输出逐位一致、教师stage函数一致，0TX/V/target/optimizer。等R02根终态且实际owned GPU handles全部退出后，GPU4/5两source真实PID/CWD/完整命令/commit/GPU PID/日志增长已核实，均已完成至少第1轮。源校准额外读取如下，仅训练cache epoch01，未读取验证/目标或身份标签。

|候选|校准IQ|L_s曝光|U_s曝光|每层patch数|校准及初始化秒|
|---|---:|---:|---:|---:|---:|
|activation_half|512|55|457|4096|4.185298|
|activation_two_thirds|512|55|457|4096|4.391543|

校准误差/梯度/损失不证明目标性能。原固定source选择和自动单候选目标流程保持，尚无本轮目标结果。
