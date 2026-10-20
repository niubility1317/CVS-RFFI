# R17 纯函数保真与分步归一化精简 R02

状态：PLANNED。用户最新要求只研发R17精简版，最大程度精简且不能降低性能；原R17 unroll_unshared/source-selected E20/4940/cd977448为固定对照。目标是原F50 seed2701/defaulthead、同168000物理样本×七视图准确率逐项不下降，容许下降0个百分点；原low urban67.216667%，70%和其他环境接近clean仍是未完成的长期目标。基准已经暴露，不称新盲测。

源依据：R01两臂完整E20/4940后波形护栏通过，但source V1800低urban共享85.944444%（−.888889pp）、半秩86.333333%（−.5pp），均未通过源资格，0新增target；原结果与失败student保留，不加载其权重。原R17权重只读分析显示步间长卷积相对共同均值Frobenius差异.28166、99%差异能量秩中位61（23至121），不支持假定步间差异天然极低秩；这不是功能/身份误差或压缩不可能性的证据。当前新机制在任何新压缩模型target结果前，由源证据固定。

|固定源候选|完整参数|新prior唯一参数|相对原4665073|
|---|---:|---:|---:|
|step_norm_fidelity|3118001|960454|减少33.1629%|
|factor_half_fidelity|3775105|1617558|减少19.0773%|

step_norm：3个Core9仅每步28个kernel>1卷积weight绑定为原3步权重均值；保留每步原conv bias、GroupNorm affine、1×1和其他参数。优化器按唯一Parameter更新一次，checkpoint复载保持绑定。内部固定迭代步索引不是TX/RX/场景路由。factor_half保持R01相同的三独立半秩SVD架构/初始化，改变蒸馏目标作源域匹配参考。两臂都只从原合法R17教师派生，0失败student初始化。coarse/action/native variance/SCALE²、3步×2CG保持，不同时减步。

新损失仅10×三步学生/冻结R17教师IQNMSE均值，配对clean仅作detach的均方能量分母（floor1e-4）和IQNMSE日志诊断，不作优化目标。原R07完整L6300/U56700/63000 cachedreceived epochs1..20（15%clean/85%residual），无额外clean拼接。E20/4940、batch256、model2701/data2026102001、AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32保持，teacher/context冻结。L126000/U1134000物理优化曝光，teacher1260000输入、每输入3步；prior继承20+新20=40，coarse20/action20冻结，共80累计串行训练pass，教师计算/父成本单列。0TX/F权重或输出/V/target训练，0头拟合。

每候选sourceV1800七视图全部预测固定后独立CPU scorer连接标签；每viewaccuracy>=原R17，mean6NMSE<=1.05×R17、cleanNMSE<=.02，合格者按最少完整参数再名称选定一个自动完整目标确认。无合格则0target；失败后不补测落选者、不依据目标调参/重排/重跑。原F50/defaulthead只用于冻结后的识别读出；整条链含原分类器源TX监督，不称全部无监督。目标七预测固定后才独立truth-last，报告逐TX/RX/day confusion/MacroF1/repair/damage、全帧/主体232/尾24波形与配对clean特征差距及准确率。原R17源/目标既有完整对照复用。

推理只用当前IQ，单一恢复IQ供全部原F分支；不读teacher/F/clean/源cache/真实身份/RX/场景/其他帧或更新状态。相同GPU/16固定sourceV IQ/batch1/2warmup对照原R17和学生的完整恢复器延迟与PyTorch峰值allocated，不含F/写盘。参数减少不保证延迟减少，分步差异状态可能增加计算。训练checkpoint不含优化器；原始IQ/权重/逐样本预测/日志/机器状态不提交，聚合结果公开。

10项聚焦检查PASS，一次独立P0/P1审查独立10PASS且无阻断问题。launcher第一步实际原R17/原F50/2Ls无querysmoke，教师stage函数完全一致、各student先验梯度、冻结context/teacher、checkpoint输出完全相同；0TX/V/target/optimizer，PASS立即资源队列。源资格不是目标成功证据。

N607普通账户，GPU4/5等R01根终态且实际owned GPU handles退出；每卡一ownworker，source≥18GB/target≥12GB/allowner<2，保留其他owner。CPU2线程/一ownscorer，等R01终态后ownscore.lock，无循环。一次新root/release preflight、归档SHA比较、远端compile及实际PID/CWD/cmd/GPU/log增长核实。技术失败保留产物/健康行继续/pendingtarget暂停/无自动重试，科学下降不作技术重启理由。精确命令、CWD、路径/seed/资源/停止规则见experiment.json。


发布前一次普通账户预检 VERIFIED：source63000/V1800与原R07缓存句柄匹配，原R17先源选择后目标、E20/4940、0TX/F/target训练及同F50基线预测句柄已核实。R01实际owned GPU handles均已退出，GPU4/5各约24GB空闲；保留其他owner，不重新验证VALIDATED_ONCE数据。实际checkpoint smoke由launcher第一步执行，通过后立即继续。
