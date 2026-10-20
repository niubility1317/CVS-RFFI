# R17 纯函数保真与分步归一化精简 R02

状态：ANALYZED。用户最新要求只研发R17精简版，最大程度精简且不能降低性能；原R17 unroll_unshared/source-selected E20/4940/cd977448为固定对照。目标是原F50 seed2701/defaulthead、同168000物理样本×七视图准确率逐项不下降，容许下降0个百分点；原low urban67.216667%，70%和其他环境接近clean仍是未完成的长期目标。基准已经暴露，不称新盲测。

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


实际发布 VERIFIED：方法commit `1515d73c5de3e962a822bbaca34182e28a6ae584` 已push且本地HEAD/远端OID独立比较一致；一次归档SHA比较、远端新模块compile及入口help通过。launcher原R17/原F50/2Ls无querysmoke PASS：教师stage函数一致、两student各步prior梯度有限非零、冻结teacher/context、严格checkpoint复载输出逐位一致，0TX/V/target/optimizer。R01根终态且实际owned GPU handles退出后，新GPU4/5各一source创建；独立/proc和GPU证据确认controller/两source实际LIVE、CWD/完整命令/commit/GPU PID匹配、源日志增长。各已完成至少第1轮，不据中间训练loss判定性能。新聚合分析工具另有10项检查PASS，拒绝把配对clean诊断误算为优化项、缺失训练轮次、未合格目标访问及目标单样本退化。尚无压缩模型目标结果。


登记字段纠正：本轮复用旧登记结构时遗留R15/R16 guard/control、旧授权文字和5个百分点长期差距阈值，现改为原R17单一对照、唯一合格候选和用户最新R17授权；接近clean未获用户数值定义。实际预登记正文、design、命令、源选择、目标范围、不可变run与运行中模型均保持。本次仅纠正公开元数据，未读取新压缩目标结果。


## 原始卷积低秩存储证据

仅原R17最终权重，84个长卷积（28/步），Float64逐核SVD，0物理IQ/forward/TX/F/V/target/训练。保留所有bias/GN/1×1，仅在两个因子总权重少于原矩阵时记为精简；它不改变正在运行的两候选。

|代数假设|完整参数|减少|改动长卷积数|原长卷积整体相对Frobenius误差|
|---|---:|---:|---:|---:|
|half_all|3775105|19.077258%|84|0.525573767|
|half_output_ge64|3821809|18.076116%|60|0.495082234|
|half_output_ge96|3999985|14.256754%|36|0.418996308|
|two_thirds|4241809|9.073041%|81|0.388968436|
|energy99|4665073|0.000000%|0|0.000000000|
|energy999|4665073|0.000000%|0|0.000000000|

统一半秩的代数参数量精确复核当前factor_half_fidelity：3775105；保留输出宽32的卷积仅少节省46704个参数（完整参数减少从19.0773%改为18.0761%），原卷积整体误差由.52557降至.49508。要求逐核保留99%或99.9%权重能量时，没有一个卷积的分解因子比原权重更小。这说明这些直接SVD存储假设不提供近乎无损的权重压缩；不能据此判定函数/身份冗余、性能下降原因或压缩不可能。完整逐层成本见[只读汇总](original_kernel_factorization.json)。3项聚焦算术/权限结构检查PASS，无额外运行审查或发布门槛。


## 完整结果

两个候选均完成固定E20/4940、L126000/U1134000与全部20条训练记录解析；教师/context更新、TX/F/V/target训练读取均为0。全部source七预测固定后独立CPU评分，controller与所有source/target真实owned进程已退出。该收集仅读取完整汇总与日志，不新增训练/预测或读取IQ、TX标签。

源域准确率（%，1800物理样本；不作为目标准确率）：

|场景|原R17|分步状态共享先验|纯蒸馏半秩先验|
|---|---:|---:|---:|
|clean|98.444444|98.444444|98.444444|
|practical_high|97.833333|98.000000|97.944444|
|practical_mid|96.777778|96.833333|96.666667|
|practical_low_suburban|95.944444|95.666667|95.777778|
|practical_high_urban|97.000000|97.166667|97.055556|
|practical_mid_urban|91.444444|91.166667|91.222222|
|practical_low_urban|86.833333|86.666667|86.222222|

源资格：{"step_norm_fidelity": false, "factor_half_fidelity": false}。固定源选择结果：`None`。各场景准确率/波形条件与完整分层、混淆矩阵及净修复算术见[完整汇总](results.json)。

本轮结论：**两候选均未通过固定源域资格，0新目标预测；未交付能保持目标性能的结构压缩版本**。完整训练和源域负结果保留，原R17继续作为部署对照。没有因源域失败改变阈值、补测或选择性重跑，不能把源域结果写成目标退化/保持结论。


实际成本（完整恢复器含冻结coarse/action与CG，FP32；模型文件不含优化器）：

|候选|完整参数|参数减少|文件字节|优化阶段秒|原R17 batch1中位ms|压缩版中位ms|延迟比|PyTorch峰值allocated比|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|step_norm_fidelity|3118001|33.1629%|12765852|1940.384|45.112|44.800|0.9931|0.7829|
|factor_half_fidelity|3775105|19.0773%|15444228|1938.903|46.150|95.819|2.0763|39.9896|

优化时间包含教师前向和20轮cache/优化，不含初始化SVD或排队。延迟对照为各行同一GPU/同16个sourceV IQ、batch1、2次warmup、顺序测量，含完整恢复图、不含F识别器/磁盘写入；热缓存、顺序及其他owner干扰可能影响结果，不声称独占GPU或统计保证。共享参数不等于减少三步前向。

全日志损失摘要（固定finalE20，训练最小值不作选模）：

|候选|损失项|首轮|末轮|20轮均值|每轮线性斜率|
|---|---|---:|---:|---:|---:|
|step_norm_fidelity|loss|0.0383331522|0.00367723329|0.00843764205|-0.00101276978|
|step_norm_fidelity|iq_nmse|0.0277841417|0.0200124492|0.0212930631|-0.000240741343|
|step_norm_fidelity|teacher_stage_nmse|0.0038333152|0.000367723325|0.000843764203|-0.000101276978|
|factor_half_fidelity|loss|0.0327135622|0.00407370609|0.00742718219|-0.00071677541|
|factor_half_fidelity|iq_nmse|0.0271405394|0.020006118|0.0210227713|-0.000187310424|
|factor_half_fidelity|teacher_stage_nmse|0.00327135622|0.000407370609|0.000742718217|-7.16775409e-05|

独立重算315组混淆矩阵、accuracy/Macro-F1、repair−damage/net与分组样本数；TX/RX/day分区分别回合至各view总矩阵。全部结果是单一训练seed及固定测试集点估计，七视图相关，不增加独立物理样本量；不称总体/多seed零退化、在轨效果或数学全局最简。实际方法commit`1515d73c5de3e962a822bbaca34182e28a6ae584`保持；新增分析代码只读取完整聚合，不修改健康run。
