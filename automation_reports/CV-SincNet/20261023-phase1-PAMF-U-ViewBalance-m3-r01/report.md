# PAMF 双锚保护下的无标签视图梯度平衡

状态：VERIFIED_RUNNING。121项检查、一次独立P0/P1审查和真实原E30 source-only smoke均PASS；三seed按固定配方续训，目标ACTIVE。CMR完成对应GPU预测后已让出卡位，UDAP健康任务保持；本假设在读取CMR／UDAP任何目标成绩前固定。

源依据见九模型reference的source_view_gradient_diagnostic.md／JSON。原F20 L-only E20和RXPSM E20共24个受控源L128／U256批次，物理U梯度范数约为普通U的5倍（中层）、8至9.5倍（默认头）；16次整体对raw真实CE作用方向相反，其中9次合并后非负但仍含负分支。这是实际分支贡献的描述，不隔离变换因果或预测目标涨幅。两个起点均有此现象，故不更换预先固定的原RXPSM E30。

从原同seed RXPSM E30／6660学生、自身EMA／高斯状态／L-only原型独立追加20轮／4440步，累计E50／11100、13L＋37LU。原T1六次L前向、全部有标签目标和默认头联合训练，以及原U半普通噪声／半Practical、软CE／原型项和置信规则保持。只把已有逐记录U loss按原实际mask分为ordinary和physical两项，两项仍除以完整U批次N、含原U系数，scalar之和必须等于原U loss；没有额外网络前向、头、教师或推理参数。

每个t2／默认头／其余可训练参数组，取原full-L梯度gL、已有两个raw L前向的平均真实CE梯度gR，以及两个原U贡献梯度gO／gP。分别把gO和gP投影到<q,gL>≥0、<q,gR>≥0的交集，选择最近欧氏向量；枚举无／单／双边界及零，双边界归一化Gram伪逆rtol1e-12。若任一anchor范数平方≤1e-12，两个U均零。投影方向范数平方≤1e-12时当前视图置零，不重分配预算；否则将其范数固定为0.5×||gL||，因此既可缩小也可放大当前视图。两视图相加总U范数由三角不等式≤||gL||，实际梯度gL＋vO＋vP。转回原dtype后双内积超出相对1e-6容差则该视图保守置零。原L＋U scalar仍用于日志，显式替代参数梯度进入原clip／optimizer／单次EMA。

固定0.5分配对应现有两类视图，无系数搜索。原全局U scalar系数在非零范数归一化后不再决定实际更新幅度，样本间置信权重仍决定方向，必须明确披露；不能把scalar unchanged称作优化权重不变。可能放大小噪声方向；有界范数／欧氏非负内积也不能保证Adam动量／预条件后的实际下降或目标收益。投影思想沿双锚方案；梯度量级平衡的动机参考[GradNorm](https://arxiv.org/abs/1711.02257)，本版没有学习任务权重或训练率控制器，不称论文复现。

每seed teacher/student U2097900、physical1048950、L-only原型1051392、EMA11100，新增raw锚4440、两视图引导4440次，无新增前向；记录逐视图梯度范数、归一化因子、双内积、有效集合／零fallback以及实际时间，不称同FLOP。最终E50学生固定，sourceV仅诊断不选模／早停。

GPU5／6／7固定seed2026092701／02／03。等待同seed CMR的raw＋DURA全部14项GPU预测结束、UDAP已派发后，占空出的槽位；每卡最多两个本任务worker，含外部最多三个计算进程，派发free≥10000MiB。单CPU scorer等UDAP终态。不可覆盖root为PROJECT/runs/20261023-phase1-PAMF-U-ViewBalance-m3-r01；相同技术失败两次停止新派发并保留健康任务和全部产物，smokePASS立即续行。

每行冻结后自动raw PAMF和固定DURA v1.0同一F各clean＋六Practical，每景168000，全14项决策固定后独立truth-last。复用同readout F20监督／F21普通PL三seed，报告逐seed／景／RX／TX／day、SD、配对纠错损伤、曝光与增量／累计成本。原raw clean至少+1pp、六景严格超过+1.5pp须同时超过两个基线，DURA单列参考；无目标反馈改规则／父来源／排序／选择性重跑，不称新独立盲测。


121项检查覆盖解析投影与双视图抵消反例、正交／非正交／共线边界、普通分支增幅／物理分支缩幅、零视图预算不转移、总U范数与双内积、近零／非有限梯度、native原CE消除与上游缩放、原U scalar拆分／完整N分母／不等分mask、无新增学生前向、隐藏U拒绝、教师／曝光／默认网络状态往返，以及三seedGPU和单CPU评分接续／raw＋固定DURA完整报告和已有SSL回归。一次独立P0/P1审查PASS，未读目标或启动实验。首个真实原E30 source-only smoke将检查两类U128／L768、原raw真实CE、两个视图预算／双内积、全部实际参数梯度与显式组合一致、默认头／教师无梯度及单次EMA；probe丢弃，正式三个seed重新加载原E30。


## 已核实三卡实际训练

实际执行commit `81c4e8ba596412d850e478e45bef23c6d5fcbec8`；一次发布包SHA本地／远端比较和远端编译PASS。独立核实CPU控制器与三个worker的PID／CWD／完整命令、实际GPU及step日志增长。真实原E30 probe raw CE0.171661，两类U各128、L768，普通／物理U各组均实际完成双锚投影和半L梯度预算分配；全部参数真实反向梯度与显式组合一致，默认头／教师无梯度／单次EMA正确。probe丢弃，正式三个seed重新加载原同seed E30。

逐条检查全部已刷盘完整step前缀，原U／physical曝光、两个view seen、原U标量精确拆分、各组每视图双内积／半范数／总U上界、激活集合和放大／丢弃累计计数正确；gzip尚未闭合，不宣称完整训练或性能收益。详见source_activation.json。GPU5／6／7每卡UVB与UDAP各一条本任务worker，含其他owner总GPU进程最多三个，未停止或热改其他健康任务。梯度归一化会放大或缩小原U方向，原全局U系数不再直接控制归一化后的范数，噪声可能被放大。固定最终E50学生继续自动raw／同一DURA各七景，完整预测后独立truth-last，CPU评分在UDAP之后。


## 三seed源训练完整，正式评测继续

VERIFIED_SOURCE_FINAL：三行各新增20轮/4440步，累计E50/11100；逐条读取全部13320条新增闭合step记录，源U曝光2097900、物理U曝光1048950、学生U视图2097900和自身EMA11100次均符合预定，父前6660步复用原完整核验记录。无U真值/目标训练或另训头。完整源证据及已发生父阶段/增量/累计时间见source_training_complete.json；程序经过时间来自共享GPU，不能等同独占GPU小时。

模型已按固定最终学生冻结，并已自动进入raw与固定DURA各七景预测；这是源训练完成，正式预测/独立评分尚未全部齐，不宣称整体实验完成或目标达成。部分目标成绩仍不读取，三seed全矩阵完成后统一分析。


## 完整目标测试结果

ANALYZED：各方法三个seed，共3模型、42项raw／固定DURA七景决策均完成独立truth-last评分，六个匹配监督／普通PL对照模型完整复用。逐景／seed／SD／Macro-F1／RX／TX／day、修复损伤及曝光／成本见target_results.md和附属完整表。原raw双门槛尚未满足，研发目标仍ACTIVE。

UVB / raw clean：84.611706349% ± 0.226492853%（三个模型 seed 样本 SD）。
UVB / DURA-LU clean：84.636904762% ± 0.188664186%（三个模型 seed 样本 SD）。
UVB / raw six_mean：71.675429894% ± 0.887589290%（三个模型 seed 样本 SD）。
UVB / DURA-LU six_mean：77.545337302% ± 0.383196721%（三个模型 seed 样本 SD）。

不依据本结果热改健康旧任务或替换父来源。
