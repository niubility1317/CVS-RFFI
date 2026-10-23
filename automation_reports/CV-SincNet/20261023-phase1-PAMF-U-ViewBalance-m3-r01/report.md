# PAMF 双锚保护下的无标签视图梯度平衡

状态：PLANNED。仅固定源依据、配方及完整矩阵；未实现、审查、发布、smoke、训练或访问本版目标。原CMR／UDAP继续运行。本假设在读取它们任何目标成绩前固定。

源依据见九模型reference的source_view_gradient_diagnostic.md／JSON。原F20 L-only E20和RXPSM E20共24个受控源L128／U256批次，物理U梯度范数约为普通U的5倍（中层）、8至9.5倍（默认头）；16次整体对raw真实CE作用方向相反，其中9次合并后非负但仍含负分支。这是实际分支贡献的描述，不隔离变换因果或预测目标涨幅。两个起点均有此现象，故不更换预先固定的原RXPSM E30。

从原同seed RXPSM E30／6660学生、自身EMA／高斯状态／L-only原型独立追加20轮／4440步，累计E50／11100、13L＋37LU。原T1六次L前向、全部有标签目标和默认头联合训练，以及原U半普通噪声／半Practical、软CE／原型项和置信规则保持。只把已有逐记录U loss按原实际mask分为ordinary和physical两项，两项仍除以完整U批次N、含原U系数，scalar之和必须等于原U loss；没有额外网络前向、头、教师或推理参数。

每个t2／默认头／其余可训练参数组，取原full-L梯度gL、已有两个raw L前向的平均真实CE梯度gR，以及两个原U贡献梯度gO／gP。分别把gO和gP投影到<q,gL>≥0、<q,gR>≥0的交集，选择最近欧氏向量；枚举无／单／双边界及零，双边界归一化Gram伪逆rtol1e-12。若任一anchor范数平方≤1e-12，两个U均零。投影方向范数平方≤1e-12时当前视图置零，不重分配预算；否则将其范数固定为0.5×||gL||，因此既可缩小也可放大当前视图。两视图相加总U范数由三角不等式≤||gL||，实际梯度gL＋vO＋vP。转回原dtype后双内积超出相对1e-6容差则该视图保守置零。原L＋U scalar仍用于日志，显式替代参数梯度进入原clip／optimizer／单次EMA。

固定0.5分配对应现有两类视图，无系数搜索。原全局U scalar系数在非零范数归一化后不再决定实际更新幅度，样本间置信权重仍决定方向，必须明确披露；不能把scalar unchanged称作优化权重不变。可能放大小噪声方向；有界范数／欧氏非负内积也不能保证Adam动量／预条件后的实际下降或目标收益。投影思想沿双锚方案；梯度量级平衡的动机参考[GradNorm](https://arxiv.org/abs/1711.02257)，本版没有学习任务权重或训练率控制器，不称论文复现。

每seed teacher/student U2097900、physical1048950、L-only原型1051392、EMA11100，新增raw锚4440、两视图引导4440次，无新增前向；记录逐视图梯度范数、归一化因子、双内积、有效集合／零fallback以及实际时间，不称同FLOP。最终E50学生固定，sourceV仅诊断不选模／早停。

GPU5／6／7固定seed2026092701／02／03。等待同seed CMR的raw＋DURA全部14项GPU预测结束、UDAP已派发后，占空出的槽位；每卡最多两个本任务worker，含外部最多三个计算进程，派发free≥10000MiB。单CPU scorer等UDAP终态。不可覆盖root为PROJECT/runs/20261023-phase1-PAMF-U-ViewBalance-m3-r01；相同技术失败两次停止新派发并保留健康任务和全部产物，smokePASS立即续行。

每行冻结后自动raw PAMF和固定DURA v1.0同一F各clean＋六Practical，每景168000，全14项决策固定后独立truth-last。复用同readout F20监督／F21普通PL三seed，报告逐seed／景／RX／TX／day、SD、配对纠错损伤、曝光与增量／累计成本。原raw clean至少+1pp、六景严格超过+1.5pp须同时超过两个基线，DURA单列参考；无目标反馈改规则／父来源／排序／选择性重跑，不称新独立盲测。
