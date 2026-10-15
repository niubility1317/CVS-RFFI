# 固定SIG_TIME V-REx：原样02/03复核

状态PLANNED，代码8ce0a02e4b0c56d1aaeda0171b7c08813d4a7811。这不是新假设：源PASS/扩展决定4056e9fb618fa92af5486c6d3670fc73bcc4194d已在首seed目标读分前上传。首seed83.2851%对匹配83.6405%的负差不改变本次矩阵。

6row=2历史oldgatecos参考＋2026092702/03各RX_BALANCED/RX_VREX两个scratch50。四新模型各33L+17U、222步/轮/11100更新，总200新epoch/44400更新；所有first01产物复用不重训。L6300/U56700隐藏TX/V27000健康诊断、split/data392005、原sourceRX/day/TX/physicalIDs保持，训练池标注10%。

网络固定SIG_TIME/joint160、195041特征参数、原6x160scale30默认cosine正常联合训练。对照均衡RX CE；候选只增加固定alpha1 presentRX CE均值population variance，缺RX排除、单RX零方差，全部L步启用。旧batch_neighbor门控/ownEMA.999/.16UCE/.01entropy/cos/AdamW.0002WD.0001/globalclip5/LS.01、原L128/U256及尾批保持；无新增信道增强、头拟合、网络分支、Sinc单位优化、RDrop或梯度机制叠加。NoExplicitCFO及内部逐包补偿保持。

普通N607既有CVS-RFFI/fullFP32；CWD PROJECT/releases/cvs_receiver_vrex_seeds_20261015_r01；命令python -u -m experiments.cvs_receiver_vrex_seeds.dispatch。不可覆盖root PROJECT/runs/20261015-phase1-receiver-vrex-oldgate-cos-e50-clean-manysig-s0203-m6-r01；本地publish输出LOCAL_ARTIFACTS/receiver_vrex_seeds_delivery_r01。首步真实firstRXBALANCED checkpoint无query smoke，仅2个合成IQ、0更新，不是初始化。四模型从各自seed零训练/ownEMA；历史参考actualsource_proof核实契约后复用预测。

四训练行就绪并行，两个参考短任务；CPU2/loader0。共同容量锁中核对所有owner实际及startup CUDA进程，每卡最多3、free>=12000MiB、优先空卡，不固定分卡；独立score空CUDA。逐row固定50轮即自动clean预测→全部prediction固定→独立truth-last→报告/登记；不等待整批训练。只允许本run实际输入/协议/非有限更新/预算/无法合法预测评分/输出冲突/系统技术失败停止，保留产物、无自动重试、不干预其他健康任务。

预期每新model完整原生/分组损失日志、E1/17/33/50共208梯度层624向量、实际特征/初始化与W独立性、默认prediction/scoring/rowcompletion；整批6row1008000决策108指标，自动合并01为9row1512000决策162指标，输出source_features/source_observations/analysis/combined_three_seed_analysis/completion。三seed均值/样本SD、配对差/正差种子数、MacroF1/最差RXday、错误修正/引入、源全曲线/门控/成本均如实报告。n2不替代三seed；84%只按完整预定义RX_VREX默认头clean均值判定，source指标不冒充DG。

本地原核心16负例/真实nativehook/数值更新及本次11负例、四实际上下文PASS；原生参数只变seed/runID/candidateID/output，两个同seed臂初始state相同，跨seed不同，PROJECT解析/隐藏U不变/首seed重训拒绝均PASS。数据VALIDATED_ONCE复用，无新数据重验或额外审批。
