# 冻结恢复器源端目标梯度几何诊断

- run_id：`20261101-diagnostic-Restore-source-gradient-geometry-s2026092701-r01`
- group_id：`restoration-source-objective-gradient-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原Ls128Us384固定512×20draw/B256，对两个冻结G的物理/输出F/latent三目标求导；同theta权重.1/1的Gram与全局clip1，0更新/TX/V/OE/target。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 固定源端梯度诊断

两个已冻结G为FeatureCond/c865与FeatureStrong/3e541，strict final30/7410；原Tri-SIG-TIME SSL/3ff8仅提供连续特征，不调用默认头。原R07训练源Ls128/Us384共512包×20存量draw，按固定B256分40批/模型，共80批。分别对IQ+.05STFT、.1输出F0特征指导和单位权重latent项使用autograd.grad，得到encoder、latent head、FiLM、decoder、all五组Gram；对同一theta报告latent权重.1/1的总梯度和全局clip1。组的实际全局因子来自all，组单独clip的数值只作明确标注的反事实。

不执行backward、optimizer或参数更新，不积累.grad。G临时允许导数计算后恢复requires_grad=False，F0始终冻结；最终逐state核保持。source Ls2实际checkpoint三目标导数smoke是第一神经工作，PASS立即完整80批。20480条G前向/20480条F0输出特征+clean512=20992条F0特征；smoke另G4/F0特征6。0TX/V/OE/novel/target读取、统计拟合、head调用、新Main推理或新恢复器版本。

该诊断基于final30/eval和固定源面板，混合七源景，不是实际训练批次/训练clip频率或AdamW更新预测；未计算动量、二阶矩及decoupled decay。main与aux的夹角只提供局部机制信号，不能据此分解Main性能百分点或建立理论界。零范数组的cos明确为null并报告有效批数。两G、全部80批/五组/两权重保留，未扫描checkpoint或根据质量取消。

设计动机来自完整源曲线的latent/IQ/输出特征取舍，代码在读取新Main数值之前已固定；不消费Main结果优化。3项聚焦测试和一次P0/P1通过。原N607 CVS-RFFI/Torch2.1+cu121/FP32-TF32off，GPU4/6 UUID准入free>=12GB、原compute<=2、仅新增1条本任务NN/单卡总<=3，其他owner不改。所有父Main技术完成才运行，无质量门槛；当前性能目标ACTIVE、效率次级、默认DURA保持。
