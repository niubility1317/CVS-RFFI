# DURA 源 RX 日期频谱重加权压力诊断

- run_id：`20261030-diagnostic-DURA-source-spectral-stress-s2026092701-r01`
- group_id：`phase1-dura-source-spectral-transfer`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源63000 L/U按15个RX/day统计PSD，构造正实增益并精确A配对，固定源512×2检查noVar恢复鲁棒性；CPU2/无标签或F指导。

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


## 固定假设与判定

源63000条L/U按现有15个RX/日期组合计算平均功率谱，固定9频点平滑，再以两个源组合的功率谱平方根比构造正实增益。每包改选另一源RX组合，保留频谱相位与原RMS，使用 `y+A(Sx-x)` 保留原模拟A和噪声。该统计混合了源采集条件与样本分布，不能称为已识别的因果接收机传函或TX身份保真证明。

固定noVar E20源模型；不使用F指导。F仅用于首个真实2Ls无query smoke，随后释放。原始分支IQ误差精确复用上一诊断相同512位置/两缓存视图的产物，不打开其F指标字段。新的两视图均须MSE增加至少10%、功率相对误差不超过2e-6、噪声回放误差不超过原功率1e-10，才另登记E20/p.5/单次G前向训练。否则本候选不训练；不放宽上一候选的门槛，也不放大其手工扰动。


## 已完成源诊断

VERIFIED / ANALYZED_SOURCE_ONLY。CPU2/0GPU耗时3.597秒，其中63000源IQ/15session的PSD校准0.252秒。唯一P0/P1审查与3项聚焦测试PASS，一次preflight/归档SHA/远端compile通过；执行commit a55b378eb47a6796ab375d2884f2e71b541a74a6。PID562975存活时独立核实release CWD、完整argv、空CUDA绑定，随后自然退出、日志3239bytes/32批与完整产物已读回。真实source2Ls G/F smoke PASS；F仅用于该smoke，未读取F优化/特征指标，G状态不变，0更新/直接TX/V/target。

独立复算57600个增益值与780个role/scene/state/lock/RX/day统计，RNG跨RX抽样和原始IQ baseline逐值复用一致。源统计增益范围0.343396至2.912086，单滤波最大条件数5.157310；PSD floor实际触发比例为0，没有手工扩大上一扰动幅度。

|源缓存视图|原始IQ误差|频谱重加权IQ误差|相对增加|功率相对误差最大值|噪声回放误差最大值|训练资格|
|---|---:|---:|---:|---:|---:|---|
|1|0.02235236|0.02957474|32.3115%|2.384e-07|3.779e-14|True|
|20|0.02112201|0.02647739|25.3545%|2.384e-07|3.971e-14|True|

两视图均通过预定10%和功率/噪声配对条件。因此下一步另登记从零E20/4940、每次0.5概率重加权、一次G前向的单阶段对照，固定noVar损失与原F50全七景。该资格说明源模型对这类变化敏感，不证明TX保真或目标收益。RMS保持的是输入信号功率；固定A下输出能量与有效SNR仍可能变化，不能把本诊断解释成已隔离接收机因果作用或一次新的native AGC仿真。默认及3pp目标保持，尚无新训练权重。
