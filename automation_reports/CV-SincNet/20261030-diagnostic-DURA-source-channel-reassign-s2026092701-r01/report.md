# DURA 保留 clean 参考的同域与跨 RX 信道重配诊断

- run_id：`20261030-diagnostic-DURA-source-channel-reassign-s2026092701-r01`
- group_id：`phase1-dura-source-channel-decoupling`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定源512×2；保持xi不变，比较同RX/day及跨RX的完整(Aj,noisej)重配，匹配scene/state/lock；CPU2/零更新/零身份标签或F指导。

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


## 固定源反事实

原生源生成器按RX/day构造模拟receiver session。这里保持每条clean参考xi逐值不变，将完整缓存算子Aj及其噪声eta_j=yj−Ajxj一起转给xi，得到Ajxi+eta_j。同RX/day与跨RX两类donor均匹配scene/state/lock；前者还排除自身。只有源metadata有双侧donor的固定位置参与，排除数公开，不能按模型误差筛样本。

两缓存视图均须至少480/512可构造、跨RX相对同RX的IQ和算子估计误差都增加至少10%，并保持参考和噪声配对，才另登记从零E20/p.5/单G前向训练。否则不训练该候选。信号和噪声功率同时报告；它是冻结源算子的反事实，不是新native AGC或接收机因果证明。F仅用于首个真实模型smoke，诊断不读F指标或身份标签。
