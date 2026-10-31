# 源训练插值与自由运行路径的误差诊断

- run_id：`20261101-diagnostic-identity-flow-forcing-s2026092701-r01`
- group_id：`phase3-identity-flow-source-forcing`；类别：`diagnostic`；阶段：`Phase3-source-interpolation-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅在固定源训练 L_s/U_s/G10 上分析 E20 含 clean 的训练插值点与自由轨迹噪声敏感度，不读取 TX、V/OE 或目标 IQ。

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

## 源训练插值与自由轨迹：VERIFIED

固定 128 条 G10 L_s 训练记录上，E20 在含 clean 的训练插值路径上，速度 NMSE 从 t=0 的 0.9155 随 t 增大降至 t=0.25 的 0.0723、t=0.75 的 0.0136。R07 L_s 对应为 0.6741、0.0567、0.0113。中间状态包含 clean，部署时不可获得；后期误差小不能解释为真实推断性能。

同一 G10 源组四步自由运行，两条轨迹平均的 IQ NMSE 在噪声幅度 0、0.005、0.02、0.04 下分别为 0.9300、0.9299、0.9289、0.9262。因此在此固定噪声模板及这批样本上，仅缩小或去掉初始噪声并未修复平均恢复误差。两路径差异较大不能单独证明噪声是主要失败原因。

该观察支持继续检验已启动的完整轨迹反传，但不能证明模型仅靠 clean 泄漏取巧，也不能证明新方案有效。原 F0/E20 状态未变，未读取 TX 标签、V/OE 或目标 IQ；无优化更新。
