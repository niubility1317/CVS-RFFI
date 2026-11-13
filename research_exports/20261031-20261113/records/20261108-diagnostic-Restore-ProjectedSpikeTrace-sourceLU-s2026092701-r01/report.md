# 投影细节尖峰的源输入与网络层幅度追踪

- run_id：`20261108-diagnostic-Restore-ProjectedSpikeTrace-sourceLU-s2026092701-r01`
- group_id：`source-projected-detail-spike-layer-trace`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用同一 E10 到 E11 的临时源复现，在首个修正 NMSE 超过100的微批中记录输入、后验基线与网络各层汇总。

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

## 源训练尖峰层级追踪完成

VERIFIED / ANALYZED。原E10/F0源检查后临时重放第11轮，25次更新后微批21/32条投影NMSE>100。最大记录接收/后验基线范数约16/15.69，提案177.05、投影221.14，实际方向放大1.249倍。层级幅度仅单条记录，原异常精确轨迹未知；零TX/V/OE/target与部署模型输出。 完整分析
