# 投影修正第11轮源域批次级技术复现

- run_id：`20261108-diagnostic-Restore-ProjectedE11Replay-sourceLU-s2026092701-r01`
- group_id：`source-projected-detail-batch-spike-reproduction`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从原 E10 技术状态按原训练轨迹临时重放 E11，遇投影修正 NMSE 大于 100 时定位源批次并测量提案和投影算子。

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

## 源训练批次复现完成

VERIFIED / ANALYZED。原E10/F0源样本检查后临时重放E11，25次更新后32条微批14条投影修正NMSE>100。最大样本原提案范数209.6703、投影后189.4231，实际方向未被投影放大；该样本算子谱范数7.543。算术与实际P×提案一致性通过；不把共享GPU复现冒充原异常的精确轨迹。零TX/V/OE/target与部署模型输出。 [完整分析](analysis.md)
