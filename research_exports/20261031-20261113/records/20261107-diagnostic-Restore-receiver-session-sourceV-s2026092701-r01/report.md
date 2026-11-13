# 固定源传播条件的模拟接收机会话对照

- run_id：`20261107-diagnostic-Restore-receiver-session-sourceV-s2026092701-r01`
- group_id：`source-paired-virtual-receiver-attribution`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原接收机、训练接收机参数、原验证参数轮换。使用同一源包、传播与底层噪声抽样。

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

## 模拟接收机配对结果已核实

固定源 V480 的低档城区，原验证会话、训练会话、验证会话轮换的恢复正确数分别为 401、399、400。raw clean 为 474。

完整分析
