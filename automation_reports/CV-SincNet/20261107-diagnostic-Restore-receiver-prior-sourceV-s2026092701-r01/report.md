# 同接收机与留出接收机的配额匹配 IQ 先验对照

- run_id：`20261107-diagnostic-Restore-receiver-prior-sourceV-s2026092701-r01`
- group_id：`source-frozen-receiver-IQ-prior-attribution`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

只用原 L_s/U_s 各拟合 12600 条同接收机和其他接收机的均值协方差。固定源 V480 七景、已知和盲估信道。

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

## 取批次技术失败

原真实 L_s 和十套拟合先验检查通过。PyTorch 2.1 与 NumPy 2 的直接数组索引兼容问题在第一景分组取批次时中断进程。独立终态和有界复现均已核实。0 完整预测景，尚无真值评分。修复改用 Python 整数列表索引，以 r02 保持同一源矩阵。
