# 源端结构化 I/Q 噪声与条件后验误差

- run_id：`20261107-diagnostic-Restore-structured-IQ-noise-sourceV-s2026092701-r01`
- group_id：`frozen-source-structured-IQ-noise-posterior`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定源 V120 七景，标量与结构化噪声分别在同一盲估和同一已知模拟 Action 上配对。全路径固定原协方差先验。

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

## 结果已核实

低档城区的盲估信道配对中，结构噪声让 IQ NMSE 从 0.058703 变为 0.058704，F(clean) 判决一致率从 81.67% 变为 82.50%，1 次修复、0 次损伤。已知模拟信道配对的 IQ NMSE 从 0.048568 变为 0.048563，一致率从 90.00% 变为 90.00%。该场景下，结构化代理的条件后验方差占源先验方差 17.417%。两条标量路径之间，已知模拟参数相对盲估有 12 次修复和 2 次损伤，净增 10 条。

完整源端分析
