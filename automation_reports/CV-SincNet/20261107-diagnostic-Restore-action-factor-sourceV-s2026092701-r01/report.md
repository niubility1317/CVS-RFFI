# 源端 Action 归因：FIR 形状与接收机调制

- run_id：`20261107-diagnostic-Restore-action-factor-sourceV-s2026092701-r01`
- group_id：`gauge-aligned-source-Action-attribution`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定源 V480 七景，冻 BF/F0，对齐 FIR 尺度与相位后分别替换 FIR、主/镜像调制及二者。原标量噪声保持到单独对照。

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

## 源端结果已核实

低档城区固定 480 条源记录。原盲估正确 401 条，替换 FIR 形状后 400 条，替换主路／镜像调制后 424 条，两者都已知时 430 条。最后额外换成真实噪声标量为 426 条，raw clean 为 474 条。只换 FIR 的修复／损伤为 7／8，只换调制为 31／8。

[完整分析](analysis.md)
