# Phase2 B 主干先运行其他未开始的方法

- run_id：`20261105-phase2-bsource-other-methods-first-m4-r05`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

沿 r02、r03、r04 复用 341 个完整任务，四处隔离共保留 129 份部分预测。13,179 个未启动任务先排 MRIOR、DADDA、Two-stage、Hu 和 POSTER，再排 ProtoNet-CDA，RadioNet 最后。记录后续注册数值异常的字段和步数，保留原训练、预测和停止规则。

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

## 恢复范围和启动规则

祖先 r02、r03、r04 共 341 项完整任务和 6,918 份预测。四处隔离的 129 份部分预测仍在原目录。其余 13,179 项保持原矩阵，前 9,660 项属于 MRIOR、DADDA、Two-stage、Hu 和 POSTER，随后是未开始的 ProtoNet-CDA 1,587 项和 RadioNet 1,932 项。新协调器只暂停已核实的 C r08 调度器派发，等待已启动 worker 自然闭合。B 队列结束或技术故障排空后恢复同一 C 调度器，独立 watcher 覆盖协调器意外退出。所有预测固定且隔离解决后才独立评分。

33 项定向测试通过，独立 P0/P1 通过。故障日志新增注册步骤和非有限字段，合成梯度验证可区分范数统计溢出和梯度元素非有限。现有四处 B 故障及两处 C 故障仍待定位。
