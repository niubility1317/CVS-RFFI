# Phase2 ProtoNet-CDA→MoPC 真实非有限支持集诊断（GPU0）

- run_id：`20261109-phase2-protonet-mopc-support-gpu0-s2026092702-r01`
- group_id：`phase2-group2-protonet-mopc-nonfinite`；类别：`diagnostic`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

单次检查 B r10 任务277 原源 checkpoint 和 K10/N10 支持集在原 GPU0 的首个非有限边界。

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

## 真实非有限故障诊断

仅在原 GPU0 上使用 r10 任务 277 的原源 checkpoint 和 K10/N10 支持集，按原 B/C 配方单次记录首个非有限边界，不读取 query 或真值，不生成预测。该正式任务曾出现非有限 loss 与实际非有限梯度；30 份部分预测及所有已有结果保留。诊断不改变原数值 guard 或训练预算；独立 P0/P1 审核通过。

## 诊断结果
