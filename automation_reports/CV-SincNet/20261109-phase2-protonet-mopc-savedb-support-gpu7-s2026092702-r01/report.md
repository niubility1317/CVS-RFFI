# Phase2 MoPC 保存 B 状态前向边界诊断（GPU7）

- run_id：`20261109-phase2-protonet-mopc-savedb-support-gpu7-s2026092702-r01`
- group_id：`phase2-group2-protonet-mopc-nonfinite`；类别：`diagnostic`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

单次从 r11 实际 B anchor 和 r13 c0192 支持集诊断前向及 BN 的非有限边界；不预测、不读取 query/truth。

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

## 诊断结果

原 GPU7 上使用 r11 的实际 B 状态及原支持集完成一次诊断，真实源检查通过，MoPC 执行 160 次更新，所检查的前向叶模块、缓冲区、优化后状态和梯度边界未观察到非有限值。r13 的原故障未复现，根因仍未知；单次正常运行不能证明修复。同步检查可能改变执行时序，且未保存故障时的完整中间状态。原 10 份部分预测与 B 状态保留；本诊断未读取 query 或真值，也未生成预测。证据：local_artifacts/20261109-phase2-protonet-mopc-savedb-support-gpu7-s2026092702-r01/completion_verified.json。
