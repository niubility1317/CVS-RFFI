# PAMF八种子原源域验证集发射机召回率诊断

- run_id：`20261019-phase1-pamf-source-class-manysig-m8-r01`
- group_id：`phase1-pamf-frozen-source-class`；类别：`diagnostic`；阶段：`Phase1 source diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen historical residual_fusion E200 checkpoints, original default head and original source V; CPU-only no adaptation/training/target access.

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

## Frozen evaluation plan

All eight historical residual_fusion E200 default-head checkpoints are fixed before this diagnostic. Evaluate original source V only (27000 packets, 4500 per TX), sequential CPU with four threads, no optimizer, adaptation, augmentation, target IQ or target truth. Existing source roles and parent scratch provenance are reused and matched exactly. Stop on first input/smoke/coverage/nonfinite/model-state/reproduction mismatch; preserve partial outputs. No performance-based selection or rerun.

Local validation: compile PASS; six focused protocol-negative checks PASS. One independent P0/P1 review PASS (source_class_review): historical checkpoint fields, source-only roles, loader metadata and non-overwrite output handling checked. Launcher first loads each real E200 checkpoint with strict original head and synthetic no-query smoke, then continues directly to original V.
