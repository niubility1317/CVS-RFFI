# Phase2 B 主干：先完成其他未启动方法

- run_id：`20261108-phase2-bsource-no-radionet-m4-r09`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Reuse 9364 completed ancestor tasks; preserve 7 quarantined failures and all existing outputs. Run 1587 unstarted ProtoNet-CDA before 634 DADDA; cancel 1932 unfinished RadioNet.

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

## 本次范围与故障保留

独立逐项清点祖先链 9,364 个完整任务（含 r08 新完成 124 项）、7 个隔离任务（共 138 份部分预测），取消 1,932 个未开始 RadioNet；余 1,587 个 ProtoNet-CDA 先于 634 个 DADDA。保留既有模型、种子、预算与数据权限，不重发失败项。所有保留范围预测固定及缺项确定前不评分。每 GPU 本任务 B/C 合计至多 2 个进程，启动前空余不少于 6,000 MiB。对祖先继承、隔离、取消、排序、资源与评分的独立 P0/P1 审核通过；独立清点确认其先决条件。远端真实 source-only/no-query smoke 仍为启动第一步。
