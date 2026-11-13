# Phase2 B 主干：取消 RadioNet 后继续其他方法

- run_id：`20261108-phase2-bsource-no-radionet-m4-r08`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Reuse 9240 closed ancestor tasks, retain 6 quarantines and 138 partial predictions; run 759 DADDA then 1587 ProtoNet-CDA unstarted tasks. Cancel 1932 RadioNet tasks.

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

## 2026-11-08 控制器已启动，正在准备


## 2026-11-08 非 RadioNet 任务已实际派发


## 2026-11-08 新故障及保留状态

新闭合 124 个任务后，任务 120（DADDA→MoPC，P03/MIX01/7-7，seed 2026092703）在 C 阶段 1 的第 19 轮第 19 次迭代报告 gradient_norm 非有限；异常梯度张量列表为空，根因未确定。该任务输出目录保留，当前无 npz 预测，原始日志保留。控制器已退出，worker 已排空，系统停止继续派发；旧完整预测及六处隔离、后继 claim 均保留。RadioNet 仍取消，预测完成屏障和评分关闭。后续只在诊断和明确恢复范围后从该 run 与祖先安全复用，不能盲重跑失败项。
