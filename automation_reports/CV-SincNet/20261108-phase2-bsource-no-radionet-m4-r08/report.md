# Phase2 B 主干：取消 RadioNet 后继续其他方法

- run_id：`20261108-phase2-bsource-no-radionet-m4-r08`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
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

独立核实 PID 1644515、start_ticks 62274143、CWD 和命令，控制器正在检查祖先进程、完整预测与隔离。检查时尚无 manifest、source-only smoke、后继 claim 或新 worker，不能称其已开始新预测。前一 r07 在创建 root 与派发前因其他用户复用旧 PID 的 cwd 权限失败，已修复并完成 23 项定向测试及独立 P0/P1 针对性复核。源代码 b46f1336a3772a8ade7e0e6017d61e0de0a5a8bb，镜像 6f7e2a1f28bb71da974780e322f26cd49bb4c0d3；单归档 SHA 与远端 compile 已核实。RadioNet 所有未完成任务取消，保留此前完整和部分结果。

## 2026-11-08 非 RadioNet 任务已实际派发

控制器 PID 1644515、start_ticks 62274143 已核实。6 个非 RadioNet 真实源 checkpoint 的 source-only/no-query smoke PASS；逐项核实引用 r02 至 r06 的 9,240 个完整任务、174,195 份预测，六处隔离的 138 份 partial 不变。1,932 个未启动 RadioNet 任务已取消；剩余 759 个 DADDA 后接 1,587 个 ProtoNet-CDA，任务顺序已逐项核对。新 DADDA 已派发，首次独立核实时新闭合 10 项，16 个真实 worker、GPU 0–7 各 2 个，failed=[]。C 的 RadioNet 已按用户要求停止，已有完整及部分结果保留；两组已知隔离未解决，全预测屏障及评分关闭。
