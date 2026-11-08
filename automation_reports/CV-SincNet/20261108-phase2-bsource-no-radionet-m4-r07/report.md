# Phase2 B 主干：取消 RadioNet 后继续其他方法

- run_id：`20261108-phase2-bsource-no-radionet-m4-r07`
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

## 2026-11-08 准备阶段失败，未派发

在祖先进程只读核验期间，旧 PID 已被其他用户进程复用，读取其 cwd 被系统拒绝。控制器已退出；失败发生在新 run 根、后继 claim、source smoke 和预测派发之前，无新 worker，所有祖先完整及部分预测保持。已修改身份检查：先检查命令行是否绑定本实验 runtime，无关进程不读取其 cwd；匹配但无法核实 cwd 时仍拒绝。新不可变后继经针对性测试与复核后继续。
