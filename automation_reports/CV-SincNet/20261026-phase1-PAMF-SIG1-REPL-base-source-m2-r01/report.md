# 完整PAMF单SIG_TIME视图：共享监督/普通PL源控制

- run_id：`20261026-phase1-PAMF-SIG1-REPL-base-source-m2-r01`
- group_id：`pamf-repl-v2-20261026`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用户新指定完整PAMF保留SIG_TIME单视图。单seed01普通PL复用已核实单视图scratchE20+10+20；监督scratch20+10+20。各final自动完整源V八视图；两个优化方向只共享这一公共控制owner。

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


## 发布准备

单视图4项聚焦检查和唯一P0/P1审查PASS；三seed E20实际张量／来源核实。只有seed01本次两行源控制。真实checkpoint联合smoke、启动和源结果尚待执行，不把本地验证写成运行成功。参见[预登记](preregistration.md)、[本地检查](local_verification.json)、[前缀张量证据](prefix_tensor_evidence.json)、[共享基座](../../../docs/goals/PAMF_SIG1_SHARED_BASELINE_20261026.md)。
