# 单视图REPL三新候选：去风格、去外推与L-only RX方向外推

- run_id：`20261026-phase1-PAMF-SIG1-REPL-RXOffset-source-m3-r01`
- group_id：`pamf-repl-v2-20261026`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

完整PAMF单SIG_TIME，三新学生同seed01/同已完成ordinaryPL E20+10+20到E50。RXOffset只外推当前真实L支持且投影掉L类别均值的RX方向，L统计跨阶段续接；全部已完成对照仅只读评估，不训练。

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


## 实现与发布准备

11项机制/协议聚焦检查PASS，唯一P0/P1审查及源门槛精确计数修正的定点复核PASS。真实checkpoint联合smoke及训练尚待执行。只训练三个新候选，两个完成参照只读评估，0新对照训练；单学生seed，不是三seed目标结果。[预登记](preregistration.md)、[本地检查](local_verification.json)。


## 实际启动与阶段记录

执行commit `b34f81891bfe379b87f1b93b78ec98dac6139cfc`，真实三个单视图E20/L128/U256联合smoke PASS，U梯度实际到单SIG_TIME/原head/body，style实际0。controller及三个worker PID/CWD/argv/GPU与跨读取记录增长独立核实。三个学生均完成新增+10段，继续自己的+20段；最新完整可读记录合计10031，不是完整训练终点。L768/AsKL0/U六视图/style0/非有限跳步0和RXOffset当前L统计续接均逐可读记录核实。两个已完成参照仅sourceV推理，未训练。无新target结果、无三seed结果。用户最新补齐缺失单视图Sup/PL六模型另记独立run，不改此run配方或源选择门槛。[启动与可用记录证据](launch_evidence.json)。
