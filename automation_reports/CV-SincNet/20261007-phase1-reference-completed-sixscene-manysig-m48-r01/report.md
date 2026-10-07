# reference_response新增完成48行补测：clean及六完整residual

- run_id：`20261007-phase1-reference-completed-sixscene-manysig-m48-r01`
- group_id：`cvs-reference-completed-snapshot-sixscene`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

按完成时刻补测尚未测试的48行：R3剩余12、R4全部32、R5 rc4全部4。与旧32行互斥，固定各自E200，不按目标指标筛选。

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

## 补测范围

新增48行与已评分32行逐row互斥：R3剩余12、R4全部32、R5 rc4全部4。均使用各自固定E200/44400步的合规从零训练模型。clean及六完整residual，每视图168000相同物理query；全部48×7预测固定后独立truth-last评分。最多4个测试worker，按GPU空余名额自动补位，每GPU总进程最多2，不影响健康训练。R5 base尚未完成，暂不能将rc4结果解释为相对base的机制增益。

## 测试完成

2026-10-07T23:26:45.254153+08:00：新增48行全部ANALYZED，共56448000次预测、4704条分层评分。远端独立truth-last复算及本地混淆矩阵复算均VERIFIED。25项定向测试通过，限定P0/P1审查通过。与前批32行合并为80行、20组各4seed。详见[80行详细结果](evidence/combined80/detailed_results_zh.md)。实际release commit=fadf4be41eb60e956d18530bd291c69ca435facc。

发布时GitHub多次HTTP500，远端分支仍为68c8e8d54b2f50908b1e019b51a61e7948299b55；保留固定本地提交，以同一git archive传输N607并核实运行。传输与运行VERIFIED，Git交付状态另行读回登记。

GitHub恢复后已正常push，并独立读回远端OID与源代码提交fadf4be41eb60e956d18530bd291c69ca435facc一致，Git代码交付VERIFIED。
