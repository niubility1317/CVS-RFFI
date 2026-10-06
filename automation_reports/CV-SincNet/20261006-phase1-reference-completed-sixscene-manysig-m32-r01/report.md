# reference_response已完成32行立即测试：clean及六完整residual

- run_id：`20261006-phase1-reference-completed-sixscene-manysig-m32-r01`
- group_id：`cvs-reference-completed-snapshot-sixscene`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

按用户要求独立测试快照时已完成的全部32行固定E200模型。16行R2及16行R3；仅由完成时间纳入，与源V或目标性能无关。后续源训练与自动选择规则保持不变，目标结果不反馈。

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

## 用户要求的即时测试与边界

原136行等待条件是本轮执行计划，不是必须等所有未来实验训练完才可测试固定模型的科学要求。本次用户明确要求立即测试已完成模型，故建立独立32行快照。2026-10-06 18:02左右完成的全部32行均纳入，取各自固定E200、44400步模型；16行R2和16行R3，未按任何源V/目标指标筛选。R3尚未完成的seed不猜补、不以部分结果冻结下一阶段。源控制器继续按既定完整四seed源域规则自动选模；本测试目录和评分不是它的输入。

独立进程只读旧checkpoint和已验证received视图，输出全新目录。最多4个测试worker，使用GPU空余名额，不停止或修改健康训练。clean及六完整residual各168000个相同物理query；全部32×7预测固定后，独立truth-last评分和第二实现复算，保留逐seed、RX、TX、混淆矩阵、实际seed数与资源指标。
