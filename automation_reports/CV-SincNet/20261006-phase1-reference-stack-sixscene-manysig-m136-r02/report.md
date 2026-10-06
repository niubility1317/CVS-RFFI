# reference_response剩余136行：clean及完整六residual环境自动测试：故障恢复r02

- run_id：`20261006-phase1-reference-stack-sixscene-manysig-m136-r02`
- group_id：`cvs-reference-phase1-stack-sixscene-evaluation`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

恢复136行clean和六个完整residual视图的自动测试；等待新父运行完成，全部预测固定后独立truth-last评分及复算。

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

## 故障恢复方案及验证

旧控制器退出竞态已复现并修复：进程退出时消失的cmdline/cwd视为已退出，再核对完成产物；真实存活身份变化仍报错。新控制器拥有新worker，不再接管旧PID。24项相关测试通过，含队列完成自动补位、R2禁止重训、目标访问门控及七视图完整性。独立审查无P0/P1。

R2保持原目录、配置、权重和日志；发布前逐行核对源契约、E200/44400、scratch来源及无目标访问。R3至R6在新目录按原方案从零训练，16并发、每卡最多2行。保留此前验证的执行加速。所有136行源域冻结后执行预登记测试；clean及六residual各168000同物理query，全部预测固定后独立truth-last评分并复算。当前尚未启动，实际commit/PID/日志将发布后登记。
