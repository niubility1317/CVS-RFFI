# Seven-view shared-context pipeline full replay

- run_id：`20261107-phase1-dura-pipelined-seven-view-replay-s2026092701-r01`
- group_id：`phase1-dura-pipelined-seven-view-replay`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 CPU owner PID 59941，较低调度优先级，真实源样本比较通过。实际进度以远端 `progress.json` 与日志为准。

## 目的与对照

对三套七景预测已完成的冻结恢复模型，逐批比较独立路径与共享上下文流水线的七类中间张量，并核对旧结果中保存的 raw 和恢复类别。历史 logits 与波形未保存，因此只能比较两条新重放路径的张量，不能声称与历史张量逐值相同。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

真实冻结 DURA、Tri-SIG-TIME SSL 与三候选的源样本 smoke 通过；三份历史七景预测和物理 ID 已核对，正在无真值重放。逐视图每 16384 条记录进度；各块轮流交换独立和流水线执行顺序。独立路径 6 线程，流水线 8 线程；计时包含计算和进程通信，不包含 IQ 读取及结果比较。出现字节差异或旧类别变化立即保存差异并停止。尚无全量结论。

## 交接

commit `7d1fc3388ffe329ff928086d182e7d00581ccd77`；已核实 PID、CWD、nice 10 和日志，源 smoke 后直接进入重放。该任务不读取目标真值、不训练、不评分、不改变现有预测。恢复先核实 PID 和远端进度，不重复启动。

## 完成结果

七场景全量重放已完成，三模型的历史 raw 和恢复类别无差异，两条新路径的七类张量逐批逐字节一致；计时与线程口径详见 [analysis.md](analysis.md)。该结论不自动覆盖其他模型，也未切换在跑预测。
