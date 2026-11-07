# 输入相关投影修正上限恢复器，从零单阶段 30 轮

- run_id：`20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r01`
- group_id：`source-input-bounded-projected-prior-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对原投影先验恢复网络的最终 IQ 修正设置当前 received IQ 范数两倍的上限，训练与部署一致。

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
# 资源并发停止

VERIFIED / STOPPED。该版 GPU 4 启动前检查到 2 个计算进程；另一任务几乎同时建立 GPU 上下文后，实际总数达到 4，超过本 run 固定的总数 3 个上限。为恢复限制，仅终止本 run 的控制器及训练子进程；独立读回确认 GPU 4 剩余 3 个其他计算进程，未向其他任务发送信号。

本 run 的真实源样本检查通过，但停止时尚未完成第 1 轮，也没有开始预测或评分。所有产物保留；相同网络和从零单阶段 30 轮计划将使用独立 r02 重新排队，并为其他任务保留一个额外并发名额。具体进程和现场证据保存在本 run 远端 `resource_stop.json`，不把这次停止的模型作为候选。
