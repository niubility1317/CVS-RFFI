# 观测条件化先验置信度，GPU6 授权共享执行

- run_id：`20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r02`
- group_id：`source-observation-conditioned-covariance-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

与 r01 保持相同模型、数据和单阶段30轮预算；仅切换本任务尚未开始训练的控制器和资源设置。

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

## GPU6 实际训练启动已核实

VERIFIED / RUNNING。控制器 4063502 与实际训练进程 4063584 的 PID、工作目录、命令、CUDA UUID、GPU6 计算 PID 和源日志已核实。启动前已有 4 个进程、剩余 17938 MiB，读回总数 5。原 r01 未开始任何 GPU 阶段，其唯一控制器按用户授权精确停止，其余七个旧控制器在线。真实源两条检查、物理/置信度梯度隔离及 E0 严格重载通过。当前首轮仍在运行，尚无新 Main 成绩。
