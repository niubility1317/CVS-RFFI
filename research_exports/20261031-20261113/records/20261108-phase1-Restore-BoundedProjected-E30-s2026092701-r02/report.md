# 输入相关投影修正上限恢复器：保留并发余量版本

- run_id：`20261108-phase1-Restore-BoundedProjected-E30-s2026092701-r02`
- group_id：`source-input-bounded-projected-prior-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

与被资源停止的 r01 相同的从零单阶段30轮有界投影网络；仅调整队列准入和本任务进程数保护。

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

## 有界版已开始训练

VERIFIED / RUNNING。GPU 2 准入时已有 1 个计算进程、空闲显存 22498 MiB；独立读回本任务实际训练进程、工作目录、资源与日志，复核时该卡共 3 个计算进程。真实原冻结 F0、两条合法源样本、完整后验、梯度隔离与 E0 重载检查 PASS，无 TX 标签、V、OE 或目标输入。已完成第 1 轮，原单阶段 30 轮、最终 E30 和双总体七景 truth-last 保持；此时没有正式性能结论。
