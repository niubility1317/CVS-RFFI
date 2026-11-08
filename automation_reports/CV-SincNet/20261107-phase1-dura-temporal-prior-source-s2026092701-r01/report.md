# 多尺度时间先验 DURA GPU 训练

- run_id：`20261107-phase1-dura-temporal-prior-source-s2026092701-r01`
- group_id：`phase1-dura-temporal-prior`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；源训练正在运行。

## 目的与对照

冻结 DURA 和源遮挡 IQ 先验，在源训练时对恢复与 clean 的先验隐藏特征进行 8/32 时间尺度中心化一致性约束。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

2026-11-07 23:28 香港时间核实源训练 PID 3672864 正在运行，日志仅记录完成第 1 轮，耗时 32010.5 秒、损失 0.02312698；该条日志距核实时约 10 小时。进程存在不能证明第 2 轮内部批次正在推进，现有日志不足以判断批次进度或推断故障。七景预测和独立评分尚未开始，暂无识别性能结果；保持原任务并继续观察。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

2026-11-08 核实预登记的 5 轮源训练完成，总耗时 136789.6 秒；源摘要记录 32768 个训练 ID，TX 标签、识别器权重和输出、验证集及目标数据读取次数均为 0，DURA 保持冻结。损失从 0.023127 降至 0.022265，只代表源域训练目标，不能推断识别性能。产物：N607 runs/20261107-phase1-dura-temporal-prior-source-s2026092701-r01/source_summary.json、final.pt。
