# 三个物理恢复模型的完整七景共享推理逐条复核

- run_id：`20261103-phase1-shared-physical-full-replay-s2026092701-r01`
- group_id：`phase1-shared-physical-full-prediction-replay`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：STOPPED（实际状态按 events.jsonl 及独立证据更新）

## 目的与对照

三个原预测全部完成后，只读重算共享推理并逐条比较冻结的raw和restored分类，核对每景门控计数；不读取目标真值，不覆盖或替代原产物。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

N607 实际 source smoke：112 条源记录的波形、门控、类别与原程序及原固定源预测逐项一致。后台 PID 2897883 已启动并等待三个原始七景预测全部完成；等待时未读取目标 IQ 或真值。

完整目标分类逐条对比尚未执行。原任务没有保存全部恢复波形或逐条门控，完整复核范围为 raw/restored 分类及各景门控总数。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 用户停止

2026-11-03 22:34（香港时间）按用户要求停止全量复核等待进程。独立读回确认 PID 已退出；源检查已完成，目标七景全量重放未开始，原输出和日志保留。
