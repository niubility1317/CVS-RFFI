# DURA 条件 IQ 流 GPU 训练

- run_id：`20261107-phase1-dura-iqflow-source-s2026092701-r01`
- group_id：`phase1-dura-conditional-iqflow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 GPU 3，真实源 checkpoint smoke 已通过，正式源训练进行中。

## 目的与对照

固定 DURA 上下文，源 IQ 路径速度匹配训练，四步确定性积分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

源 smoke 使用 64 个合法源 ID × 三视图，状态为 SOURCE_TRAINING_SMOKE_PASS；不读 F、TX 标签、V 或目标。正式源训练 PID 3823069 已完成第 1/5 轮，训练累计计时 3836.3 秒、流损失 0.0076553；其余轮次仍在运行，尚无七景识别结果。训练损失不是识别准确率。

## 交接

固定五轮训练完成后，既有控制器自动执行源预测 smoke、七景预测和独立 truth-last；恢复先核实原 owner，不重复启动。
