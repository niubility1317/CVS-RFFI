# 独立多视图教师学生 GPU 训练

- run_id：`20261107-phase1-direct-multiview-distill-source-s2026092701-r01`
- group_id：`phase1-direct-multiview-distill`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 GPU 1 控制器 PID 3972911、训练 PID 3973073 已核实。

## 目的与对照

从头训练单输入独立时频学生，使用同一源记录三个模拟视图的 DURA 平均输出作为训练教师。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

真实源 checkpoint smoke 使用 64 个源 ID × 三视图，通过；正式 GPU 训练已完成第 8/20 轮，训练累计日志计时 567.4 秒，第 8 轮源损失 0.420796。训练未读取 F、TX 标签、V 或目标；DURA 三视图均值只在源训练时充当波形教师，推理只使用单条 received IQ。源损失不是识别准确率，尚无七景识别结果。

## 交接

固定 20 轮完成后自动执行源预测 smoke、七景预测及独立评分；恢复时核实唯一 owner，不重复启动。
