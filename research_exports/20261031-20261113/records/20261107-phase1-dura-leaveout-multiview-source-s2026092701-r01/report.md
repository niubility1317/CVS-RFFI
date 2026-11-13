# DURA 留一视图教师 GPU 训练

- run_id：`20261107-phase1-dura-leaveout-multiview-source-s2026092701-r01`
- group_id：`phase1-dura-leaveout-multiview`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：experiment.json；状态记录：events.jsonl

## 目的与对照

冻结 DURA，多尺度修正训练时使用同一源记录另外两视图的 DURA 输出均值作为教师。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

本机真实 DURA GPU 技术检查通过；远端源训练、七景预测和独立 truth-last 均未完成，尚无识别结果。

## 交接

队列满足资源条件后自动执行源 smoke、固定五轮训练和七景预测，再独立评分；恢复时核实唯一 owner，不重复启动。
