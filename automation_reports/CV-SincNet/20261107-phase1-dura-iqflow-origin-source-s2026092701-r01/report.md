# DURA 起点覆盖流 GPU 训练

- run_id：`20261107-phase1-dura-iqflow-origin-source-s2026092701-r01`
- group_id：`phase1-dura-iqflow-origin`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 GPU 5 上控制器 PID 3955844、训练 PID 3956029 已核实。

## 目的与对照

固定 DURA 上下文，半数概率选择实际积分起点，其余均匀采样路径，源 IQ 速度匹配。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

N607 真实 DURA 源 smoke 使用 64 个源 ID × 三视图，通过；固定源训练进行中，尚无完整轮次。训练不读 F、TX 标签、V 或目标。正式七景预测和独立 truth-last 尚未开始，暂无识别结果。

## 交接

固定五轮训练完成后自动执行七景预测和独立评分；恢复时核实唯一 owner，不重复启动。
