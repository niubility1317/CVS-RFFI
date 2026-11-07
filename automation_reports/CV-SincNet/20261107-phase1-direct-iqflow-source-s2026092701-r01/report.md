# 独立 IQ 流 GPU 训练

- run_id：`20261107-phase1-direct-iqflow-source-s2026092701-r01`
- group_id：`phase1-direct-iqflow`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；N607 GPU 2 控制器 PID 3992852、训练 PID 3993062 已核实。

## 目的与对照

以单条 received IQ 为锚点，用源 clean IQ 训练路径速度，四步确定性积分；部署不需要 DURA。训练流程复用含 DURA 上下文的源缓存，并读取冻结 DURA 进行源检查，因此不称整个训练流程从未读取 DURA；流损失路径本身以 received 到 clean 为目标。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

N607 真实源 checkpoint smoke 使用 64 个源 ID × 三视图，通过；固定 20 轮 GPU 训练进行中，尚无完整轮次。训练不读取 F、TX 标签、V 或目标，七景预测和独立 truth-last 尚未开始，暂无识别结果。

## 交接

固定 20 轮完成后自动执行源预测 smoke、七景预测及独立评分；恢复时核实唯一 owner，不重复启动。
