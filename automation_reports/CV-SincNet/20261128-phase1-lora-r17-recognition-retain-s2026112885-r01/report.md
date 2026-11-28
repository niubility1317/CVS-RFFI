# LoRa R17 实增益与源识别保持约束

- run_id：`20261128-phase1-lora-r17-recognition-retain-s2026112885-r01`
- group_id：`phase1-lora-r17-recognition-retention`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 5 执行。

## 目的与对照

冻结 LoRa 实增益、R02/R07/R17 与 E104，训练双方向实增益头；使用合法源 L 与 U 伪标签，约束分类器相对实增益基线的损伤。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验及远端编译通过，控制器 PID 3677838，训练进程 PID 3677967；已核实命令、CWD、GPU、首步和后续日志增长。执行提交 `b9b99c508bc7c4203ac41ab85320d90090350a20`。

## 结果与覆盖

训练后对排除前 660 包后的 RX b200_1 固定 300 包 clean／low urban 预测，再由独立 CPU 评分。历史源 V 已用于 E104 选模，历史目标基准已曝光。固定 DURA v1 的 LoRa 匹配对照与目标七景仍待完成；本候选筛查结果尚未产生。

## 交接

恢复先检查本 run 的 `state.json`、PID 和日志，避免重复启动。
