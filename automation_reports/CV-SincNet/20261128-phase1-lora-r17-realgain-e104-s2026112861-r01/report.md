# LoRa R17 与物理残差实增益融合、冻结 E104 源域指导

- run_id：`20261128-phase1-lora-r17-realgain-e104-s2026112861-r01`
- group_id：`phase1-lora-r17-real-gain-supervised-fusion`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练正在 N607 GPU 5 执行。

## 目的与对照

首轮 source V 筛查未通过后，固定新候选：R17 和 R07 伴随残差双方向频率实增益，源域真实标签及冻结 E104 高置信 U 伪标签训练，独立物理 V 子集筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
发布归档传输校验与远端编译通过。控制器 PID 3535905，首个训练 worker PID 3535917，物理 GPU 5；实际进程、CWD、显存与日志增长已读回。真实 checkpoint 首步确认冻结 E104 的监督损失对两条新增方向均有有限非零梯度。

## 结果与覆盖

训练完成后，队列自动固定与第一轮 60 包互斥的 300 包源 V clean／low urban 预测，再由独立 CPU 评分。该流程尚无本候选筛查结果；目标集七景及固定 DURA v1 的 LoRa 匹配对照待完成。E104 历史选模曾使用源 V，既往完整目标成绩也已曝光。

## 交接

恢复时先只读核对本 run `state.json`、所属 PID 与输出，保留唯一 owner；不要重复启动。训练、预测与评分按原预登记顺序自动接续。
