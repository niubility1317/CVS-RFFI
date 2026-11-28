# WiSig 冻结 R17 clean 教师一致性与实增益筛查

- run_id：`20261128-phase1-wisig-r17-consistency-gain-s2026112881-r01`
- group_id：`phase1-wisig-r17-consistency-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 7 执行。

## 目的与对照

冻结完整 R17，以其 clean 输出为 IQ 教师训练受扰输入的实频谱增益，源域 RX8 筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过；控制器 PID 3605401，训练进程 PID 3605415。实际 GPU、命令、真实权重首步及日志增长已核实；执行提交 `ff6ff3a184e481f266d792c8155975842ae6ff14`。登记文件和事件已写入，索引重建遇到其他任务持有的锁，作为非阻断记录保留。

## 结果与覆盖

训练后自动固定 RX8 的 360 包 clean／low urban 预测，再独立评分；此前 RX1、RX3、RX4、RX6 已用于筛查。历史源 V 和目标成绩已曝光。固定 DURA v1 对照与目标七景待完成，当前无本候选筛查结果。

## 交接

恢复先核实本 run 的 `state.json`、PID 和日志，避免重复启动。
