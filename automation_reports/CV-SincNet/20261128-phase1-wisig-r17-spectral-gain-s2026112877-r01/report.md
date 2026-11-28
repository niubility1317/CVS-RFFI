# WiSig R17 保持约束实频谱增益源域筛查

- run_id：`20261128-phase1-wisig-r17-spectral-gain-s2026112877-r01`
- group_id：`phase1-wisig-r17-spectral-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 7 执行。

## 目的与对照

冻结完整 R17，学习接收信号与其输出差值的实频谱增益，使用源域波形和 R17 保持损失，在未筛查源 RX6 上评估。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验与远端编译通过。控制器 PID 3591851，训练进程 PID 3591867；已核实实际命令、GPU、首步及日志增长。真实 R17 checkpoint 与合法源 U 的无查询检查确认零初始增益恒等且梯度有限非零。执行提交 `2a9eaba95d648a37bbaff5c2315972b0ef75b886`。

## 结果与覆盖

训练后自动固定 RX6 的 360 包 clean 和 low urban 预测，再由独立 CPU 评分。该接收机与前次 RX1、RX3、RX4 筛查互斥；历史源 V 和目标基准已曝光。当前无本候选筛查结果，完整七景及固定 DURA v1 的匹配正式对照仍待完成。

## 交接

恢复时先核查原 run 的 `state.json`、PID 与日志，避免重复启动。
