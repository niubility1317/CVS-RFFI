# WiSig R17 实增益低城区 full 源训练

- run_id：`20261128-phase1-wisig-r17-urban-source-s2026112893-r01`
- group_id：`phase1-wisig-r17-real-gain-urban-source`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 7 执行。

## 目的与对照

冻结 R17，只用已登记源 U low-urban full 信道与 clean 配对训练实增益；在排除既有筛查物理包的 RX3 新样本上筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过；控制器 PID 3702940，训练进程 PID 3702955；实际 GPU、命令、首步及日志增长已核实。执行提交 `272f0df285fa4bf94b869c6882893afb87b1d0e1`。源 U 低城区 Full 训练池实核为 7517 条。

## 结果与覆盖

训练后在源 RX3 三个日期各固定 100 条先前未筛查包，生成 clean／low urban 预测，再由独立 CPU 评分。前 2100 条筛查记录和曾用于信道技术检查的 1 条源 V 包已排除；源 V 历史及目标基准均已有曝光。本候选尚无筛查结果，DURA v1 与目标七景对照待完成。

## 交接

恢复时先核查本 run 的 `state.json`、PID 与日志，避免重复启动。
