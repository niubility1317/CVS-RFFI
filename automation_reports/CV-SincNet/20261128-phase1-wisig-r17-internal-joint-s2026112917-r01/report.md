# WiSig R17 内部实增益与展开先验联合训练

- run_id：`20261128-phase1-wisig-r17-internal-joint-s2026112917-r01`
- group_id：`phase1-wisig-r17-internal-gain-joint`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 7 执行。

## 目的与对照

冻结 R02/R07 和 WiSig 实增益，使用实际 Full 低城区源 U 配对联合微调 R17 三个先验及内部三方向增益。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过；控制器 PID 3770291、训练进程 PID 3770305，已核实实际 GPU、命令和真实 checkpoint 首步。执行提交 `07527481cffb0f81942b917845259500ed81e554`。

## 结果与覆盖

训练后固定 RX1 三个日期各 100 条 clean／low urban 预测，再独立 CPU 评分，排除前 3300 条筛查记录和 1 条信道技术探针。源 V 历史及目标基准均已有曝光；当前无本候选筛查结果，固定 DURA v1 与目标七景对照待完成。

## 交接

恢复先检查本 run 的 `state.json`、PID 与日志，避免重复启动。
