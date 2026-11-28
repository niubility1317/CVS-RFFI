# WiSig R17 实增益使用原生 full 信道源训练

- run_id：`20261128-phase1-wisig-r17-full-source-s2026112889-r01`
- group_id：`phase1-wisig-r17-real-gain-full-source`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 7 执行。

## 目的与对照

冻结 R17，使用已登记源 U full 信道与 clean 配对训练实增益，测试排除既有源筛查物理包的 RX1 新样本。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验及远端编译通过；控制器 PID 3687557，训练进程 PID 3687742，已核实 GPU、命令、真实 checkpoint 首步与后续日志增长。执行提交 `d7208489d750d6609eeaa13c390058d2f291939a`。

## 结果与覆盖

训练结束后按原 Full 仿真配置，固定 RX1 三个日期各 100 包的 clean／low urban 预测，再独立连接源 V 标签评分。已排除前 1800 个筛查包。技术检查额外读取的源 V ID `5fa0fd382eb8af0fbc4cc08b` 未读取标签或识别结果，并从本轮与今后未曝光筛查中排除；完整源 V 和历史目标成绩曾曝光。当前尚无本候选筛查结果，DURA v1 对照及目标七景待完成。

## 交接

恢复时先核实本 run 的 `state.json`、PID 与日志，避免重复启动。
