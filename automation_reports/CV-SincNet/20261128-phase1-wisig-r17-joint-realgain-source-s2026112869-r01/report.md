# WiSig 完整 R17 与实增益源域联合微调

- run_id：`20261128-phase1-wisig-r17-joint-realgain-source-s2026112869-r01`
- group_id：`phase1-wisig-r17-joint-real-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练正在 N607 GPU 7 执行。

## 目的与对照

前两项冻结 R17 后附加恢复头均未通过筛查；固定 R02/R07 并联合微调 R17 三个 prior 与双方向实增益头，通过同契约原 R17 IQ 教师约束偏移，筛查未用过的源 RX4。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
发布归档传输校验及远端编译通过。控制器 PID 3562183，首个训练 worker PID 3562201，物理 GPU 7；已读回进程、CWD、首步及后续日志增长。首步检查 R17 prior 与两条增益方向梯度有限且非零，R02/R07 保持冻结。

## 结果与覆盖

训练后自动固定源 RX 4 的 360 条记录在 clean 与 low urban 的预测，再独立连接源 V 标签评分。RX 4 与前两次筛查的 RX 1、RX 3 物理记录互斥；源 V 和历史目标基准已有曝光。本候选尚无筛查结果，完整目标七景与 DURA v1 配对比较待完成。

## 交接

恢复先只读核查本 run 状态、PID 及日志，避免重复启动。训练、预测、评分按预登记自动接续。
