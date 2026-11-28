# WiSig R17 每轮物理求解内部实增益

- run_id：`20261128-phase1-wisig-r17-internal-gain-s2026112909-r01`
- group_id：`phase1-wisig-r17-internal-real-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 7 执行。

## 目的与对照

冻结 WiSig R17 和实增益先验，将三方向实增益插入 R17 每轮先验与两步加权求解之间，源 U 低城区 Full 配对训练。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验及远端编译通过；控制器 PID 3750044，训练进程 PID 3750122；已核实 GPU、命令、真实 checkpoint 首步与日志增长。执行提交 `a3e38fb86dbd86abbf4403e38628383cf040f3bd`。状态事件已写入；索引重建遇共享锁，登记及事件保留。

## 结果与覆盖

训练后在 RX8 三个日期各固定 100 条 clean／low urban 预测，再独立 CPU 评分。前 3000 条筛查记录和 1 条源 V 信道探针已排除；源 V 历史及目标基准均已有曝光。当前无本候选筛查结果，固定 DURA v1 与目标七景对照待完成。

## 交接

恢复先核实本 run 的 `state.json`、PID 与日志，避免重复启动。
