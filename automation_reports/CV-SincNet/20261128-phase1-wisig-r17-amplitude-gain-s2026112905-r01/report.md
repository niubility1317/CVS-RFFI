# WiSig R17 三方向实频谱幅度融合

- run_id：`20261128-phase1-wisig-r17-amplitude-gain-s2026112905-r01`
- group_id：`phase1-wisig-r17-amplitude-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 7 执行。

## 目的与对照

冻结 WiSig R17 和实增益先验，在低城区 Full 源 U 配对上训练三个频率实增益方向，RX6 新物理包筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过；控制器 PID 3728277、训练进程 PID 3728299。实际 GPU、命令、真实 checkpoint 首步及日志增长已核实。执行提交 `bd478df3cb57a839fd30f900158015e6f13b90a2`。状态事件已写入，索引更新遇共享锁，登记及事件保留。

## 结果与覆盖

训练后固定 RX6 三个日期各 100 包的 clean／low urban 预测，再独立 CPU 评分。已排除前 2700 条筛查记录和 1 条源 V 技术探针；源 V 历史及目标基准均已有曝光。当前无本候选筛查结果，固定 DURA v1 与目标七景对照待完成。

## 交接

恢复先检查本 run 的 `state.json`、PID 与日志，避免重复启动。
