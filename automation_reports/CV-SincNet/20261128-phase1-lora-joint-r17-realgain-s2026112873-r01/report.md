# LoRa R17 与实增益主干源域联合微调

- run_id：`20261128-phase1-lora-joint-r17-realgain-s2026112873-r01`
- group_id：`phase1-lora-joint-r17-real-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练在 N607 GPU 5 执行。

## 目的与对照

冻结 R02/R07/R17 和 E104，联合微调实增益主干及 R17 物理残差双方向头，源域无泄漏筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
归档传输校验和远端编译通过。控制器 PID 3586351，训练进程 PID 3586422；已独立核对 CWD、命令、GPU 与日志增长。真实 checkpoint 首步检查了零增益恒等、双方向增益与实增益 core 梯度及 tracker 冻结。代码提交为 `6ffe09217d1021ac9d3d7129cbc97008804ef8c0`。

## 结果与覆盖

训练后自动固定源 RX b200_1 的 300 个此前未筛查物理包的 clean 与 low urban 预测，再由 CPU 独立连接真值；晋级规则见登记。当前尚无本候选筛查结果。历史源 V 已参与 E104 选模，历史目标基准也已曝光；同契约 DURA v1 LoRa 对照及完整目标七景待完成。

## 交接

恢复时先核对原 run 的 `state.json`、PID 与日志；每行只有一个运行 owner，避免重复启动。
