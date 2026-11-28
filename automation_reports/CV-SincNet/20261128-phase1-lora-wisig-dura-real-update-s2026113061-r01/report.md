# LoRa 与 WiSig 同构双阶段 DURA 与实增益观测更新

- run_id：`20261128-phase1-lora-wisig-dura-real-update-s2026113061-r01`
- group_id：`phase1-lora-wisig-gain-dura`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：FAILED（LoRa 预测的 E104 加载失败）

## 目的与对照

两套数据统一使用两阶段 DURA 恢复后接逐频点非负实增益。增益由 DURA 估计的作用与方差、接收 IQ 和 DURA 输出通过两步投影共轭梯度求解。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 技术失败与恢复

WiSig RX3 的 160 个源验证包，两景五条读出均已冻结。LoRa 预测未完成，独立评分未调用。

LoRa 预测在 W24 归档加载期间遇到旧物理模块与当前恢复代码的导入冲突。失败发生在读取本次 V 预测样本和接入真值之前。已完成部分原样保留。技术恢复登记在 [20261128-phase1-lora-wisig-dura-real-update-s2026113061-r02](../20261128-phase1-lora-wisig-dura-real-update-s2026113061-r02/report.md)，只补 LoRa 预测和独立评分。

[原始冻结产物和失败日志](/E:/type10-7/local_artifacts/20261128-phase1-lora-wisig-dura-real-update-s2026113061-r01/readback)
