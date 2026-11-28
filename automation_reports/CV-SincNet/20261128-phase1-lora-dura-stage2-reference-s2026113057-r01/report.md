# LoRa 原生 8192 点双阶段 DURA 结构对照

- run_id：`20261128-phase1-lora-dura-stage2-reference-s2026113057-r01`
- group_id：`phase1-lora-wisig-gain-dura`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：FAILED（LoRa 预测的 E104 加载失败）

## 目的与对照

以 LoRa 源 U_s IQ 从零训练与 WiSig R69 参数结构相同的两阶段 DURA。检验相对 LoRa 三阶段 R17、实增益和直通的差异，固定旧 160 源包，补齐对照。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 技术失败与恢复

LoRa 18,000 步训练完成。从 25,200 个合法 U_s 包抽样，源域拟合共 44.62 分钟，峰值已分配显存 0.75 GiB。冻结权重与拟合记录完整。

LoRa 预测在 W24 归档加载期间遇到旧物理模块与当前恢复代码的导入冲突。失败发生在读取本次 V 预测样本和接入真值之前。已完成部分原样保留。技术恢复登记在 [20261128-phase1-lora-dura-stage2-reference-s2026113057-r02](../20261128-phase1-lora-dura-stage2-reference-s2026113057-r02/report.md)，只补 LoRa 预测和独立评分。

[原始冻结产物和失败日志](/E:/type10-7/local_artifacts/20261128-phase1-lora-dura-stage2-reference-s2026113057-r01/readback)
