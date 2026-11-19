# SatIQ PAMF same-satellite hard-negative batching diagnostic

- run_id：`20261119-diagnostic-satiq-pamf-same-sat-batches-s2026092701-r01`
- group_id：`phase1-satiq-ccs2023-fully-supervised`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Match PAMF-Metric architecture and objective for one full source epoch. Group half-batches by satellite so in-batch triplets encounter wrong beams of the same satellite.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

第一轮真实训练清单已逐批核对。原方案只有 11.04% 的训练锚点在批次内遇到同星错波束，平均 0.116 个候选。新版达到 93.43% 和 2.638 个。两者的跨 UTC 日期正对比例均约为 93.42%。两者各遍历 903,200 条互异样本，首轮共享 903,172 条。首 5,000 步由于调换批次，仅共享 28,356 条，因此以一整轮结果为主要配对对照。

该诊断保留 PAMF-Metric 的 11,000 点输入、模型、所有损失、训练随机初值和验证面板。正式三组百轮训练继续运行。[正式实验](../20261119-phase1-satiq-ccs2023-sup-s2026092701-r03/report.md)

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

N607 GPU 0 received the new job with no incumbent workers. PID 1967567 passed the real-packet smoke and produced a growing step log, using one source epoch and no augmentation.
