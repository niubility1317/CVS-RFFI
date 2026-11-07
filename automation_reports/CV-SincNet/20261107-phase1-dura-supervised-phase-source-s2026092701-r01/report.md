# DURA 源配对相位监督 GPU 训练

- run_id：`20261107-phase1-dura-supervised-phase-source-s2026092701-r01`
- group_id：`phase1-dura-supervised-phase`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：N607 GPU 0 正式源训练运行中，worker PID 459768；远端真实 DURA checkpoint 和源样本训练 smoke 已通过。实时状态以进程、日志和产物为准。

## 目的与对照

冻结 DURA，以合法源 clean 配对监督完整恢复输出的相对整体相位。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

冻结 DURA，独立从零训练 32867 参数的复矩条件相位网络。除原 IQ 恢复损失外，仅在合法源 clean 配对上，以 clean 与 DURA anchor 的受限整体相位为监督，约束两个分支组合后的最终输出相位；相位损失权重 1，按可观测噪声门控。源 clean 和理想角均不在部署时读取。

固定 32768 个源 ID × 三视图、5 轮、AdamW 学习率 0.0003，末轮冻结；训练不读取 F、TX 标签、V 或目标。该训练仅改变相位监督，源数据、结构、seed 和预算与复矩对照匹配。七项聚焦测试、本机真实源数据六步 GPU 检查、独立 P0/P1 审查及对完整输出相位的针对性复核通过；远端 64 条源 ID × 三视图 smoke 已通过。训练结束后自动进行七景预测和独立 truth-last；尚无性能结论。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
