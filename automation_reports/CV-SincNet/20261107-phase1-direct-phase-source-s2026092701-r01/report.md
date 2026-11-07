# 独立相位等变恢复 GPU 训练

- run_id：`20261107-phase1-direct-phase-source-s2026092701-r01`
- group_id：`phase1-direct-phase`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：N607 GPU 7 的固定 20 轮源训练已完成，最终权重已保存，七景预测运行中。真实源训练与源预测 smoke 均已通过；运行状态以进程、日志及产物为准。

## 目的与对照

从接收 IQ 直接恢复，采用复数卷积和幅度门控保持全局相位等变。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

训练从零初始化 62496 参数复数残差网络，使用 32768 个源 ID × 三视图、固定 20 轮和 batch 32。训练损失为按 clean 能量归一化的输出重建误差，加 0.02 倍输出相对 received 的归一化改变量。训练不读 DURA 输出、F、TX 标签、V 或目标；DURA 在训练后用于源诊断及正式对照。GPU smoke 使用 64 个源 ID × 三视图，核验 F/TX/V/target 及 DURA 训练读取均为 0。七项聚焦测试与一次独立 P0/P1 审查通过。

远端训练日志记录 20/20 轮，`source_summary.json` 和末轮 `final.pt` 已生成；112 条源预测 smoke 通过，正式七景预测已启动。当前尚无七景识别结果或性能提升结论。按预登记 artifact 保留每 row 与 RX/day/TX/scene/K/seed 的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
