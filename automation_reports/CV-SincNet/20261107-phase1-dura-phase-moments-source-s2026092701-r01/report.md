# DURA 复矩条件相位 GPU 训练

- run_id：`20261107-phase1-dura-phase-moments-source-s2026092701-r01`
- group_id：`phase1-dura-phase-moments`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：N607 GPU 0 正式源训练运行中，训练 worker PID 404028；远端真实 DURA checkpoint 与源样本训练 smoke 已通过。实时状态以远端进程、日志和产物为准。

## 目的与对照

冻结 DURA，以逐记录二阶和四阶复数统计增强条件相位混合。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

冻结 DURA，新加 32867 参数。在条件相位双分支中，为 received、DURA anchor 和物理残差增加单记录归一化二阶、四阶复数统计；条件分支据此与原均值及噪声信息共同预测混合权重、整体相位与幅度修正。它从零初始化，并不继承条件相位基线或其他候选权重。完整候选不声称严格全局相位等变。

与条件相位基线固定相同的 32768 个合法源 ID、三视图、5 轮、AdamW 学习率 0.0003 与末轮冻结；训练不读取 F 权重和输出、TX 标签、V 或目标。先前 64 条源 ID × 三视图的小样本检查发现复数统计非零，但不能证明其预测能力或身份效用。

本机真实 DURA 源样本六步 GPU 检查、七项聚焦测试、原条件相位模型重构前后逐元素一致检查和一次独立 P0/P1 审查通过；远端真实权重 smoke 已通过，正式训练已自动启动。完成后自动进入七景预测及独立 truth-last；尚无识别性能结果。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
