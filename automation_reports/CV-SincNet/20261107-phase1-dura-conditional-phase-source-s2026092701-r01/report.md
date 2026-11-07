# DURA 条件相位双分支 GPU 训练

- run_id：`20261107-phase1-dura-conditional-phase-source-s2026092701-r01`
- group_id：`phase1-dura-conditional-phase`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：N607 GPU 0 正式源训练运行中。真实 DURA checkpoint 和 64 条源记录 × 三视图的远端训练 smoke 已通过；训练 worker PID 387065。实时状态以进程、日志和产物为准。

## 目的与对照

冻结 DURA，以逐记录观测融合局部复数修正与整体相位幅度修正。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

冻结 DURA v1.0，新加 32483 参数：局部复数修正支路与整体相位、幅度修正支路通过逐记录可观测信息学习混合；初始输出精确等于 DURA。整体相位分支有意依赖绝对相位，完整候选不宣称严格全局相位等变。

源训练固定 32768 个合法源 ID × 三视图、5 轮、AdamW 学习率 0.0003，末轮冻结；不读取 F 权重和输出、TX 标签、V 或目标。真实 DURA 源样本六步 GPU 检查、七项聚焦测试及独立 P0/P1 审查通过；远端 smoke 证实冻结 DURA 和权限计数。源训练完成后自动对同一冻结 Tri-SIG-TIME SSL 默认头完成七景预测，全部固定后独立 truth-last 和 DURA 逐记录配对。

本候选依据原 DURA 的源相位旋转诊断设计；该合成诊断不证明真实信道分布或身份性能。正式结果尚缺，不能宣称提升。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
