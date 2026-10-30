# DURA 源频谱配对增强单阶段恢复对照

- run_id：`20261030-phase1-DURA-source-spectralMix-s2026092701-r01`
- group_id：`phase1-dura-source-spectral-transfer`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源统计校准频谱增益，每次p.5精确A配对增强，一次G前向；从零单阶段E20/4940，无F/恢复器教师，自动全七景truth-last。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 与固定对照的唯一方法差异

原noVar E20从零单阶段对照保持损失、学习率、批量、物理源数据与20份模拟缓存。本版只在每次访问以0.5概率选择源统计频谱重加权：均匀选另一源RX的日期组合，使用正实增益，保输入RMS，以 `y+A(Sx-x)` 同步改变配对参考和接收IQ。未选样本原样，一次G前向；没有TX标签、F指导、神经教师或恢复器祖先。频谱统计预计算0.252秒，单独计成本，不计作额外神经训练阶段。冻结推理不读取该统计或RX元数据。

源诊断达到预定双视图条件，足以检验这条候选，不代表识别已改善。固定最终E20后自动完成同一F50的1800×7源面板及168000×7目标预测，全部七景固定后独立truth-last。实际增强比例、L/U曝光、跨源组合计数及完整耗时必须随结果报告，默认DURA保持。
