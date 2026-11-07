# 条件均值与耦合协方差联合恢复器，从零单阶段 30 轮

- run_id：`20261107-phase1-Restore-JointPrior-E30-s2026092701-r01`
- group_id：`joint-source-mean-coupled-covariance-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在固定耦合协方差结构上增加共享上下文的130参数均值输出头，按原先验逐坐标标准差限幅，仍经同一物理后验恢复IQ。

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

## N607 联合先验恢复器发布与实际排队

VERIFIED / QUEUED。联合均值/协方差版已发布，独立读回实际控制器PID/CWD/命令、CUDA为空、无子进程、等待日志增长及零GPU占用。代码4bba9df3、发布包SHA对比与远端编译通过，11项针对性检查及一次P0/P1审查通过。固定等待Coupled版最后GPU终态，再按GPU1正常总数最多3个、已有最多2个且剩余至少12000 MiB准入；不读取其分数，不影响其他owner。新增130参数的先验均值头与原耦合协方差共用上下文，初始输出一致，均值按原先验每坐标1个标准差限幅。原L_s/U_s、实际带噪IQ/STFT/连续冻结F0损失、15global+15otherRX、从零单阶段30/7410保持。真实source2/F0/均值及协方差梯度/E0检查仍待GPU训练首步，PASS后直接继续完整训练与Main七景；尚无本版性能结果。 [发布后独立证据](launch_evidence.json)
