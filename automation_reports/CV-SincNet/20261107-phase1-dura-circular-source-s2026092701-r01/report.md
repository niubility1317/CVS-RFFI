# DURA 局部共轭相位修正 GPU 训练

- run_id：`20261107-phase1-dura-circular-source-s2026092701-r01`
- group_id：`phase1-dura-circular`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING。N607 GPU0 队列 PID 347424、控制器 PID 347435、训练 worker PID 347628；远端真实 DURA checkpoint 与源样本 GPU smoke 已通过，正式训练开始。实时状态以进程、日志和产物为准。

## 目的与对照

冻结 DURA，将局部共轭特征引入条件相位等变修正分支。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

冻结 DURA v1.0，从零训练 35008 参数的修正分支；输入为 received、DURA 恢复输出、物理残差信息及噪声估计。三个复信号均添加 17、65、129 点局部共轭特征，训练 32768 个合法源 ID × 三视图、固定 5 轮。损失为源 IQ 归一化重建误差，加 0.02 倍相对 DURA 的改变量和 0.002 倍相对 DURA 的正超额误差项。DURA 全部 266951 参数保持冻结；训练不读 F、TX 标签、V 或目标。

修正分支仅在三个输入复信号同时旋转且方差不变时保证相位等变；不能将此扩展为完整 DURA 链路的性质。真实源样本六步本机 GPU 检查、十项聚焦测试和一次独立 P0/P1 审查通过；远端 64 源 ID × 三视图训练 smoke 证实 DURA 冻结、F/TX/V/target 读取均为 0。正式训练及全部七景独立评分尚未完成，没有识别性能结论。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
