# 共轭相位等变恢复 GPU 训练

- run_id：`20261107-phase1-direct-circular-source-s2026092701-r01`
- group_id：`phase1-direct-circular`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING。N607 GPU3 队列 PID 330487、控制器 PID 330504、训练 worker PID 330604；真实源样本 GPU smoke 通过，三个源视图准备完成，正式训练进行中。实际状态以进程、日志和产物为准。

## 目的与对照

从接收 IQ 直接恢复，增加局部共轭特征并保持全局相位等变。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

从零初始化 63456 参数网络，仅使用接收 IQ；在基础相位等变模型上增加三个局部共轭特征：分别用 17、65、129 个采样点的 `E[z²]/E[|z|²] × conj(z)`。这些特征随整体相位共同旋转，但不保证保留真实设备指纹。固定训练 32768 个源 ID × 三视图、20 轮、batch 32；损失为按 clean 能量归一化的重建误差加 0.02 倍输出相对 received 的归一化改变量。训练不读取 DURA 输出、F、TX 标签、V 或目标；DURA 仅在训练后用作源诊断和正式对照。

远端 GPU smoke 使用 64 个源 ID × 三视图，核验 DURA、F、TX、V 和目标训练读取均为 0；八项针对性测试、真实源数据六步本机 GPU 检查及一次独立 P0/P1 审查通过。正式训练及后续七景独立评分尚未完成，没有识别性能结论。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
