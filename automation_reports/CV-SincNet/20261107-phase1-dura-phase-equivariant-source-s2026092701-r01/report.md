# Phase-equivariant DURA correction GPU source training

- run_id：`20261107-phase1-dura-phase-equivariant-source-s2026092701-r01`
- group_id：`phase1-dura-phase-equivariant`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：TRAINING_COMPLETE；N607 GPU 3 完成全部五轮，正式七景评分已完成。

## 目的与对照

冻结 DURA，使用接收 IQ、DURA 输出、物理方向和方差作为输入，以无偏置复数卷积、幅度／噪声门控及有界径向残差学习修正。接收 IQ、DURA 输出和物理方向共同旋转、方差固定时，修正器的输出随之旋转；该条件性质不证明冻结 DURA 本身或整个系统相位等变。

修正网络 32128 个可训练参数，另加冻结 DURA 266951 参数；零初始化时输出逐值等于 DURA。训练使用 32768 个合法源记录的三个固定视图、五轮、batch 32，源损失为归一化 IQ 误差、0.02 倍 DURA 改变量及 0.002 倍归一化正超额误差。训练不读取 F、TX 标签、V 或目标，最终轮按预定规则冻结。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

10 项针对性检查通过，独立 P0/P1 审查未发现阻断问题。N607 GPU 3 的真实 DURA 源 IQ smoke 已通过：64 个源 ID × 三视图、六次更新，损失 0.02017534；DURA 冻结，未读取 F、TX 标签、V 或目标。正式训练五轮完成，末轮损失 0.02252737，日志累计约 1003.1 秒。独立源诊断 672 条 IQ、14 个接收机／场景组合中，相对 DURA 有 11 组归一化波形误差下降、1 组相同、2 组略升。两组 low urban 分别为 0.03740450→0.03740678 和 0.10067829→0.10068485。波形误差不是身份识别准确率，不能据此判断正式识别收益。原始结果见 [source_summary.json](source_summary.json)。按预登记完成七景预测后独立 truth-last 评分。

## 交接

执行 commit `9c4687fbcb936983b6a367e2e7e912155d3ee0ac`；启动时 GPU 3 空闲 18501 MiB、其他两个 GPU 应用。已核实 trainer PID、CWD、GPU 占用和真实源 smoke；恢复先查原 PID、controller 状态和日志，不重复启动。
