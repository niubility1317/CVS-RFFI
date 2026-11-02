# DURA 内部物理后验：真实源域接口检查

- 范围：Phase1 源训练池的 4 条 L_s 接收波形（保护门开启、关闭各 2 条），不读取设备身份标签、U_s 真值、V 或地面测试。
- 模型：固定 DURA v1.0、固定 Tri-SIG-TIME SSL、既有源训练盲信道／高斯混合后验与 0.8 保护门；两个 DURA 内部新修正模块零初始化。
- 检查：两条门控路径上的初始化输出与 DURA 完全一致，两个新模块均有有限非零源重建及特征损失梯度，四个已有网络保持冻结。检查只做反向传播，不进行优化更新。
- 运行：N607 CPU 2 线程，不使用 GPU；负载超过 CPU 核数减 24 时等待。代码使用本登记所在 Git 版本，固定基础运行环境为项目 CVS-RFFI Python 和已有 Phase1 来源发布包。
- 命令：在本次只读 release 目录中运行 `python -m experiments.phase1_dura_posterior.smoke_controller`；由控制器调用 `python -m experiments.phase1_dura_posterior.physical_source_smoke --root /home/szu2070436088/2510044040/CV-SincNet`。
- 唯一 release：`releases/20261102-phase1-dura-posterior-internal-smoke-s2026092701-r01-release`；唯一输出：`runs/20261102-phase1-dura-posterior-internal-smoke-s2026092701-r01`。发生异常时保留输出并停止，不按分数改变行为。
- 产物：`status.json`、`smoke.log`，仅证明接口和梯度；这不是训练、准确率或 Phase1 七景结果。后续正式训练与评估须单独登记。
