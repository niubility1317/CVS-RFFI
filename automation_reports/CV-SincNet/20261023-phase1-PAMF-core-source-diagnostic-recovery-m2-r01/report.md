# 源物理诊断技术恢复：已分析

ABC101/ABC111 seed2026092701的两条新预测和独立评分已完成，覆盖原已登记七景与源侧诊断。只修复 NumPy2/Torch2.1 显式int64索引接口；原 F/G 权重复用，没有新训练、IQ/ID或预算变化。实际执行提交117fbaecc1b2eda8080fb6503a16ce4006cc45ca。

完整结果见 [三核心分析](../20261023-phase1-PAMF-core-factorial-recovery-r01/core_results.md)，复核见 [数值验证](../20261023-phase1-PAMF-core-factorial-recovery-r01/core_verification.json)。三个实际 F seed 条件于每种固定 G，不是三次独立完整系统。所有负值保留；目标结果不回流训练、默认方法或队列排序。其他实验继续，监控未结束。
