# 一次独立 P0/P1 启动前审查

2026-10-12，独立只读 reviewer：未发现 P0/P1 缺陷。

范围：新 `experiments/cvs_relation_complement` 及实际执行依赖。整源 RX 留出、零初始化融合、完整 native 身份路径与原 cosine 分类头、两种分支相同容量与频率处理、优化器覆盖、30 轮/6660 更新均核实。弱幅相不变声明仅限新增通道。

5 个复用 CE30 完成证据 VERIFIED：配置匹配，fitL5040/heldV5400，留出 RX 未参与训练，无初始化 checkpoint、U 或目标访问，冻结后评估且状态未变。新训练的采样、优化器及预算可比。实际 run/preflight/checks/summarize/dispatch 的归档依赖检查 PASS；单 owner、不可覆盖输出、每 GPU 最多 3 个总进程保护有效。

本地 CUDA 检查 PASS：两个分支同为 197177 参数；零初始化原模型等价；第二次更新梯度进入分支；构建时禁止 torch.load 仍成功；常幅相不变性与普通关系反例；零 IQ 保留并有限；优化器完整且无重复；4 项违规配置负测。真实 checkpoint 无 query smoke 随 launcher 首步运行，PASS 后继续。
