# 接收机划分二的低 U 权重独立种子复核

**VERIFIED / ANALYZED**。发布提交 `e369c8795f30fa69ab7cf6b70f8e1e24079943b2`。真实 E20 冒烟 2/2，GPU 0、1 的 PID、CWD、命令和日志均已核对。两条训练、预测、clean 独立评分和 U 诊断共 8 项全部完成，无失败。见 [结果](analysis.md)。

探索剂量时，接收机划分二用融合 RX 均衡及总 U 权重 3，在种子 02 上达到 84.0238%，相对同机制权重 6 净增 990 条，相对原 Tri 净增 712 条。这里固定权重 3、伪标签门槛 0.95、融合正对混合系数 0.5 和原梯度保护，在种子 01、03 上只验证这一个配方。两个种子均沿用其同源角色的原 E20，再到固定 E50。02 用于剂量探索，01、03 未用于剂量选择。目标接收机与日期为已有历史设置。[剂量实验](../20261119-phase1-pamf-repl-rxbalance-dose-clean-m12-r01/analysis.md)

固定 E50 后对三个 raw clean 子集预测，随后独立 truth-last。合并 `cross_rx + cross_both`，逐物理 ID 比较同机制权重 6、原 Tri 和普通 PL。逐种子报告是否提升，以及对原 Tri 优势的影响，再汇总平均。同时读训练接受率、梯度、源验证及冻结后独立 U 真值诊断。

N607 原环境、GPUs 0–6，每卡总进程上限两条、准入保留 12 GiB。真实同源 E20 冒烟后直接开跑。发布入口是 `experiments.pamf_repl_source_adapt.transfer_publish --receiver-balance-dose-confirm`。阻断性错误时停止新派发，已有产物保留。[逐行登记](experiment.json)
