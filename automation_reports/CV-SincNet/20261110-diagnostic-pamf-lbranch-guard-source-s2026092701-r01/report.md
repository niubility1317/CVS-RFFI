# 冻结 PAMF：真实标签分支关系与梯度保护源域检验

状态：REGISTERED。详细数据句柄、模型、角色、命令、矩阵和停止条件见 [experiment.json](experiment.json)。用户明确授权源域冻结模型验证，不训练、不访问目标集。

## 历史与本次差异

此前的普通 PL 冻结模型全 V 五分支真实标签与伪标签关系梯度探针记录于 `E:/type10-7/local_artifacts/pamf_branch_training_20261110/source_relation_gradient_probe_all_v.json`；参考梯度预算及方向保护的研究见独立 worktree `docs/reports/pamf_physics_branch_ssl_20261110/report.md`。旧 SIG-TIME 单步增量诊断见 `E:/type10-7/automation_reports/CV-SincNet/20261014-phase1-SIGTIME-crossRX-credit-sourceL-manysig-s2026092701-r01/report.md`。这些分别不是本次固定完整 Tri-SIG-TIME E50 的真实 L 分支梯度、与独立 V 分组方向的同参数支持集比较。

## 固定设计与证据边界

读取预登记模型原始自身 source-only E20+10+20/final50、seed2026092701，已有同 Main split/F0 契约 `VALIDATED_ONCE`。检查冻结读取器/分段祖先/数据索引，不读取 U 或目标。复用 SourceDataset：完整载入 L6300 和 V27000，12 个 batch 每个 TX×RX×day 单元各一条，实际 L1080、独立 V1080 参与前向；原有物理 ID 角色互斥，按序号仅用于分层，不是同步同一事件。

真实 L 原始 clean CE（LS0.01）作为融合原目标的局部代理；五个部署路径各自用真标签构造同 TX/day、不同 RX 的正对，与相同 RX/day、不同 TX 的负对，温度0.2、单支权重0.01。分别计算实际参数梯度和单支汇总；每个分支仅在自身梯度有定义的参数支持集，与相同参数的 L-CE 梯度比较。若辅助梯度与 CE 负内积，则移除 CE 方向上的冲突分量；不增加分类头或跨不相关参数构造梯度。source V 按五 RX 的 CE 梯度只用于分析 `g_V·g_aux` 和 `g_V·g_guard`，不用于确定保护因子，不拟合任何新东西。模型 eval、0优化器、0参数更新、0目标/隐藏 U 读取；此数值是局部 SGD 一阶方向代理，既不是实际 Adam 更新，也不代表完整监督＋REPL 训练目标、其他权重、泛化或准确率。

保护并非真实完整 L/U 梯度保护，原模型的 MixStyle、As、KL、对比及 U 全部没有在代理 CE 中计算。数据因首次使用模型而检查来源，但不重复验证既有数据。独立 P0/P1 检查发现冻结推理 Sinc 使用 no_grad；本次仅在内存恢复原可微滤波合成并清缓存，真实 checkpoint 合成 no-query smoke 在打开 L/V 前验证两个 Sinc 坐标梯度有限；不改权重与推理定义。失败保留独立 root，不覆盖其他工作或自动重试。
