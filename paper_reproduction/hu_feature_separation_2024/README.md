# Feature Separation论文复现

本目录是Hu等人2024年论文的独立复现，不调用`paper_reproduction/cvs_aligned`。其论文确定使用WiSig ManySig、6个TX、12个RX、30 samples/TX训练、Adam lr=0.005、batch size=256和每TX25样本微调。

论文未报告λ、epoch、seed、Welch分段/窗口、增强应用概率与微调冻结策略。本实现把它们写成`implementation_choice`，不把选择冒充为论文设定。
