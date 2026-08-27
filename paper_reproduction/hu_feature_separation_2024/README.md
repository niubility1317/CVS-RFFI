# Feature Separation论文复现

本目录是Hu等人2024年论文的独立复现，不调用`paper_reproduction/cvs_aligned`。其论文确定使用WiSig ManySig、6个TX、12个RX、30 samples/TX训练、Adam lr=0.005、batch size=256和每TX25样本微调。本实现提供同步/前导提取、均衡、归一化、Welch-PSD、图6优先主干、特征分离loss、三类信道增强、训练选模、TX-only微调和六类论文评价矩阵。

论文未报告λ、epoch、seed、Welch分段/窗口、增强应用概率与微调冻结策略。本实现把常用可复现设置固定在`strict_method.json`，并在训练结果的`method_metadata.unpublished_defaults`中返回；这些选择不能冒充为论文作者设定。
