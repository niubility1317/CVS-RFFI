# 论文原始复现矩阵

本矩阵只记录原论文设定和本轮WiSig复现实现。CVSStage2-C、OA-MSE、`z_id/z_dom`、satellite/LEO view、unknown gate、DRIFT/RIEI/RA-Collab替代结果均不进入本表。

## Cross-Domain Adaptation for RF Fingerprinting Using Prototypical Networks

|核对项|原论文设定|本轮WiSig复现状态|
|---|---|---|
|数据集|原文讨论ORACLE、CORES和WiSig；本轮按用户指定使用WiSig。|配置固定为WiSig。|
|source/target域|WiSig以采集日/域变化构造source和target domain；目标侧只有very few labeled examples。|`configs/protonet_cda_wisig.json`记录source days和target days；具体receiver编号若原文未给出，标`paper-unspecified`。|
|support/query|episode creation包含support set、query set和base truth；support用于估计prototype，query用于评估。|`EpisodeBatch`和`validate_closed_set_episode`检查support/query、K-shot、N-way、ID泄漏和target receiver元数据。|
|N-way K-shot|原文强调very few labeled support；具体N-way/K-shot网格未完整给出。|`paper-unspecified`；正式实验配置必须显式写入，属于implementation choice。|
|backbone|原文称backbone network；CNNbaseline使用相同网络但最后层替换为分类输出，层表未完整给出。|本轮保留backbone接口要求；具体层表为`paper-unspecified`，不能声称作者结构完全复刻。|
|prototype|每个label的support embedding均值作为prototype。|`compute_prototypes`已实现。|
|距离函数|Euclidean distance。|正式配置锁定Euclidean distance；`cosine`和`sqeuclidean`只作为implementation choice，不用于论文主结果。|
|损失|ProtoNet用query cross-entropy反传；CNNbaseline用negative log likelihood。|`prototypical_nll`实现query CE；CNNbaseline仍需训练入口接入。|
|优化器|Stochastic gradient descent。|配置记录SGD；lr、batch size、epoch、seed为`paper-unspecified`。|
|evaluation protocol|比较CNNbaseline和ProtoNet，报告source/target domain accuracy，并分析normalization影响。|矩阵已记录；正式结果前仍需实现WiSig评估脚本和结果表。|
|指标|source/target accuracy；图2/图3为主要证据。|结果尚未生成；不得声称正式复现完成。|

## Few-shot Cross-Receiver Radio Frequency Fingerprinting Identification Based on Feature Separation

|核对项|原论文设定|本轮WiSig复现状态|
|---|---|---|
|数据集|WiSig ManySig。|配置固定为WiSig ManySig。|
|数据与预处理|I/Q时域`2×256`，Welch功率谱`1×256`，融合为`3×256`；包含归一化、Gaussian noise SNR`[15,30] dB`、multipath、Doppler`[-15,15] Hz`增强。|`build_wisig_fusion_representation`生成`3×256`融合输入；增强参数进入配置，完整真实数据loader仍待接入。|
|source/target receiver|source receivers用于训练，target receiver用于跨接收机验证/测试。|配置记录`source_receivers`和`target_receiver`；`validate_closed_set_episode`检查source/target receiver disjoint。|
|训练样本|ManySig上选择两个source receivers、六类TX，每类随机30个训练样本。|配置记录30 samples per transmitter。|
|验证/测试|target receiver的Day1按train/val/test为6:2:2。|配置记录6:2:2 split；真实loader仍待接入。|
|fine-tuning|新receiver每类25 samples/class继续微调。|配置记录25 samples/class；具体冻结策略原文未明确，标`paper-unspecified`。|
|模型结构|`3×256`融合输入，带channel-attention的ResNet18/2D Conv/ResBlock，后接Dense/BN1D分支。|`FeatureSeparationNet`实现3通道输入、attention ResNet18风格共享encoder、TX/RX分支和分类头。|
|损失|`LFS=LCE+λ1LSim+λ2LCLFEtx+λ3LCLFErx`。|`feature_separation_loss`实现TX/RX CE、similarity loss和TX/RX entropy loss。|
|similarity loss|`C=X1^T X2`后取Frobenius norm。|`similarity_loss`按该形式实现。|
|优化器|Adam，lr 0.005，batch size 256。|配置记录Adam、0.005、batch size 256。|
|epoch/seed/lambda|原文未完整说明。|`paper-unspecified`；配置中的lambda是implementation choice，不能写成论文设定。|
|指标|Accuracy均值和论文表格。|结果尚未生成；不得声称正式复现完成。|

