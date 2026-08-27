# 两篇RFFI论文的一比一复现可追溯审计

## 审计目标与边界

本记录核对Gaskin等人《Deep Learning Model Portability for Domain-Agnostic Device Fingerprinting》和Hu等人《Few-shot cross-receiver radio frequency fingerprinting identification based on feature separation》的原文与本地实现。审计结论只在原文明确给出且代码已实现、配置已闭合、端到端训练与评价路径存在时标记为`EXACT`。论文未公开的超参数、数据切分细节或实现选择必须标记为`PAPER_UNSPECIFIED`，不能据此宣称一比一复现。

## 审计对象

|论文|实现目录|监督角色|状态|
|---|---|---|---|
|Gaskin等，2023，Tweak|`paper_reproduction/gaskin_tweak_2023`|独立论文—代码核对|完成|
|Hu等，2024，Feature Separation|`paper_reproduction/hu_feature_separation_2024`|独立论文—代码核对|完成|
|两者的本地可执行性与历史模块边界|`paper_reproduction`、`tests`|独立集成核对|完成|

## 判定规则

- `EXACT`：原文要求和代码行为可逐项对上，且所需配置值和执行路径存在。
- `DIVERGENT`：原文要求明确，当前实现采用不同做法。
- `MISSING`：原文要求明确，当前实现没有对应能力或端到端路径。
- `PAPER_UNSPECIFIED`：原文没有给出足以固定实现的细节；本地选择可运行，但不能叫作论文一比一复现。

## Gaskin等（2023）：Tweak

|原文要求|原文证据|本地证据|判定|影响|
|---|---|---|---|---|
|`2×128`输入和Conv128(k7)→Conv128(k5)→pool/BN→Conv256(k7)→Conv256(k5)→pool/BN→FC256→FC12拓扑|p.9|`gaskin_tweak_2023/model.py:7-35`|`EXACT`|结构级一致；未公开LeakyReLU负斜率、BN参数、卷积bias和初始化仍是`PAPER_UNSPECIFIED`。|
|孪生共享网络|p.9|只有可单独调用的`TweakEncoder`，没有三支共享训练封装：`model.py:7-36`|`DIVERGENT`|数学上可由外部共享调用实现，但当前没有论文训练闭环。|
|Triplet公式和`margin=0.1`|p.6，式(1)|`triplet.py:22-28`|`EXACT`|目标函数和margin一致。|
|hard-negative/高损失triplet采样|p.6、p.9|每anchor取最远正例和最近负例：`triplet.py:7-19`|`PAPER_UNSPECIFIED`|论文没有给出完整batch构造或筛选规则，不能认证为数值级同一挖掘器。|
|欧氏距离的数值实现|p.6|挖掘用`torch.cdist`，损失用有默认`eps`的`F.pairwise_distance`：`triplet.py:11,26-27`|`DIVERGENT`|会造成微小但真实的逐数值差异。|
|每类质心和平均L2半径|p.6，Algorithm1|`calibration.py:15-31`|`EXACT`|与算法式一致。|
|每类固定N校准样本、同设备多域校准|p.6-7；默认N为训练集10%见p.14|函数接受任意样本数，且每个label只有一个质心/半径：`calibration.py:16-30`；配置只记录比例：`wisig_surrogate.json:15`|`MISSING`|没有N约束、采样或多域状态。|
|以M个输入embedding均值作每次决策，默认M=10|p.7-8，p.14|只记录`decision_inputs_m=10`，无聚合实现：`wisig_surrogate.json:16`|`MISSING`|阻断closed/open-set论文决策的端到端复现。|
|closed-set和open-set判定|p.7-8，Algorithm2-3|`closed_set_predict`和`open_set_admit`：`calibration.py:41-55`|`EXACT`|边界分别使用`<`和`<=`，但输入被假定已完成M聚合。|
|原始LoRa数据、128点切段、10 known/15 unknown、75/25单次传输切分|p.8、p.11、p.14|`wisig_surrogate.json:2-4`明确WiSig替代；没有数据加载或切分代码|`MISSING`|原始数据不在本地；替代数据只能叫`METHOD_REPRODUCTION_ON_SURROGATE_DATA`。|
|SGD(momentum=.9)、batch=64、LR搜索、100epoch、最佳checkpoint|p.9|配置记录部分值：`wisig_surrogate.json:7-14`；无训练器、LR搜索或checkpoint|`MISSING`|无法执行论文训练。|
|三种闭集迁移场景、准确率；5次5 known/5 unknown开集抽样、最小质心距离AUROC、TPR/FPR|p.10-11|`metrics.py:6-13`只平均外部传入的指标|`MISSING`|没有正式评测、抽样或指标计算。|

Tweak的8项局部单元测试已通过；覆盖输入形状、挖掘、loss梯度、质心/半径和两个判定规则（`tests/test_gaskin_tweak_2023.py:10-62`）。这证明核心算子，不证明论文实验或原文数值被复现。

## Hu等（2024）：Feature Separation

|原文要求|原文证据|本地证据|判定|影响|
|---|---|---|---|---|
|`2×256`IQ加`1×256`Welch-PSD得到`3×256`|p.1490，式(13)-(15)|`representation.py:21-23`|`EXACT`|三路输入尺寸和拼接一致。|
|Welch的窗口、多段平均和重叠|p.1490，式(13)-(15)|固定为单个Hann窗256点FFT：`representation.py:6-18`|`PAPER_UNSPECIFIED`|代码公开了选择，但单段FFT不是论文足以确认的Welch实现，不能叫一比一。|
|同步、前导提取、信道均衡和归一化|p.1492|目录没有数据加载或预处理模块|`MISSING`|不能生成论文输入。|
|ManySig的6TX/12RX/4天/每收发对1000段、Table1的receiver/date角色和6:2:2划分|p.1492及Table1|配置仅登记数据集名、TX数和30样本：`configs/manysig_paper_choices.json:2-7`|`MISSING`|没有固定的训练、验证、测试数据。|
|带channel-attention的ResNet18和图6主干|p.1491-1492、Fig.6|5个各含2个残差块的阶段，所有残差块都插入attention，另加512维门控：`model.py:26-79`|`DIVERGENT`|实现并非标准ResNet18；图6不能支撑“每个残差块均有注意力”的做法。|
|TX/RX特征和分类分支|p.1491、Fig.6|`512→256+BN+LeakyReLU`和线性分类器：`model.py:82-105`|`PARTIAL`|图中分类器后的BN、LeakyReLU、Softmax没有逐层实现；交叉熵内含softmax在损失层面可等价。|
|两项交叉熵、Frobenius相似度和正号熵项|p.1490-1491，式(16)、(20)-(31)|`losses.py:7-40`|`EXACT`（公式表达）|PyTorch CE默认按batch取均值，而论文式子写样本求和；与未归一化相似度项的相对尺度未锁定。|
|`λ1/λ2/λ3`、epoch、seed、scheduler|论文未报告具体值|配置明确为`null`：`manysig_paper_choices.json:18-26`|`PAPER_UNSPECIFIED`|配置不闭合，未获作者源码/确认不能严格数值复现。|
|每epoch噪声、多径、Doppler增强|p.1488-1489，Algorithm1|配置只登记范围：`manysig_paper_choices.json:10-16`；无实现或调用|`MISSING`|多径的“最大时延→每路径时延”层级也未编码。|
|Adam(lr=.005)、batch=256、调度、每epoch验证保存、loss停滞后取最高验证准确率模型|p.1490、p.1492，Algorithm1|无训练循环、scheduler、验证或checkpoint|`MISSING`|无法训练或选出论文模型。|
|无新接收机/跨日/通道增强/损失消融/25样本TX微调评价矩阵|p.1493-1496，Table2-5和Fig.9-13|只有一次TX交叉熵step：`finetune.py:17-29`|`MISSING`|无法复现83.6%或98.25%等论文结果。|
|少样本微调仅用TX标签|p.1491，式(32)；p.1496|优化encoder、TX分支和TX分类器：`finetune.py:12-14`|`PARTIAL`|冻结策略原文未公开；而`model.train()`仍会更新RX分支BatchNorm运行统计量，注释中的“RX path stays frozen”不严格成立。|

Hu的5项局部单元测试已通过，覆盖融合形状、前向张量、loss梯度和一次TX微调（`tests/test_hu_feature_separation_2024.py:10-57`）。它们没有覆盖WiSig构建、预处理、增强、训练选模、切分或论文评价矩阵。

## 本地集成与既有模块边界

- 用户目录`E:\type10-7\paper_reproduction\gaskin_tweak_2023`和`hu_feature_separation_2024`的源码、说明和配置与隔离worktree同名内容一致；用户目录额外保存论文PDF，仓库测试位于`tests/`。该镜像关系为`EXACT`（源码语义）。
- 既有`paper_reproduction/feature_separation_crossrx`不是Hu论文的新实现：它使用log+标准化单FFT（`feature_separation_crossrx/model.py:8-18`）、不同的ResNet样式和分支（`:64-151`），并将按batch归一化的相似度、跨支熵以负号加入loss（`losses.py:7-14,42-44`）。它是CVS基线扩展，不能与本次Hu目录混用或报告为Hu论文复现。
- 隔离worktree以禁用pytest缓存的方式运行两份目标测试，13项均通过。该测试结果仅为`CORE_IMPLEMENTED_AND_UNIT_TESTED`。

## 当前冻结结论与下一步

三路独立只读审计一致判定：两份本地实现当前均为`CORE_IMPLEMENTED_AND_UNIT_TESTED`，整体为`NOT_ONE_TO_ONE_REPRODUCIBLE`。阻断原因不是CVS实验协议，也不是当前单元测试失败，而是论文明确要求的端到端数据、训练和评测路径缺失；Hu还存在可见主干偏离；两篇论文均含未公开实现细节。

要把状态提升到“可执行的paper-informed reproduction”，先补齐两条端到端pipeline、固定可取得的数据与划分、实现正式训练/选模/评测，并把所有原文未公开的选择单独登记。要称为“一比一复现”，还必须取得原始数据或可证明等价的发布版本，以及作者源码或作者确认来闭合未披露超参数、Welch细节、采样和冻结策略；在此之前任何结果不得与论文原始表格作同等复现宣称。
