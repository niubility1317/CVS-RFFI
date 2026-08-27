# 两篇跨域RFFI论文复现设计

## 目标

在`E:\type10-7\paper_reproduction`创建并维护两个相互隔离的论文复现目录，同时在本Git仓库的`paper_reproduction`保存完全相同的版本化实现。每个目录必须包含用户提供的论文PDF、论文忠实代码、单元测试、可复现配置、运行说明与结果边界说明。

本文档的“论文复现”仅表示实现论文规定的方法并在可得数据上运行。只有数据集、receiver/day划分、训练设置和指标同时匹配时，才能比较论文数值；WiSig替代数据上的Tweak结果必须标记为`METHOD_REPRODUCTION_ON_SURROGATE_DATA`，不得写为Gaskin等人的原始数值复现。

## 目录与职责

两个用户可见目录为：

```text
E:\type10-7\paper_reproduction\gaskin_tweak_2023\
E:\type10-7\paper_reproduction\hu_feature_separation_2024\
```

两者分别镜像到Git仓库的`paper_reproduction/gaskin_tweak_2023/`和`paper_reproduction/hu_feature_separation_2024/`。每个目录固定包含`paper.pdf`、`src/`、`tests/`、`configs/`和`README.md`。根目录副本面向阅读和直接使用；Git仓库副本是唯一的版本控制与发布承载面。已有`feature_separation_crossrx`和`cvs_aligned`保持不动，它们是既有实现/项目扩展，不能替代新的paper-faithful实现。

## Gaskin等，2023：Tweak

实现输入为`[B,2,128]`IQ片段的共享1D编码器：Conv1D(128,k=7)、Conv1D(128,k=5)、MaxPool(2)、BatchNorm、Conv1D(256,k=7)、Conv1D(256,k=5)、MaxPool(2)、BatchNorm、FC(256)和FC(12)，全程LeakyReLU。训练采用margin=0.1的batch-hard triplet loss、SGD momentum=0.9、batch size=64和最多100 epochs；学习率是每个可复现实验配置显式记录的搜索结果。

校准不得改变网络权重：每类从N个标注目标域样本构造embedding centroid与平均欧氏半径。决策按论文的M个片段平均embedding执行closed-set分类与open-set Admit/Reject。评估包含closed-set accuracy，以及5次随机known/unknown划分均值的AUROC、TPR和FPR。测试覆盖triplet挖掘、校准几何、三种决策分支、权重不变性和输入形状。

论文原始LoRa测试床为25个发射机和2台USRP B210；在其数据未提供前，Tweak仅在WiSig上做方法复现和运行验证。此路径不创建或声称硬件、日期或LoRa配置的原文结果表。

## Hu等，2024：Feature Separation

实现使用WiSig ManySig：每个样本的I/Q`2x256`与256点Welch PSD组成`3x256`输入。共享骨干严格以论文图6的注意力ResNet18结构实现；TX与RX支路均为`512→256→BN→LeakyReLU`，再经独立单层分类器得到TX/RX logits。

训练总损失为`L_FS=L_CE+λ1L_Sim+λ2L_CLFEtx+λ3L_CLFErx`，其中`L_CE`是TX/RX交叉熵，`L_Sim=||X_tx^T X_rx||_F`，熵项作用于相应分类器概率。论文没有给出λ、epoch、seed、增强应用概率与微调冻结策略；这些值只能作为配置中的`implementation_choice`写入，不得标为论文参数。基础实验采用论文给出的30 samples/TX、Adam lr=0.005、batch size=256、噪声SNR[15,30]dB、多径和Doppler[-15,15]Hz增强，并分开报告未知receiver的zero-shot与每TX25样本的fine-tuning。

测试覆盖三路输入、PSD实现、结构输出形状、每项损失的符号与梯度、特征相关性约束和只有TX标签的fine-tuning。实验输出将分别对齐论文的同日跨receiver、跨日期、增强消融和微调表格；任何无法由公开文本确定的实验轴均显式标识。

## 数据、运行与验收

两个实现均使用`ssr-gpu`环境。训练之前先通过小型synthetic样本测试所有路径，再根据数据清单核验WiSig receiver标签而非原始索引。实际训练产物写至各目录下的`runs/`，不覆盖已有实验；独立评分只读取已封存prediction。Tweak与Hu的结果报告各自保留训练数据、receiver/day划分、TX集合、样本数、seed、方法/数据复现状态、已知与未知指标。

源代码变更遵循测试先行。每个小组件先新增预期失败的测试，再以最小实现使其通过；完成后运行该模块完整测试、代码静态检查与配置解析检查。正式交付只提交本次新增的两个paper-faithful目录、镜像说明和必要测试，不暂存当前工作树中的其他未跟踪文件。
