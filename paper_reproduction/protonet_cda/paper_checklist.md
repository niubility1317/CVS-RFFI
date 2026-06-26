# Cross-Domain Adaptation for RF Fingerprinting Using Prototypical Networks复现检查表

## 论文任务

- [x] 本轮数据集按用户指定设为WiSig。
- [x] source域学习embedding network，target域very few labeled support估计prototype。
- [x] target query按到prototype的Euclidean distance softmax分类。
- [x] support/query闭集标签集合一致性由`validate_closed_set_episode`检查。
- [x] support/query样本ID泄漏由`validate_closed_set_episode`检查。
- [x] receiver/domain元数据可强制检查，防止source/target receiver混用。

## 模型结构

- [x] 原型分类核心已实现：`compute_prototypes`、`distance_logits`、`prototypical_nll`。
- [ ] embedding backbone层表：`paper-unspecified`，论文只称backbone network，未给完整层表。
- [ ] CNNbaseline训练入口：尚未接入；论文要求与ProtoNet使用same datasets/domain splits比较。

## 损失与优化

- [x] query CE/NLL已实现。
- [x] 正式配置锁定Euclidean distance。
- [x] optimizer记录为SGD。
- [ ] lr、batch size、epoch、seed：`paper-unspecified`。

## 数据划分与指标

- [x] WiSig source/target domain由配置记录。
- [x] support/query同target receiver的强校验已实现。
- [ ] 原论文未给出的具体N-way/K-shot网格仍为`paper-unspecified`。
- [ ] 正式source/target accuracy、图2/图3复现结果尚未生成。

## implementation choice

- 本轮按用户指定使用WiSig，而不是同时跑ORACLE/CORES。
- `cosine`和`sqeuclidean`仅保留为代码扩展选项；论文主配置不得启用。

