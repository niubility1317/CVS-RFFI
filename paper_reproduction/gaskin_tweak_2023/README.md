# Tweak论文复现

本实现覆盖Tweak的共享encoder三元组训练、batch-hard挖掘、SGD(momentum=.9)、100epoch接口、五点学习率搜索、最佳checkpoint、固定N多域校准、M=10聚合、closed/open-set决策和五次均衡开集评价。论文原始测试床是25个LoRa发射机和2台USRP B210；其数据不在当前工作区。因此任何WiSig实验只能标为`METHOD_REPRODUCTION_ON_SURROGATE_DATA`，不能比较或宣称论文的原始数值。

论文没有公开的LeakyReLU/BatchNorm细节、初始化、完整挖掘策略和离散学习率候选固定在`strict_method.json`中，并在训练结果的`method_metadata.unpublished_defaults`中返回。
