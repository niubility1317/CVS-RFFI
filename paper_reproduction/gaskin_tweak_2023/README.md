# Tweak论文复现

本实现覆盖Tweak的共享encoder平方L2三元组训练、严格`dAN<dAP`的mini-batch全量挖掘、SGD(momentum=.9)、100epoch接口、五点学习率搜索、固定N多域校准、M=10聚合及closed/open-set决策。官方LoRa配置实验入口为`official_lora_experiment`，已使用官方数据；WiSig替代数据实验仍只能标为`METHOD_REPRODUCTION_ON_SURROGATE_DATA`。全量miner是论文未唯一确定的重建选择，不代表作者代码。

论文没有公开的LeakyReLU/BatchNorm细节、初始化、miner聚合、物理帧75/25选择、离散学习率候选和最佳checkpoint规则记录在`strict_method.json`及产物的`method_metadata.unpublished_defaults`。官方runner对每个record先作seeded affine双射、再切分75%/25%。V7先作五个1epoch的fresh LR probe，再由固定source训练池监控准确率选择下降的LR，从头训练100epoch并保留source监控准确率最高的epoch（相同准确率保留较早epoch）。监控参考和判别frame互斥，但均来自训练池，不能称为独立validation或泛化指标。

校准/聚合使用float64，欧氏距离禁用MM消减路径，以避免近邻embedding的距离被抵消为零。每行预测和用于复算的校准状态先写出，再写单独truth文件并评分；V1—V6产物保留。`training.fit_tweak`是另一个接受显式validation batch的通用接口，仍执行完整grid；它不等同于官方runner的1+100调度，不能将二者结果混用。
