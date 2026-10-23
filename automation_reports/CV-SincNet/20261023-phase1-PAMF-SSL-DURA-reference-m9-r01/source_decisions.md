# 源域诊断与后续来源决定

仅在CPU读取固定720条源V诊断记录，每模型eval/冻结，TX标签只来自合法V；没有U真值、target、GPU任务或状态更新。加载前核对实际E30/6660来源，F20实际checkpoint的U/EMA关闭；RXPSM使用其原E30学生及自身EMA。

| E30来源 | 三seed raw源准确率均值 | 连续权重中的错误质量比例均值 |
|---|---:|---:|
| F20_student | 97.638889% | 0.539121% |
| RXPSM_student | 97.685185% | 0.368801% |
| RXPSM_own_EMA | 97.685185% | 0.365057% |

720条是固定平衡源诊断，不能外推U正确率、未见RX或显著性。F20使用固定初始高斯(.95,.01)，RXPSM使用其E30状态和固定L-only其他RX原型；连续权重错误比例比较并非单因素教师因果实验。现有证据没有显示换到监督E30教师更可靠，当前和下一条独立候选继续沿原RXPSM E30，不换父权重。

恢复视图训练假设尚未实现、加载或发布。DURA R69原登记的checkpoint.provenance_verdict披露“humanchoiceafterhistoricalformalexposure”，permissions.claim_scope为历史工程基准；当前记录不足以确认该人工选取链满足新F训练的source-only继承条件。因此本次保留它在用户指定的冻结参考评测中的语义，不将其作为新F教师/训练视图来源。这是PROJECT_PROTOCOL.md中checkpoint输入权限判断，不是新增审批或额外门禁，也不改变已经授权的DURA参考指标。这里不把“曾被评分”本身等同于训练污染，只记录尚缺source-only选择依据。

下一条独立源域假设使用标签保护的U特征统计混合，不加载G，不改变数据角色。MixStyle官方代码/论文给出p=0.5、alpha=0.1和低层特征统计混合；本项目固定使用原time_down/t2.pool，并以教师预测同TX、原始不同源RX和现有权重>=0.5限制配对。该保护规则是本项目研究改动，不称论文完整复现或已证明收益。

来源：[MixStyle官方代码](https://github.com/KaiyangZhou/mixstyle-release)、[ICLR2021论文](https://openreview.net/forum?id=6xHJ37MVxxp)。原实现仅为原理与固定超参数依据，不能证明PAMF的目标涨幅。
