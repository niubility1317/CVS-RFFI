# 默认训练日志

从本次日志修订起，本包所有经过`xuc_fusion.runtime.train`的新训练默认输出详细日志，无需开启额外开关。当前已在N607运行的六行继续使用其不可变发布版本，本次不停止、不重启、不热改这些进程，也不改写其原始日志。

## 人工阅读

启动时输出实际resolved参数；每50个接受步输出进度、loss、梯度范数和实际学习率。每个epoch沿用CVS的分块风格：`EPOCH-BEGIN`、`TRAIN`、`LR`、`LOSS-ORIGIN-MEAN`、`WEIGHTS-LAST`、`VAL-SOURCE`、`DAOT-RC4`、`IR`、`EXPOSURE`、`SATELLITE`、`RESOURCE`、`EPOCH-END`。

这是在IR训练入口中恢复CVS日志的可读性。原生`SSDG.train_ssdg.format_ssdg_epoch_block`仍保留。IR入口的字段和权限不同，不直接调用依赖大量原生训练状态、默认值及target结果的旧formatter，也不为套齐版式构造虚假值。

## AI读取

|文件|用途|
|---|---|
|`training.log`|持久保存的详细文本，与标准输出一致，不依赖外层重定向|
|`training_config.json`|实际生效参数，含方法、seed、优化器设置、调度和数据路径|
|`training_metrics.jsonl`|每epoch一条紧凑记录；优先用于AI分析，无物理ID数组及大混淆矩阵|
|`training_metrics.csv`|与紧凑JSONL相同数值的宽表；每epoch一行，晚出现的IR字段扩展列，早期缺项为空|
|`actions.jsonl`|完整逐步证据、物理曝光、路由和原始参数，保持原格式|
|`logs.jsonl`|完整epoch验证指标和分组结果，保持原格式|
|`ir_steps.jsonl`|IR方法的完整逐步响应诊断，保持原格式|

紧凑记录的schema为`cvs_training_epoch_v1`。数值字段显式标注`.mean/.min/.max/.last/.observed_count`；布尔状态分别保存`.true_count/.observed_count`，区分配置与实际执行。平均数只使用实际记录的有限值，缺失值不补零。所有accuracy及F1均为0至1的比例。

`loss_terms`是原点目标记录的各项原生分量，不把它们未经归一化及加权直接相加解释为总loss。`weights`是实际ticket权重，`learning_rates`是实际optimizer参数组学习率。`epoch_seconds`是相邻epoch summary时间戳差，包含区间内的数据、训练、验证及前一轮保存/日志开销，不代表纯GPU时间。IR和DAOT诊断只有实际测量时才出现，稀疏探针通过observed_count标明覆盖步数。

本训练入口尚未保存training accuracy，明确写为null/N/A；不能从loss反推。源域V accuracy、CE及整体Macro-F1可直接读取。训练期间不执行target评估，target字段为null并写明原因。缺项不虚构，也不触发额外前向、数据读取或目标评估。

日志模块仅消费已接受步和已完成的源域验证结果，不导入torch、不访问模型、优化器、数据集或checkpoint，不改变任何训练决策。
