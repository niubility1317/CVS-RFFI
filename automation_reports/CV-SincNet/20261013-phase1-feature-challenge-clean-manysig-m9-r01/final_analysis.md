# 浅中层特征自挑战：完整 clean 结果

VERIFIED：20261013-phase1-feature-challenge-clean-manysig-m9-r01，代码提交 a9ee32cd6e05323d11b2b5fcf4e28305f495dbba。九个从零训练 30 轮模型均完成固定 finalE30 冻结、168000 个 clean 预测及独立 truth-last 评分。84% 目标尚未达成。

|方案|三种子准确率均值|SD（百分点）|Macro-F1 均值|最差 RX 准确率均值|
|---|---:|---:|---:|---:|
|CE30|79.5202%|1.6345|79.1829%|70.8778%|
|RANDOM_MASK|78.2518%|0.9818|77.7889%|67.9431%|
|GRAD_MASK|79.1185%|0.8892|78.7074%|69.2125%|

|方案|seed 2026092701|seed 2026092702|seed 2026092703|
|---|---:|---:|---:|
|CE30|80.4976%|77.6333%|80.4298%|
|RANDOM_MASK|78.6845%|77.1280%|78.9429%|
|GRAD_MASK|79.3423%|78.1387%|79.8744%|

梯度遮挡相对纯 CE 均值降低 0.4018 个百分点，仅 1/3 seed 正向；相对随机遮挡增加 0.8667 个百分点，3/3 seed 正向。随机遮挡相对纯 CE 降低 1.2685 个百分点。按梯度选择优于同预算随机删除，但未证明本配置改善跨接收机识别，也未证明去除了 RX 信息。

完整读取并核验九份 30 epoch 日志、59940 条 step 记录、九份 stdout、真实 final 权重、1512000 个原始预测和 162 份总体/RX/day/TX 混淆矩阵；独立重算 accuracy 与 Macro-F1 一致。无技术失败、nonfinite 或遗漏评分。U、teacher、EMA、卫星训练计数均为 0。

遮挡实际发生 33300 次，每个遮挡模型 5550 次；time_down 的 48 通道与 t2.pool 的 96 通道各 2775 次，每包准确删除 25% 通道。梯度组被删除通道平均承载 42.89%、44.80%、45.68% 的绝对梯度激活分数，随机组约 25%，排名与扰动确已执行。三组末轮 L 准确率约 98.7% 至 99.0%，CE 约 0.033 至 0.038。没有源训练崩溃或遮挡未执行的证据。

训练耗时包含合法 V 诊断，在实际并发环境下：CE 三个 seed 为 215.87/225.46/216.65 秒，随机遮挡 393.56/386.67/388.42 秒，梯度遮挡 387.20/393.53/393.01 秒。CE 均值 219.33 秒，梯度遮挡 391.24 秒，约 1.78 倍。这是本次并发实测，不是硬件独立基准。各组身份可训练参数为 177025；推理无遮挡、排名或新参数。总 L 包曝光为 7672320，未改变有标签比例。

实际源 RX 索引为 1/3/4/6/8、源 day 为 1/2/3，全部 6 TX，split 为 392005。L/U/V 为 6300/56700/27000，比例为 7/63/30。源 V 只作健康诊断，不选 epoch 或候选。目标 clean 仅在固定末轮后逐包预测，prediction 完成后独立 scorer 连接 truth。无 support/K/新类注册，本报告不构成 Phase2 或卫星部署结果。

全部 9 行在首次目标访问前固定，不按目标成绩修改权重、比例、层位、seed 或重跑。原测试基准已暴露，不声称新的独立盲测。本路线与另一活动对话的 matching/sampling/update/transform/directed/sets 分开，使用独立分支、owner 和 runroot；未停止或热修改其他任务。

方法受 [RSC 原论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3018_ECCV_2020_paper.php) 启发。此处采用绝对激活乘类别梯度、浅中层交替和 clean CE 锚定，是射频改编，不宣称论文复现。

证据保留在不可变 N607 runroot 的 source/step_metrics.jsonl.gz、epoch_metrics.jsonl、identity.pth、prediction/predictions.npz、scores/metrics.json、analysis.json 和 completion.json。本地全量核验位于独立 worktree 的 local_artifacts/feature_challenge_delivery/full_audit.json。
