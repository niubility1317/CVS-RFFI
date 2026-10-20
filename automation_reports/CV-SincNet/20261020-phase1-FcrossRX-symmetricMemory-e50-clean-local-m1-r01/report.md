# F 跨 RX：保留当前批次梯度的短期记忆

状态：LOCAL_VERIFIED，尚未训练。

本候选 SYM4 仍以原 F＋跨 RX 同 TX 对比学习为基底，目标是提升源域固定弱 TX 14-7、20-19 在 clean 目标域的性能，同时保持其余 TX。整体超过 84% 不代表弱类目标达成。匹配基线复用本机既有三种子 BASE，历史 85.1139% 只作方法锚点。

SYM4 在当前批次的锚点与参照特征两侧保留梯度，只让历史参照停止梯度。空历史时，损失值和完整特征梯度在数值检查容差内等同原 SupCon。最近三个真实 L 批次提供短期历史，每个浅/中层、每次 RDrop 单独维护，按物理 ID 保留最新项并排除当前重复项。当前批次 detached 均值/标准差统一用于当前与历史 raw mean/std 描述子，再单位化。原温度 0.2、跨 RX 同 TX 正样本、不同 TX 负样本、−log 正样本数常数与权重日程保留。

这是独立登记、独立输出的第三行，使用空闲 worker 名额；既有 ANCHOR/MEM4 两行保持原实现与运行计划。候选在其目标结果产生前固定，不依据其分数修改 SYM4。SYM4 不用额外弱类权重、头、前向或 IQ；历史只由可见 L 更新，不由 U/V/教师/EMA/目标更新，不导出到推理。

默认头正常联合训练；原 SIG_TIME_X3、统一 Sinc 物理更新及 clip5、RDrop1、LS0.01、普通增强、旧门控＋ownEMA＋cos 均保持。从零 50＝33L＋17U、11100 次更新，无星地增强，冻结后自动只测 clean。

先完成种子 2026092701 固定预算；按 experiment.json 中已固定源域默认头规则决定原样扩展 2702/03：两个弱 TX recall 均严格增加，其他 TX 不低于 BASE 0.25 个百分点，整体不下降，最差源 RX 不低于 BASE 0.25 个百分点，特征无零向量。该决定先于 SYM4 目标访问。首种子无论源判定如何均 clean 预测并独立 truth-last 评分；负结果保留。

完整矩阵、数据、实际来源、命令/环境/CWD、路径、GPU0 全局最多 3 worker、RAM 下限 4.5 GiB、CPU scorer 最多 1 和停止规则见 experiment.json。现有两行计入全局资源，不覆盖或停止健康任务。

聚焦数学、CPU/CUDA 实际 F 检查通过，共 14 个协议负例。默认头 6 个参数辅助梯度均为 None，主联合梯度逐值与原生一致；Sinc、浅/中层梯度非零，原物理优化器与 clip、EMA eval、精确导出通过。一次真实 checkpoint 无 query 检查通过，模型丢弃、不作训练初始化。一次独立 direct P0/P1 检查通过。

[XBM 原论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)提供跨批次记忆思路；当前无收益预设，更多独立配对和陈旧性不能混作纯难负样本效果。
