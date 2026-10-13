# Phase1 可迁移特征提取器目标

主任务仅 Phase1：持续复盘、合法源机制诊断、最小实现、公平实验与结果交付。研究目标是默认原头 clean 三预定义 seed 均值 84%，同时检查真实 joint160 特征接线与可迁移性证据；不承诺必达，不用目标分数调参。

最新用户时序设计覆盖此前普通cos换头诊断许可：本轮默认 ResidualPhysicalHead 6×160/scale30 正常随骨干联合训练，禁止单独分类头训练、冻结换头、原型或 support 调参。旧已完成换头结果保留为历史，不继续新增。Phase2/newTX/未知拒识不在范围。

最新优先路线为 TIME“覆盖范围 × 分支内非线性”四组同容量实验。紧凑（3,1）/（5,1）/（5,1）与长覆盖（3,1）/（5,1）/（5,2），分别线性或分支SiLU，仅改t1/t2/t3，外部归一化/激活/池化/dropout与PA/frequency/reference不变。新配方固定旧门控＋EMA＋cos、33 label＋17 pseudo、从零50轮/11100更新，各3seed；历史 OLD_GATE_COS50 外部基线同33＋17，历史TIME是否复用必须核对实际执行与RNG路径，不能只看名字。此前32＋18的LEGACY结果单列，不合并。

每Phase1骨干≤50轮，U真值隐藏，原L6300/U56700/V27000，训练池标注10%，source split392005/RX/day/TX/物理ID不变；V不反向传播。NoExplicitCFO及内部逐包补偿保留，无新星地训练。固定终轮后自动clean逐包预测、独立truth-last评分；目标IQ/truth/统计/误差不参与结构、参数、父权重、停止、重跑选择。已暴露基准不称独立盲测或部署验证。

PA旧门控＋cos32＋18完整9行已核验：baseline80.6454%，CONV80.7738%，CROSS80.8133%；零单独head更新。实际执行1f8814c046a8c68eba8a334e66a0aceb76e491e0，300新model-epochs/66600更新、1512000默认决定、162条评分核验完毕。84%仍未达到。

当前最多两个动作：先对已有TIME真实权重做核折叠与源L固定720包/seed分支撤除；再发布TIME四组控制。Sinc真实9模型kernelrelativeL2≤2.4071e−8的近静止证据保留，参数坐标修复排队，不混入本轮。诊断零梯度/optimizer/headfit，不把核秩、源几何、人工gradient当作迁移效果。

每次实现/config/同轮实际结果与负结果提交push，并独立比较远端OID。主报告集中在 [CVS_PHASE1_FEATURE_REVIEW_20261014.md](CVS_PHASE1_FEATURE_REVIEW_20261014.md) 与机器台账；大原始数据、权重、预测、日志保留原存储。
