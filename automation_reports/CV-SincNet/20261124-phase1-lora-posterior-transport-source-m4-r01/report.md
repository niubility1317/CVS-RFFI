# LoRa 条件后验方差调节恢复量

[预登记](experiment.json) · [运行事件](events.jsonl)

六条实收前导使用既有接收去扫和原采样回写。粗分支在多频复数先验上同时拟合清晰的条件均值和尺度。正则化相对作用把粗恢复映回接收域，细分支用前向残差、接收估计方差和粗后验继续更新。实收外带估计复噪声方差，半盲两组四节点留出作用估计显式进入反馈。两类估计均依赖当包接收，原采样尺度和载波保留。

标量高斯工作模型中，设测得噪声方差为 s²，条件方差为 v，则 k=v/s²。对均值校正量乘 1/(1+sqrt(k))，使高斯已知条件下的边际收缩系数由 k 变为 sqrt(k)。实际用实收 s² 与学习 v，并将 k 截到 [0.002,1]。公式针对高斯工作模型；实收残差相关且信道未知，实际性能由对照判定。两臂各连续训练 40,000 步，唯一训练读出差异为此系数或固定 0.75。先验拟合、均值／条件方差损失、观测反馈和细化更新共享同一 U 合法数据契约。所有方差诊断只使用互斥合法 U 及独立通用扰动，固定门低于阈值时逐样点旁路。 [Tweedie 后验矩原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3325056/)

Noise2Score 用 score 函数连接噪声模型与后验恢复。本实验用合法地面 IQ 对条件均值与方差作联合监督，恢复响应用实收噪声和相对作用计算，实际判据是固定 Tri-set 的逐包识别。感知与失真之间的理论差距说明降低均方误差并不自动保留信号分布。本轮的独立机制是同权、同输入下的方差调节，不使用标签或识别器作拟合。 [Noise2Score 原论文](https://proceedings.neurips.cc/paper/2021/hash/077b83af57538aa183971a2fe0971ec1-Abstract.html) · [作者实现](https://github.com/cubeyoung/Noise2Score) · [失真与感知原论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)

两条路线共享每臂终步权重和阈值。固定 Tri-set 2026092702 在 900 同物理源 V 包上预测 clean 和六景，随后独立后置评分。主读出只有在 Full、Residual 都超过匹配半量 DURA 且 clean 持平时才进入正式矩阵。
