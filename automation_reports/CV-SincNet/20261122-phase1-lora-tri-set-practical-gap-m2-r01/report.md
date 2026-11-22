# LoRa Tri 的源域星地损失自适应

从相同 LoRa 源契约的 E140 纯监督基座出发，使用种子 2026092702。E140 的源验证 clean 为 73.51%，六种 practical 为 41.13% 至 66.03%，说明场景退化程度差异较大。原 Tri-set 的高可信 U 硬标签由源域规则筛出，但只覆盖 6 个预测类别。本轮保持 Tri-set 的源标签、全部 U 损失和观测不变，仅检验源验证驱动的真实标签星地交叉熵权重。基准为固定有标签 practical CE 0.45 和原 Tri-set，同列普通 PL 与纯监督供比较。

从 E140 的源 V 固定评估取得各类 clean 正确率和各景 practical 正确率，取二者的非负差。场景组先在类别轴取平均，场景与类别组保留 6×10 项。权重为 1 加每项与全体均值之差，将偏移限在 ±0.25。按批次有效权重总和归一，固定 practical CE 系数 0.45。clean CE、原有类别权重、训练样本和场景采样序列保持配对，每组 28 轮 × 88 步。实际权重、应用前后的损失与伪标签使用量同步保存。计算表时只读取对应的 E140 源验证与真实 V 类别计数，不使用任何目标或 U 真值。

两组全量预测 30,000 个包、15 台未见接收机、clean 加六种 practical，矩阵全部固定后本地独立评分。以固定 0.45 组为主配对对照，关注六景均值、低强度城区及 clean 正确数。初筛要求六景均值高于固定 0.45 组，且 clean 最多下降 0.05 个百分点。通过仅代表单种子证据。固定源 E140 权重和最终 checkpoint，不按目标结果改训练配方或选 checkpoint。

N607 普通账户，从 GPU 1、4、5、0、6、2、3 按实际空闲容量分派，每卡一个本 run 进程，最多两个并行，最低空闲 12 GiB。CWD 为 `CV-SincNet/releases/20261122-phase1-lora-tri-set-practical-gap-m2-r01-release`，独立输出位于 `CV-SincNet/runs/20261122-phase1-lora-tri-set-practical-gap-m2-r01`。本聊天作为统一 owner，使用 `full_adaptive_publish --practical-gap --practical-submit …` 启动，真实 E140 权重、源波形和源实际权重表 smoke 后训练。源合同、损失非有限、完整预算或预测矩阵异常时保留产物并停止。等待 210 份预测闭合后用 `full_adaptive_score --practical-control …` 做独立目标评分和训练 U 诊断。
