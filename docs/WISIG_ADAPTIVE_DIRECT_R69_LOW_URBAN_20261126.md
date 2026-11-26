# Adaptive 直蒸馏 R69 的 Low Urban 续训

实验 `20261126-phase1-wisig-adaptive-direct-r69-low-urban-m1-r01` 从 `20261126-phase1-wisig-r17-direct-r69-adaptive-m1-r01/student/final.pt` 的 E20 权重开始。该权重由相同 Adaptive R17 直接蒸馏得到，谱系跳过 R38、R44、R46、R65。R69 保留 266,951 个参数、两级恢复先验和 4 步共轭梯度求解。续训更新两个恢复先验，冻结粗恢复及信道作用和有效噪声方差估计模块。

每轮将原 WiSig Main 源训练的 63,000 个 L/U 物理包随机分配至四组。默认 Full Low Urban 和加重 Full Low Urban 各占 25%，沿用已登记的加重定义（参考 SNR 降 6 dB、残余相位跟踪标准差乘 1.5）。这两组用干净参考的 IQ NMSE 加 0.05 倍多尺度谱误差。37.5% 回放旧 R07 混合实收 IQ，约束输出接近冻结直蒸馏 R69 的输出。其余 12.5% 以 clean 作为输入和重建目标。

新增 Low Urban 观测由固定源训练命名空间生成，保留源物理 ID 和接收机/日期分组。模拟器与训练之间只传 IQ。训练访问源域 L/U 的 IQ，模拟器的实际信道作用、真实噪声方差、TX 标签、PAMF 权重和目标数据均隔离。使用新的 AdamW 状态，权重衰减 1e-4、梯度范数裁剪 1。学习率在前 100 步从 2e-5 升至 5e-5，按余弦降至 5e-6。固定 10 轮、2,470 步、批大小 256，使用最后权重。

正式矩阵仅含默认和加重 Full，共用 clean 对照。每档使用 WiSig Main 原 168,000 个配对物理包（cross-RX 126,000，cross-both 42,000）、六景及同三份冻结 PAMF。所有 clean 和星地 IQ 都先经过 DURA，再输入 PAMF。四个预测分片固定后由独立 CPU 评分，报告 Low Urban、六景均值、clean、准确率、Macro-F1 和同包净纠错。固定对照是续训前的直蒸馏 R69、同一 Adaptive R17 的逐级 R69，以及原版 R69 和 Raw。

N607 启动使用 `experiments.pamf_unrolled_restore.wisig_r69_low_urban publish`，固定代码提交 `93051483ce0709da8c27db72c869c9e5c25f1ea2`。训练单 GPU，预测最多双 GPU。输出位于该运行的独立 `runs/<run>`，保留源增强与源 GPU 检查记录、`student/first_update.json`、逐轮损失、末轮权重、`predictions/` 和 `scores.json`。数据契约不符、非有限损失/梯度、系统故障或预测无法正常闭合均停止本运行。

## 启动状态

11 月 26 日 13:48（香港时间）在 N607 GPU 6 上核实首个优化步。源 IQ 检查通过，默认和加重 Low Urban 各已生成 63,000 条源域观测。首步损失 0.061114，裁剪前梯度范数 0.604902，可训练参数 81,336。正式测试在固定 E10 权重后接续。
