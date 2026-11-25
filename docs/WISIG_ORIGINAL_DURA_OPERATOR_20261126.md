# WiSig 原版 DURA 算子监督对照

原版 DURA v1.0 保留 R02 初始化及 R07 至 R69 的全部阶段。两组独立训练只改变 R07 信道作用监督。模型种子为 `2026092701`，每个新阶段 20 轮、4,940 步，七段共 34,580 步。每轮使用原版同一批 63,000 个 L/U 源样本及同一轮的原始实收 IQ。R07 的两组白噪声探针、0.1 倍对数方差误差以及原学习率、优化器和固定最后权重均沿用。

| 实验 | 信道作用目标 | 方差目标 |
| --- | --- | --- |
| `20261126-phase1-wisig-original-dura-operator-m1-r01` | 全包 Adaptive 保留获选候选的时延、主和镜像系数、时变与分数时延基。直接将算子系数转换为原版的反射 FIR 和逐时刻主、镜像增益。 | 与原波形反拟合组相同的 Adaptive 估计。 |
| `20261126-phase1-wisig-original-dura-fast-operator-m1-r01` | 冻结的 Practical 混合快估，诊断 `20261126-diagnostic-wisig-practical-mixture-s2026112512-r01`。其合法源 U 协方差和参数由原 WiSig 源池拟合。保留混合模型的时延坐标后使用同一个算子转换。 | 逐包直接复用原波形反拟合组已封存的估计方差，与同一物理样本及同一轮接收 IQ 配对。 |

保留原算子支持的 0 至 63 采样点滞后。主系数与共轭镜像系数联合做秩一逼近，得到共享的原版 64 抽头 FIR。若某个拟合仅含原格式之外的滞后，用原 IQ 波形转换兜底，逐轮记录次数。恒等样本仍用原恒等算子和方差下界。快估组的 Adaptive 方差来源只提供已估计方差，生成器与训练器不读取模拟器作用、噪声或目标信息。R02 权重来自同合同源 IQ 的原固定波形选择；R07 后只沿用本组新权重。

主要诊断是冻结 R07 的同参考随机探针 NMSE，报告整体均值和尾部误差，以及 Low Urban 的逐景结果。实际 IQ NMSE 和对数方差误差辅助定位，最终以冻结 R69 到 PAMF 的同包识别、Macro-F1、净纠错和 clean 保持为准。正式矩阵与上轮一致，Main split cross-RX 126,000 和 cross-both 42,000 条，默认和加重两档，Full 和 Residual，clean 和六种 Practical，三个相同冻结识别器。每条 IQ 走完整 R69，八片预测封存后独立读真值。目标成绩不回流训练或选模。

发布入口 `experiments.pamf_unrolled_restore.wisig_original_dura publish` 使用 `--action-supervision operator` 或 `--action-supervision fast_mixture`。N607 各组一个低负载训练 GPU、最低 12 GiB 可用，避开本对照组已登记的训练 GPU，每组最多两张卡并行预测。CPU 目标生成每组最多 16 个单线程进程。快估组优先提交。发布环境、命令和来源见相应 experiment record，远端工作目录 `releases/<run>-release`，输出 `runs/<run>`。预期产物包括源无 query 检查、逐轮估计完成记录及耗时、首步优化证据、各级固定最后权重、八片逐包预测和独立 `scores.json`。以合同不匹配、非有限数值、系统性进程失败或预测无法合法闭合为停止条件。
