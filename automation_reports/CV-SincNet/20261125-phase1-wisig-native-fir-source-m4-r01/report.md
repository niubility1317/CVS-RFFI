# WiSig 原生窗双向 FIR 观测恢复

原 R69 源 V 消融显示，保留其估计 FIR 相对单位作用在两路线各多判对 1,306 个，同一模型直接代入短训练差分方差则损伤 clean。本实验比较两套从零训练的 81 抽头 FIR。单侧对照覆盖当前潜在窗的 0 至 80 点延迟，双向组覆盖 −40 至 +40 点。两个模型的网络、参数量、源 IQ、通用扰动和优化预算一致。各组在对应零延迟位置以单位冲激初始化，反射延拓只用于有限观测窗的线性作用。输出保持 256 点原坐标，两次求解均使用该 FIR 的实内积伴随。 [R69 观察消融](../20261125-diagnostic-wisig-r69-observation-source-m2-r01/report.md) · [模型实现](../../../experiments/pamf_unrolled_restore/wisig_native_fir.py)

输入为当前 25 MS/s 的 256 点实收复 IQ。包采满后计算 RMS 和时频粗恢复，随后单次估计全窗复 FIR、逐点主复系数与共轭项以及全窗每实分量有效方差。两阶段用相同估计作用，各含一个独立多尺度学习先验和两次原 CG 更新。FIR 抽头的范数受限。方差从同一接收估计路径输出，以 `Â(x_source)−y` 的停梯度残差功率拟合，外部噪声观测、仿真真实作用和真实噪声均无需读取。该残差混合通用噪声、前向作用误差与反射边界误差。方差损失参与同一优化循环，波形求解分支停止对方差的反向梯度。门控取归一化 FIR 相对其零延迟冲激的偏差及全窗方差，两门限在 4,096 个源 U clean 观测上一次固定。每组同一权重和门限覆盖两路线。

两臂各 576,089 参数，使用同一 52,092 个合法源 U 包、512 包留出、独立 4,096 包校准。每臂从零训练固定 40,000 步，batch 24，AdamW 初始学习率 `3e-4`，余弦降到 `3e-5`。每包产生两份独立的通用变化，其中一份配合真实源包之间的弱复谱差异做同扰动的差异保真。保持 3 个源包的身份映射。损失同时覆盖粗层、两个先验和求解波形、正向波形预测、方差似然、跨扰动一致性、真实源差异和弱复谱。生成只返回 IQ，恢复拟合与校准隔离真实仿真参数、识别器、TX、源 V 及目标。粗恢复和方差估计不继承 R69 权重。

UDKE 在图像超分辨中交替展开模糊核及图像子问题，以已知缩放倍数和噪声水平调节求解。本实验只从当前窗估计一次实线性复作用与有效方差，并用相同估计驱动两次信号求解。Noise2Noise 用独立受扰观测互作训练目标。这里用合法地面源 IQ 作波形参考，两份通用扰动只用于一致性，真实源差异另作保真检验。最终指标由冻结设备识别器独立评分。 [UDKE 原论文](https://arxiv.org/abs/2203.05568) · [UDKE 作者实现](https://github.com/natezhenghy/UDKE) · [Noise2Noise 原论文](https://proceedings.mlr.press/v80/lehtinen18a.html) · [Noise2Noise 作者实现](https://github.com/NVlabs/noise2noise)

终步后固定同一 Main split 三视图 SIG-TIME SSL 种子 `2026092701`。Full 和 Residual 用此前同序 1,800 个源 V 物理包，各 clean 和六景，同包对照直通、原 R69 以及两组全部阶段。预测矩阵闭合后进行独立 CPU truth-last 评分。预登记主读出为 `centered_gated`，源 V 同时报告两路六景、低强城区、clean 与相对 R69 的配对不确定性。只有另行按目标协议登记后才进入多种子 Main split。

N607 普通账号，优先空闲 GPU，每臂一个独立训练进程。运行目录 `${N607_PROJECT}/runs/20261125-phase1-wisig-native-fir-source-m4-r01`，工作目录为同名 `-release`。通过 `python -m experiments.pamf_unrolled_restore.wisig_native_fir_source publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config>` 发布。任一臂真实源 IQ 首步、数据身份、有限性、终点无目标检查或预测矩阵闭合失败时停止本运行并保留产物。预期两臂各有 `*-complete.json`、冻结权重和完整日志，以及 `smoke.json`、两路 14 个盲矩阵与 `source-scores.json`。记录参数量、优化时长及交付前冻结权重 B1 时延。 [执行入口](../../../experiments/pamf_unrolled_restore/wisig_native_fir_source.py)
