# 原版 DURA 与 Tri-SIG-TIME 的 LoRa 配对验证

固定种子 2026092702。使用同合同源域独立训练并冻结的 R69 与 Tri U=6、Tri U=3。三种识别器（原对照 Tri-set 和两种原 Tri 适配）分别比较直通、R69 全量和 R69 半量。保留固定 DURA、形状保护和上一轮高相干半量对照。组合在任何这些候选的测试成绩可见前预先确定。此轮不新增训练，冻结的 G 不读取 F、TX 或目标，组合只进行推理。

同一 30,000 包及 15 台未见 RX，clean 和六种 practical。固定 12 条路径、1,260 个盲预测数组、84 个评分单元。复用 840 个既有盲数组。两组原 Tri 的恢复输出从相同冻结门控调用相同 R69，保留完整 256 点 IQ；门控旁路预测逐项复用本识别器直通。预测全部冻结后独立评分。每条联合路径都报告相对各自 clean 的六景平均和最难场景差距、相对该识别器直通的净纠错，以及相对已验证强基线的实际增益。目标是星地接近 clean，并维持此前 clean 基线的绝对表现。

普通 N607 账户、GPU0–7 允许容量调度。调用 `python -m experiments.pamf_lora_main.full_dura_publish --native-tri-submit ... --r69-submit ...`。输出 `CV-SincNet/runs/20261122-phase1-lora-native-dura-tri-joint-r01`，CWD `CV-SincNet/releases/20261122-phase1-lora-native-dura-tri-joint-r01-release`。先等待两个 preregistered 原版适配矩阵预测冻结，再用真实源 IQ 检查冻结 G 与两种 F 的连通性，继续盲预测和评分。输入合同、权重编号、门控、非有限或不完整矩阵错误为技术停止条件。
