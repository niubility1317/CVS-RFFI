# LoRa W24 共用与独有修正的同能量对照

复用星地增强 W24 E104 和 40,000 步冻结 W24 DURA。在两个路线的 clean 与六种星地场景，复用已冻结的无门控半量中心 IQ 和同一批 300 个源 V 物理包。重新生成同包的实收 W24 张量以恢复原生边缘，逐点核对中心 IQ 与归档一致。原识别对照的直通与半量类别必须逐条复现。两路线的每个包均已实际执行原冻结网络。

对每个包的实际半量修正按原跟踪器相位旋转到共同坐标，保留各 chirp 的实收幅度尺度。沿八条 chirp 取均值得到共用修正，原修正减去该均值得到独有修正。对照包含直通、原半量、只用共用分量、只用独有分量、偏共用和偏独有。两条偏向对照分别对共用／独有分量乘 1.25／0.75 和 0.75／1.25，然后逐包归一化到原半量修正的总能量。修正沿原相位回写中心窗，保持原生边缘的实测 IQ。分量和权重对所有包与两条路线固定。

N607 普通账号，使用原 W24 发布工作目录，一张有资源余量的 GPU 和八个 CPU 信道工人。入口 `python -m experiments.pamf_lora_main.w24_repeat_partition publish --repo . --context <源域上下文> --ssh-config <直连配置> --output <新发布目录>`，新输出根 `runs/20261125-diagnostic-lora-w24-repeat-partition-s2026112559-r01`。先用真实源 U 核对权重、相位模长、修正能量和分类输出。两路线的全部预测冻结后，独立 CPU 按物理 ID 读取源 V 标签。报告逐景和逐 RX 正确数、净纠错、配对区间及共用分量能量占比。出现实收 IQ、原预测复现、数值或覆盖异常时停止下游并保留产物。预期 `smoke.json` 及两路 `source-predictions.json`、`predictions.npz` 和 `source-scores.json`。 [登记](experiment.json) · [冻结半量对照](../../../docs/analysis/LORA_W24_READOUT_SWEEP_20261125.md)
