# LoRa 四分量修正方向的复标量隔离诊断

固定原 8,192 点 Practical 父恢复器、新四分量方向头的 48,000 步末版权重、W24 E104 和默认分类头。复用此前四分量训练的 300 个源 V 包、五台源接收机、七个 Full 视图和相同源信道实现。此前等总量的源域析因发现分量内复方向贡献大于能量占比。另一版沿学习方向使用真值最优幅度也未提高识别。本次检验新方向头是否已接近可通过逐包幅相标量纠正的范围。 [四分量源训练与冻结](../20261128-phase1-lora-w24-component-directions-s2026112991-r01/report.md) · [固定总量源域析因](../20261128-diagnostic-lora-w24-component-allocation-s2026112559-r02/report.md) · [旧版幅度对照](../20261127-phase1-lora-w24-regional-dose-s2026112991-r01/report.md)

六条固定读出为直通、新学习方向、分量内最优非负实增益、等范数整体相位校正、最优复增益，以及中心精确扣除新增噪声。前两条逐包核对原冻结源预测。其余四条使用源仿真隔离参考，保留父版原生边缘及各读出的原 W24 定标。令 `g` 为某分量中已学到的复向量，`n` 为该分量的正确扣噪向量。复增益为 `gᴴn / (gᴴg)`。实增益只取其非负实部，整体相位只保留其单位复相位。复增益无法改变分量内部各坐标之间的相对方向。洁净参考上四条干预保留学习读出。

比较幅度、相位和二者合用相对学习方向的同包净变化，也比较完整中心扣噪相对复增益的净变化。若复增益仍只能挽回少量错误，单一分量标量校准的研究依据较弱。这个判断只用于设计下一轮恢复机制。源仿真新增噪声仅在诊断读取，整个运行不训练、不选模、不读取目标或报告正式性能。

运行标识 `20261128-diagnostic-lora-w24-component-scalars-source-s2026112559-r01`。N607 使用原 Full CPU 发布的 Python 环境与原 W24 工作目录，两个推理线程、两个并行重放进程、低优先级 CPU。复用 `20261128-phase1-lora-w24-formal-component-directions-cpu-s2026092702-r01` 的 `runtime.json` 绑定，分开执行 `probe_component_scalars.py predict <runtime.json>` 和 `probe_component_scalars.py score <runtime.json>`。代码以本登记提交为准。真实权重在合法源 U_s 冒烟，任一路径、原读出逐包对应或全量封存校验出错时停止并保留产物。七景六路预测封存后，独立评分才读取源真值。独占输出包含 `smoke.json`、`source-predictions.json`、`predictions.npz`、`source-scores.json` 与运行日志。 [诊断入口](probe_component_scalars.py)
