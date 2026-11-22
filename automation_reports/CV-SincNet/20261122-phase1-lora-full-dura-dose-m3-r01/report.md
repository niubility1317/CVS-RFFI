# 完整 LoRa DURA 固定与自适应剂量的三种子配对验证

## 源域恢复

开发小试显示扩大 DURA 的触发范围会同时增加纠正与误修。这个验证区分能否识别真实受扰和能否利用当前恢复方向。完整物理数据来自 `20261122-phase1-lora-full-aligned-data-r01` 的 1 MHz 对齐缓存。2800 条 L 和 25200 条 U 的原始、模拟配对只提供 IQ。DURA 从随机初始化训练双阶段展开骨架，不复用小试 G 权重。共享的源 IQ 门控用 2800 条 clean L 和同包六景观测拟合 45 轮，用全部 clean L 与 U 的最高分加余量校准旁路；门控只读取当前包的 IQ，固定后不调参。DURA 固定剂量与自适应剂量共享该旁路、相同 G 和相同识别器。

三个独立 G seed 为 2026092701、2026092702、2026092703，各 1600 次实际更新，每步 24 对。每步各取一半可恢复 L 实用视图和可恢复训练 U，配准 clean 到观测的相位与微小频偏后，以逐 chirp 波形误差和同包可重复误差训练固定 G。按既有物理行在 U 中每五包留出一包专供剂量，其余进 G；优化 G 的队列不含留出的对应物理 ID。G 冻结后，通过当前输出的实际残差与正确方向的实部内积，求使配对误差最小的 0–1 剂量，训练剂量器 50 轮。剂量器只看观测、已经冻结的恢复残差及同包可靠性，保持原最大恢复剂量，保守地缩小误修。恢复训练和门控不采样源 V，不读 F、设备类别、目标数据。冻结后从 V 的每 12 个物理行取一条，共 1000 包，做仅波形的固定与剂量审计；它不参与选模。

## 固定识别器与盲评分

在主全量 `20261122-phase1-lora-full-adaptive-m12-r01` 目标预测和全 G 冻结后，每个 G seed 只接到同 seed、已冻结的 adaptive_tri F。三个读出为复用同权重的 direct、同 G 固定剂量与同 G 学习剂量。直接识别的原生产预测严格校验 seed 和目标盲矩阵，旁路 IQ 必须逐点等同原观测，用同样的 direct 决策。源域只看波形，完整目标首次成绩不反哺 G 权重、阈值或候选选择。每种子每读出读取 30000 条同物理包与七种 practical/clean 视图，共 63 格。独立报告 clean 逐点透传数、相对 direct/fixed 的错变对与对变错、逐 RX/TX、三种子波动及六景均值。clean 保持与跨种子的实际 practical 净纠错用于判断改进。

## 发布

`python -m experiments.pamf_lora_main.full_dura_publish --repo REPO --output OUT --context CONTEXT --ssh-config SSH_CONFIG --data-submit DATA --tri-submit TRI`。普通 N607 CVS-RFFI 环境，全量对齐数据只复用。工作目录 `CV-SincNet/releases/20261122-phase1-lora-full-dura-dose-m3-r01-release`，输出 `CV-SincNet/runs/20261122-phase1-lora-full-dura-dose-m3-r01`。GPU0–6 按真实容量跨卡，单卡本次一个进程，全卡至多四个，排除 GPU7。训练与源审计进程不接收目标路径。输入同源契约/父模型不合、非有限梯度、IQ 精确旁路或预测计数失效即停止后续派发并保留产物。独立 `full_dura_score` 在全预测冻结后核验每个 2000 条 RX 数组并打开官方真值。预期包含源 G 步数、剂量训练记录、无 TX 的源 `source_wave`、冻结预测与 63 格 scores。已本地核对真实源 cold-start 前反向、已知 0.25 解析剂量与完整分数矩阵。实际数据上的 source-only smoke 在远端使用冻结的同源门控先执行。
