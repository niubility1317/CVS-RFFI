# 为 LoRa 恢复器生成独立源 practical 配对

在已登记的 2800 L 与 25200 盲 U 的每个原 1 MHz 物理包上各生成一份独立 practical 观测。保留原流式信道、本 RX 的固定硬件条件、源角色和场景归组。唯一变化为随机流键附加固定的第二次实现编号。L 按原物理行号分配六景之一，U 与本包的第一份场景对应。补偿与原实验一致的 24 点数值时延，从实际相邻上下文截取并抗混叠重采样为六条 chirp。只读原始源 L/U IQ 和配对索引，clean 窗口沿用原对齐缓存。TX 真值、V 与目标无需打开。配对身份和物理行顺序沿用原合同。

数据准备阶段无训练与目标推理。先用六条真实 U 逐点复算原缓存，检查新观测与旧观测不同且重复生成逐点相同。预期产物为 `L_s_practical.npy`、`U_s_practical.npy`、`smoke.json` 和带实际角色计数的 `complete.json`。为后续 G 的独立场景实现训练对照提供固定数据句柄。

N607 普通账户，既有 LoRa Python 和最多八个 CPU 工作进程。工作目录 `CV-SincNet/releases/20261122-phase1-lora-source-practical-draw-r01-release`，唯一输出为 `CV-SincNet/runs/20261122-phase1-lora-source-practical-draw-r01`。命令为 `python -m experiments.pamf_lora_main.full_restore_draw publish`，传入原 1 MHz 数据 release receipt。原数据或对齐状态不符、随机流校验/重现性失败、输出非有限即保留已经产生的产物并记录失败。
