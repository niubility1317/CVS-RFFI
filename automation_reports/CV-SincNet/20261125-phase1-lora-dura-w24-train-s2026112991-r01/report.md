# LoRa W24 DURA 从零训练

按用户要求以扩容 DURA 的粗恢复、实收观测与双阶段两步 CG 为骨架，从随机初始化训练 W24 权重。窗口包含全部八条上扫频 chirp。每条中心 1 ms 以 384 kS/s 表示，首尾各保留 24 个 1 MS/s 样点。首尾能量和归一化相邻点复相关以六个量进入粗恢复、观测估计和两个学习先验。输出保留实际首尾 IQ，中心使用末步半量修正。固定 clean 旁路门来自合法源 U 外带观测。 [恢复模型](../../../experiments/pamf_lora_main/dura_w24_model.py)

单种子 `2026112991`。源域 `lora_full_packets_v1`、盲角色 `U_s`，固定五台源 RX，拟合 20,000 包、校准 2,500 包、留出 500 包。通用随机扰动先施加到 1 MHz IQ，再分别投影到中心和边缘采样坐标。匹配原扩容 DURA 的 batch 16、AdamW、初始学习率 `2e-4`、余弦下降至 `2e-5`、梯度范数 1、40,000 步末冻结。一次优化循环联合各阶段波形、观测残差尺度、源弱复谱变化和双视图一致性。读取权限限源 U IQ 和其分组索引。 [训练配置](../../../experiments/pamf_lora_main/dura_w24_train.py)

N607 普通账号，优先选择空闲 GPU。提交由已推送 HEAD 固定，并写入 `release/submit.json`。运行目录为 `${N607_PROJECT}/runs/20261125-phase1-lora-dura-w24-train-s2026112991-r01`，发布目录以同名 `-release` 结尾。使用私有已验证 Python、`runtime_source.json` 和 `n607_ssh_config`，通过 `python -m experiments.pamf_lora_main.dura_w24_train publish --repo <checkout> --output <artifact-root>/release --context <runtime_source.json> --ssh-config <config>` 启动。真实源检查、数据契约、有限性或产物闭合失败时仅停止本链并保留已有产物。预期包含 `smoke-w24.json`、完整 `logs/fit.log`、`w24-complete.json`、`w24-model.pt` 和独立终态读回。

冻结后按已固定的 W24 单种子矩阵预测，待完整预测闭合后独立评分。 [正式测试登记](../20261125-phase1-lora-dura-w24-io-s2026092702-r01/report.md)
