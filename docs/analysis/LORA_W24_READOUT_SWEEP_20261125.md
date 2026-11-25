# LoRa W24 无门控源域回写幅度对照

运行 `20261125-diagnostic-lora-w24-readout-sweep-source-s2026112559-r01`。沿用原生 Full 与原生 Residual 的同一 300 个合法源 V 包、原信道随机键、星地增强 W24 E104 和冻结 40,000 步 W24 DURA。每包执行一次完整恢复网络，在实收中心坐标使用同一末步修正的 0.5、0.75 和 1.0 倍读出，首尾保留原生实收 IQ。三档在 clean 和六景均使用实际恢复输出。另保留同包直通。两路线只改变读出幅度，模型、估计量、输入和识别器保持一致。精确噪声扫描的信息从该诊断隔离。

N607 普通账号，使用归档原生 W24 工作目录，单 GPU、八个 CPU 信道工人。入口 `python -m experiments.pamf_lora_main.w24_native_residual_source publish --readout-sweep --repo <checkout> --context <LoRa-context> --ssh-config <config> --output <new-release>`。使用新的固定运行根 `runs/20261125-diagnostic-lora-w24-readout-sweep-source-s2026112559-r01`。真实权重的源 U 检查后预测，逐路核对旧面板物理 ID 和直通类别。两路线全部冻结后另起 CPU 读取源 V 标签。按路线、场景和接收机报告正确数、净变化和配对区间。来源、复现、数值或冻结错误停止本诊断下游，保留既有产物。预期 `smoke.json`，两套 `source-predictions.json`、`predictions.npz`、`source-scores.json`。零目标读取，零训练，运行不改变正在进行的正式矩阵。 [原双路线源域对照](../../automation_reports/CV-SincNet/20261125-diagnostic-lora-w24-native-residual-source-s2026112559-r02/report.md)
