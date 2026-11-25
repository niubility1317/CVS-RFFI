# LoRa W24 无门控源域回写幅度对照

运行 `20261125-diagnostic-lora-w24-readout-sweep-source-s2026112559-r01`。沿用原生 Full 与原生 Residual 的同一 300 个合法源 V 包、原信道随机键、星地增强 W24 E104 和冻结 40,000 步 W24 DURA。每包执行一次完整恢复网络，在实收中心坐标使用同一末步修正的 0.5、0.75 和 1.0 倍读出，首尾保留原生实收 IQ。三档在 clean 和六景均使用实际恢复输出。另保留同包直通。两路线只改变读出幅度，模型、估计量、输入和识别器保持一致。精确噪声扫描的信息从该诊断隔离。

N607 普通账号，使用归档原生 W24 工作目录，单 GPU、八个 CPU 信道工人。入口 `python -m experiments.pamf_lora_main.w24_native_residual_source publish --readout-sweep --repo <checkout> --context <LoRa-context> --ssh-config <config> --output <new-release>`。使用新的固定运行根 `runs/20261125-diagnostic-lora-w24-readout-sweep-source-s2026112559-r01`。真实权重的源 U 检查后预测，逐路核对旧面板物理 ID 和直通类别。两路线全部冻结后另起 CPU 读取源 V 标签。按路线、场景和接收机报告正确数、净变化和配对区间。来源、复现、数值或冻结错误停止本诊断下游，保留既有产物。预期 `smoke.json`，两套 `source-predictions.json`、`predictions.npz`、`source-scores.json`。零目标读取，零训练，运行不改变正在进行的正式矩阵。 [原双路线源域对照](../../automation_reports/CV-SincNet/20261125-diagnostic-lora-w24-native-residual-source-s2026112559-r02/report.md)

## 独立源域评分

**VERIFIED**。N607 真实源 U 的 clean 与受扰检查通过。两条路线逐景重现旧直通，随后在全量冻结后独立评分。每路六景分母 1,800，低强度城区和 clean 分母均为 300。

| 读出 | Full 六景 | Full 低强度城区 | Residual 六景 | Residual 低强度城区 | 各路线 clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 直通 | 1,394 | 150 | 1,197 | 127 | 297 |
| 0.50 倍 | 1,434 | 164 | 1,217 | 131 | 293 |
| 0.75 倍 | 1,415 | 174 | 1,188 | 123 | 292 |
| 1.00 倍 | 1,339 | 158 | 1,109 | 102 | 292 |

半量的两路六景净增分别为 40 和 20，95% 配对区间分别为 [17，65] 和 [−12，54]。0.75 倍在 Full 低强度城区多判对 24 个，区间约为 [9，39]，但两路六景分别较半量少判对 19 和 29 个。全量的两路六景相对直通少判对 55 和 88 个，区间分别为 [−89，−22] 和 [−130，−46]。三档恢复的 clean 均低于同包直通。当前冻结权重的源域证据支持保留半量作为此权重的既定读出，扩大回写幅度未同时改善双路线和 clean。两套 `source-scores.json` 保留逐景和逐接收机结果。
