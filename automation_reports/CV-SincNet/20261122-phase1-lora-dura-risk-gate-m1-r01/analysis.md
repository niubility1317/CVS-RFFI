# LoRa DURA 学习旁路配对结果

**VERIFIED / SCORED_COMPLETE**。源 IQ 门控完成 45 轮。保持旧逐 chirp G 和同一冻结 adaptive_tri F，四路 28 格都在完整冻结后独立评分。

| 读出 | clean | practical 六景均值 | 比直接识别新增正确 / 新增错误 | 净正确变化 |
|---|---:|---:|---:|---:|
| 直接识别 | 68.6979% | 52.4479% | 0 / 0 | 0 |
| 旧逐 chirp DURA | 68.6979% | 52.6649% | 71 / 46 | +25 |
| 只过滤已有恢复 | 68.6979% | 52.6736% | 71 / 45 | +26 |
| 过滤并扩展恢复 | 68.6979% | 52.6215% | 98 / 78 | +20 |

三个恢复读出均对 1920 条目标 clean IQ 逐点透传，源 V 800 条 clean 无触发。过滤组对比旧 G 仅少一条新错。扩展组相对旧 G 多纠正 27 条但多产生 32 条错误，净少 5 条。源 V practical 中城区低触发 410/800（旧 382/800），城区中 239/800（旧 190/800），但两个场景的净正确变化分别为 −1 和 −2。与之前保守投影的结果合在一起，源 IQ 区分物理受扰与未受扰可以调节覆盖，不能代替预测恢复残差有益的能力。

后续应在完整同源合同上以 G 冷启动。用 L/U 真实 IQ 配对监督实际修正方向与剂量，并把相同 G 的原始旁路、以配对波形收益训练的剂量和直接 F 预先固定，保持 TX、F 与目标不进入 G 拟合。当前旧逐 chirp 仍是已验证的实用对照。开发区的 +26/11520 不能说明完整主划分可重复。

[28 格独立评分](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/dura_risk_gate_m1_r01/readout/scores.json) · [触发与净纠错](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/dura_risk_gate_m1_r01/analysis.json) · [实际训练](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/dura_risk_gate_m1_r01/readout/G/gate/training.json)
