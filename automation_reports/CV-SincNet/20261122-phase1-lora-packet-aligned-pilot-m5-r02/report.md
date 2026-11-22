# LoRa 输入修正试验 r02

r01 的数据归档已校验。N607 源域检查发现归一化后 IQ 被提升到 float64，当时没有派发训练或预测。r02 使用新运行目录，将归一化 IQ 明确保存为 float32，复用已校验的原采样率 IQ。候选矩阵、信道、物理角色、预算和评分保持原登记。

在 50 条分层源 V 包上，以同一个 practical 通道输出比较补偿前后。郊区高场景的去啁啾峰偏差中位数从 2,937.5 Hz 降至 62.5 Hz。城区低从 2,937.5 Hz 降至 187.5 Hz，保留该条件的实际模拟残差。DURA 在 30 条干净源包抽检中均逐点透传，在 8 条城区低样本中有 6 条启用恢复。这些结果检验输入和保护规则，正式识别收益以冻结后评分为准。

[原登记](../20261122-phase1-lora-packet-aligned-pilot-m5-r01/report.md) · [六景时序实测](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/source_alignment_audit.json)
