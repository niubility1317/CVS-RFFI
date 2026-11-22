# LoRa 高相干恢复步幅对照

固定种子 2026092702 的冻结恢复器与 Tri-set。在训练以外的 3,000 个源 V 包上，两档缩小步幅的波形使用同一门控，均在两个等距物理包半区提高识别。该源域诊断不读取目标数据。此次只比较源域诊断中预先固定的 0.25 和 0.50 两个修正倍数，不改恢复器、校准阈值、载波坐标、原形状保护路径、冻结 F 或相同批次的目标输入。

| 去载波修正 | 源 V 半区一净纠正 | 源 V 半区二净纠正 | 六景合计 |
|---|---:|---:|---:|
| 1.00 | −47 | +37 | −10 |
| 0.25 | +39 | +45 | +84 |
| 0.50 | +38 | +79 | +117 |
| 仅共享分量 | −11 | +14 | +3 |
| 仅各 chirp 独有分量 | +4 | +21 | +25 |

目标两条新路径分别把既有高相干分支的复数 IQ 修正限制到完整提案的四分之一和二分之一，最大额外校正分别为原 IQ RMS 的 3.75% 和 7.50%。目标每场景固定 30,000 个包与 15 台未见 RX，包含 clean 和六种 practical。沿用已冻结的 Tri-set 直通、固定 DURA、形状保护、高相干全量四个对照，共 630 份盲预测数组、42 个评分单元。报告两档相对形状保护的净纠错、场景、RX 和 clean 变更。两档均完整测试。扩大种子需观察较形状保护六景净改善且 clean 无净损失，继续报告 IQ 旁路和各 RX 效果。

普通 N607 账户，允许 GPU0–7，容量调度选择绑定。使用 `python -m experiments.pamf_lora_main.full_dura_publish --draw-submit ... --stack-submit ... --detail-dose-submit ...`，其他输入与高相干分支原发布一致，Git commit 定义实现。发布 CWD 为 `CV-SincNet/releases/20261122-phase1-lora-dura-coherent-dose-m2-r01-release`，唯一运行根为 `CV-SincNet/runs/20261122-phase1-lora-dura-coherent-dose-m2-r01`。真实源 IQ 与冻结恢复器 smoke 后，源审计、盲预测及独立评分。若合同、权重继承、限幅、原门控激活或盲预测矩阵出现技术错误则停止。

[源域步幅与分量消融](E:/type10-7/local_artifacts/lora_packet_pilot_20261122/full_v1/dura_fingerprint_audit/coherent_detail_source_ablation.json)
