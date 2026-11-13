# 冻结原先验的源域结果

从原 DURA 独立副本出发，冻结原先验，以合法源 IQ 与 0.01 权重模拟信道作用响应联合训练上下文估计，固定 5 轮；未用 F、TX 标签、V 或目标数据。训练实际 851.7 秒。完成后同一 672 条源 pilot 的 14 个分组进行 IQ 评估，每组 48 条；候选未在 pilot 拟合，但原 DURA 可能见过对应源域。

| 源分组 | 原 DURA NMSE | 冻结先验候选 NMSE | 先验与上下文联合训练候选 NMSE |
|---|---:|---:|---:|
| RX4/6 clean | 0.00004552 | 0.00003968 | 0.00004096 |
| RX4/6 low urban | 0.037405 | 0.039023 | 0.043010 |
| RX8 clean | 0.00003287 | 0.00002790 | 0.00002706 |
| RX8 low urban | 0.100678 | 0.099894 | 0.102617 |

冻结先验候选在 RX8 的 mid urban、low urban 两组及两组 clean 的平均 NMSE 低于原 DURA，另外 10 个受损场景分组高于原 DURA。与同源 IQ、同预算的联合训练候选相比，冻结先验降低了两组 low urban 的误差，但 RX4/6 仍未超过原 DURA。该结果只反映源波形，不能推断身份识别收益或目标达标。

目标为固定 Tri-SIG-TIME SSL 的完整 Phase1 七景；预测 smoke 已通过，独立 truth-last 评分待七景全部完成。远端源汇总：[计算环境] runs/20261106-phase1-dura-fixed-prior-action-source-s2026092701-r01/source_summary.json。
