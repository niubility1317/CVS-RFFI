# 条件对抗恢复：源训练完成

固定 DURA 和合法源 IQ，条件对抗修正完成 32768 个源 ID × 三视图 × 五轮，batch32。真实 DURA 源 smoke 使用 64 个源 ID × 三视图，冻结模型源预测 smoke 覆盖 112 条；训练不读 F、TX 标签、V 或目标，判别器不进入部署。第五轮 G 总损失 0.03155785，包括源配对波形项和固定权重对抗项；该数值不是身份准确率，也不与其他损失配方直接比较。

此前已探索的源 pilot 共 672 条，RX4/6 与 RX8 各七个分组、每组 48 条。对照 DURA，13 组波形 NMSE 上升，1 组相同，0 组下降。该 pilot 未进入本候选拟合，但不能据此推断 DURA 从未见过相应源域；波形 NMSE 不代表 TX 识别性能。按事前固定矩阵执行七景测试，不因源域诊断改变或取消。

| 源分组 | DURA NMSE | 对抗候选 NMSE |
| --- | ---: | ---: |
| RX4_6/clean | 0.00004552 | 0.00004999 |
| RX4_6/practical_high | 0.00542070 | 0.00768384 |
| RX4_6/practical_mid | 0.00802796 | 0.01071004 |
| RX4_6/practical_low_suburban | 0.00871864 | 0.01182579 |
| RX4_6/practical_high_urban | 0.00909436 | 0.01132891 |
| RX4_6/practical_mid_urban | 0.02324280 | 0.02767345 |
| RX4_6/practical_low_urban | 0.03740450 | 0.04394917 |
| RX8/clean | 0.00003287 | 0.00003287 |
| RX8/practical_high | 0.01545082 | 0.01878365 |
| RX8/practical_mid | 0.01540366 | 0.01882070 |
| RX8/practical_low_suburban | 0.02475799 | 0.02860489 |
| RX8/practical_high_urban | 0.01530391 | 0.01831853 |
| RX8/practical_mid_urban | 0.05165612 | 0.05721689 |
| RX8/practical_low_urban | 0.10067829 | 0.10477861 |

正式七景预测进行中，完整预测固定后独立 truth-last；当前没有本版身份识别结果。原始证据见 [source_summary.json](source_summary.json)。
