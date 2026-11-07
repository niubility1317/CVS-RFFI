# 反向配对流：源训练完成

固定 DURA 和合法源 IQ，五轮 GPU 训练完成，使用 32768 个源 ID × 三视图、batch32。训练不读 F、TX 标签、V 或目标；最终轮速度匹配损失 0.01206723，它不是身份准确率，也不能与其他损失配方直接比较。原版与双路径概率平均分别复用同一个固定 final5，二者的正式七景预测已启动。

此前已探索的源 pilot 共 672 条，RX4/6 与 RX8 各七组、每组 48 条。与 DURA 的波形 NMSE 相比，本版 5 组下降、1 组相同、8 组上升。该 pilot 不参与本候选拟合，但原 DURA 可能接触过相应源域。源波形误差不代表身份识别准确率，不用于调整或取消已登记测试。

| 源分组 | DURA NMSE | 本版 NMSE |
| --- | ---: | ---: |
| RX4_6/clean | 0.00004552 | 0.00004552 |
| RX4_6/practical_high | 0.00542070 | 0.00543952 |
| RX4_6/practical_mid | 0.00802796 | 0.00801190 |
| RX4_6/practical_low_suburban | 0.00871864 | 0.00869771 |
| RX4_6/practical_high_urban | 0.00909436 | 0.00911708 |
| RX4_6/practical_mid_urban | 0.02324280 | 0.02339833 |
| RX4_6/practical_low_urban | 0.03740450 | 0.03741842 |
| RX8/clean | 0.00003287 | 0.00003287 |
| RX8/practical_high | 0.01545082 | 0.01558228 |
| RX8/practical_mid | 0.01540366 | 0.01542650 |
| RX8/practical_low_suburban | 0.02475799 | 0.02471229 |
| RX8/practical_high_urban | 0.01530391 | 0.01528727 |
| RX8/practical_mid_urban | 0.05165612 | 0.05172013 |
| RX8/practical_low_urban | 0.10067829 | 0.10088495 |

完整七景预测全部固定后，分别独立 truth-last；目前没有身份识别结果。原始记录见 [source_summary.json](source_summary.json)。
