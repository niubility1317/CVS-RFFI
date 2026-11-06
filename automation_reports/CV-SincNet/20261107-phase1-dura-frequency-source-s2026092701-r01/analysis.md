# DURA 频域回归：源拟合完成

N607 GPU 2 对 32768 条合法源记录 × 三视图完成四组复岭回归拟合，各组 24576 条视图，系数固定。真实 DURA checkpoint 源 smoke 与冻结后的 112 条源预测 smoke 均通过。一次解析拟合计时 4.236 秒；不含源缓存加载、DURA 原训练与缓存生成、源评估和地面预测成本，不能与其他完整训练时长直接比较。

此前探索过的 672 条源 pilot，每组 48 条，原 DURA 可能见过相同源域。频域候选在 12/12 个受损分组的平均 IQ NMSE 低于 DURA，两个 clean 分组高于 DURA；IQ NMSE 不是身份准确率。训练不读取 F、TX 标签、V 或目标，完整七景预测正在运行。

| 源组 | 条数 | DURA NMSE | 频域候选 NMSE |
| --- | ---: | ---: | ---: |
| RX4_6/clean | 48 | 0.00004552 | 0.00004901 |
| RX4_6/practical_high | 48 | 0.00542070 | 0.00529102 |
| RX4_6/practical_mid | 48 | 0.00802796 | 0.00783631 |
| RX4_6/practical_low_suburban | 48 | 0.00871864 | 0.00864239 |
| RX4_6/practical_high_urban | 48 | 0.00909436 | 0.00900034 |
| RX4_6/practical_mid_urban | 48 | 0.02324280 | 0.02310981 |
| RX4_6/practical_low_urban | 48 | 0.03740450 | 0.03736195 |
| RX8/clean | 48 | 0.00003287 | 0.00004572 |
| RX8/practical_high | 48 | 0.01545082 | 0.01520789 |
| RX8/practical_mid | 48 | 0.01540366 | 0.01524313 |
| RX8/practical_low_suburban | 48 | 0.02475799 | 0.02452529 |
| RX8/practical_high_urban | 48 | 0.01530391 | 0.01514195 |
| RX8/practical_mid_urban | 48 | 0.05165612 | 0.05104020 |
| RX8/practical_low_urban | 48 | 0.10067829 | 0.10001916 |

[源摘要](source_summary.json)、[源训练 smoke](training_smoke.json)。
