# DURA 时变复数修正源域结果

冻结原 DURA，训练一个 28614 参数的逐时间 9 抽头复数／共轭滤波及波形修正网络，输入为当前 received IQ、DURA 输出、物理残差和估计方差；零初始化时输出与 DURA 相同。训练只使用合法源配对 IQ、原 DURA 输出和 DURA-relative guard，未读取 F、TX 标签、V 或目标。固定 5 轮，GPU 训练计时 52.3 秒，另源数据及 DURA 上下文准备约 69.3 秒。完整模型还包含原 DURA 的 266951 参数和物理求解成本。

同一 672 条源 pilot 的 14 个分组各 48 条，以下为波形 NMSE（越低越好）：

| 源分组 | 原 DURA | 时变修正 |
|---|---:|---:|
| RX4/6 clean | 0.00004552 | 0.00004791 |
| RX4/6 low urban | 0.037405 | 0.037485 |
| RX8 clean | 0.00003287 | 0.00003875 |
| RX8 low urban | 0.100678 | 0.100556 |

12 个受损场景分组中，11 组平均 NMSE 低于原 DURA；RX4/6 low urban 及两个 clean 分组高于原 DURA。单组样本量小，绝对差值也小，不能据此断言统计显著或身份识别正收益。源 pilot 不在新修正网络拟合集中，但原 DURA 可能见过对应源域。

候选已冻结，真实权重预测 smoke 通过，七景预测和独立 truth-last 评分正在进行。远端源证据：N607 runs/20261106-phase1-dura-timevarying-source-s2026092701-r01/source_summary.json。
