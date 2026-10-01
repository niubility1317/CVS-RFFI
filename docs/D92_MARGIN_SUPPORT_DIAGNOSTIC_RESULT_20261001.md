本文件保留原始support摘要中的数学、训练与资源诊断。真实地面A/B/C现已另行补齐，见[完整三阶段报告](D92_MARGIN_GROUND_A_PAIRED_SUPPORT_RESULT_20261001.md)；以下原始摘要的A=N/A表示该原始summary未合入A，不表示当前缺少A证据。

# MarginJoint 完整 support 报告

run：`20261001-phase2-d92-margin-joint-support-m2-r01`；已核验 160 个 parent、1800 条物理路径。
本报告只描述合法 support 的持出诊断，不代表 query 准确率或新增独立数据验证。
A 与 B−A 为 N/A：实际地面头尚未接入。R0/B0 是 support 分类器，不能代替 A。
B/C 按同一 row、同一旧 support 与物理旧 held 样本配对；C 对全部注册类统一竞争并继承当次实际 B。
H、注册下降和绝对新旧差先逐 parent 计算，再等 parent 汇总；proxy 先对全部 anchor 求 parent 内均值。
true K1 只有全 support 解析头，无独立 held；其 OOF/proxy 指标单列 N/A。proxy 的 train K1 不等于真实 K1 独立证据。
理想方向为适应提升至少 10 个百分点、注册旧类下降至多 1 个百分点、新旧绝对差至多 3 个百分点；均为 soft 目标，不触发晋级、选模或重跑。

| 路径 | A旧 | B旧 | C旧 | C新 | H | B−A | B−B0 | 注册下降 | 平均绝对差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | N/A | 71.0417 | 63.7674 | 54.8464 | 58.4168 | N/A | 0.0000 | 7.2743 | 10.7405 |
| R_MARGIN_seq | N/A | 70.2083 | 66.5538 | 52.2734 | 57.8275 | N/A | -0.8333 | 3.6545 | 15.3220 |

准确率/H 用 %，差值用百分点。整体仅含 K5/10/20、新增2/5/10/20 的可测 parent。
注册下降变小需要同时检查 B 与 C，不能把 B 下降导致的差值缩小解释为保旧改善。

| 相对 R0 的变化 | B_old_accuracy | C_old_accuracy | C_new_accuracy | C_h | adaptation_gain_B_minus_A | support_adaptation_B_minus_B0 | total_old_accuracy_drop | C_abs_new_old_gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R_MARGIN_seq | -0.8333 | 2.7865 | -2.5729 | -0.5893 | N/A | -0.8333 | -3.6198 | 4.5816 |

完整 K×新增类数矩阵

| diagnostic | path | k | old_count | new_count | registered_count | measured_parents | A旧 | B旧 | C旧 | C新 | H | B−A | B−B0 | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | R0 | 1 | 6 | 0 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 6 | 2 | 8 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 6 | 5 | 11 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 6 | 10 | 16 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 6 | 20 | 26 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 5 | 6 | 0 | 6 | 8 | N/A | 63.3333 | 63.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 6 | 2 | 8 | 8 | N/A | 63.3333 | 61.2500 | 50.0000 | 53.5219 | N/A | 0.0000 | 2.0833 | 17.0833 |
| oof | R0 | 5 | 6 | 5 | 11 | 8 | N/A | 63.3333 | 54.5833 | 50.0000 | 51.9179 | N/A | 0.0000 | 8.7500 | 7.0833 |
| oof | R0 | 5 | 6 | 10 | 16 | 8 | N/A | 63.3333 | 53.3333 | 48.2500 | 50.3611 | N/A | 0.0000 | 10.0000 | 8.0833 |
| oof | R0 | 5 | 6 | 20 | 26 | 8 | N/A | 63.3333 | 48.7500 | 46.2500 | 46.5387 | N/A | 0.0000 | 14.5833 | 10.5000 |
| oof | R0 | 10 | 6 | 0 | 6 | 8 | N/A | 73.5417 | 73.5417 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 6 | 2 | 8 | 8 | N/A | 73.5417 | 71.6667 | 51.8750 | 59.4513 | N/A | 0.0000 | 1.8750 | 20.2083 |
| oof | R0 | 10 | 6 | 5 | 11 | 8 | N/A | 73.5417 | 67.9167 | 56.5000 | 61.4086 | N/A | 0.0000 | 5.6250 | 11.8333 |
| oof | R0 | 10 | 6 | 10 | 16 | 8 | N/A | 73.5417 | 65.6250 | 57.8750 | 61.4557 | N/A | 0.0000 | 7.9167 | 8.0833 |
| oof | R0 | 10 | 6 | 20 | 26 | 8 | N/A | 73.5417 | 64.1667 | 56.5000 | 60.0243 | N/A | 0.0000 | 9.3750 | 7.7500 |
| oof | R0 | 20 | 6 | 0 | 6 | 8 | N/A | 76.2500 | 76.2500 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 6 | 2 | 8 | 8 | N/A | 76.2500 | 74.6875 | 53.1250 | 61.5905 | N/A | 0.0000 | 1.5625 | 21.5625 |
| oof | R0 | 20 | 6 | 5 | 11 | 8 | N/A | 76.2500 | 70.1042 | 61.2500 | 65.0848 | N/A | 0.0000 | 6.1458 | 8.8542 |
| oof | R0 | 20 | 6 | 10 | 16 | 8 | N/A | 76.2500 | 68.0208 | 63.6250 | 65.7027 | N/A | 0.0000 | 8.2292 | 4.3958 |
| oof | R0 | 20 | 6 | 20 | 26 | 8 | N/A | 76.2500 | 65.1042 | 62.9062 | 63.9441 | N/A | 0.0000 | 11.1458 | 3.4479 |
| oof | R_MARGIN_seq | 1 | 6 | 0 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 6 | 2 | 8 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 6 | 5 | 11 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 6 | 10 | 16 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 6 | 20 | 26 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 5 | 6 | 0 | 6 | 8 | N/A | 62.0833 | 62.0833 | N/A | N/A | N/A | -1.2500 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 6 | 2 | 8 | 8 | N/A | 62.0833 | 61.2500 | 46.2500 | 50.2835 | N/A | -1.2500 | 0.8333 | 20.8333 |
| oof | R_MARGIN_seq | 5 | 6 | 5 | 11 | 8 | N/A | 62.0833 | 57.9167 | 47.0000 | 51.1703 | N/A | -1.2500 | 4.1667 | 12.2500 |
| oof | R_MARGIN_seq | 5 | 6 | 10 | 16 | 8 | N/A | 62.0833 | 56.6667 | 45.5000 | 49.8690 | N/A | -1.2500 | 5.4167 | 12.0000 |
| oof | R_MARGIN_seq | 5 | 6 | 20 | 26 | 8 | N/A | 62.0833 | 53.7500 | 44.7500 | 48.2214 | N/A | -1.2500 | 8.3333 | 11.8333 |
| oof | R_MARGIN_seq | 10 | 6 | 0 | 6 | 8 | N/A | 72.0833 | 72.0833 | N/A | N/A | N/A | -1.4583 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 6 | 2 | 8 | 8 | N/A | 72.0833 | 71.4583 | 50.0000 | 57.6631 | N/A | -1.4583 | 0.6250 | 23.1250 |
| oof | R_MARGIN_seq | 10 | 6 | 5 | 11 | 8 | N/A | 72.0833 | 69.3750 | 54.2500 | 60.3711 | N/A | -1.4583 | 2.7083 | 15.1250 |
| oof | R_MARGIN_seq | 10 | 6 | 10 | 16 | 8 | N/A | 72.0833 | 68.9583 | 55.6250 | 61.4768 | N/A | -1.4583 | 3.1250 | 13.3333 |
| oof | R_MARGIN_seq | 10 | 6 | 20 | 26 | 8 | N/A | 72.0833 | 67.9167 | 54.4375 | 60.3799 | N/A | -1.4583 | 4.1667 | 13.4792 |
| oof | R_MARGIN_seq | 20 | 6 | 0 | 6 | 8 | N/A | 76.4583 | 76.4583 | N/A | N/A | N/A | 0.2083 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 6 | 2 | 8 | 8 | N/A | 76.4583 | 75.2083 | 51.8750 | 60.5651 | N/A | 0.2083 | 1.2500 | 23.3333 |
| oof | R_MARGIN_seq | 20 | 6 | 5 | 11 | 8 | N/A | 76.4583 | 73.6458 | 57.5000 | 63.9920 | N/A | 0.2083 | 2.8125 | 16.1458 |
| oof | R_MARGIN_seq | 20 | 6 | 10 | 16 | 8 | N/A | 76.4583 | 71.9792 | 59.7500 | 65.0278 | N/A | 0.2083 | 4.4792 | 12.2292 |
| oof | R_MARGIN_seq | 20 | 6 | 20 | 26 | 8 | N/A | 76.4583 | 70.5208 | 60.3438 | 64.9104 | N/A | 0.2083 | 5.9375 | 10.1771 |
| proxy | R0 | 1 | 6 | 0 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 6 | 2 | 8 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 6 | 5 | 11 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 6 | 10 | 16 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 6 | 20 | 26 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 5 | 6 | 0 | 6 | 8 | N/A | 50.9375 | 50.9375 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 6 | 2 | 8 | 8 | N/A | 50.9375 | 49.2708 | 33.1250 | 33.3428 | N/A | 0.0000 | 1.6667 | 27.1875 |
| proxy | R0 | 5 | 6 | 5 | 11 | 8 | N/A | 50.9375 | 45.7292 | 39.6250 | 41.0697 | N/A | 0.0000 | 5.2083 | 13.4375 |
| proxy | R0 | 5 | 6 | 10 | 16 | 8 | N/A | 50.9375 | 42.2917 | 37.4375 | 38.7831 | N/A | 0.0000 | 8.6458 | 10.3125 |
| proxy | R0 | 5 | 6 | 20 | 26 | 8 | N/A | 50.9375 | 38.4375 | 33.2812 | 34.1491 | N/A | 0.0000 | 12.5000 | 12.2812 |
| proxy | R0 | 10 | 6 | 0 | 6 | 8 | N/A | 53.0324 | 53.0324 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 6 | 2 | 8 | 8 | N/A | 53.0324 | 51.0185 | 25.5556 | 29.9442 | N/A | 0.0000 | 2.0139 | 29.5370 |
| proxy | R0 | 10 | 6 | 5 | 11 | 8 | N/A | 53.0324 | 47.1991 | 36.3611 | 39.8694 | N/A | 0.0000 | 5.8333 | 14.1991 |
| proxy | R0 | 10 | 6 | 10 | 16 | 8 | N/A | 53.0324 | 43.7731 | 36.9306 | 39.0686 | N/A | 0.0000 | 9.2593 | 11.7407 |
| proxy | R0 | 10 | 6 | 20 | 26 | 8 | N/A | 53.0324 | 39.6991 | 32.4653 | 34.5480 | N/A | 0.0000 | 13.3333 | 13.2847 |
| proxy | R0 | 20 | 6 | 0 | 6 | 8 | N/A | 52.4726 | 52.4726 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 6 | 2 | 8 | 8 | N/A | 52.4726 | 49.6436 | 28.9803 | 33.0784 | N/A | 0.0000 | 2.8289 | 25.3235 |
| proxy | R0 | 20 | 6 | 5 | 11 | 8 | N/A | 52.4726 | 45.4825 | 37.0526 | 39.5978 | N/A | 0.0000 | 6.9901 | 12.3575 |
| proxy | R0 | 20 | 6 | 10 | 16 | 8 | N/A | 52.4726 | 41.6338 | 36.0888 | 37.6858 | N/A | 0.0000 | 10.8388 | 10.3980 |
| proxy | R0 | 20 | 6 | 20 | 26 | 8 | N/A | 52.4726 | 38.0702 | 32.3882 | 34.1667 | N/A | 0.0000 | 14.4024 | 10.1743 |
| proxy | R_MARGIN_seq | 1 | 6 | 0 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 6 | 2 | 8 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 6 | 5 | 11 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 6 | 10 | 16 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 6 | 20 | 26 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 5 | 6 | 0 | 6 | 8 | N/A | 50.9375 | 50.9375 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 6 | 2 | 8 | 8 | N/A | 50.9375 | 50.5208 | 27.5000 | 28.7583 | N/A | 0.0000 | 0.4167 | 30.3125 |
| proxy | R_MARGIN_seq | 5 | 6 | 5 | 11 | 8 | N/A | 50.9375 | 48.0208 | 33.7500 | 37.8817 | N/A | 0.0000 | 2.9167 | 17.6875 |
| proxy | R_MARGIN_seq | 5 | 6 | 10 | 16 | 8 | N/A | 50.9375 | 46.2500 | 33.8125 | 38.1120 | N/A | 0.0000 | 4.6875 | 14.4792 |
| proxy | R_MARGIN_seq | 5 | 6 | 20 | 26 | 8 | N/A | 50.9375 | 44.2708 | 30.9375 | 35.2556 | N/A | 0.0000 | 6.6667 | 15.2292 |
| proxy | R_MARGIN_seq | 10 | 6 | 0 | 6 | 8 | N/A | 53.0324 | 53.0324 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 6 | 2 | 8 | 8 | N/A | 53.0324 | 52.1296 | 20.9722 | 26.1998 | N/A | 0.0000 | 0.9028 | 32.6389 |
| proxy | R_MARGIN_seq | 10 | 6 | 5 | 11 | 8 | N/A | 53.0324 | 50.6713 | 30.9444 | 37.2502 | N/A | 0.0000 | 2.3611 | 20.1898 |
| proxy | R_MARGIN_seq | 10 | 6 | 10 | 16 | 8 | N/A | 53.0324 | 48.0787 | 32.7639 | 38.2434 | N/A | 0.0000 | 4.9537 | 16.6759 |
| proxy | R_MARGIN_seq | 10 | 6 | 20 | 26 | 8 | N/A | 53.0324 | 45.8333 | 29.9514 | 35.4963 | N/A | 0.0000 | 7.1991 | 17.8819 |
| proxy | R_MARGIN_seq | 20 | 6 | 0 | 6 | 8 | N/A | 52.4726 | 52.4726 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 6 | 2 | 8 | 8 | N/A | 52.4726 | 50.8882 | 23.3059 | 28.5282 | N/A | 0.0000 | 1.5844 | 29.5998 |
| proxy | R_MARGIN_seq | 20 | 6 | 5 | 11 | 8 | N/A | 52.4726 | 48.6129 | 31.1579 | 36.5426 | N/A | 0.0000 | 3.8596 | 18.7971 |
| proxy | R_MARGIN_seq | 20 | 6 | 10 | 16 | 8 | N/A | 52.4726 | 46.0800 | 32.2796 | 36.8615 | N/A | 0.0000 | 6.3925 | 15.9649 |
| proxy | R_MARGIN_seq | 20 | 6 | 20 | 26 | 8 | N/A | 52.4726 | 43.7061 | 30.0329 | 34.8775 | N/A | 0.0000 | 8.7664 | 15.2675 |

by_receiver_scene

| diagnostic | path | k | new_count | cohort | receiver | scenario | measured_parents | A旧 | B旧 | C旧 | C新 | H | B−A | B−B0 | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | R0 | 1 | 0 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 0 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 0 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 0 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 10 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 98.3333 | 98.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 50.0000 | 50.0000 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 96.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 49.1667 | 49.1667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 98.3333 | 96.6667 | 82.5000 | 89.0224 | N/A | 0.0000 | 1.6667 | 14.1667 |
| oof | R0 | 10 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 50.0000 | 39.1667 | 37.5000 | 38.2582 | N/A | 0.0000 | 10.8333 | 3.0000 |
| oof | R0 | 10 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 88.3333 | 78.0000 | 82.8378 | N/A | 0.0000 | 8.3333 | 10.3333 |
| oof | R0 | 10 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 49.1667 | 38.3333 | 33.5000 | 35.7044 | N/A | 0.0000 | 10.8333 | 4.8333 |
| oof | R0 | 10 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 98.3333 | 98.3333 | 57.5000 | 72.5349 | N/A | 0.0000 | 0.0000 | 40.8333 |
| oof | R0 | 10 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 50.0000 | 45.8333 | 25.0000 | 32.3505 | N/A | 0.0000 | 4.1667 | 20.8333 |
| oof | R0 | 10 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 96.6667 | 80.0000 | 87.5407 | N/A | 0.0000 | 0.0000 | 16.6667 |
| oof | R0 | 10 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 49.1667 | 45.8333 | 45.0000 | 45.3790 | N/A | 0.0000 | 3.3333 | 2.5000 |
| oof | R0 | 10 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 98.3333 | 95.0000 | 77.5000 | 85.3597 | N/A | 0.0000 | 3.3333 | 17.5000 |
| oof | R0 | 10 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 50.0000 | 39.1667 | 37.5000 | 38.2908 | N/A | 0.0000 | 10.8333 | 2.0000 |
| oof | R0 | 10 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 88.3333 | 79.0000 | 83.4030 | N/A | 0.0000 | 8.3333 | 9.3333 |
| oof | R0 | 10 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 49.1667 | 34.1667 | 32.0000 | 33.0437 | N/A | 0.0000 | 15.0000 | 2.1667 |
| oof | R0 | 10 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 98.3333 | 96.6667 | 76.0000 | 85.0820 | N/A | 0.0000 | 1.6667 | 20.6667 |
| oof | R0 | 10 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 50.0000 | 41.6667 | 30.0000 | 34.7871 | N/A | 0.0000 | 8.3333 | 11.6667 |
| oof | R0 | 10 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 90.0000 | 88.0000 | 88.9509 | N/A | 0.0000 | 6.6667 | 3.6667 |
| oof | R0 | 10 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 49.1667 | 43.3333 | 32.0000 | 36.8142 | N/A | 0.0000 | 5.8333 | 11.3333 |
| oof | R0 | 20 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 56.6667 | 56.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 94.5833 | 94.5833 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 57.0833 | 57.0833 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 93.3333 | 88.2500 | 90.7174 | N/A | 0.0000 | 3.3333 | 5.0833 |
| oof | R0 | 20 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 56.6667 | 43.7500 | 41.0000 | 42.3007 | N/A | 0.0000 | 12.9167 | 2.7500 |
| oof | R0 | 20 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 94.5833 | 90.4167 | 87.5000 | 88.9131 | N/A | 0.0000 | 4.1667 | 2.9167 |
| oof | R0 | 20 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 57.0833 | 44.5833 | 37.7500 | 40.8797 | N/A | 0.0000 | 12.5000 | 6.8333 |
| oof | R0 | 20 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 95.8333 | 68.7500 | 80.0053 | N/A | 0.0000 | 0.8333 | 27.0833 |
| oof | R0 | 20 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 56.6667 | 55.0000 | 32.5000 | 40.8007 | N/A | 0.0000 | 1.6667 | 22.5000 |
| oof | R0 | 20 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 94.5833 | 94.5833 | 81.2500 | 87.4026 | N/A | 0.0000 | 0.0000 | 13.3333 |
| oof | R0 | 20 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 57.0833 | 53.3333 | 30.0000 | 38.1534 | N/A | 0.0000 | 3.7500 | 23.3333 |
| oof | R0 | 20 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 92.9167 | 86.8750 | 89.7913 | N/A | 0.0000 | 3.7500 | 6.0417 |
| oof | R0 | 20 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 56.6667 | 40.0000 | 42.5000 | 41.1989 | N/A | 0.0000 | 16.6667 | 2.5000 |
| oof | R0 | 20 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 94.5833 | 88.7500 | 86.1250 | 87.4072 | N/A | 0.0000 | 5.8333 | 2.6250 |
| oof | R0 | 20 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 57.0833 | 38.7500 | 36.1250 | 37.3791 | N/A | 0.0000 | 18.3333 | 2.6250 |
| oof | R0 | 20 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 93.3333 | 85.5000 | 89.2402 | N/A | 0.0000 | 3.3333 | 7.8333 |
| oof | R0 | 20 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 56.6667 | 47.5000 | 38.0000 | 42.2214 | N/A | 0.0000 | 9.1667 | 9.5000 |
| oof | R0 | 20 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 94.5833 | 90.8333 | 89.0000 | 89.9054 | N/A | 0.0000 | 3.7500 | 1.8333 |
| oof | R0 | 20 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 57.0833 | 48.7500 | 32.5000 | 38.9720 | N/A | 0.0000 | 8.3333 | 16.2500 |
| oof | R0 | 5 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 93.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 35.0000 | 35.0000 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 91.6667 | 91.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 33.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 83.3333 | 67.0000 | 74.2426 | N/A | 0.0000 | 10.0000 | 16.3333 |
| oof | R0 | 5 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 35.0000 | 20.0000 | 15.0000 | 17.1242 | N/A | 0.0000 | 15.0000 | 5.0000 |
| oof | R0 | 5 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 91.6667 | 85.0000 | 80.0000 | 82.4052 | N/A | 0.0000 | 6.6667 | 5.0000 |
| oof | R0 | 5 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 25.0000 | 31.0000 | 27.6723 | N/A | 0.0000 | 8.3333 | 6.0000 |
| oof | R0 | 5 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 93.3333 | 50.0000 | 65.1163 | N/A | 0.0000 | 0.0000 | 43.3333 |
| oof | R0 | 5 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 35.0000 | 35.0000 | 45.0000 | 39.1304 | N/A | 0.0000 | 0.0000 | 10.0000 |
| oof | R0 | 5 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 91.6667 | 88.3333 | 80.0000 | 83.7234 | N/A | 0.0000 | 3.3333 | 8.3333 |
| oof | R0 | 5 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 28.3333 | 25.0000 | 26.1176 | N/A | 0.0000 | 5.0000 | 6.6667 |
| oof | R0 | 5 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 83.3333 | 60.0000 | 69.7674 | N/A | 0.0000 | 10.0000 | 23.3333 |
| oof | R0 | 5 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 35.0000 | 16.6667 | 28.5000 | 21.0196 | N/A | 0.0000 | 18.3333 | 11.8333 |
| oof | R0 | 5 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 91.6667 | 76.6667 | 74.0000 | 75.2988 | N/A | 0.0000 | 15.0000 | 2.6667 |
| oof | R0 | 5 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 18.3333 | 22.5000 | 20.0690 | N/A | 0.0000 | 15.0000 | 4.1667 |
| oof | R0 | 5 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 85.0000 | 70.0000 | 76.7301 | N/A | 0.0000 | 8.3333 | 15.0000 |
| oof | R0 | 5 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 35.0000 | 26.6667 | 22.0000 | 24.0602 | N/A | 0.0000 | 8.3333 | 4.6667 |
| oof | R0 | 5 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 91.6667 | 83.3333 | 88.0000 | 85.5589 | N/A | 0.0000 | 8.3333 | 4.6667 |
| oof | R0 | 5 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 23.3333 | 20.0000 | 21.3225 | N/A | 0.0000 | 10.0000 | 4.0000 |
| oof | R_MARGIN_seq | 1 | 0 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 0 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 0 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 0 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 10 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 10 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 10 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 10 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 2 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 2 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 2 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 2 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 20 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 20 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 20 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 20 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 5 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 5 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 5 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 5 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 10 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 47.5000 | N/A | N/A | N/A | -2.5000 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 96.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 47.5000 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | 80.0000 | 87.5472 | N/A | -1.6667 | 0.0000 | 16.6667 |
| oof | R_MARGIN_seq | 10 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 43.3333 | 31.0000 | 36.1435 | N/A | -2.5000 | 4.1667 | 12.3333 |
| oof | R_MARGIN_seq | 10 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 91.6667 | 78.5000 | 84.5526 | N/A | 0.0000 | 5.0000 | 13.1667 |
| oof | R_MARGIN_seq | 10 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 44.1667 | 33.0000 | 37.6639 | N/A | -1.6667 | 3.3333 | 11.1667 |
| oof | R_MARGIN_seq | 10 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | 50.0000 | 65.9091 | N/A | -1.6667 | 0.0000 | 46.6667 |
| oof | R_MARGIN_seq | 10 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 47.5000 | 25.0000 | 32.5962 | N/A | -2.5000 | 0.0000 | 22.5000 |
| oof | R_MARGIN_seq | 10 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 96.6667 | 77.5000 | 86.0239 | N/A | 0.0000 | 0.0000 | 19.1667 |
| oof | R_MARGIN_seq | 10 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 45.0000 | 47.5000 | 46.1234 | N/A | -1.6667 | 2.5000 | 4.1667 |
| oof | R_MARGIN_seq | 10 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 95.8333 | 75.5000 | 84.4552 | N/A | -1.6667 | 0.8333 | 20.3333 |
| oof | R_MARGIN_seq | 10 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 41.6667 | 34.2500 | 37.5956 | N/A | -2.5000 | 5.8333 | 7.4167 |
| oof | R_MARGIN_seq | 10 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 91.6667 | 77.0000 | 83.6802 | N/A | 0.0000 | 5.0000 | 14.6667 |
| oof | R_MARGIN_seq | 10 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 42.5000 | 31.0000 | 35.7885 | N/A | -1.6667 | 5.0000 | 11.5000 |
| oof | R_MARGIN_seq | 10 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | 71.0000 | 81.8648 | N/A | -1.6667 | 0.0000 | 25.6667 |
| oof | R_MARGIN_seq | 10 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 45.8333 | 25.0000 | 32.3466 | N/A | -2.5000 | 1.6667 | 20.8333 |
| oof | R_MARGIN_seq | 10 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 90.8333 | 87.0000 | 88.8659 | N/A | 0.0000 | 5.8333 | 3.8333 |
| oof | R_MARGIN_seq | 10 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 44.1667 | 34.0000 | 38.4071 | N/A | -1.6667 | 3.3333 | 10.1667 |
| oof | R_MARGIN_seq | 20 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 97.0833 | N/A | N/A | N/A | 0.4167 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 57.5000 | N/A | N/A | N/A | 0.8333 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 95.4167 | N/A | N/A | N/A | 0.8333 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 55.8333 | N/A | N/A | N/A | -1.2500 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 95.4167 | 83.5000 | 89.0602 | N/A | 0.4167 | 1.6667 | 11.9167 |
| oof | R_MARGIN_seq | 20 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 52.9167 | 35.7500 | 42.6715 | N/A | 0.8333 | 4.5833 | 17.1667 |
| oof | R_MARGIN_seq | 20 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 92.0833 | 85.7500 | 88.8038 | N/A | 0.8333 | 3.3333 | 6.3333 |
| oof | R_MARGIN_seq | 20 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 47.5000 | 34.0000 | 39.5757 | N/A | -1.2500 | 8.3333 | 13.5000 |
| oof | R_MARGIN_seq | 20 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 96.2500 | 72.5000 | 82.7033 | N/A | 0.4167 | 0.8333 | 23.7500 |
| oof | R_MARGIN_seq | 20 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 55.8333 | 27.5000 | 36.8500 | N/A | 0.8333 | 1.6667 | 28.3333 |
| oof | R_MARGIN_seq | 20 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 95.0000 | 81.2500 | 87.5760 | N/A | 0.8333 | 0.4167 | 13.7500 |
| oof | R_MARGIN_seq | 20 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 53.7500 | 26.2500 | 35.1311 | N/A | -1.2500 | 2.0833 | 27.5000 |
| oof | R_MARGIN_seq | 20 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 95.8333 | 84.7500 | 89.9508 | N/A | 0.4167 | 1.2500 | 11.0833 |
| oof | R_MARGIN_seq | 20 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 49.5833 | 37.5000 | 42.6748 | N/A | 0.8333 | 7.9167 | 12.0833 |
| oof | R_MARGIN_seq | 20 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 91.2500 | 84.6250 | 87.8125 | N/A | 0.8333 | 4.1667 | 6.6250 |
| oof | R_MARGIN_seq | 20 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 45.4167 | 34.5000 | 39.2035 | N/A | -1.2500 | 10.4167 | 10.9167 |
| oof | R_MARGIN_seq | 20 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 95.8333 | 79.5000 | 86.8641 | N/A | 0.4167 | 1.2500 | 16.3333 |
| oof | R_MARGIN_seq | 20 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 55.8333 | 32.0000 | 40.6806 | N/A | 0.8333 | 1.6667 | 23.8333 |
| oof | R_MARGIN_seq | 20 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 92.5000 | 87.0000 | 89.6539 | N/A | 0.8333 | 2.9167 | 5.5000 |
| oof | R_MARGIN_seq | 20 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 50.4167 | 31.5000 | 38.7694 | N/A | -1.2500 | 5.4167 | 18.9167 |
| oof | R_MARGIN_seq | 5 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 93.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 33.3333 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 88.3333 | N/A | N/A | N/A | -3.3333 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 33.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 90.0000 | 62.0000 | 73.4211 | N/A | 0.0000 | 3.3333 | 28.0000 |
| oof | R_MARGIN_seq | 5 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 26.6667 | 14.0000 | 18.3108 | N/A | -1.6667 | 6.6667 | 12.6667 |
| oof | R_MARGIN_seq | 5 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 81.6667 | 76.0000 | 78.6266 | N/A | -3.3333 | 6.6667 | 5.6667 |
| oof | R_MARGIN_seq | 5 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 28.3333 | 30.0000 | 29.1176 | N/A | 0.0000 | 5.0000 | 1.6667 |
| oof | R_MARGIN_seq | 5 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 93.3333 | 40.0000 | 55.2608 | N/A | 0.0000 | 0.0000 | 53.3333 |
| oof | R_MARGIN_seq | 5 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 33.3333 | 45.0000 | 38.1818 | N/A | -1.6667 | 0.0000 | 11.6667 |
| oof | R_MARGIN_seq | 5 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 86.6667 | 75.0000 | 80.1913 | N/A | -3.3333 | 1.6667 | 11.6667 |
| oof | R_MARGIN_seq | 5 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 31.6667 | 25.0000 | 27.5000 | N/A | 0.0000 | 1.6667 | 6.6667 |
| oof | R_MARGIN_seq | 5 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 90.0000 | 60.0000 | 71.9952 | N/A | 0.0000 | 3.3333 | 30.0000 |
| oof | R_MARGIN_seq | 5 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 20.0000 | 24.0000 | 21.4664 | N/A | -1.6667 | 13.3333 | 4.6667 |
| oof | R_MARGIN_seq | 5 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 80.0000 | 72.0000 | 75.6983 | N/A | -3.3333 | 8.3333 | 8.0000 |
| oof | R_MARGIN_seq | 5 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 25.0000 | 23.0000 | 23.7259 | N/A | 0.0000 | 8.3333 | 4.6667 |
| oof | R_MARGIN_seq | 5 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 88.3333 | 64.0000 | 74.1887 | N/A | 0.0000 | 5.0000 | 24.3333 |
| oof | R_MARGIN_seq | 5 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 31.6667 | 18.0000 | 22.5249 | N/A | -1.6667 | 1.6667 | 13.6667 |
| oof | R_MARGIN_seq | 5 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 83.3333 | 86.0000 | 84.6342 | N/A | -3.3333 | 5.0000 | 2.6667 |
| oof | R_MARGIN_seq | 5 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 28.3333 | 20.0000 | 23.3333 | N/A | 0.0000 | 5.0000 | 8.3333 |
| proxy | R0 | 1 | 0 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 0 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 0 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 0 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 10 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 86.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 27.7778 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 72.4074 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 25.2778 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 77.9630 | 57.5556 | 66.0681 | N/A | 0.0000 | 8.7037 | 20.4074 |
| proxy | R0 | 10 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 16.2963 | 16.5000 | 14.6198 | N/A | 0.0000 | 11.4815 | 8.3519 |
| proxy | R0 | 10 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 65.0926 | 57.2222 | 60.4127 | N/A | 0.0000 | 7.3148 | 11.9815 |
| proxy | R0 | 10 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 15.7407 | 16.4444 | 15.1739 | N/A | 0.0000 | 9.5370 | 6.2222 |
| proxy | R0 | 10 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 86.9444 | 22.7778 | 34.0903 | N/A | 0.0000 | -0.2778 | 64.1667 |
| proxy | R0 | 10 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 23.5185 | 18.0556 | 18.4267 | N/A | 0.0000 | 4.2593 | 10.6481 |
| proxy | R0 | 10 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 71.8519 | 40.8333 | 48.9143 | N/A | 0.0000 | 0.5556 | 31.2037 |
| proxy | R0 | 10 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 21.7593 | 20.5556 | 18.3454 | N/A | 0.0000 | 3.5185 | 12.1296 |
| proxy | R0 | 10 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 74.5370 | 48.0556 | 58.2718 | N/A | 0.0000 | 12.1296 | 26.4815 |
| proxy | R0 | 10 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 14.1667 | 16.9167 | 14.1893 | N/A | 0.0000 | 13.6111 | 8.3611 |
| proxy | R0 | 10 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 58.3333 | 49.3889 | 52.7587 | N/A | 0.0000 | 14.0741 | 13.8148 |
| proxy | R0 | 10 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 11.7593 | 15.5000 | 12.9723 | N/A | 0.0000 | 13.5185 | 4.4815 |
| proxy | R0 | 10 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 82.2222 | 54.1111 | 64.9709 | N/A | 0.0000 | 4.4444 | 28.1111 |
| proxy | R0 | 10 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 20.9259 | 15.1111 | 15.2597 | N/A | 0.0000 | 6.8519 | 11.6667 |
| proxy | R0 | 10 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 66.7593 | 60.8889 | 63.0969 | N/A | 0.0000 | 5.6481 | 10.4630 |
| proxy | R0 | 10 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 18.8889 | 15.3333 | 16.1501 | N/A | 0.0000 | 6.3889 | 6.5556 |
| proxy | R0 | 20 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 84.9342 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 29.8465 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 69.0132 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 26.0965 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 72.2807 | 52.2895 | 60.3003 | N/A | 0.0000 | 12.6535 | 20.2456 |
| proxy | R0 | 20 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 18.2675 | 16.8026 | 16.4183 | N/A | 0.0000 | 11.5789 | 6.7719 |
| proxy | R0 | 20 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 60.6579 | 58.5526 | 59.2824 | N/A | 0.0000 | 8.3553 | 7.0614 |
| proxy | R0 | 20 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 15.3289 | 16.7105 | 14.7419 | N/A | 0.0000 | 10.7675 | 7.5132 |
| proxy | R0 | 20 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 83.4868 | 23.4868 | 35.7343 | N/A | 0.0000 | 1.4474 | 60.0000 |
| proxy | R0 | 20 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 26.2939 | 18.0921 | 19.3971 | N/A | 0.0000 | 3.5526 | 13.7281 |
| proxy | R0 | 20 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 67.5658 | 55.9868 | 60.1251 | N/A | 0.0000 | 1.4474 | 15.9211 |
| proxy | R0 | 20 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 21.2281 | 18.3553 | 17.0572 | N/A | 0.0000 | 4.8684 | 11.6447 |
| proxy | R0 | 20 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 69.8026 | 48.5132 | 56.9408 | N/A | 0.0000 | 15.1316 | 21.4123 |
| proxy | R0 | 20 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 15.7237 | 15.8553 | 15.0344 | N/A | 0.0000 | 14.1228 | 6.0921 |
| proxy | R0 | 20 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 54.5614 | 51.2632 | 52.5007 | N/A | 0.0000 | 14.4518 | 7.7500 |
| proxy | R0 | 20 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 12.1930 | 13.9211 | 12.1910 | N/A | 0.0000 | 13.9035 | 5.4430 |
| proxy | R0 | 20 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 77.1053 | 54.6579 | 63.5566 | N/A | 0.0000 | 7.8289 | 22.4561 |
| proxy | R0 | 20 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 22.8728 | 17.0789 | 17.8391 | N/A | 0.0000 | 6.9737 | 10.3904 |
| proxy | R0 | 20 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 63.3114 | 60.2895 | 61.4108 | N/A | 0.0000 | 5.7018 | 7.9605 |
| proxy | R0 | 20 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 18.6404 | 16.1842 | 15.5847 | N/A | 0.0000 | 7.4561 | 8.6228 |
| proxy | R0 | 5 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 90.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 20.4167 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 66.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 25.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 80.4167 | 60.7500 | 68.4001 | N/A | 0.0000 | 10.4167 | 20.6667 |
| proxy | R0 | 5 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 14.5833 | 15.5000 | 14.5360 | N/A | 0.0000 | 5.8333 | 3.5833 |
| proxy | R0 | 5 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 60.4167 | 57.7500 | 58.5113 | N/A | 0.0000 | 6.2500 | 10.3333 |
| proxy | R0 | 5 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 13.7500 | 15.7500 | 13.6851 | N/A | 0.0000 | 12.0833 | 6.6667 |
| proxy | R0 | 5 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 90.0000 | 21.2500 | 31.9568 | N/A | 0.0000 | 0.8333 | 68.7500 |
| proxy | R0 | 5 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 20.0000 | 32.5000 | 24.2687 | N/A | 0.0000 | 0.4167 | 12.5000 |
| proxy | R0 | 5 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 66.6667 | 66.2500 | 65.3404 | N/A | 0.0000 | 0.0000 | 13.7500 |
| proxy | R0 | 5 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 20.4167 | 12.5000 | 11.8055 | N/A | 0.0000 | 5.4167 | 13.7500 |
| proxy | R0 | 5 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 77.5000 | 46.2500 | 57.3160 | N/A | 0.0000 | 13.3333 | 31.2500 |
| proxy | R0 | 5 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 14.1667 | 20.0000 | 16.1178 | N/A | 0.0000 | 6.2500 | 6.5000 |
| proxy | R0 | 5 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 50.8333 | 52.5000 | 51.4330 | N/A | 0.0000 | 15.8333 | 5.2500 |
| proxy | R0 | 5 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 11.2500 | 14.3750 | 11.7294 | N/A | 0.0000 | 14.5833 | 6.1250 |
| proxy | R0 | 5 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 85.0000 | 58.5000 | 68.9117 | N/A | 0.0000 | 5.8333 | 26.5000 |
| proxy | R0 | 5 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 19.5833 | 15.5000 | 15.8714 | N/A | 0.0000 | 0.8333 | 6.4167 |
| proxy | R0 | 5 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 62.0833 | 72.0000 | 66.0133 | N/A | 0.0000 | 4.5833 | 14.9167 |
| proxy | R0 | 5 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 16.2500 | 12.5000 | 13.4822 | N/A | 0.0000 | 9.5833 | 5.9167 |
| proxy | R_MARGIN_seq | 1 | 0 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 0 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 0 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 0 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 10 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 10 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 10 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 10 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 2 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 2 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 2 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 2 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 20 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 20 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 20 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 20 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 5 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 5 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 5 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 5 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 10 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 86.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 27.7778 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 72.4074 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 25.2778 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 83.3333 | 50.3333 | 62.6100 | N/A | 0.0000 | 3.3333 | 33.0000 |
| proxy | R_MARGIN_seq | 10 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 20.6481 | 13.6667 | 15.5018 | N/A | 0.0000 | 7.1296 | 8.0926 |
| proxy | R_MARGIN_seq | 10 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 69.0741 | 51.6667 | 58.6806 | N/A | 0.0000 | 3.3333 | 18.3704 |
| proxy | R_MARGIN_seq | 10 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 19.2593 | 15.3889 | 16.1812 | N/A | 0.0000 | 6.0185 | 7.2407 |
| proxy | R_MARGIN_seq | 10 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 87.0370 | 17.7778 | 27.7050 | N/A | 0.0000 | -0.3704 | 69.2593 |
| proxy | R_MARGIN_seq | 10 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 25.0926 | 14.1667 | 16.1532 | N/A | 0.0000 | 2.6852 | 12.2222 |
| proxy | R_MARGIN_seq | 10 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 73.1481 | 35.2778 | 44.0618 | N/A | 0.0000 | -0.7407 | 37.8704 |
| proxy | R_MARGIN_seq | 10 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 23.2407 | 16.6667 | 16.8793 | N/A | 0.0000 | 2.0370 | 11.2037 |
| proxy | R_MARGIN_seq | 10 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 81.2963 | 44.3611 | 57.2565 | N/A | 0.0000 | 5.3704 | 36.9352 |
| proxy | R_MARGIN_seq | 10 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 19.0741 | 14.9722 | 15.9181 | N/A | 0.0000 | 8.7037 | 7.2685 |
| proxy | R_MARGIN_seq | 10 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 65.9259 | 46.1944 | 53.6932 | N/A | 0.0000 | 6.4815 | 22.0648 |
| proxy | R_MARGIN_seq | 10 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 17.0370 | 14.2778 | 15.1175 | N/A | 0.0000 | 8.2407 | 5.2593 |
| proxy | R_MARGIN_seq | 10 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 85.6481 | 44.7778 | 58.5026 | N/A | 0.0000 | 1.0185 | 40.8704 |
| proxy | R_MARGIN_seq | 10 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 24.7222 | 11.7778 | 14.4322 | N/A | 0.0000 | 3.0556 | 13.5000 |
| proxy | R_MARGIN_seq | 10 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 70.3704 | 53.8889 | 60.4704 | N/A | 0.0000 | 2.0370 | 16.5926 |
| proxy | R_MARGIN_seq | 10 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 21.9444 | 13.3333 | 15.5954 | N/A | 0.0000 | 3.3333 | 9.7963 |
| proxy | R_MARGIN_seq | 20 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 84.9342 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 29.8465 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 69.0132 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 26.0965 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 78.5965 | 45.2368 | 56.9723 | N/A | 0.0000 | 6.3377 | 33.3596 |
| proxy | R_MARGIN_seq | 20 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 22.3465 | 14.5526 | 16.4500 | N/A | 0.0000 | 7.5000 | 10.3114 |
| proxy | R_MARGIN_seq | 20 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 64.7588 | 54.4737 | 58.7518 | N/A | 0.0000 | 4.2544 | 12.0658 |
| proxy | R_MARGIN_seq | 20 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 18.6184 | 14.8553 | 15.2719 | N/A | 0.0000 | 7.4781 | 8.1228 |
| proxy | R_MARGIN_seq | 20 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 84.5395 | 18.5526 | 29.5798 | N/A | 0.0000 | 0.3947 | 65.9868 |
| proxy | R_MARGIN_seq | 20 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 27.6096 | 12.5000 | 14.6697 | N/A | 0.0000 | 2.2368 | 18.3553 |
| proxy | R_MARGIN_seq | 20 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 68.6842 | 47.6316 | 54.6335 | N/A | 0.0000 | 0.3289 | 21.5789 |
| proxy | R_MARGIN_seq | 20 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 22.7193 | 14.5395 | 15.2297 | N/A | 0.0000 | 3.3772 | 12.4781 |
| proxy | R_MARGIN_seq | 20 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 77.0833 | 44.3816 | 55.9805 | N/A | 0.0000 | 7.8509 | 32.7018 |
| proxy | R_MARGIN_seq | 20 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 20.3289 | 14.4605 | 16.2312 | N/A | 0.0000 | 9.5175 | 7.8421 |
| proxy | R_MARGIN_seq | 20 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 61.3158 | 48.2895 | 53.6887 | N/A | 0.0000 | 7.6974 | 14.0000 |
| proxy | R_MARGIN_seq | 20 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 16.0965 | 13.0000 | 13.6097 | N/A | 0.0000 | 10.0000 | 6.5263 |
| proxy | R_MARGIN_seq | 20 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 81.5570 | 44.5263 | 57.0240 | N/A | 0.0000 | 3.3772 | 37.0307 |
| proxy | R_MARGIN_seq | 20 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 25.6579 | 13.9211 | 16.0415 | N/A | 0.0000 | 4.1886 | 13.9474 |
| proxy | R_MARGIN_seq | 20 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 66.3158 | 51.9474 | 57.7175 | N/A | 0.0000 | 2.6974 | 15.0263 |
| proxy | R_MARGIN_seq | 20 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 20.9211 | 14.2368 | 15.3873 | N/A | 0.0000 | 5.1754 | 9.1842 |
| proxy | R_MARGIN_seq | 5 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 90.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 20.4167 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 66.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 25.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 86.6667 | 52.7500 | 64.8308 | N/A | 0.0000 | 4.1667 | 34.0833 |
| proxy | R_MARGIN_seq | 5 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 16.2500 | 12.5000 | 13.3164 | N/A | 0.0000 | 4.1667 | 6.2500 |
| proxy | R_MARGIN_seq | 5 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 64.5833 | 55.5000 | 59.2539 | N/A | 0.0000 | 2.0833 | 10.7500 |
| proxy | R_MARGIN_seq | 5 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 17.5000 | 14.5000 | 15.0470 | N/A | 0.0000 | 8.3333 | 6.8333 |
| proxy | R_MARGIN_seq | 5 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 90.4167 | 13.7500 | 21.7112 | N/A | 0.0000 | 0.4167 | 76.6667 |
| proxy | R_MARGIN_seq | 5 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 20.0000 | 26.2500 | 20.2951 | N/A | 0.0000 | 0.4167 | 13.7500 |
| proxy | R_MARGIN_seq | 5 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 67.9167 | 62.5000 | 63.7977 | N/A | 0.0000 | -1.2500 | 14.5833 |
| proxy | R_MARGIN_seq | 5 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 23.7500 | 7.5000 | 9.2292 | N/A | 0.0000 | 2.0833 | 16.2500 |
| proxy | R_MARGIN_seq | 5 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 84.1667 | 42.3750 | 55.7662 | N/A | 0.0000 | 6.6667 | 41.7917 |
| proxy | R_MARGIN_seq | 5 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 17.0833 | 16.5000 | 16.2712 | N/A | 0.0000 | 3.3333 | 4.9167 |
| proxy | R_MARGIN_seq | 5 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 60.0000 | 51.2500 | 55.1119 | N/A | 0.0000 | 6.6667 | 8.7500 |
| proxy | R_MARGIN_seq | 5 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 15.8333 | 13.6250 | 13.8731 | N/A | 0.0000 | 10.0000 | 5.4583 |
| proxy | R_MARGIN_seq | 5 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 86.6667 | 48.0000 | 61.1600 | N/A | 0.0000 | 4.1667 | 38.6667 |
| proxy | R_MARGIN_seq | 5 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 20.0000 | 12.0000 | 13.7556 | N/A | 0.0000 | 0.4167 | 8.3333 |
| proxy | R_MARGIN_seq | 5 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 65.0000 | 66.5000 | 64.9121 | N/A | 0.0000 | 1.6667 | 11.8333 |
| proxy | R_MARGIN_seq | 5 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 20.4167 | 8.5000 | 11.6990 | N/A | 0.0000 | 5.4167 | 11.9167 |

by_model_cohort

| diagnostic | path | k | new_count | model_seed | cohort | measured_parents | A旧 | B旧 | C旧 | C新 | H | B−A | B−B0 | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | R0 | 1 | 0 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 0 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 0 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 0 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 10 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 2 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 20 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 5 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 10 | 0 | 2026092701 | rx1 | 2 | N/A | 74.1667 | 74.1667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 0 | 2026092701 | rx3 | 2 | N/A | 74.1667 | 74.1667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 0 | 2026092702 | rx1 | 2 | N/A | 74.1667 | 74.1667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 0 | 2026092702 | rx3 | 2 | N/A | 71.6667 | 71.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 10 | 10 | 2026092701 | rx1 | 2 | N/A | 74.1667 | 69.1667 | 60.0000 | 64.2542 | N/A | 0.0000 | 5.0000 | 9.1667 |
| oof | R0 | 10 | 10 | 2026092701 | rx3 | 2 | N/A | 74.1667 | 62.5000 | 53.0000 | 57.3456 | N/A | 0.0000 | 11.6667 | 9.5000 |
| oof | R0 | 10 | 10 | 2026092702 | rx1 | 2 | N/A | 74.1667 | 66.6667 | 60.0000 | 63.0264 | N/A | 0.0000 | 7.5000 | 8.0000 |
| oof | R0 | 10 | 10 | 2026092702 | rx3 | 2 | N/A | 71.6667 | 64.1667 | 58.5000 | 61.1966 | N/A | 0.0000 | 7.5000 | 5.6667 |
| oof | R0 | 10 | 2 | 2026092701 | rx1 | 2 | N/A | 74.1667 | 71.6667 | 42.5000 | 53.3346 | N/A | 0.0000 | 2.5000 | 29.1667 |
| oof | R0 | 10 | 2 | 2026092701 | rx3 | 2 | N/A | 74.1667 | 71.6667 | 62.5000 | 66.7321 | N/A | 0.0000 | 2.5000 | 9.1667 |
| oof | R0 | 10 | 2 | 2026092702 | rx1 | 2 | N/A | 74.1667 | 72.5000 | 40.0000 | 51.5508 | N/A | 0.0000 | 1.6667 | 32.5000 |
| oof | R0 | 10 | 2 | 2026092702 | rx3 | 2 | N/A | 71.6667 | 70.8333 | 62.5000 | 66.1876 | N/A | 0.0000 | 0.8333 | 10.0000 |
| oof | R0 | 10 | 20 | 2026092701 | rx1 | 2 | N/A | 74.1667 | 69.1667 | 58.0000 | 63.0424 | N/A | 0.0000 | 5.0000 | 11.1667 |
| oof | R0 | 10 | 20 | 2026092701 | rx3 | 2 | N/A | 74.1667 | 60.0000 | 53.5000 | 56.5602 | N/A | 0.0000 | 14.1667 | 6.5000 |
| oof | R0 | 10 | 20 | 2026092702 | rx1 | 2 | N/A | 74.1667 | 65.0000 | 57.0000 | 60.6081 | N/A | 0.0000 | 9.1667 | 8.3333 |
| oof | R0 | 10 | 20 | 2026092702 | rx3 | 2 | N/A | 71.6667 | 62.5000 | 57.5000 | 59.8865 | N/A | 0.0000 | 9.1667 | 5.0000 |
| oof | R0 | 10 | 5 | 2026092701 | rx1 | 2 | N/A | 74.1667 | 68.3333 | 55.0000 | 60.9457 | N/A | 0.0000 | 5.8333 | 13.3333 |
| oof | R0 | 10 | 5 | 2026092701 | rx3 | 2 | N/A | 74.1667 | 65.8333 | 61.0000 | 62.9865 | N/A | 0.0000 | 8.3333 | 6.5000 |
| oof | R0 | 10 | 5 | 2026092702 | rx1 | 2 | N/A | 74.1667 | 70.0000 | 51.0000 | 58.9234 | N/A | 0.0000 | 4.1667 | 19.0000 |
| oof | R0 | 10 | 5 | 2026092702 | rx3 | 2 | N/A | 71.6667 | 67.5000 | 59.0000 | 62.7786 | N/A | 0.0000 | 4.1667 | 8.5000 |
| oof | R0 | 20 | 0 | 2026092701 | rx1 | 2 | N/A | 76.6667 | 76.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 0 | 2026092701 | rx3 | 2 | N/A | 75.4167 | 75.4167 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 0 | 2026092702 | rx1 | 2 | N/A | 76.6667 | 76.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 0 | 2026092702 | rx3 | 2 | N/A | 76.2500 | 76.2500 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 20 | 10 | 2026092701 | rx1 | 2 | N/A | 76.6667 | 67.5000 | 65.2500 | 66.3522 | N/A | 0.0000 | 9.1667 | 2.2500 |
| oof | R0 | 20 | 10 | 2026092701 | rx3 | 2 | N/A | 75.4167 | 67.5000 | 64.2500 | 65.7574 | N/A | 0.0000 | 7.9167 | 3.2500 |
| oof | R0 | 20 | 10 | 2026092702 | rx1 | 2 | N/A | 76.6667 | 69.5833 | 64.0000 | 66.6659 | N/A | 0.0000 | 7.0833 | 5.5833 |
| oof | R0 | 20 | 10 | 2026092702 | rx3 | 2 | N/A | 76.2500 | 67.5000 | 61.0000 | 64.0353 | N/A | 0.0000 | 8.7500 | 6.5000 |
| oof | R0 | 20 | 2 | 2026092701 | rx1 | 2 | N/A | 76.6667 | 75.4167 | 47.5000 | 58.1423 | N/A | 0.0000 | 1.2500 | 27.9167 |
| oof | R0 | 20 | 2 | 2026092701 | rx3 | 2 | N/A | 75.4167 | 73.7500 | 58.7500 | 65.1061 | N/A | 0.0000 | 1.6667 | 15.0000 |
| oof | R0 | 20 | 2 | 2026092702 | rx1 | 2 | N/A | 76.6667 | 75.4167 | 53.7500 | 62.6636 | N/A | 0.0000 | 1.2500 | 21.6667 |
| oof | R0 | 20 | 2 | 2026092702 | rx3 | 2 | N/A | 76.2500 | 74.1667 | 52.5000 | 60.4498 | N/A | 0.0000 | 2.0833 | 21.6667 |
| oof | R0 | 20 | 20 | 2026092701 | rx1 | 2 | N/A | 76.6667 | 65.4167 | 64.8750 | 65.0635 | N/A | 0.0000 | 11.2500 | 4.4583 |
| oof | R0 | 20 | 20 | 2026092701 | rx3 | 2 | N/A | 75.4167 | 63.7500 | 61.3750 | 62.5081 | N/A | 0.0000 | 11.6667 | 2.3750 |
| oof | R0 | 20 | 20 | 2026092702 | rx1 | 2 | N/A | 76.6667 | 67.5000 | 64.5000 | 65.9267 | N/A | 0.0000 | 9.1667 | 4.0833 |
| oof | R0 | 20 | 20 | 2026092702 | rx3 | 2 | N/A | 76.2500 | 63.7500 | 60.8750 | 62.2781 | N/A | 0.0000 | 12.5000 | 2.8750 |
| oof | R0 | 20 | 5 | 2026092701 | rx1 | 2 | N/A | 76.6667 | 69.1667 | 61.5000 | 65.0387 | N/A | 0.0000 | 7.5000 | 7.6667 |
| oof | R0 | 20 | 5 | 2026092701 | rx3 | 2 | N/A | 75.4167 | 69.1667 | 61.5000 | 64.7081 | N/A | 0.0000 | 6.2500 | 7.6667 |
| oof | R0 | 20 | 5 | 2026092702 | rx1 | 2 | N/A | 76.6667 | 71.6667 | 62.0000 | 66.4230 | N/A | 0.0000 | 5.0000 | 9.6667 |
| oof | R0 | 20 | 5 | 2026092702 | rx3 | 2 | N/A | 76.2500 | 70.4167 | 60.0000 | 64.1693 | N/A | 0.0000 | 5.8333 | 10.4167 |
| oof | R0 | 5 | 0 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 65.0000 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 0 | 2026092701 | rx3 | 2 | N/A | 61.6667 | 61.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 0 | 2026092702 | rx1 | 2 | N/A | 63.3333 | 63.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 0 | 2026092702 | rx3 | 2 | N/A | 63.3333 | 63.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R0 | 5 | 10 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 51.6667 | 40.0000 | 45.0880 | N/A | 0.0000 | 13.3333 | 11.6667 |
| oof | R0 | 5 | 10 | 2026092701 | rx3 | 2 | N/A | 61.6667 | 53.3333 | 52.0000 | 52.4762 | N/A | 0.0000 | 8.3333 | 6.0000 |
| oof | R0 | 5 | 10 | 2026092702 | rx1 | 2 | N/A | 63.3333 | 51.6667 | 42.0000 | 46.2788 | N/A | 0.0000 | 11.6667 | 9.6667 |
| oof | R0 | 5 | 10 | 2026092702 | rx3 | 2 | N/A | 63.3333 | 56.6667 | 59.0000 | 57.6013 | N/A | 0.0000 | 6.6667 | 5.0000 |
| oof | R0 | 5 | 2 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 65.0000 | 45.0000 | 51.6886 | N/A | 0.0000 | 0.0000 | 23.3333 |
| oof | R0 | 5 | 2 | 2026092701 | rx3 | 2 | N/A | 61.6667 | 56.6667 | 50.0000 | 52.8411 | N/A | 0.0000 | 5.0000 | 10.0000 |
| oof | R0 | 5 | 2 | 2026092702 | rx1 | 2 | N/A | 63.3333 | 63.3333 | 50.0000 | 52.5581 | N/A | 0.0000 | 0.0000 | 30.0000 |
| oof | R0 | 5 | 2 | 2026092702 | rx3 | 2 | N/A | 63.3333 | 60.0000 | 55.0000 | 57.0000 | N/A | 0.0000 | 3.3333 | 5.0000 |
| oof | R0 | 5 | 20 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 50.0000 | 43.5000 | 45.1891 | N/A | 0.0000 | 15.0000 | 16.8333 |
| oof | R0 | 5 | 20 | 2026092701 | rx3 | 2 | N/A | 61.6667 | 43.3333 | 44.5000 | 43.5504 | N/A | 0.0000 | 18.3333 | 5.5000 |
| oof | R0 | 5 | 20 | 2026092702 | rx1 | 2 | N/A | 63.3333 | 50.0000 | 45.0000 | 45.5980 | N/A | 0.0000 | 13.3333 | 18.3333 |
| oof | R0 | 5 | 20 | 2026092702 | rx3 | 2 | N/A | 63.3333 | 51.6667 | 52.0000 | 51.8174 | N/A | 0.0000 | 11.6667 | 1.3333 |
| oof | R0 | 5 | 5 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 55.0000 | 46.0000 | 50.0552 | N/A | 0.0000 | 10.0000 | 9.0000 |
| oof | R0 | 5 | 5 | 2026092701 | rx3 | 2 | N/A | 61.6667 | 53.3333 | 50.0000 | 51.3242 | N/A | 0.0000 | 8.3333 | 4.0000 |
| oof | R0 | 5 | 5 | 2026092702 | rx1 | 2 | N/A | 63.3333 | 56.6667 | 46.0000 | 50.7350 | N/A | 0.0000 | 6.6667 | 10.6667 |
| oof | R0 | 5 | 5 | 2026092702 | rx3 | 2 | N/A | 63.3333 | 53.3333 | 58.0000 | 55.5572 | N/A | 0.0000 | 10.0000 | 4.6667 |
| oof | R_MARGIN_seq | 1 | 0 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 0 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 0 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 0 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 10 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 10 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 10 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 10 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 2 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 2 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 2 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 2 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 20 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 20 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 20 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 20 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 5 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 5 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 5 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 1 | 5 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_MARGIN_seq | 10 | 0 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 71.6667 | N/A | N/A | N/A | -2.5000 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 0 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 70.8333 | N/A | N/A | N/A | -3.3333 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 0 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 72.5000 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 0 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 73.3333 | N/A | N/A | N/A | 1.6667 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 10 | 10 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 70.0000 | 55.5000 | 61.8453 | N/A | -2.5000 | 1.6667 | 14.5000 |
| oof | R_MARGIN_seq | 10 | 10 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 68.3333 | 53.0000 | 59.5507 | N/A | -3.3333 | 2.5000 | 15.3333 |
| oof | R_MARGIN_seq | 10 | 10 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 70.0000 | 55.5000 | 61.8453 | N/A | -1.6667 | 2.5000 | 14.5000 |
| oof | R_MARGIN_seq | 10 | 10 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 67.5000 | 58.5000 | 62.6658 | N/A | 1.6667 | 5.8333 | 9.0000 |
| oof | R_MARGIN_seq | 10 | 2 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 70.8333 | 35.0000 | 46.8007 | N/A | -2.5000 | 0.8333 | 35.8333 |
| oof | R_MARGIN_seq | 10 | 2 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 70.8333 | 60.0000 | 64.8209 | N/A | -3.3333 | 0.0000 | 10.8333 |
| oof | R_MARGIN_seq | 10 | 2 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 73.3333 | 40.0000 | 51.7045 | N/A | -1.6667 | -0.8333 | 33.3333 |
| oof | R_MARGIN_seq | 10 | 2 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 70.8333 | 65.0000 | 67.3264 | N/A | 1.6667 | 2.5000 | 12.5000 |
| oof | R_MARGIN_seq | 10 | 20 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 69.1667 | 54.5000 | 60.9555 | N/A | -2.5000 | 2.5000 | 14.6667 |
| oof | R_MARGIN_seq | 10 | 20 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 66.6667 | 51.2500 | 57.8320 | N/A | -3.3333 | 4.1667 | 15.4167 |
| oof | R_MARGIN_seq | 10 | 20 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 68.3333 | 55.2500 | 61.0953 | N/A | -1.6667 | 4.1667 | 13.0833 |
| oof | R_MARGIN_seq | 10 | 20 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 67.5000 | 56.7500 | 61.6367 | N/A | 1.6667 | 5.8333 | 10.7500 |
| oof | R_MARGIN_seq | 10 | 5 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 69.1667 | 47.0000 | 55.6627 | N/A | -2.5000 | 2.5000 | 22.1667 |
| oof | R_MARGIN_seq | 10 | 5 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 66.6667 | 60.0000 | 62.9015 | N/A | -3.3333 | 4.1667 | 6.6667 |
| oof | R_MARGIN_seq | 10 | 5 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 73.3333 | 49.0000 | 58.5487 | N/A | -1.6667 | -0.8333 | 24.3333 |
| oof | R_MARGIN_seq | 10 | 5 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 68.3333 | 61.0000 | 64.3715 | N/A | 1.6667 | 5.0000 | 7.3333 |
| oof | R_MARGIN_seq | 20 | 0 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 77.5000 | N/A | N/A | N/A | 0.8333 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 0 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 75.8333 | N/A | N/A | N/A | 0.4167 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 0 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 77.0833 | N/A | N/A | N/A | 0.4167 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 0 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 75.4167 | N/A | N/A | N/A | -0.8333 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 20 | 10 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 74.1667 | 59.2500 | 65.6475 | N/A | 0.8333 | 3.3333 | 14.9167 |
| oof | R_MARGIN_seq | 20 | 10 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 69.5833 | 61.0000 | 64.8884 | N/A | 0.4167 | 6.2500 | 8.5833 |
| oof | R_MARGIN_seq | 20 | 10 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 74.1667 | 60.0000 | 66.0842 | N/A | 0.4167 | 2.9167 | 14.1667 |
| oof | R_MARGIN_seq | 20 | 10 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 70.0000 | 58.7500 | 63.4911 | N/A | -0.8333 | 5.4167 | 11.2500 |
| oof | R_MARGIN_seq | 20 | 2 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 75.8333 | 50.0000 | 59.6998 | N/A | 0.8333 | 1.6667 | 25.8333 |
| oof | R_MARGIN_seq | 20 | 2 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 74.1667 | 56.2500 | 63.2810 | N/A | 0.4167 | 1.6667 | 17.9167 |
| oof | R_MARGIN_seq | 20 | 2 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 76.2500 | 50.0000 | 59.8536 | N/A | 0.4167 | 0.8333 | 26.2500 |
| oof | R_MARGIN_seq | 20 | 2 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 74.5833 | 51.2500 | 59.4261 | N/A | -0.8333 | 0.8333 | 23.3333 |
| oof | R_MARGIN_seq | 20 | 20 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 71.2500 | 61.2500 | 65.8312 | N/A | 0.8333 | 6.2500 | 10.0000 |
| oof | R_MARGIN_seq | 20 | 20 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 67.5000 | 59.5000 | 63.1477 | N/A | 0.4167 | 8.3333 | 8.0000 |
| oof | R_MARGIN_seq | 20 | 20 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 74.1667 | 61.0000 | 66.7944 | N/A | 0.4167 | 2.9167 | 13.1667 |
| oof | R_MARGIN_seq | 20 | 20 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 69.1667 | 59.6250 | 63.8684 | N/A | -0.8333 | 6.2500 | 9.5417 |
| oof | R_MARGIN_seq | 20 | 5 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 75.4167 | 53.5000 | 62.2116 | N/A | 0.8333 | 2.0833 | 21.9167 |
| oof | R_MARGIN_seq | 20 | 5 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 72.0833 | 61.0000 | 65.4959 | N/A | 0.4167 | 3.7500 | 11.0833 |
| oof | R_MARGIN_seq | 20 | 5 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 76.2500 | 58.0000 | 65.3331 | N/A | 0.4167 | 0.8333 | 18.2500 |
| oof | R_MARGIN_seq | 20 | 5 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 70.8333 | 57.5000 | 62.9274 | N/A | -0.8333 | 4.5833 | 13.3333 |
| oof | R_MARGIN_seq | 5 | 0 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 65.0000 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 0 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 63.3333 | N/A | N/A | N/A | 1.6667 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 0 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 61.6667 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 0 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 58.3333 | N/A | N/A | N/A | -5.0000 | 0.0000 | N/A |
| oof | R_MARGIN_seq | 5 | 10 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 61.6667 | 39.0000 | 47.5213 | N/A | 0.0000 | 3.3333 | 22.6667 |
| oof | R_MARGIN_seq | 5 | 10 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 56.6667 | 51.0000 | 53.6266 | N/A | 1.6667 | 6.6667 | 5.6667 |
| oof | R_MARGIN_seq | 5 | 10 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 55.0000 | 37.0000 | 44.2105 | N/A | -1.6667 | 6.6667 | 18.0000 |
| oof | R_MARGIN_seq | 5 | 10 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 53.3333 | 55.0000 | 54.1176 | N/A | -5.0000 | 5.0000 | 1.6667 |
| oof | R_MARGIN_seq | 5 | 2 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 63.3333 | 50.0000 | 52.5581 | N/A | 0.0000 | 1.6667 | 30.0000 |
| oof | R_MARGIN_seq | 5 | 2 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 61.6667 | 45.0000 | 51.8750 | N/A | 1.6667 | 1.6667 | 16.6667 |
| oof | R_MARGIN_seq | 5 | 2 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 63.3333 | 35.0000 | 40.8845 | N/A | -1.6667 | -1.6667 | 35.0000 |
| oof | R_MARGIN_seq | 5 | 2 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 56.6667 | 55.0000 | 55.8163 | N/A | -5.0000 | 1.6667 | 1.6667 |
| oof | R_MARGIN_seq | 5 | 20 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 58.3333 | 43.5000 | 49.5222 | N/A | 0.0000 | 6.6667 | 14.8333 |
| oof | R_MARGIN_seq | 5 | 20 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 53.3333 | 43.5000 | 47.8912 | N/A | 1.6667 | 10.0000 | 9.8333 |
| oof | R_MARGIN_seq | 5 | 20 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 51.6667 | 40.5000 | 43.9395 | N/A | -1.6667 | 10.0000 | 19.8333 |
| oof | R_MARGIN_seq | 5 | 20 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 51.6667 | 51.5000 | 51.5330 | N/A | -5.0000 | 6.6667 | 2.8333 |
| oof | R_MARGIN_seq | 5 | 5 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 61.6667 | 46.0000 | 52.6877 | N/A | 0.0000 | 3.3333 | 15.6667 |
| oof | R_MARGIN_seq | 5 | 5 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 56.6667 | 54.0000 | 55.1660 | N/A | 1.6667 | 6.6667 | 3.3333 |
| oof | R_MARGIN_seq | 5 | 5 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 58.3333 | 36.0000 | 44.0260 | N/A | -1.6667 | 3.3333 | 22.3333 |
| oof | R_MARGIN_seq | 5 | 5 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 55.0000 | 52.0000 | 52.8016 | N/A | -5.0000 | 3.3333 | 7.6667 |
| proxy | R0 | 1 | 0 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 0 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 0 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 0 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 10 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 2 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 20 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 5 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 10 | 0 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 58.1481 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 0 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 49.5370 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 0 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 56.2963 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 0 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 48.1481 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 10 | 10 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 47.3148 | 36.2778 | 40.0132 | N/A | 0.0000 | 10.8333 | 15.0370 |
| proxy | R0 | 10 | 10 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 40.7407 | 37.1667 | 38.1006 | N/A | 0.0000 | 8.7963 | 9.3148 |
| proxy | R0 | 10 | 10 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 46.9444 | 37.7778 | 40.6747 | N/A | 0.0000 | 9.3519 | 13.7222 |
| proxy | R0 | 10 | 10 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 40.0926 | 36.5000 | 37.4861 | N/A | 0.0000 | 8.0556 | 8.8889 |
| proxy | R0 | 10 | 2 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 56.2963 | 20.0000 | 25.4298 | N/A | 0.0000 | 1.8519 | 39.6296 |
| proxy | R0 | 10 | 2 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 47.5926 | 32.2222 | 34.8752 | N/A | 0.0000 | 1.9444 | 20.9259 |
| proxy | R0 | 10 | 2 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 54.1667 | 20.8333 | 27.0872 | N/A | 0.0000 | 2.1296 | 35.1852 |
| proxy | R0 | 10 | 2 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 46.0185 | 29.1667 | 32.3845 | N/A | 0.0000 | 2.1296 | 22.4074 |
| proxy | R0 | 10 | 20 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 44.5370 | 31.7222 | 35.7715 | N/A | 0.0000 | 13.6111 | 18.0370 |
| proxy | R0 | 10 | 20 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 35.8333 | 32.2500 | 33.2682 | N/A | 0.0000 | 13.7037 | 8.7685 |
| proxy | R0 | 10 | 20 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 44.1667 | 33.2500 | 36.6895 | N/A | 0.0000 | 12.1296 | 16.8056 |
| proxy | R0 | 10 | 20 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 34.2593 | 32.6389 | 32.4628 | N/A | 0.0000 | 13.8889 | 9.5278 |
| proxy | R0 | 10 | 5 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 52.1296 | 33.0000 | 39.2443 | N/A | 0.0000 | 6.0185 | 21.5000 |
| proxy | R0 | 10 | 5 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 43.4259 | 40.0000 | 40.8612 | N/A | 0.0000 | 6.1111 | 8.6111 |
| proxy | R0 | 10 | 5 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 51.0185 | 36.2222 | 40.9863 | N/A | 0.0000 | 5.2778 | 18.2778 |
| proxy | R0 | 10 | 5 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 42.2222 | 36.2222 | 38.3858 | N/A | 0.0000 | 5.9259 | 8.4074 |
| proxy | R0 | 20 | 0 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 57.3904 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 0 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 47.3465 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 0 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 57.3904 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 0 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 47.7632 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 20 | 10 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 44.3421 | 33.8026 | 37.4966 | N/A | 0.0000 | 13.0482 | 13.4342 |
| proxy | R0 | 20 | 10 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 38.4649 | 37.6579 | 37.3332 | N/A | 0.0000 | 8.8816 | 7.2281 |
| proxy | R0 | 20 | 10 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 46.2061 | 35.2895 | 39.2221 | N/A | 0.0000 | 11.1842 | 13.5833 |
| proxy | R0 | 20 | 10 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 37.5219 | 37.6053 | 36.6911 | N/A | 0.0000 | 10.2412 | 7.3465 |
| proxy | R0 | 20 | 2 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 54.6272 | 21.2500 | 27.9260 | N/A | 0.0000 | 2.7632 | 36.4474 |
| proxy | R0 | 20 | 2 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 44.6053 | 36.8421 | 38.6104 | N/A | 0.0000 | 2.7412 | 14.2544 |
| proxy | R0 | 20 | 2 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 55.1535 | 20.3289 | 27.2055 | N/A | 0.0000 | 2.2368 | 37.2807 |
| proxy | R0 | 20 | 2 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 44.1886 | 37.5000 | 38.5719 | N/A | 0.0000 | 3.5746 | 13.3114 |
| proxy | R0 | 20 | 20 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 41.9956 | 31.9211 | 35.5309 | N/A | 0.0000 | 15.3947 | 13.3640 |
| proxy | R0 | 20 | 20 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 33.9254 | 32.6513 | 32.6245 | N/A | 0.0000 | 13.4211 | 7.1469 |
| proxy | R0 | 20 | 20 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 43.5307 | 32.4474 | 36.4443 | N/A | 0.0000 | 13.8596 | 14.1404 |
| proxy | R0 | 20 | 20 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 32.8289 | 32.5329 | 32.0672 | N/A | 0.0000 | 14.9342 | 6.0461 |
| proxy | R0 | 20 | 5 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 49.4956 | 34.6579 | 39.6697 | N/A | 0.0000 | 7.8947 | 17.2851 |
| proxy | R0 | 20 | 5 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 41.2500 | 37.5526 | 38.3645 | N/A | 0.0000 | 6.0965 | 8.1623 |
| proxy | R0 | 20 | 5 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 50.4825 | 37.0789 | 41.7260 | N/A | 0.0000 | 6.9079 | 15.5614 |
| proxy | R0 | 20 | 5 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 40.7018 | 38.9211 | 38.6311 | N/A | 0.0000 | 7.0614 | 8.4211 |
| proxy | R0 | 5 | 0 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 54.5833 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 0 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 45.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 0 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 56.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 0 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 46.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R0 | 5 | 10 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 46.6667 | 38.7500 | 41.6564 | N/A | 0.0000 | 7.9167 | 11.4167 |
| proxy | R0 | 5 | 10 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 35.8333 | 37.0000 | 35.4072 | N/A | 0.0000 | 10.0000 | 9.1667 |
| proxy | R0 | 5 | 10 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 48.3333 | 37.5000 | 41.2797 | N/A | 0.0000 | 8.3333 | 12.8333 |
| proxy | R0 | 5 | 10 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 38.3333 | 36.5000 | 36.7893 | N/A | 0.0000 | 8.3333 | 7.8333 |
| proxy | R0 | 5 | 2 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 54.1667 | 25.0000 | 26.4204 | N/A | 0.0000 | 0.4167 | 40.0000 |
| proxy | R0 | 5 | 2 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 41.6667 | 37.5000 | 37.1785 | N/A | 0.0000 | 4.1667 | 12.5000 |
| proxy | R0 | 5 | 2 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 55.8333 | 28.7500 | 29.8051 | N/A | 0.0000 | 0.8333 | 41.2500 |
| proxy | R0 | 5 | 2 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 45.4167 | 41.2500 | 39.9673 | N/A | 0.0000 | 1.2500 | 15.0000 |
| proxy | R0 | 5 | 20 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 45.8333 | 33.1250 | 36.6343 | N/A | 0.0000 | 8.7500 | 18.7083 |
| proxy | R0 | 5 | 20 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 31.6667 | 32.3750 | 31.1482 | N/A | 0.0000 | 14.1667 | 6.2917 |
| proxy | R0 | 5 | 20 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 45.8333 | 33.1250 | 36.7996 | N/A | 0.0000 | 10.8333 | 19.0417 |
| proxy | R0 | 5 | 20 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 30.4167 | 34.5000 | 32.0143 | N/A | 0.0000 | 16.2500 | 5.0833 |
| proxy | R0 | 5 | 5 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 50.4167 | 37.0000 | 41.9536 | N/A | 0.0000 | 4.1667 | 13.7500 |
| proxy | R0 | 5 | 5 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 38.3333 | 41.5000 | 39.1207 | N/A | 0.0000 | 7.5000 | 9.6667 |
| proxy | R0 | 5 | 5 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 54.1667 | 37.0000 | 42.8295 | N/A | 0.0000 | 2.5000 | 19.1667 |
| proxy | R0 | 5 | 5 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 40.0000 | 43.0000 | 40.3748 | N/A | 0.0000 | 6.6667 | 11.1667 |
| proxy | R_MARGIN_seq | 1 | 0 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 0 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 0 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 0 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 10 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 10 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 10 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 10 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 2 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 2 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 2 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 2 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 20 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 20 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 20 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 20 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 5 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 5 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 5 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 1 | 5 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_MARGIN_seq | 10 | 0 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 58.1481 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 0 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 49.5370 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 0 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 56.2963 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 0 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 48.1481 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 10 | 10 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 52.5000 | 31.0000 | 38.5275 | N/A | 0.0000 | 5.6481 | 21.7963 |
| proxy | R_MARGIN_seq | 10 | 10 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 44.6296 | 33.5000 | 37.5679 | N/A | 0.0000 | 4.9074 | 13.4259 |
| proxy | R_MARGIN_seq | 10 | 10 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 51.4815 | 33.0000 | 39.5843 | N/A | 0.0000 | 4.8148 | 19.2963 |
| proxy | R_MARGIN_seq | 10 | 10 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 43.7037 | 33.5556 | 37.2939 | N/A | 0.0000 | 4.4444 | 12.1852 |
| proxy | R_MARGIN_seq | 10 | 2 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 56.7593 | 14.7222 | 19.9145 | N/A | 0.0000 | 1.3889 | 43.1481 |
| proxy | R_MARGIN_seq | 10 | 2 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 48.1481 | 26.9444 | 31.3964 | N/A | 0.0000 | 1.3889 | 24.1667 |
| proxy | R_MARGIN_seq | 10 | 2 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 55.3704 | 17.2222 | 23.9437 | N/A | 0.0000 | 0.9259 | 38.3333 |
| proxy | R_MARGIN_seq | 10 | 2 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 48.2407 | 25.0000 | 29.5447 | N/A | 0.0000 | -0.0926 | 24.9074 |
| proxy | R_MARGIN_seq | 10 | 20 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 50.5556 | 29.1111 | 36.2431 | N/A | 0.0000 | 7.5926 | 22.9630 |
| proxy | R_MARGIN_seq | 10 | 20 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 42.2222 | 29.9722 | 34.6036 | N/A | 0.0000 | 7.3148 | 14.0093 |
| proxy | R_MARGIN_seq | 10 | 20 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 49.8148 | 30.2222 | 36.9316 | N/A | 0.0000 | 6.4815 | 21.2407 |
| proxy | R_MARGIN_seq | 10 | 20 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 40.7407 | 30.5000 | 34.2072 | N/A | 0.0000 | 7.4074 | 13.3148 |
| proxy | R_MARGIN_seq | 10 | 5 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 55.6481 | 26.6667 | 35.3025 | N/A | 0.0000 | 2.5000 | 29.1296 |
| proxy | R_MARGIN_seq | 10 | 5 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 46.4815 | 35.0000 | 38.9933 | N/A | 0.0000 | 3.0556 | 12.4444 |
| proxy | R_MARGIN_seq | 10 | 5 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 54.7222 | 29.8889 | 37.6324 | N/A | 0.0000 | 1.5741 | 25.2407 |
| proxy | R_MARGIN_seq | 10 | 5 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 45.8333 | 32.2222 | 37.0726 | N/A | 0.0000 | 2.3148 | 13.9444 |
| proxy | R_MARGIN_seq | 20 | 0 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 57.3904 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 0 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 47.3465 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 0 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 57.3904 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 0 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 47.7632 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 20 | 10 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 49.7368 | 29.0921 | 35.8703 | N/A | 0.0000 | 7.6535 | 21.8202 |
| proxy | R_MARGIN_seq | 20 | 10 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 41.7325 | 34.6579 | 37.0097 | N/A | 0.0000 | 5.6140 | 10.2588 |
| proxy | R_MARGIN_seq | 20 | 10 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 51.2061 | 30.6974 | 37.5520 | N/A | 0.0000 | 6.1842 | 21.8509 |
| proxy | R_MARGIN_seq | 20 | 10 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 41.6447 | 34.6711 | 37.0141 | N/A | 0.0000 | 6.1184 | 9.9298 |
| proxy | R_MARGIN_seq | 20 | 2 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 55.8333 | 15.1974 | 21.6159 | N/A | 0.0000 | 1.5570 | 42.2588 |
| proxy | R_MARGIN_seq | 20 | 2 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 45.7018 | 31.1184 | 35.1818 | N/A | 0.0000 | 1.6447 | 16.4693 |
| proxy | R_MARGIN_seq | 20 | 2 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 56.3158 | 15.8553 | 22.6336 | N/A | 0.0000 | 1.0746 | 42.0833 |
| proxy | R_MARGIN_seq | 20 | 2 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 45.7018 | 31.0526 | 34.6814 | N/A | 0.0000 | 2.0614 | 17.5877 |
| proxy | R_MARGIN_seq | 20 | 20 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 48.1579 | 29.3026 | 35.8723 | N/A | 0.0000 | 9.2325 | 19.7588 |
| proxy | R_MARGIN_seq | 20 | 20 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 38.8377 | 30.6382 | 33.6526 | N/A | 0.0000 | 8.5088 | 10.5592 |
| proxy | R_MARGIN_seq | 20 | 20 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 49.2544 | 29.5395 | 36.3394 | N/A | 0.0000 | 8.1360 | 20.7851 |
| proxy | R_MARGIN_seq | 20 | 20 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 38.5746 | 30.6513 | 33.6458 | N/A | 0.0000 | 9.1886 | 9.9671 |
| proxy | R_MARGIN_seq | 20 | 5 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 53.0921 | 27.5789 | 34.9806 | N/A | 0.0000 | 4.2982 | 26.4868 |
| proxy | R_MARGIN_seq | 20 | 5 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 43.7719 | 32.5789 | 36.3369 | N/A | 0.0000 | 3.5746 | 12.4474 |
| proxy | R_MARGIN_seq | 20 | 5 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 54.1228 | 30.8684 | 38.0849 | N/A | 0.0000 | 3.2675 | 24.4912 |
| proxy | R_MARGIN_seq | 20 | 5 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 43.4649 | 33.6053 | 36.7679 | N/A | 0.0000 | 4.2982 | 11.7632 |
| proxy | R_MARGIN_seq | 5 | 0 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 54.5833 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 0 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 45.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 0 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 56.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 0 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 46.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_MARGIN_seq | 5 | 10 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 50.0000 | 32.7500 | 38.6764 | N/A | 0.0000 | 4.5833 | 18.4167 |
| proxy | R_MARGIN_seq | 5 | 10 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 39.5833 | 34.5000 | 36.2728 | N/A | 0.0000 | 6.2500 | 7.7500 |
| proxy | R_MARGIN_seq | 5 | 10 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 52.9167 | 32.5000 | 39.4708 | N/A | 0.0000 | 3.7500 | 21.9167 |
| proxy | R_MARGIN_seq | 5 | 10 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 42.5000 | 35.5000 | 38.0281 | N/A | 0.0000 | 4.1667 | 9.8333 |
| proxy | R_MARGIN_seq | 5 | 2 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 54.1667 | 18.7500 | 20.7090 | N/A | 0.0000 | 0.4167 | 44.5833 |
| proxy | R_MARGIN_seq | 5 | 2 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 43.7500 | 30.0000 | 33.2127 | N/A | 0.0000 | 2.0833 | 13.7500 |
| proxy | R_MARGIN_seq | 5 | 2 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 56.2500 | 21.2500 | 21.2974 | N/A | 0.0000 | 0.4167 | 45.8333 |
| proxy | R_MARGIN_seq | 5 | 2 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 47.9167 | 40.0000 | 39.8141 | N/A | 0.0000 | -1.2500 | 17.0833 |
| proxy | R_MARGIN_seq | 5 | 20 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 50.0000 | 29.5000 | 35.9519 | N/A | 0.0000 | 4.5833 | 22.3333 |
| proxy | R_MARGIN_seq | 5 | 20 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 37.0833 | 31.2500 | 33.4403 | N/A | 0.0000 | 8.7500 | 7.4167 |
| proxy | R_MARGIN_seq | 5 | 20 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 51.2500 | 29.3750 | 36.0855 | N/A | 0.0000 | 5.4167 | 24.3750 |
| proxy | R_MARGIN_seq | 5 | 20 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 38.7500 | 33.6250 | 35.5446 | N/A | 0.0000 | 7.9167 | 6.7917 |
| proxy | R_MARGIN_seq | 5 | 5 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 52.5000 | 30.0000 | 37.2540 | N/A | 0.0000 | 2.0833 | 22.8333 |
| proxy | R_MARGIN_seq | 5 | 5 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 40.8333 | 36.0000 | 36.9348 | N/A | 0.0000 | 5.0000 | 11.3333 |
| proxy | R_MARGIN_seq | 5 | 5 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 54.1667 | 30.0000 | 37.6615 | N/A | 0.0000 | 2.5000 | 24.1667 |
| proxy | R_MARGIN_seq | 5 | 5 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 44.5833 | 39.0000 | 39.6762 | N/A | 0.0000 | 2.0833 | 12.4167 |

模型与资源口径

目标仅为 pooled class-RMS CE。分类梯度为归档的 g_Z，不包含 +Z 近端梯度；坐标受 0.5 硬球约束，Armijo 使用投影后的实际 delta。
C 使用原始 Gaussian 核和带自由截距的凸 margin QP；A 与实际工作集因子的尝试/完成、完整约束扫描、谱诊断、RHS 和缓存分别按实际执行记录，不能套用固定因子数。
KKT 伴随只在工作集独立且严格互补的可微范围核对；非光滑或技术失败保留实际状态，不静默退回无约束头。完整 K/L 两端梯度传入 adapter。
全部物理旧 support 对全部注册类的间隔约束只作用于训练 support。正原间隔可保持其严格正确；负原间隔不保证每对赢家不变，也不保证未来 query 保旧。远处自由截距 b 仍影响完整分数。
两个 QP 峰值跨阶段/行取 MAX，其余实际执行计数取 SUM；factor guard计live scratch输入与Cholesky输出，不统一覆盖返回状态冻结时的布局/bytes瞬时复制、QP构建临时数组、BLAS workspace、归档或整个进程RSS。
以下保留所有实测资源字段；n²×RHS、数组字节和参数计数为数值代理，不是实测 FLOPs、设备峰值或完整部署包。未测项为 N/A；嵌套累计计时不可相加当墙钟。

| 资源字段 | 原值 |
|---|---|
| qp_resources | {"max_transitions": 4096, "max_factor_buffer_bytes": 167772160} |
| parent_wall_seconds_sum | 5715.596793881996 |
| base_fit_maximum_training_feature_bytes | 2143232 |
| base_fit_maximum_gram_bytes | 1059968 |
| base_fit_maximum_kernel_bytes | 1059968 |
| base_fit_maximum_coefficient_bytes | 75712 |
| base_fit_effective_degrees_of_freedom_seconds_sum | 0.3500220605637878 |
| base_fit_solve_seconds_sum | 0.47668890956265386 |
| base_fit_gram_seconds_sum | 52.19785044870514 |
| base_fit_fit_seconds_sum | 53.02627824997762 |
| base_fit_score_seconds_sum | 156.04874619898328 |
| base_fit_maximum_persistent_state_bytes | 2224800 |
| base_fit_fit_and_score_seconds_sum | 230.66740419031703 |
| margin_preparation_maximum_qp_max_factor_buffer_bytes | 167772160 |
| margin_preparation_preparation_seconds_sum | 549.6789269950823 |
| margin_preparation_maximum_retained_vector_record_bytes | 0 |
| margin_preparation_maximum_prepared_numeric_state_bytes | 31599200 |
| candidate_fit_maximum_qp_max_factor_buffer_bytes | 167772160 |
| candidate_fit_maximum_margin_qp_peak_factor_buffer_bytes | 2212352 |
| candidate_fit_maximum_margin_qp_peak_explicit_solve_temporary_bytes | 489216 |
| candidate_fit_fit_seconds_sum | 3275.441623888313 |
| candidate_fit_maximum_retained_vector_record_bytes | 0 |
| candidate_fit_maximum_prepared_numeric_state_bytes | 31599200 |
| candidate_fit_maximum_resident_numeric_state_bytes | 21351672 |
| candidate_fit_maximum_deployment_numeric_state_bytes | 5498976 |
| candidate_fit_maximum_persistent_state_bytes | 21351672 |
| candidate_fit_score_seconds_sum | 782.4958286680194 |
| objective_forward_seconds_sum | 931.1751513555791 |
| objective_backward_seconds_sum | 597.8796697496437 |
| final_head_forward_seconds_sum | 84.01718686881941 |
| margin_preparation_prior_score_seconds_sum | 297.29011160790105 |
| candidate_fit_maximum_max_forward_cache_bytes | 58871488 |
| candidate_fit_maximum_max_simultaneous_forward_cache_bytes | 89091368 |
| outer_residual_raw_distance_evaluation_count_sum | 407680 |
| outer_residual_raw_distance_pair_count_sum | 12305936 |
| outer_residual_reference_distance_evaluation_count_sum | 407680 |
| outer_residual_reference_distance_pair_count_sum | 5180448 |
| outer_residual_kernel_evaluation_count_sum | 382200 |
| outer_residual_kernel_pair_count_sum | 8695088 |
| outer_residual_adapter_physical_evaluation_count_sum | 382200 |
| outer_residual_dictionary_physical_evaluation_count_sum | 382200 |
| outer_residual_intercept_addition_count_sum | 5447400 |
| outer_residual_score_seconds_sum | 488.3507896382944 |
| outer_prior_raw_distance_evaluation_count_sum | 273280 |
| outer_prior_raw_distance_pair_count_sum | 3472608 |
| outer_prior_reference_distance_evaluation_count_sum | 273280 |
| outer_prior_reference_distance_pair_count_sum | 3472608 |
| outer_prior_kernel_evaluation_count_sum | 256200 |
| outer_prior_kernel_pair_count_sum | 2453664 |
| outer_prior_adapter_physical_evaluation_count_sum | 256200 |
| outer_prior_dictionary_physical_evaluation_count_sum | 256200 |
| outer_prior_intercept_addition_count_sum | 1537200 |
| outer_prior_score_seconds_sum | 290.9935948061029 |
| actual_counters | {"episodes": 160, "k1_episodes": 40, "oof_episodes": 120, "proxy_anchor_count": 1400, "sequence_paths": 1800, "baseline_head_fit_count": 3240, "baseline_factorization_count": 3240, "baseline_head_triangular_solve_count": 6480, "baseline_effective_df_triangular_solve_count": 6480, "baseline_triangular_solve_count": 12960, "head_fit_count": 17064, "factorization_count": 86837, "trained_margin_stage_count": 648, "diagnostic_fit_count": 0, "candidate_preparation_count": 3240, "candidate_stage_count": 3240, "final_score_evaluation_count": 6336, "final_score_physical_count": 764400, "ajlr_preparation_count": 3240, "latent_svd_count": 3240, "dictionary_physical_evaluation_count": 77168, "prepared_distance_evaluation_count": 8856, "original_distance_pair_count": 14492912, "prior_head_fit_count": 864, "prior_factorization_count": 864, "prior_triangular_solve_count": 1728, "prior_triangular_rhs_count": 12096, "prior_triangular_rhs_element_count": 376320, "prior_triangular_dense_work_unit_count": 15160320, "prior_intercept_fit_count": 864, "prior_intercept_addition_count": 1167168, "prior_score_evaluation_count": 3168, "prior_score_physical_count": 154208, "reference_distance_evaluation_count": 314248, "reference_distance_pair_count": 40441536, "raw_distance_evaluation_count": 310504, "raw_distance_pair_count": 63990128, "kernel_evaluation_count": 178616, "kernel_pair_count": 88386192, "adapter_physical_evaluation_count": 1036096, "ajlr_stage_count": 3240, "optimizer_steps": 2592, "optimizer_iterations": 2592, "inner_objective_evaluation_count": 3240, "inner_head_fit_count": 9720, "inner_factorization_count": 72052, "final_head_fit_count": 3240, "final_factorization_count": 10681, "head_triangular_solve_count": 263380, "head_triangular_rhs_count": 976084, "head_triangular_rhs_element_count": 67471510, "head_triangular_dense_work_unit_count": 9203612162, "intercept_fit_count": 12960, "intercept_addition_count": 11994656, "derivative_triangular_solve_count": 24556, "ce_adjoint_solve_count": 7776, "derivative_triangular_rhs_count": 238294, "derivative_triangular_rhs_element_count": 20433582, "derivative_triangular_dense_work_unit_count": 2825703170, "backward_evaluation_count": 2592, "accepted_trial_count": 2592, "rejected_trial_count": 0, "trial_count": 2592, "trial_attempt_count": 2592, "ajlr_forward_evaluation_count": 12960, "completed_factorization_count": 82733, "margin_qp_forward_transitions": 78109, "margin_qp_forward_full_constraint_scans": 83869, "margin_qp_forward_spectral_checks": 5760, "margin_qp_forward_compact_snapshot_rebuilds": 78109, "margin_qp_forward_compact_snapshot_dense_work_units": 3608687880, "margin_qp_forward_spectral_cubic_dimension_units": 8434172080, "margin_qp_forward_independence_checks": 34861, "margin_qp_forward_factorization_attempts": 75533, "margin_qp_forward_factorizations_completed": 75533, "margin_qp_forward_condition_estimation_calls": 75533, "margin_qp_forward_triangular_calls": 248980, "margin_qp_forward_triangular_rhs_columns": 875284, "margin_qp_forward_triangular_rhs_elements": 64763350, "margin_qp_forward_triangular_dense_work_units": 9093982082, "margin_qp_adjoint_transitions": 0, "margin_qp_adjoint_full_constraint_scans": 0, "margin_qp_adjoint_spectral_checks": 0, "margin_qp_adjoint_compact_snapshot_rebuilds": 0, "margin_qp_adjoint_compact_snapshot_dense_work_units": 0, "margin_qp_adjoint_spectral_cubic_dimension_units": 0, "margin_qp_adjoint_independence_checks": 0, "margin_qp_adjoint_factorization_attempts": 0, "margin_qp_adjoint_factorizations_completed": 0, "margin_qp_adjoint_condition_estimation_calls": 0, "margin_qp_adjoint_triangular_calls": 15916, "margin_qp_adjoint_triangular_rhs_columns": 186454, "margin_qp_adjoint_triangular_rhs_elements": 18820782, "margin_qp_adjoint_triangular_dense_work_units": 2760730370, "margin_qp_peak_explicit_solve_temporary_bytes": 489216, "margin_qp_peak_factor_buffer_bytes": 2212352} |
| structural_budget | {"exact": {"episodes": 160, "k1_episodes": 40, "oof_episodes": 120, "proxy_anchor_count": 1400, "sequence_paths": 1800, "baseline_head_fit_count": 3240, "candidate_preparation_count": 3240, "candidate_stage_count": 3240, "diagnostic_fit_count": 0}, "maximum": {"trained_margin_stage_count": 648, "optimizer_steps": 2592, "optimizer_iterations": 2592, "trial_count": 31104, "trial_attempt_count": 31104, "inner_objective_evaluation_count": 31752, "inner_head_fit_count": 95256, "inner_factorization_count": 173503512, "final_head_fit_count": 3240, "final_factorization_count": 5901480, "prior_head_fit_count": 864, "prior_factorization_count": 864, "baseline_factorization_count": 3240, "head_fit_count": 102600, "factorization_count": 179409096, "margin_qp_forward_factorization_attempts": 179350272, "margin_qp_forward_transitions": 179306496, "margin_qp_forward_spectral_checks": 43776}} |
| run_wall_seconds | 3135.413212776184 |
| lane_wall_seconds_sum | 6264.5754248640005 |
| maximum_lane_peak_rss_bytes | 1319796736 |
| peak_gpu_memory_bytes | N/A |
| deployment_package_bytes | N/A |
| incremental_transmission_bytes | N/A |
| additional_ground_data_payload_bytes | 0 |
| additional_ground_statistics_bytes | 0 |
| separate_spectral_seconds | N/A |
| independent_analysis_work | {"independent_general_solve_count": 23563, "independent_general_rhs_column_count": 683971, "independent_general_rhs_element_count": 65661171, "independent_dense_cubic_work_unit_count": 27104715969, "independent_dense_rhs_work_unit_count": 9424978489, "independent_spectral_diagnostic_count": 13536, "independent_spectral_cubic_work_unit_count": 20841062320} |
| triangular_rhs_count_scope | "Per actual system: columns=r; elements=n*r; dense=n*n*r; proxy not measured FLOPs" |
| resident_state_scope | "Core resident numeric buffers include actual B; archive bytes, minimum deployment and process RSS are distinct. Alias layouts cannot be reconstructed from NPZ copies." |
| positive_kernel_evidence | "Independently rebuilt full physical raw Gaussian/equivalence Gram, PSD spectra, primal/dual/KKT/all physical inequalities and saved affine/active factor residuals" |
| unmeasured_reason | "CPU only; no deployment transport/energy measured; candidate raw-kernel spectra are charged in actual head timing, not separately timed" |
| hardware | [{"row_id": "rx3-cvs-daot-rc4-s2026092701", "hardware": {"platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "processor": "x86_64", "cpu_count": 96, "dtype": "float64", "gpu_use": false}, "blas_environment": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}, {"row_id": "rx3-cvs-daot-rc4-s2026092702", "hardware": {"platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "processor": "x86_64", "cpu_count": 96, "dtype": "float64", "gpu_use": false}, "blas_environment": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}, {"row_id": "rx1-cvs-daot-rc4-s2026092701", "hardware": {"platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "processor": "x86_64", "cpu_count": 96, "dtype": "float64", "gpu_use": false}, "blas_environment": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}, {"row_id": "rx1-cvs-daot-rc4-s2026092702", "hardware": {"platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "processor": "x86_64", "cpu_count": 96, "dtype": "float64", "gpu_use": false}, "blas_environment": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}] |

| 实际执行计数 | 数值 |
|---|---:|
| episodes | 160 |
| k1_episodes | 40 |
| oof_episodes | 120 |
| proxy_anchor_count | 1400 |
| sequence_paths | 1800 |
| baseline_head_fit_count | 3240 |
| baseline_factorization_count | 3240 |
| baseline_head_triangular_solve_count | 6480 |
| baseline_effective_df_triangular_solve_count | 6480 |
| baseline_triangular_solve_count | 12960 |
| head_fit_count | 17064 |
| factorization_count | 86837 |
| trained_margin_stage_count | 648 |
| diagnostic_fit_count | 0 |
| candidate_preparation_count | 3240 |
| candidate_stage_count | 3240 |
| final_score_evaluation_count | 6336 |
| final_score_physical_count | 764400 |
| ajlr_preparation_count | 3240 |
| latent_svd_count | 3240 |
| dictionary_physical_evaluation_count | 77168 |
| prepared_distance_evaluation_count | 8856 |
| original_distance_pair_count | 14492912 |
| prior_head_fit_count | 864 |
| prior_factorization_count | 864 |
| prior_triangular_solve_count | 1728 |
| prior_triangular_rhs_count | 12096 |
| prior_triangular_rhs_element_count | 376320 |
| prior_triangular_dense_work_unit_count | 15160320 |
| prior_intercept_fit_count | 864 |
| prior_intercept_addition_count | 1167168 |
| prior_score_evaluation_count | 3168 |
| prior_score_physical_count | 154208 |
| reference_distance_evaluation_count | 314248 |
| reference_distance_pair_count | 40441536 |
| raw_distance_evaluation_count | 310504 |
| raw_distance_pair_count | 63990128 |
| kernel_evaluation_count | 178616 |
| kernel_pair_count | 88386192 |
| adapter_physical_evaluation_count | 1036096 |
| ajlr_stage_count | 3240 |
| optimizer_steps | 2592 |
| optimizer_iterations | 2592 |
| inner_objective_evaluation_count | 3240 |
| inner_head_fit_count | 9720 |
| inner_factorization_count | 72052 |
| final_head_fit_count | 3240 |
| final_factorization_count | 10681 |
| head_triangular_solve_count | 263380 |
| head_triangular_rhs_count | 976084 |
| head_triangular_rhs_element_count | 67471510 |
| head_triangular_dense_work_unit_count | 9203612162 |
| intercept_fit_count | 12960 |
| intercept_addition_count | 11994656 |
| derivative_triangular_solve_count | 24556 |
| ce_adjoint_solve_count | 7776 |
| derivative_triangular_rhs_count | 238294 |
| derivative_triangular_rhs_element_count | 20433582 |
| derivative_triangular_dense_work_unit_count | 2825703170 |
| backward_evaluation_count | 2592 |
| accepted_trial_count | 2592 |
| rejected_trial_count | 0 |
| trial_count | 2592 |
| trial_attempt_count | 2592 |
| ajlr_forward_evaluation_count | 12960 |
| completed_factorization_count | 82733 |
| margin_qp_forward_transitions | 78109 |
| margin_qp_forward_full_constraint_scans | 83869 |
| margin_qp_forward_spectral_checks | 5760 |
| margin_qp_forward_compact_snapshot_rebuilds | 78109 |
| margin_qp_forward_compact_snapshot_dense_work_units | 3608687880 |
| margin_qp_forward_spectral_cubic_dimension_units | 8434172080 |
| margin_qp_forward_independence_checks | 34861 |
| margin_qp_forward_factorization_attempts | 75533 |
| margin_qp_forward_factorizations_completed | 75533 |
| margin_qp_forward_condition_estimation_calls | 75533 |
| margin_qp_forward_triangular_calls | 248980 |
| margin_qp_forward_triangular_rhs_columns | 875284 |
| margin_qp_forward_triangular_rhs_elements | 64763350 |
| margin_qp_forward_triangular_dense_work_units | 9093982082 |
| margin_qp_adjoint_transitions | 0 |
| margin_qp_adjoint_full_constraint_scans | 0 |
| margin_qp_adjoint_spectral_checks | 0 |
| margin_qp_adjoint_compact_snapshot_rebuilds | 0 |
| margin_qp_adjoint_compact_snapshot_dense_work_units | 0 |
| margin_qp_adjoint_spectral_cubic_dimension_units | 0 |
| margin_qp_adjoint_independence_checks | 0 |
| margin_qp_adjoint_factorization_attempts | 0 |
| margin_qp_adjoint_factorizations_completed | 0 |
| margin_qp_adjoint_condition_estimation_calls | 0 |
| margin_qp_adjoint_triangular_calls | 15916 |
| margin_qp_adjoint_triangular_rhs_columns | 186454 |
| margin_qp_adjoint_triangular_rhs_elements | 18820782 |
| margin_qp_adjoint_triangular_dense_work_units | 2760730370 |
| margin_qp_peak_explicit_solve_temporary_bytes | 489216 |
| margin_qp_peak_factor_buffer_bytes | 2212352 |

独立分析来源：`E:\type10-7\automation_reports\CV-SincNet\20261001-phase2-d92-margin-joint-support-m2-r01\results\support_summary\summary.json`；训练 commit：`e50de0ad4e7d0570dce62ba88c3791fdf1c14951`；分析 commit：`ccb7f3469cc1a28ea8bbd7eb7adce937b41a430a`。
