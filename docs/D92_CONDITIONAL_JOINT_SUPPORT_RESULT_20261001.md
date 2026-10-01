# ConditionalJoint 完整 support 报告

run：`20261001-phase2-d92-conditional-joint-support-m2-r01`；已核验 160 个 parent、1800 条物理路径。
本报告只描述合法 support 的持出诊断，不代表 query 准确率或新增独立数据验证。
A 与 B−A 为 N/A：实际地面头尚未接入。R0/B0 是 support 分类器，不能代替 A。
B/C 按同一 row、同一旧 support 与物理旧 held 样本配对；C 对全部注册类统一竞争并继承当次实际 B。
H、注册下降和绝对新旧差先逐 parent 计算，再等 parent 汇总；proxy 先对全部 anchor 求 parent 内均值。
true K1 只有全 support 解析头，无独立 held；其 OOF/proxy 指标单列 N/A。proxy 的 train K1 不等于真实 K1 独立证据。
理想方向为适应提升至少 10 个百分点、注册旧类下降至多 1 个百分点、新旧绝对差至多 3 个百分点；均为 soft 目标，不触发晋级、选模或重跑。

| 路径 | A旧 | B旧 | C旧 | C新 | H | B−A | B−B0 | 注册下降 | 平均绝对差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | N/A | 71.0417 | 63.7674 | 54.8464 | 58.4168 | N/A | 0.0000 | 7.2743 | 10.7405 |
| R_CONDITIONAL_seq | N/A | 70.2083 | 61.3281 | 57.6094 | 58.1416 | N/A | -0.8333 | 8.8802 | 12.3247 |

准确率/H 用 %，差值用百分点。整体仅含 K5/10/20、新增2/5/10/20 的可测 parent。
注册下降变小需要同时检查 B 与 C，不能把 B 下降导致的差值缩小解释为保旧改善。

| 相对 R0 的变化 | B_old_accuracy | C_old_accuracy | C_new_accuracy | C_h | adaptation_gain_B_minus_A | support_adaptation_B_minus_B0 | total_old_accuracy_drop | C_abs_new_old_gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R_CONDITIONAL_seq | -0.8333 | -2.4392 | 2.7630 | -0.2752 | N/A | -0.8333 | 1.6059 | 1.5842 |

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
| oof | R_CONDITIONAL_seq | 1 | 6 | 0 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 6 | 2 | 8 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 6 | 5 | 11 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 6 | 10 | 16 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 6 | 20 | 26 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 5 | 6 | 0 | 6 | 8 | N/A | 62.0833 | 62.0833 | N/A | N/A | N/A | -1.2500 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 6 | 2 | 8 | 8 | N/A | 62.0833 | 54.5833 | 65.0000 | 51.9057 | N/A | -1.2500 | 7.5000 | 35.4167 |
| oof | R_CONDITIONAL_seq | 5 | 6 | 5 | 11 | 8 | N/A | 62.0833 | 53.3333 | 51.5000 | 51.8093 | N/A | -1.2500 | 8.7500 | 10.0000 |
| oof | R_CONDITIONAL_seq | 5 | 6 | 10 | 16 | 8 | N/A | 62.0833 | 49.5833 | 50.2500 | 48.6904 | N/A | -1.2500 | 12.5000 | 11.1667 |
| oof | R_CONDITIONAL_seq | 5 | 6 | 20 | 26 | 8 | N/A | 62.0833 | 47.0833 | 47.6250 | 45.6812 | N/A | -1.2500 | 15.0000 | 13.6250 |
| oof | R_CONDITIONAL_seq | 10 | 6 | 0 | 6 | 8 | N/A | 72.0833 | 72.0833 | N/A | N/A | N/A | -1.4583 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 6 | 2 | 8 | 8 | N/A | 72.0833 | 67.2917 | 60.0000 | 61.6891 | N/A | -1.4583 | 4.7917 | 20.6250 |
| oof | R_CONDITIONAL_seq | 10 | 6 | 5 | 11 | 8 | N/A | 72.0833 | 64.3750 | 58.2500 | 60.8186 | N/A | -1.4583 | 7.7083 | 8.5417 |
| oof | R_CONDITIONAL_seq | 10 | 6 | 10 | 16 | 8 | N/A | 72.0833 | 63.1250 | 58.7500 | 60.6423 | N/A | -1.4583 | 8.9583 | 7.7917 |
| oof | R_CONDITIONAL_seq | 10 | 6 | 20 | 26 | 8 | N/A | 72.0833 | 61.6667 | 56.0000 | 58.3895 | N/A | -1.4583 | 10.4167 | 9.2083 |
| oof | R_CONDITIONAL_seq | 20 | 6 | 0 | 6 | 8 | N/A | 76.4583 | 76.4583 | N/A | N/A | N/A | 0.2083 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 6 | 2 | 8 | 8 | N/A | 76.4583 | 71.7708 | 60.9375 | 65.7610 | N/A | 0.2083 | 4.6875 | 11.0417 |
| oof | R_CONDITIONAL_seq | 20 | 6 | 5 | 11 | 8 | N/A | 76.4583 | 68.8542 | 60.1250 | 64.0865 | N/A | 0.2083 | 7.6042 | 8.7292 |
| oof | R_CONDITIONAL_seq | 20 | 6 | 10 | 16 | 8 | N/A | 76.4583 | 67.8125 | 61.4375 | 64.4026 | N/A | 0.2083 | 8.6458 | 6.5417 |
| oof | R_CONDITIONAL_seq | 20 | 6 | 20 | 26 | 8 | N/A | 76.4583 | 66.4583 | 61.4375 | 63.8230 | N/A | 0.2083 | 10.0000 | 5.2083 |
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
| proxy | R_CONDITIONAL_seq | 1 | 6 | 0 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 6 | 2 | 8 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 6 | 5 | 11 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 6 | 10 | 16 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 6 | 20 | 26 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 6 | 0 | 6 | 8 | N/A | 50.9375 | 50.9375 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 6 | 2 | 8 | 8 | N/A | 50.9375 | 43.6458 | 47.8125 | 36.1417 | N/A | 0.0000 | 7.2917 | 35.4167 |
| proxy | R_CONDITIONAL_seq | 5 | 6 | 5 | 11 | 8 | N/A | 50.9375 | 40.0000 | 43.2500 | 38.6680 | N/A | 0.0000 | 10.9375 | 17.7500 |
| proxy | R_CONDITIONAL_seq | 5 | 6 | 10 | 16 | 8 | N/A | 50.9375 | 38.4375 | 38.8750 | 36.9418 | N/A | 0.0000 | 12.5000 | 11.8125 |
| proxy | R_CONDITIONAL_seq | 5 | 6 | 20 | 26 | 8 | N/A | 50.9375 | 34.3750 | 33.5000 | 31.8192 | N/A | 0.0000 | 16.5625 | 13.8333 |
| proxy | R_CONDITIONAL_seq | 10 | 6 | 0 | 6 | 8 | N/A | 53.0324 | 53.0324 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 6 | 2 | 8 | 8 | N/A | 53.0324 | 44.0741 | 40.7639 | 33.7795 | N/A | 0.0000 | 8.9583 | 31.5972 |
| proxy | R_CONDITIONAL_seq | 10 | 6 | 5 | 11 | 8 | N/A | 53.0324 | 41.1111 | 39.6111 | 37.5271 | N/A | 0.0000 | 11.9213 | 16.4259 |
| proxy | R_CONDITIONAL_seq | 10 | 6 | 10 | 16 | 8 | N/A | 53.0324 | 38.7269 | 38.1528 | 36.2993 | N/A | 0.0000 | 14.3056 | 12.5926 |
| proxy | R_CONDITIONAL_seq | 10 | 6 | 20 | 26 | 8 | N/A | 53.0324 | 36.5509 | 33.0972 | 32.6556 | N/A | 0.0000 | 16.4815 | 13.5231 |
| proxy | R_CONDITIONAL_seq | 20 | 6 | 0 | 6 | 8 | N/A | 52.4726 | 52.4726 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 6 | 2 | 8 | 8 | N/A | 52.4726 | 43.0976 | 44.5395 | 36.9171 | N/A | 0.0000 | 9.3750 | 28.1634 |
| proxy | R_CONDITIONAL_seq | 20 | 6 | 5 | 11 | 8 | N/A | 52.4726 | 39.5943 | 39.3947 | 36.8457 | N/A | 0.0000 | 12.8783 | 15.0702 |
| proxy | R_CONDITIONAL_seq | 20 | 6 | 10 | 16 | 8 | N/A | 52.4726 | 37.0779 | 37.3717 | 35.2232 | N/A | 0.0000 | 15.3947 | 12.4803 |
| proxy | R_CONDITIONAL_seq | 20 | 6 | 20 | 26 | 8 | N/A | 52.4726 | 35.1425 | 32.8026 | 32.4152 | N/A | 0.0000 | 17.3300 | 11.1820 |

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
| oof | R_CONDITIONAL_seq | 1 | 0 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 0 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 0 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 0 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 10 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 10 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 10 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 10 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 2 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 2 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 2 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 2 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 20 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 20 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 20 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 20 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 5 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 5 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 5 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 5 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 10 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 47.5000 | N/A | N/A | N/A | -2.5000 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 96.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 47.5000 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | 81.0000 | 88.1393 | N/A | -1.6667 | 0.0000 | 15.6667 |
| oof | R_CONDITIONAL_seq | 10 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 36.6667 | 38.0000 | 37.3021 | N/A | -2.5000 | 10.8333 | 1.6667 |
| oof | R_CONDITIONAL_seq | 10 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 87.5000 | 79.0000 | 83.0079 | N/A | 0.0000 | 9.1667 | 8.5000 |
| oof | R_CONDITIONAL_seq | 10 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 31.6667 | 37.0000 | 34.1200 | N/A | -1.6667 | 15.8333 | 5.3333 |
| oof | R_CONDITIONAL_seq | 10 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | 60.0000 | 74.0426 | N/A | -1.6667 | 0.0000 | 36.6667 |
| oof | R_CONDITIONAL_seq | 10 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 40.0000 | 42.5000 | 40.5155 | N/A | -2.5000 | 7.5000 | 10.8333 |
| oof | R_CONDITIONAL_seq | 10 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 95.0000 | 80.0000 | 86.8505 | N/A | 0.0000 | 1.6667 | 15.0000 |
| oof | R_CONDITIONAL_seq | 10 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 37.5000 | 57.5000 | 45.3479 | N/A | -1.6667 | 10.0000 | 20.0000 |
| oof | R_CONDITIONAL_seq | 10 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 95.8333 | 75.5000 | 84.4552 | N/A | -1.6667 | 0.8333 | 20.3333 |
| oof | R_CONDITIONAL_seq | 10 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 35.0000 | 38.0000 | 36.3886 | N/A | -2.5000 | 12.5000 | 3.0000 |
| oof | R_CONDITIONAL_seq | 10 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 86.6667 | 77.2500 | 81.6399 | N/A | 0.0000 | 10.0000 | 9.4167 |
| oof | R_CONDITIONAL_seq | 10 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 29.1667 | 33.2500 | 31.0741 | N/A | -1.6667 | 18.3333 | 4.0833 |
| oof | R_CONDITIONAL_seq | 10 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 96.6667 | 96.6667 | 73.0000 | 83.1789 | N/A | -1.6667 | 0.0000 | 23.6667 |
| oof | R_CONDITIONAL_seq | 10 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 47.5000 | 37.5000 | 34.0000 | 35.6248 | N/A | -2.5000 | 10.0000 | 3.5000 |
| oof | R_CONDITIONAL_seq | 10 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 96.6667 | 89.1667 | 87.0000 | 88.0699 | N/A | 0.0000 | 7.5000 | 2.1667 |
| oof | R_CONDITIONAL_seq | 10 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 47.5000 | 34.1667 | 39.0000 | 36.4010 | N/A | -1.6667 | 13.3333 | 4.8333 |
| oof | R_CONDITIONAL_seq | 20 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 97.0833 | N/A | N/A | N/A | 0.4167 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 57.5000 | N/A | N/A | N/A | 0.8333 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 95.4167 | N/A | N/A | N/A | 0.8333 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 55.8333 | N/A | N/A | N/A | -1.2500 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 95.0000 | 84.0000 | 89.1549 | N/A | 0.4167 | 2.0833 | 11.0000 |
| oof | R_CONDITIONAL_seq | 20 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 44.5833 | 39.0000 | 41.5417 | N/A | 0.8333 | 12.9167 | 5.5833 |
| oof | R_CONDITIONAL_seq | 20 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 90.8333 | 85.5000 | 88.0810 | N/A | 0.8333 | 4.5833 | 5.3333 |
| oof | R_CONDITIONAL_seq | 20 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 40.8333 | 37.2500 | 38.8325 | N/A | -1.2500 | 15.0000 | 4.2500 |
| oof | R_CONDITIONAL_seq | 20 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 96.2500 | 75.0000 | 84.3062 | N/A | 0.4167 | 0.8333 | 21.2500 |
| oof | R_CONDITIONAL_seq | 20 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 51.2500 | 47.5000 | 49.3030 | N/A | 0.8333 | 6.2500 | 3.7500 |
| oof | R_CONDITIONAL_seq | 20 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 94.5833 | 82.5000 | 88.1290 | N/A | 0.8333 | 0.8333 | 12.0833 |
| oof | R_CONDITIONAL_seq | 20 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 45.0000 | 38.7500 | 41.3057 | N/A | -1.2500 | 10.8333 | 7.0833 |
| oof | R_CONDITIONAL_seq | 20 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 95.4167 | 85.0000 | 89.9037 | N/A | 0.4167 | 1.6667 | 10.4167 |
| oof | R_CONDITIONAL_seq | 20 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 42.5000 | 40.5000 | 41.4321 | N/A | 0.8333 | 15.0000 | 2.7500 |
| oof | R_CONDITIONAL_seq | 20 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 90.4167 | 84.7500 | 87.4903 | N/A | 0.8333 | 5.0000 | 5.6667 |
| oof | R_CONDITIONAL_seq | 20 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 37.5000 | 35.5000 | 36.4659 | N/A | -1.2500 | 18.3333 | 2.0000 |
| oof | R_CONDITIONAL_seq | 20 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 97.0833 | 95.4167 | 81.5000 | 87.8842 | N/A | 0.4167 | 1.6667 | 13.9167 |
| oof | R_CONDITIONAL_seq | 20 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 57.5000 | 48.3333 | 37.5000 | 42.2284 | N/A | 0.8333 | 9.1667 | 10.8333 |
| oof | R_CONDITIONAL_seq | 20 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 95.4167 | 91.2500 | 87.0000 | 89.0444 | N/A | 0.8333 | 4.1667 | 4.2500 |
| oof | R_CONDITIONAL_seq | 20 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 55.8333 | 40.4167 | 34.5000 | 37.1889 | N/A | -1.2500 | 15.4167 | 5.9167 |
| oof | R_CONDITIONAL_seq | 5 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 93.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 33.3333 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 88.3333 | N/A | N/A | N/A | -3.3333 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 33.3333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 85.0000 | 65.0000 | 73.6663 | N/A | 0.0000 | 8.3333 | 20.0000 |
| oof | R_CONDITIONAL_seq | 5 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 18.3333 | 24.0000 | 20.4024 | N/A | -1.6667 | 15.0000 | 5.6667 |
| oof | R_CONDITIONAL_seq | 5 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 78.3333 | 80.0000 | 79.1155 | N/A | -3.3333 | 10.0000 | 3.6667 |
| oof | R_CONDITIONAL_seq | 5 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 16.6667 | 32.0000 | 21.5775 | N/A | 0.0000 | 16.6667 | 15.3333 |
| oof | R_CONDITIONAL_seq | 5 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 91.6667 | 50.0000 | 64.7010 | N/A | 0.0000 | 1.6667 | 41.6667 |
| oof | R_CONDITIONAL_seq | 5 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 23.3333 | 80.0000 | 36.0000 | N/A | -1.6667 | 10.0000 | 56.6667 |
| oof | R_CONDITIONAL_seq | 5 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 83.3333 | 75.0000 | 78.7234 | N/A | -3.3333 | 5.0000 | 8.3333 |
| oof | R_CONDITIONAL_seq | 5 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 20.0000 | 55.0000 | 28.1984 | N/A | 0.0000 | 13.3333 | 35.0000 |
| oof | R_CONDITIONAL_seq | 5 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 85.0000 | 61.0000 | 71.0208 | N/A | 0.0000 | 8.3333 | 24.0000 |
| oof | R_CONDITIONAL_seq | 5 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 15.0000 | 28.5000 | 18.9894 | N/A | -1.6667 | 18.3333 | 13.5000 |
| oof | R_CONDITIONAL_seq | 5 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 73.3333 | 74.0000 | 73.5810 | N/A | -3.3333 | 15.0000 | 5.0000 |
| oof | R_CONDITIONAL_seq | 5 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 15.0000 | 27.0000 | 19.1339 | N/A | 0.0000 | 18.3333 | 12.0000 |
| oof | R_CONDITIONAL_seq | 5 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 93.3333 | 90.0000 | 68.0000 | 77.4026 | N/A | 0.0000 | 3.3333 | 22.0000 |
| oof | R_CONDITIONAL_seq | 5 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 33.3333 | 21.6667 | 28.0000 | 24.4101 | N/A | -1.6667 | 11.6667 | 6.3333 |
| oof | R_CONDITIONAL_seq | 5 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 88.3333 | 80.0000 | 84.0000 | 81.7880 | N/A | -3.3333 | 8.3333 | 7.3333 |
| oof | R_CONDITIONAL_seq | 5 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 33.3333 | 21.6667 | 26.0000 | 23.6364 | N/A | 0.0000 | 11.6667 | 4.3333 |
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
| proxy | R_CONDITIONAL_seq | 1 | 0 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 0 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 0 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 0 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 10 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 10 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 10 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 10 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 2 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 2 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 2 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 2 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 20 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 20 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 20 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 20 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 5 | rx1 | 20-19 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 5 | rx1 | 20-19 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 5 | rx3 | 19-1 | practical_high | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 5 | rx3 | 19-1 | practical_low_urban | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 86.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 27.7778 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 72.4074 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 25.2778 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 75.5556 | 56.7222 | 64.5986 | N/A | 0.0000 | 11.1111 | 18.8333 |
| proxy | R_CONDITIONAL_seq | 10 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 14.0741 | 19.1111 | 14.6613 | N/A | 0.0000 | 13.7037 | 9.6667 |
| proxy | R_CONDITIONAL_seq | 10 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 58.7963 | 57.8333 | 57.6941 | N/A | 0.0000 | 13.6111 | 9.1852 |
| proxy | R_CONDITIONAL_seq | 10 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 6.4815 | 18.9444 | 8.2432 | N/A | 0.0000 | 18.7963 | 12.6852 |
| proxy | R_CONDITIONAL_seq | 10 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 85.3704 | 33.6111 | 46.1399 | N/A | 0.0000 | 1.2963 | 51.7593 |
| proxy | R_CONDITIONAL_seq | 10 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 15.6481 | 35.5556 | 20.7930 | N/A | 0.0000 | 12.1296 | 20.0926 |
| proxy | R_CONDITIONAL_seq | 10 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 67.1296 | 50.8333 | 55.9931 | N/A | 0.0000 | 5.2778 | 19.6296 |
| proxy | R_CONDITIONAL_seq | 10 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 8.1481 | 43.0556 | 12.1921 | N/A | 0.0000 | 17.1296 | 34.9074 |
| proxy | R_CONDITIONAL_seq | 10 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 74.1667 | 48.0833 | 58.1921 | N/A | 0.0000 | 12.5000 | 26.0833 |
| proxy | R_CONDITIONAL_seq | 10 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 12.7778 | 17.3056 | 13.5244 | N/A | 0.0000 | 15.0000 | 7.4167 |
| proxy | R_CONDITIONAL_seq | 10 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 53.5185 | 50.6111 | 51.3250 | N/A | 0.0000 | 18.8889 | 9.9444 |
| proxy | R_CONDITIONAL_seq | 10 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 5.7407 | 16.3889 | 7.5808 | N/A | 0.0000 | 19.5370 | 10.6481 |
| proxy | R_CONDITIONAL_seq | 10 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 86.6667 | 81.4815 | 52.0000 | 63.1567 | N/A | 0.0000 | 5.1852 | 29.4815 |
| proxy | R_CONDITIONAL_seq | 10 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 27.7778 | 14.8148 | 20.3333 | 15.6244 | N/A | 0.0000 | 12.9630 | 11.2593 |
| proxy | R_CONDITIONAL_seq | 10 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 72.4074 | 60.6481 | 63.2222 | 61.4309 | N/A | 0.0000 | 11.7593 | 9.4259 |
| proxy | R_CONDITIONAL_seq | 10 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.2778 | 7.5000 | 22.8889 | 9.8963 | N/A | 0.0000 | 17.7778 | 15.5370 |
| proxy | R_CONDITIONAL_seq | 20 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 84.9342 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 29.8465 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 69.0132 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 26.0965 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 71.7105 | 51.5132 | 59.4705 | N/A | 0.0000 | 13.2237 | 20.2500 |
| proxy | R_CONDITIONAL_seq | 20 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 13.1360 | 19.5526 | 14.1606 | N/A | 0.0000 | 16.7105 | 9.7149 |
| proxy | R_CONDITIONAL_seq | 20 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 54.6053 | 59.0132 | 56.2767 | N/A | 0.0000 | 14.4079 | 8.9254 |
| proxy | R_CONDITIONAL_seq | 20 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 8.8596 | 19.4079 | 10.9850 | N/A | 0.0000 | 17.2368 | 11.0307 |
| proxy | R_CONDITIONAL_seq | 20 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 80.5482 | 34.2763 | 46.9099 | N/A | 0.0000 | 4.3860 | 46.2719 |
| proxy | R_CONDITIONAL_seq | 20 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 17.1053 | 40.9211 | 22.3904 | N/A | 0.0000 | 12.7412 | 24.5614 |
| proxy | R_CONDITIONAL_seq | 20 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 63.7281 | 61.5132 | 61.8796 | N/A | 0.0000 | 5.2851 | 11.3816 |
| proxy | R_CONDITIONAL_seq | 20 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 11.0088 | 41.4474 | 16.4884 | N/A | 0.0000 | 15.0877 | 30.4386 |
| proxy | R_CONDITIONAL_seq | 20 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 69.8684 | 48.1118 | 56.6114 | N/A | 0.0000 | 15.0658 | 21.9189 |
| proxy | R_CONDITIONAL_seq | 20 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 12.7412 | 16.4803 | 13.1212 | N/A | 0.0000 | 17.1053 | 7.4057 |
| proxy | R_CONDITIONAL_seq | 20 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 49.5833 | 51.8355 | 50.2262 | N/A | 0.0000 | 19.4298 | 7.8838 |
| proxy | R_CONDITIONAL_seq | 20 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 8.3772 | 14.7829 | 9.7020 | N/A | 0.0000 | 17.7193 | 7.5197 |
| proxy | R_CONDITIONAL_seq | 20 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 84.9342 | 76.6009 | 50.7368 | 60.4402 | N/A | 0.0000 | 8.3333 | 25.8640 |
| proxy | R_CONDITIONAL_seq | 20 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 29.8465 | 14.2544 | 23.5000 | 15.7086 | N/A | 0.0000 | 15.5921 | 12.5263 |
| proxy | R_CONDITIONAL_seq | 20 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 69.0132 | 57.8289 | 59.9474 | 58.4660 | N/A | 0.0000 | 11.1842 | 7.9781 |
| proxy | R_CONDITIONAL_seq | 20 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 26.0965 | 9.6930 | 23.3947 | 12.7679 | N/A | 0.0000 | 16.4035 | 13.9123 |
| proxy | R_CONDITIONAL_seq | 5 | 0 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 90.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 0 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 20.4167 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 0 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 66.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 0 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 25.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 10 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 77.5000 | 60.2500 | 67.2146 | N/A | 0.0000 | 13.3333 | 17.9167 |
| proxy | R_CONDITIONAL_seq | 5 | 10 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 11.6667 | 17.2500 | 12.8687 | N/A | 0.0000 | 8.7500 | 8.0833 |
| proxy | R_CONDITIONAL_seq | 5 | 10 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 56.2500 | 60.0000 | 57.5352 | N/A | 0.0000 | 10.4167 | 10.2500 |
| proxy | R_CONDITIONAL_seq | 5 | 10 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 8.3333 | 18.0000 | 10.1489 | N/A | 0.0000 | 17.5000 | 11.0000 |
| proxy | R_CONDITIONAL_seq | 5 | 2 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 89.5833 | 32.5000 | 47.1783 | N/A | 0.0000 | 1.2500 | 57.0833 |
| proxy | R_CONDITIONAL_seq | 5 | 2 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 11.6667 | 61.2500 | 19.0828 | N/A | 0.0000 | 8.7500 | 49.5833 |
| proxy | R_CONDITIONAL_seq | 5 | 2 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 62.0833 | 68.7500 | 64.0622 | N/A | 0.0000 | 4.5833 | 15.8333 |
| proxy | R_CONDITIONAL_seq | 5 | 2 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 11.2500 | 28.7500 | 14.2434 | N/A | 0.0000 | 14.5833 | 19.1667 |
| proxy | R_CONDITIONAL_seq | 5 | 20 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 73.7500 | 45.6250 | 55.9226 | N/A | 0.0000 | 17.0833 | 28.1250 |
| proxy | R_CONDITIONAL_seq | 5 | 20 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 11.6667 | 20.0000 | 14.4565 | N/A | 0.0000 | 8.7500 | 8.3333 |
| proxy | R_CONDITIONAL_seq | 5 | 20 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 45.0000 | 53.6250 | 48.5495 | N/A | 0.0000 | 21.6667 | 11.2083 |
| proxy | R_CONDITIONAL_seq | 5 | 20 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 7.0833 | 14.7500 | 8.3481 | N/A | 0.0000 | 18.7500 | 7.6667 |
| proxy | R_CONDITIONAL_seq | 5 | 5 | rx1 | 20-19 | practical_high | 2 | N/A | 90.8333 | 82.5000 | 55.5000 | 65.7185 | N/A | 0.0000 | 8.3333 | 27.0000 |
| proxy | R_CONDITIONAL_seq | 5 | 5 | rx1 | 20-19 | practical_low_urban | 2 | N/A | 20.4167 | 11.6667 | 24.0000 | 14.8094 | N/A | 0.0000 | 8.7500 | 13.0000 |
| proxy | R_CONDITIONAL_seq | 5 | 5 | rx3 | 19-1 | practical_high | 2 | N/A | 66.6667 | 57.9167 | 72.5000 | 63.6055 | N/A | 0.0000 | 8.7500 | 17.9167 |
| proxy | R_CONDITIONAL_seq | 5 | 5 | rx3 | 19-1 | practical_low_urban | 2 | N/A | 25.8333 | 7.9167 | 21.0000 | 10.5385 | N/A | 0.0000 | 17.9167 | 13.0833 |

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
| oof | R_CONDITIONAL_seq | 1 | 0 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 0 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 0 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 0 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 10 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 10 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 10 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 10 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 2 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 2 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 2 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 2 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 20 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 20 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 20 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 20 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 5 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 5 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 5 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 1 | 5 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_CONDITIONAL_seq | 10 | 0 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 71.6667 | N/A | N/A | N/A | -2.5000 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 0 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 70.8333 | N/A | N/A | N/A | -3.3333 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 0 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 72.5000 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 0 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 73.3333 | N/A | N/A | N/A | 1.6667 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 10 | 10 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 67.5000 | 60.0000 | 63.4486 | N/A | -2.5000 | 4.1667 | 7.5000 |
| oof | R_CONDITIONAL_seq | 10 | 10 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 60.0000 | 56.5000 | 57.9864 | N/A | -3.3333 | 10.8333 | 7.8333 |
| oof | R_CONDITIONAL_seq | 10 | 10 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 65.8333 | 59.0000 | 61.9928 | N/A | -1.6667 | 6.6667 | 9.8333 |
| oof | R_CONDITIONAL_seq | 10 | 10 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 59.1667 | 59.5000 | 59.1415 | N/A | 1.6667 | 14.1667 | 6.0000 |
| oof | R_CONDITIONAL_seq | 10 | 2 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 70.0000 | 47.5000 | 56.3830 | N/A | -2.5000 | 1.6667 | 22.5000 |
| oof | R_CONDITIONAL_seq | 10 | 2 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 65.0000 | 70.0000 | 65.8355 | N/A | -3.3333 | 5.8333 | 18.3333 |
| oof | R_CONDITIONAL_seq | 10 | 2 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 66.6667 | 55.0000 | 58.1751 | N/A | -1.6667 | 5.8333 | 25.0000 |
| oof | R_CONDITIONAL_seq | 10 | 2 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 67.5000 | 67.5000 | 66.3629 | N/A | 1.6667 | 5.8333 | 16.6667 |
| oof | R_CONDITIONAL_seq | 10 | 20 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 66.6667 | 56.0000 | 60.6493 | N/A | -2.5000 | 5.0000 | 11.0000 |
| oof | R_CONDITIONAL_seq | 10 | 20 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 57.5000 | 52.7500 | 54.7987 | N/A | -3.3333 | 13.3333 | 8.4167 |
| oof | R_CONDITIONAL_seq | 10 | 20 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 64.1667 | 57.5000 | 60.1946 | N/A | -1.6667 | 8.3333 | 12.3333 |
| oof | R_CONDITIONAL_seq | 10 | 20 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 58.3333 | 57.7500 | 57.9153 | N/A | 1.6667 | 15.0000 | 5.0833 |
| oof | R_CONDITIONAL_seq | 10 | 5 | 2026092701 | rx1 | 2 | N/A | 71.6667 | 68.3333 | 54.0000 | 60.2924 | N/A | -2.5000 | 3.3333 | 14.3333 |
| oof | R_CONDITIONAL_seq | 10 | 5 | 2026092701 | rx3 | 2 | N/A | 70.8333 | 61.6667 | 64.0000 | 62.6762 | N/A | -3.3333 | 9.1667 | 4.3333 |
| oof | R_CONDITIONAL_seq | 10 | 5 | 2026092702 | rx1 | 2 | N/A | 72.5000 | 65.8333 | 53.0000 | 58.5112 | N/A | -1.6667 | 6.6667 | 12.8333 |
| oof | R_CONDITIONAL_seq | 10 | 5 | 2026092702 | rx3 | 2 | N/A | 73.3333 | 61.6667 | 62.0000 | 61.7947 | N/A | 1.6667 | 11.6667 | 2.6667 |
| oof | R_CONDITIONAL_seq | 20 | 0 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 77.5000 | N/A | N/A | N/A | 0.8333 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 0 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 75.8333 | N/A | N/A | N/A | 0.4167 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 0 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 77.0833 | N/A | N/A | N/A | 0.4167 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 0 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 75.4167 | N/A | N/A | N/A | -0.8333 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 20 | 10 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 68.3333 | 61.0000 | 64.4321 | N/A | 0.8333 | 9.1667 | 7.3333 |
| oof | R_CONDITIONAL_seq | 20 | 10 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 65.4167 | 63.7500 | 64.5593 | N/A | 0.4167 | 10.4167 | 2.3333 |
| oof | R_CONDITIONAL_seq | 20 | 10 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 71.2500 | 62.0000 | 66.2645 | N/A | 0.4167 | 5.8333 | 9.2500 |
| oof | R_CONDITIONAL_seq | 20 | 10 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 66.2500 | 59.0000 | 62.3543 | N/A | -0.8333 | 9.1667 | 7.2500 |
| oof | R_CONDITIONAL_seq | 20 | 2 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 73.7500 | 61.2500 | 66.8211 | N/A | 0.8333 | 3.7500 | 12.5000 |
| oof | R_CONDITIONAL_seq | 20 | 2 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 69.1667 | 63.7500 | 66.2638 | N/A | 0.4167 | 6.6667 | 6.2500 |
| oof | R_CONDITIONAL_seq | 20 | 2 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 73.7500 | 61.2500 | 66.7881 | N/A | 0.4167 | 3.3333 | 12.5000 |
| oof | R_CONDITIONAL_seq | 20 | 2 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 70.4167 | 57.5000 | 63.1709 | N/A | -0.8333 | 5.0000 | 12.9167 |
| oof | R_CONDITIONAL_seq | 20 | 20 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 67.5000 | 63.2500 | 65.2549 | N/A | 0.8333 | 10.0000 | 5.0000 |
| oof | R_CONDITIONAL_seq | 20 | 20 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 63.7500 | 60.7500 | 62.2113 | N/A | 0.4167 | 12.0833 | 3.0000 |
| oof | R_CONDITIONAL_seq | 20 | 20 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 70.4167 | 62.2500 | 66.0809 | N/A | 0.4167 | 6.6667 | 8.1667 |
| oof | R_CONDITIONAL_seq | 20 | 20 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 64.1667 | 59.5000 | 61.7449 | N/A | -0.8333 | 11.2500 | 4.6667 |
| oof | R_CONDITIONAL_seq | 20 | 5 | 2026092701 | rx1 | 2 | N/A | 77.5000 | 71.2500 | 58.0000 | 63.9406 | N/A | 0.8333 | 6.2500 | 13.2500 |
| oof | R_CONDITIONAL_seq | 20 | 5 | 2026092701 | rx3 | 2 | N/A | 75.8333 | 65.4167 | 63.0000 | 64.1597 | N/A | 0.4167 | 10.4167 | 2.4167 |
| oof | R_CONDITIONAL_seq | 20 | 5 | 2026092702 | rx1 | 2 | N/A | 77.0833 | 72.5000 | 61.0000 | 66.1719 | N/A | 0.4167 | 4.5833 | 11.5000 |
| oof | R_CONDITIONAL_seq | 20 | 5 | 2026092702 | rx3 | 2 | N/A | 75.4167 | 66.2500 | 58.5000 | 62.0736 | N/A | -0.8333 | 9.1667 | 7.7500 |
| oof | R_CONDITIONAL_seq | 5 | 0 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 65.0000 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 0 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 63.3333 | N/A | N/A | N/A | 1.6667 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 0 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 61.6667 | N/A | N/A | N/A | -1.6667 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 0 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 58.3333 | N/A | N/A | N/A | -5.0000 | 0.0000 | N/A |
| oof | R_CONDITIONAL_seq | 5 | 10 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 55.0000 | 45.0000 | 49.2982 | N/A | 0.0000 | 10.0000 | 10.6667 |
| oof | R_CONDITIONAL_seq | 5 | 10 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 50.0000 | 54.0000 | 51.4937 | N/A | 1.6667 | 13.3333 | 6.0000 |
| oof | R_CONDITIONAL_seq | 5 | 10 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 48.3333 | 44.0000 | 44.7705 | N/A | -1.6667 | 13.3333 | 15.0000 |
| oof | R_CONDITIONAL_seq | 5 | 10 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 45.0000 | 58.0000 | 49.1993 | N/A | -5.0000 | 13.3333 | 13.0000 |
| oof | R_CONDITIONAL_seq | 5 | 2 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 58.3333 | 65.0000 | 52.1429 | N/A | 0.0000 | 6.6667 | 46.6667 |
| oof | R_CONDITIONAL_seq | 5 | 2 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 51.6667 | 70.0000 | 52.1849 | N/A | 1.6667 | 11.6667 | 35.0000 |
| oof | R_CONDITIONAL_seq | 5 | 2 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 56.6667 | 65.0000 | 48.5581 | N/A | -1.6667 | 5.0000 | 51.6667 |
| oof | R_CONDITIONAL_seq | 5 | 2 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 51.6667 | 60.0000 | 54.7368 | N/A | -5.0000 | 6.6667 | 8.3333 |
| oof | R_CONDITIONAL_seq | 5 | 20 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 53.3333 | 44.0000 | 47.2907 | N/A | 0.0000 | 11.6667 | 16.3333 |
| oof | R_CONDITIONAL_seq | 5 | 20 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 45.0000 | 47.0000 | 45.5504 | N/A | 1.6667 | 18.3333 | 6.3333 |
| oof | R_CONDITIONAL_seq | 5 | 20 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 46.6667 | 45.5000 | 42.7194 | N/A | -1.6667 | 15.0000 | 21.1667 |
| oof | R_CONDITIONAL_seq | 5 | 20 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 43.3333 | 54.0000 | 47.1645 | N/A | -5.0000 | 15.0000 | 10.6667 |
| oof | R_CONDITIONAL_seq | 5 | 5 | 2026092701 | rx1 | 2 | N/A | 65.0000 | 58.3333 | 54.0000 | 55.3191 | N/A | 0.0000 | 6.6667 | 13.6667 |
| oof | R_CONDITIONAL_seq | 5 | 5 | 2026092701 | rx3 | 2 | N/A | 63.3333 | 53.3333 | 54.0000 | 53.5436 | N/A | 1.6667 | 10.0000 | 4.0000 |
| oof | R_CONDITIONAL_seq | 5 | 5 | 2026092702 | rx1 | 2 | N/A | 61.6667 | 53.3333 | 42.0000 | 46.4935 | N/A | -1.6667 | 8.3333 | 14.6667 |
| oof | R_CONDITIONAL_seq | 5 | 5 | 2026092702 | rx3 | 2 | N/A | 58.3333 | 48.3333 | 56.0000 | 51.8808 | N/A | -5.0000 | 10.0000 | 7.6667 |
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
| proxy | R_CONDITIONAL_seq | 1 | 0 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 0 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 0 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 0 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 10 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 10 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 10 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 10 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 2 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 2 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 2 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 2 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 20 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 20 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 20 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 20 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 5 | 2026092701 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 5 | 2026092701 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 5 | 2026092702 | rx1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 1 | 5 | 2026092702 | rx3 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 0 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 58.1481 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 0 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 49.5370 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 0 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 56.2963 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 0 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 48.1481 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 10 | 10 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 45.3704 | 36.8889 | 39.2261 | N/A | 0.0000 | 12.7778 | 15.2963 |
| proxy | R_CONDITIONAL_seq | 10 | 10 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 33.5185 | 38.0000 | 33.2492 | N/A | 0.0000 | 16.0185 | 10.6296 |
| proxy | R_CONDITIONAL_seq | 10 | 10 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 44.2593 | 38.9444 | 40.0338 | N/A | 0.0000 | 12.0370 | 13.2037 |
| proxy | R_CONDITIONAL_seq | 10 | 10 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 31.7593 | 38.7778 | 32.6882 | N/A | 0.0000 | 16.3889 | 11.2407 |
| proxy | R_CONDITIONAL_seq | 10 | 2 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 51.2963 | 35.2778 | 34.6850 | N/A | 0.0000 | 6.8519 | 35.8333 |
| proxy | R_CONDITIONAL_seq | 10 | 2 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 37.5926 | 48.0556 | 33.8380 | N/A | 0.0000 | 11.9444 | 29.3519 |
| proxy | R_CONDITIONAL_seq | 10 | 2 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 49.7222 | 33.8889 | 32.2479 | N/A | 0.0000 | 6.5741 | 36.0185 |
| proxy | R_CONDITIONAL_seq | 10 | 2 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 37.6852 | 45.8333 | 34.3471 | N/A | 0.0000 | 10.4630 | 25.1852 |
| proxy | R_CONDITIONAL_seq | 10 | 20 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 44.0741 | 32.1389 | 35.7974 | N/A | 0.0000 | 14.0741 | 17.3981 |
| proxy | R_CONDITIONAL_seq | 10 | 20 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 30.2778 | 33.0000 | 29.4621 | N/A | 0.0000 | 19.2593 | 10.5556 |
| proxy | R_CONDITIONAL_seq | 10 | 20 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 42.8704 | 33.2500 | 35.9191 | N/A | 0.0000 | 13.4259 | 16.1019 |
| proxy | R_CONDITIONAL_seq | 10 | 20 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 28.9815 | 34.0000 | 29.4437 | N/A | 0.0000 | 19.1667 | 10.0370 |
| proxy | R_CONDITIONAL_seq | 10 | 5 | 2026092701 | rx1 | 2 | N/A | 58.1481 | 48.6111 | 34.8889 | 38.8483 | N/A | 0.0000 | 9.5370 | 21.2407 |
| proxy | R_CONDITIONAL_seq | 10 | 5 | 2026092701 | rx3 | 2 | N/A | 49.5370 | 34.5370 | 43.8889 | 36.2135 | N/A | 0.0000 | 15.0000 | 12.5741 |
| proxy | R_CONDITIONAL_seq | 10 | 5 | 2026092702 | rx1 | 2 | N/A | 56.2963 | 47.6852 | 37.4444 | 39.9327 | N/A | 0.0000 | 8.6111 | 19.5000 |
| proxy | R_CONDITIONAL_seq | 10 | 5 | 2026092702 | rx3 | 2 | N/A | 48.1481 | 33.6111 | 42.2222 | 35.1136 | N/A | 0.0000 | 14.5370 | 12.3889 |
| proxy | R_CONDITIONAL_seq | 20 | 0 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 57.3904 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 0 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 47.3465 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 0 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 57.3904 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 0 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 47.7632 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 20 | 10 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 41.6228 | 34.8289 | 36.0878 | N/A | 0.0000 | 15.7675 | 14.9079 |
| proxy | R_CONDITIONAL_seq | 20 | 10 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 32.6974 | 38.8158 | 34.0224 | N/A | 0.0000 | 14.6491 | 9.8289 |
| proxy | R_CONDITIONAL_seq | 20 | 10 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 43.2237 | 36.2368 | 37.5434 | N/A | 0.0000 | 14.1667 | 15.0570 |
| proxy | R_CONDITIONAL_seq | 20 | 10 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 30.7675 | 39.6053 | 33.2393 | N/A | 0.0000 | 16.9956 | 10.1272 |
| proxy | R_CONDITIONAL_seq | 20 | 2 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 48.6842 | 37.4342 | 34.5119 | N/A | 0.0000 | 8.7061 | 35.1096 |
| proxy | R_CONDITIONAL_seq | 20 | 2 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 38.3772 | 51.6447 | 39.8272 | N/A | 0.0000 | 8.9693 | 21.0746 |
| proxy | R_CONDITIONAL_seq | 20 | 2 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 48.9693 | 37.7632 | 34.7884 | N/A | 0.0000 | 8.4211 | 35.7237 |
| proxy | R_CONDITIONAL_seq | 20 | 2 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 36.3596 | 51.3158 | 38.5409 | N/A | 0.0000 | 11.4035 | 20.7456 |
| proxy | R_CONDITIONAL_seq | 20 | 20 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 40.9868 | 32.0395 | 34.6397 | N/A | 0.0000 | 16.4035 | 14.5789 |
| proxy | R_CONDITIONAL_seq | 20 | 20 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 30.2412 | 33.2434 | 30.6525 | N/A | 0.0000 | 17.1053 | 7.6820 |
| proxy | R_CONDITIONAL_seq | 20 | 20 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 41.6228 | 32.5526 | 35.0929 | N/A | 0.0000 | 15.7675 | 14.7456 |
| proxy | R_CONDITIONAL_seq | 20 | 20 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 27.7193 | 33.3750 | 29.2756 | N/A | 0.0000 | 20.0439 | 7.7215 |
| proxy | R_CONDITIONAL_seq | 20 | 5 | 2026092701 | rx1 | 2 | N/A | 57.3904 | 45.0658 | 35.9474 | 37.2674 | N/A | 0.0000 | 12.3246 | 19.5746 |
| proxy | R_CONDITIONAL_seq | 20 | 5 | 2026092701 | rx3 | 2 | N/A | 47.3465 | 34.9123 | 40.9211 | 36.0141 | N/A | 0.0000 | 12.4342 | 10.3947 |
| proxy | R_CONDITIONAL_seq | 20 | 5 | 2026092702 | rx1 | 2 | N/A | 57.3904 | 45.7895 | 38.2895 | 38.8814 | N/A | 0.0000 | 11.6009 | 18.8158 |
| proxy | R_CONDITIONAL_seq | 20 | 5 | 2026092702 | rx3 | 2 | N/A | 47.7632 | 32.6096 | 42.4211 | 35.2198 | N/A | 0.0000 | 15.1535 | 11.4956 |
| proxy | R_CONDITIONAL_seq | 5 | 0 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 54.5833 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 0 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 45.8333 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 0 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 56.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 0 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 46.6667 | N/A | N/A | N/A | 0.0000 | 0.0000 | N/A |
| proxy | R_CONDITIONAL_seq | 5 | 10 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 44.1667 | 39.0000 | 40.1239 | N/A | 0.0000 | 10.4167 | 12.5000 |
| proxy | R_CONDITIONAL_seq | 5 | 10 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 32.0833 | 39.7500 | 33.8945 | N/A | 0.0000 | 13.7500 | 11.8333 |
| proxy | R_CONDITIONAL_seq | 5 | 10 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 45.0000 | 38.5000 | 39.9594 | N/A | 0.0000 | 11.6667 | 13.5000 |
| proxy | R_CONDITIONAL_seq | 5 | 10 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 32.5000 | 38.2500 | 33.7896 | N/A | 0.0000 | 14.1667 | 9.4167 |
| proxy | R_CONDITIONAL_seq | 5 | 2 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 50.0000 | 46.2500 | 32.9250 | N/A | 0.0000 | 4.5833 | 52.0833 |
| proxy | R_CONDITIONAL_seq | 5 | 2 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 36.6667 | 45.0000 | 37.1190 | N/A | 0.0000 | 9.1667 | 17.5000 |
| proxy | R_CONDITIONAL_seq | 5 | 2 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 51.2500 | 47.5000 | 33.3362 | N/A | 0.0000 | 5.4167 | 54.5833 |
| proxy | R_CONDITIONAL_seq | 5 | 2 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 36.6667 | 52.5000 | 41.1866 | N/A | 0.0000 | 10.0000 | 17.5000 |
| proxy | R_CONDITIONAL_seq | 5 | 20 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 42.5000 | 33.1250 | 35.4550 | N/A | 0.0000 | 12.0833 | 17.2083 |
| proxy | R_CONDITIONAL_seq | 5 | 20 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 27.0833 | 33.3750 | 28.6340 | N/A | 0.0000 | 18.7500 | 8.8750 |
| proxy | R_CONDITIONAL_seq | 5 | 20 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 42.9167 | 32.5000 | 34.9240 | N/A | 0.0000 | 13.7500 | 19.2500 |
| proxy | R_CONDITIONAL_seq | 5 | 20 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 25.0000 | 35.0000 | 28.2637 | N/A | 0.0000 | 21.6667 | 10.0000 |
| proxy | R_CONDITIONAL_seq | 5 | 5 | 2026092701 | rx1 | 2 | N/A | 54.5833 | 46.2500 | 42.5000 | 41.5664 | N/A | 0.0000 | 8.3333 | 18.0833 |
| proxy | R_CONDITIONAL_seq | 5 | 5 | 2026092701 | rx3 | 2 | N/A | 45.8333 | 32.0833 | 45.5000 | 36.0222 | N/A | 0.0000 | 13.7500 | 15.5833 |
| proxy | R_CONDITIONAL_seq | 5 | 5 | 2026092702 | rx1 | 2 | N/A | 56.6667 | 47.9167 | 37.0000 | 38.9615 | N/A | 0.0000 | 8.7500 | 21.9167 |
| proxy | R_CONDITIONAL_seq | 5 | 5 | 2026092702 | rx3 | 2 | N/A | 46.6667 | 33.7500 | 48.0000 | 38.1218 | N/A | 0.0000 | 12.9167 | 15.4167 |

模型与资源口径

目标仅为 pooled class-RMS CE。分类梯度为归档的 g_Z，不包含 +Z 近端梯度；坐标受 0.5 硬球约束，Armijo 使用投影后的实际 delta。
正条件核前向分别分解 projection 与 residual 系统；每个系统各两次 triangular solve，另计三项谱诊断。完整伴随复用因子，两组伴随分别记 RHS。
旧训练锚点 residual=0 是核约束，不等于所有未来 query 保旧保证；远处常数项 v 也不能忽略。
以下保留所有实测资源字段；n²×RHS、数组字节和参数计数为数值代理，不是实测 FLOPs、设备峰值或完整部署包。未测项为 N/A；嵌套累计计时不可相加当墙钟。

| 资源字段 | 原值 |
|---|---|
| parent_wall_seconds_sum | 5282.9065068588825 |
| base_fit_maximum_training_feature_bytes | 2143232 |
| base_fit_maximum_gram_bytes | 1059968 |
| base_fit_maximum_kernel_bytes | 1059968 |
| base_fit_maximum_coefficient_bytes | 75712 |
| base_fit_effective_degrees_of_freedom_seconds_sum | 0.3416131058038445 |
| base_fit_solve_seconds_sum | 0.4591314028657507 |
| base_fit_gram_seconds_sum | 50.18323901918484 |
| base_fit_fit_seconds_sum | 50.98563895659754 |
| base_fit_score_seconds_sum | 152.78482149064075 |
| base_fit_maximum_persistent_state_bytes | 2224800 |
| base_fit_fit_and_score_seconds_sum | 224.7882036791416 |
| conditional_preparation_preparation_seconds_sum | 532.1651418876863 |
| conditional_preparation_maximum_prepared_numeric_state_bytes | 4360352 |
| conditional_preparation_maximum_retained_vector_record_bytes | 0 |
| candidate_fit_fit_seconds_sum | 3006.230551746179 |
| candidate_fit_maximum_resident_numeric_state_bytes | 21048224 |
| candidate_fit_maximum_deployment_numeric_state_bytes | 8146896 |
| candidate_fit_maximum_retained_vector_record_bytes | 0 |
| candidate_fit_maximum_prepared_numeric_state_bytes | 4360352 |
| candidate_fit_maximum_persistent_state_bytes | 21048224 |
| candidate_fit_score_seconds_sum | 767.3000077917677 |
| objective_forward_seconds_sum | 873.7529089414311 |
| objective_backward_seconds_sum | 577.6857958019973 |
| final_head_forward_seconds_sum | 78.26221300986072 |
| conditional_preparation_prior_score_seconds_sum | 289.07384882833867 |
| candidate_fit_maximum_max_forward_cache_bytes | 59329624 |
| candidate_fit_maximum_max_simultaneous_forward_cache_bytes | 90305072 |
| outer_residual_raw_distance_evaluation_count_sum | 407680 |
| outer_residual_raw_distance_pair_count_sum | 12305936 |
| outer_residual_reference_distance_evaluation_count_sum | 407680 |
| outer_residual_reference_distance_pair_count_sum | 5180448 |
| outer_residual_kernel_evaluation_count_sum | 382200 |
| outer_residual_kernel_pair_count_sum | 8695088 |
| outer_residual_adapter_physical_evaluation_count_sum | 382200 |
| outer_residual_dictionary_physical_evaluation_count_sum | 382200 |
| outer_residual_intercept_addition_count_sum | 5447400 |
| outer_residual_score_seconds_sum | 480.15049507997173 |
| outer_prior_raw_distance_evaluation_count_sum | 273280 |
| outer_prior_raw_distance_pair_count_sum | 3472608 |
| outer_prior_reference_distance_evaluation_count_sum | 273280 |
| outer_prior_reference_distance_pair_count_sum | 3472608 |
| outer_prior_kernel_evaluation_count_sum | 256200 |
| outer_prior_kernel_pair_count_sum | 2453664 |
| outer_prior_adapter_physical_evaluation_count_sum | 256200 |
| outer_prior_dictionary_physical_evaluation_count_sum | 256200 |
| outer_prior_intercept_addition_count_sum | 1537200 |
| outer_prior_score_seconds_sum | 284.06225020013517 |
| actual_counters | {"episodes": 160, "k1_episodes": 40, "oof_episodes": 120, "proxy_anchor_count": 1400, "sequence_paths": 1800, "baseline_head_fit_count": 3240, "baseline_factorization_count": 3240, "baseline_head_triangular_solve_count": 6480, "baseline_effective_df_triangular_solve_count": 6480, "baseline_triangular_solve_count": 12960, "head_fit_count": 17064, "factorization_count": 22824, "trained_conditional_stage_count": 648, "diagnostic_fit_count": 0, "candidate_preparation_count": 3240, "candidate_stage_count": 3240, "final_score_evaluation_count": 6336, "final_score_physical_count": 764400, "ajlr_preparation_count": 3240, "latent_svd_count": 3240, "dictionary_physical_evaluation_count": 77168, "prepared_distance_evaluation_count": 8856, "original_distance_pair_count": 14492912, "prior_head_fit_count": 864, "prior_factorization_count": 864, "prior_triangular_solve_count": 1728, "prior_triangular_rhs_count": 12096, "prior_triangular_rhs_element_count": 376320, "prior_triangular_dense_work_unit_count": 15160320, "prior_intercept_fit_count": 864, "prior_intercept_addition_count": 1167168, "prior_score_evaluation_count": 3168, "prior_score_physical_count": 154208, "reference_distance_evaluation_count": 314248, "reference_distance_pair_count": 40441536, "raw_distance_evaluation_count": 310504, "raw_distance_pair_count": 63990128, "kernel_evaluation_count": 178616, "kernel_pair_count": 88386192, "adapter_physical_evaluation_count": 1036096, "ajlr_stage_count": 3240, "optimizer_steps": 2592, "optimizer_iterations": 2592, "inner_objective_evaluation_count": 3240, "inner_head_fit_count": 9720, "inner_factorization_count": 14040, "final_head_fit_count": 3240, "final_factorization_count": 4680, "head_triangular_solve_count": 37440, "head_triangular_rhs_count": 765152, "head_triangular_rhs_element_count": 32018336, "head_triangular_dense_work_unit_count": 2148468896, "intercept_fit_count": 12960, "intercept_addition_count": 11994656, "derivative_triangular_solve_count": 22464, "ce_adjoint_solve_count": 7776, "derivative_triangular_rhs_count": 262656, "derivative_triangular_rhs_element_count": 11621120, "derivative_triangular_dense_work_unit_count": 937473280, "backward_evaluation_count": 2592, "accepted_trial_count": 2592, "rejected_trial_count": 0, "trial_count": 2592, "trial_attempt_count": 2592, "ajlr_forward_evaluation_count": 12960, "projection_factorization_count": 5760, "projection_triangular_solve_count": 11520, "projection_triangular_rhs_count": 488672, "projection_triangular_rhs_element_count": 19625280, "projection_triangular_dense_work_unit_count": 968031360, "residual_factorization_count": 12960, "residual_triangular_solve_count": 25920, "residual_triangular_rhs_count": 276480, "residual_triangular_rhs_element_count": 12393056, "residual_triangular_dense_work_unit_count": 1180437536, "projection_adjoint_factorization_count": 0, "projection_adjoint_triangular_solve_count": 6912, "projection_adjoint_triangular_rhs_count": 105408, "projection_adjoint_triangular_rhs_element_count": 3279360, "projection_adjoint_triangular_dense_work_unit_count": 132111360, "residual_adjoint_factorization_count": 0, "residual_adjoint_triangular_solve_count": 15552, "residual_adjoint_triangular_rhs_count": 157248, "residual_adjoint_triangular_rhs_element_count": 8341760, "residual_adjoint_triangular_dense_work_unit_count": 805361920, "spectral_diagnostic_count": 17280, "completed_factorization_count": 18720} |
| structural_budget | {"exact": {"episodes": 160, "k1_episodes": 40, "oof_episodes": 120, "proxy_anchor_count": 1400, "sequence_paths": 1800, "baseline_head_fit_count": 3240, "candidate_preparation_count": 3240, "candidate_stage_count": 3240, "diagnostic_fit_count": 0}, "maximum": {"trained_conditional_stage_count": 648, "optimizer_steps": 2592, "optimizer_iterations": 2592, "trial_count": 31104, "trial_attempt_count": 31104, "inner_objective_evaluation_count": 31752, "inner_head_fit_count": 95256, "inner_factorization_count": 137592, "final_head_fit_count": 3240, "final_factorization_count": 4680, "prior_head_fit_count": 864, "prior_factorization_count": 864, "baseline_factorization_count": 3240, "head_fit_count": 102600, "factorization_count": 146376, "projection_factorization_count": 43776, "residual_factorization_count": 98496, "spectral_diagnostic_count": 131328}} |
| run_wall_seconds | 2876.2192981243134 |
| lane_wall_seconds_sum | 5699.25445125901 |
| maximum_lane_peak_rss_bytes | 1154347008 |
| peak_gpu_memory_bytes | N/A |
| deployment_package_bytes | N/A |
| incremental_transmission_bytes | N/A |
| additional_ground_data_payload_bytes | 0 |
| additional_ground_statistics_bytes | 0 |
| separate_spectral_seconds | N/A |
| independent_analysis_work | {"independent_general_solve_count": 7776, "independent_general_rhs_column_count": 78624, "independent_general_rhs_element_count": 5889184, "independent_dense_cubic_work_unit_count": 5885701216, "independent_dense_rhs_work_unit_count": 751517344, "independent_spectral_diagnostic_count": 0} |
| triangular_rhs_count_scope | "Per actual system: columns=r; elements=n*r; dense=n*n*r; proxy not measured FLOPs" |
| resident_state_scope | "Core resident numeric buffers include actual B; archive bytes, minimum deployment and process RSS are distinct. Alias layouts cannot be reconstructed from NPZ copies." |
| positive_kernel_evidence | "Independently rebuilt raw Gaussian/equivalence geometry plus actual saved projection/residual factor residuals; residual-only certificate positive_kernel_checked remains false" |
| unmeasured_reason | "CPU only; no deployment transport/energy measured; three candidate spectral diagnostics included in head timing but not independently timed" |
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
| factorization_count | 22824 |
| trained_conditional_stage_count | 648 |
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
| inner_factorization_count | 14040 |
| final_head_fit_count | 3240 |
| final_factorization_count | 4680 |
| head_triangular_solve_count | 37440 |
| head_triangular_rhs_count | 765152 |
| head_triangular_rhs_element_count | 32018336 |
| head_triangular_dense_work_unit_count | 2148468896 |
| intercept_fit_count | 12960 |
| intercept_addition_count | 11994656 |
| derivative_triangular_solve_count | 22464 |
| ce_adjoint_solve_count | 7776 |
| derivative_triangular_rhs_count | 262656 |
| derivative_triangular_rhs_element_count | 11621120 |
| derivative_triangular_dense_work_unit_count | 937473280 |
| backward_evaluation_count | 2592 |
| accepted_trial_count | 2592 |
| rejected_trial_count | 0 |
| trial_count | 2592 |
| trial_attempt_count | 2592 |
| ajlr_forward_evaluation_count | 12960 |
| projection_factorization_count | 5760 |
| projection_triangular_solve_count | 11520 |
| projection_triangular_rhs_count | 488672 |
| projection_triangular_rhs_element_count | 19625280 |
| projection_triangular_dense_work_unit_count | 968031360 |
| residual_factorization_count | 12960 |
| residual_triangular_solve_count | 25920 |
| residual_triangular_rhs_count | 276480 |
| residual_triangular_rhs_element_count | 12393056 |
| residual_triangular_dense_work_unit_count | 1180437536 |
| projection_adjoint_factorization_count | 0 |
| projection_adjoint_triangular_solve_count | 6912 |
| projection_adjoint_triangular_rhs_count | 105408 |
| projection_adjoint_triangular_rhs_element_count | 3279360 |
| projection_adjoint_triangular_dense_work_unit_count | 132111360 |
| residual_adjoint_factorization_count | 0 |
| residual_adjoint_triangular_solve_count | 15552 |
| residual_adjoint_triangular_rhs_count | 157248 |
| residual_adjoint_triangular_rhs_element_count | 8341760 |
| residual_adjoint_triangular_dense_work_unit_count | 805361920 |
| spectral_diagnostic_count | 17280 |
| completed_factorization_count | 18720 |

独立分析来源：`E:\type10-7\automation_reports\CV-SincNet\20261001-phase2-d92-conditional-joint-support-m2-r01\results\support_summary\summary.json`；训练 commit：`7204161b126a9d3a17096aee5c75bce8ccfee6ec`；分析 commit：`0145234e381b5799306acfb3d36a4e17865c1814`。
