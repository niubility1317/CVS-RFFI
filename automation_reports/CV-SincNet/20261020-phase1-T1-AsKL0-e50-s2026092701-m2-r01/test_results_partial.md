# T1 E50：已完成续训组的详细测试结果

核实时间：2026-10-20 11:48:28 UTC+8。run `20261020-phase1-T1-AsKL0-e50-s2026092701-m2-r01`，model seed `2026092701`，split `392005`。

本次只完成两条路径中的续训组：原T1 E30学生及自身EMA追加20轮，累计E50。scratch组截至本次读回完成E40/50，仍在健康训练，尚无冻结最终checkpoint或正式测试分数；不能将本次单行结果视为两组完整比较。

两组既定测试范围为clean＋六种practical。续训组7×168000＝1176000条预测均已完整固定，再由独立CPU scorer读取truth，126条overall/RX/day/TX指标完整；status=TRAINED_AND_SCORED/SCORED_COMPLETE，independent_recount=VERIFIED。原E30与续训组引用同一个固定formal_manifest、相同视图和168000个样本；本文再次用保存的混淆矩阵复算accuracy与Macro-F1一致，不重新运行预测或读取truth。

以下accuracy和Macro-F1均以百分数表示，差值单位为百分点。Macro-F1是六个TX类别F1的均值；跨环境均值是各场景指标的算术均值，不是合并混淆矩阵后的F1。

| 环境 | 原E30 accuracy | E30＋20 accuracy | Δaccuracy | 原E30 Macro-F1 | E30＋20 Macro-F1 | ΔMacro-F1 | 净增正确数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 85.0720 | 85.2458 | +0.1738 | 84.7851 | 84.9731 | +0.1881 | +292 |
| practical_high | 81.0405 | 81.2982 | +0.2577 | 80.7375 | 81.0618 | +0.3243 | +433 |
| practical_mid | 78.3411 | 78.8506 | +0.5095 | 78.0565 | 78.6147 | +0.5583 | +856 |
| practical_low_suburban | 75.6173 | 76.2393 | +0.6220 | 75.3454 | 75.9666 | +0.6212 | +1045 |
| practical_high_urban | 77.4101 | 77.9274 | +0.5173 | 77.1047 | 77.6970 | +0.5923 | +869 |
| practical_mid_urban | 62.6149 | 63.7024 | +1.0875 | 62.2891 | 63.3225 | +1.0335 | +1827 |
| practical_low_urban | 51.1780 | 52.4161 | +1.2381 | 50.7736 | 51.7733 | +0.9996 | +2080 |

| 场景组 | 原E30 accuracy均值 | E30＋20 accuracy均值 | Δaccuracy | E30＋20 Macro-F1均值 | ΔMacro-F1 |
|---|---:|---:|---:|---:|---:|
| 六种星地环境 | 71.0336 | 71.7390 | +0.7054 | 71.4060 | +0.6882 |
| 城市三场景 | 63.7343 | 64.6819 | +0.9476 | 64.2643 | +0.8751 |
| 非城市三场景 | 78.3329 | 78.7960 | +0.4631 | 78.5477 | +0.5012 |

## 目标接收机分层

原始RX编号与TX类别编号是不同角色。每个RX在每个视图有24000条测试样本。下表为续训E50准确率与相对原E30差值。

| 目标RX | clean | Δclean | mid urban | Δmid urban | low urban | Δlow urban |
|---|---:|---:|---:|---:|---:|---:|
| 1-1 | 80.5833 | +1.0625 | 62.7583 | +1.2708 | 51.8083 | +1.4500 |
| 14-7 | 77.1292 | -1.2958 | 57.4333 | +0.4750 | 47.2417 | +0.3500 |
| 2-1 | 92.3042 | +0.2667 | 67.0375 | +1.0583 | 55.2958 | +1.3083 |
| 20-1 | 83.1333 | +0.8333 | 60.6750 | +2.0542 | 49.9583 | +1.7625 |
| 7-14 | 94.4208 | +0.1667 | 68.9583 | +0.9375 | 56.7875 | +1.1917 |
| 7-7 | 87.0833 | -0.2375 | 65.3792 | +1.1750 | 53.0625 | +1.4958 |
| 8-8 | 82.0667 | +0.4208 | 63.6750 | +0.6417 | 52.7583 | +1.1083 |

## TX类别识别率

各TX识别率为该真实TX样本中预测正确的比例，即该类召回率；每类每场景28000条。

| TX | clean | Δclean | low urban | Δlow urban |
|---|---:|---:|---:|---:|
| 14-10 | 97.2500 | +0.6536 | 52.0250 | +0.9607 |
| 14-7 | 64.3500 | +2.2679 | 40.1286 | +1.2000 |
| 20-15 | 78.4607 | -0.0143 | 48.4679 | +0.5929 |
| 20-19 | 72.5214 | -2.1464 | 34.3321 | -0.1857 |
| 6-15 | 99.8250 | +0.0393 | 55.5821 | +1.8750 |
| 8-20 | 99.0679 | +0.2429 | 83.9607 | +2.9857 |

## 完整RX/day/TX测试指标

| 场景 | 分层维度 | 原始编号 | 数量 | accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| clean | receiver | 1-1 | 24000 | 80.5833 | 78.2711 |
| clean | receiver | 14-7 | 24000 | 77.1292 | 75.3934 |
| clean | receiver | 2-1 | 24000 | 92.3042 | 92.2939 |
| clean | receiver | 20-1 | 24000 | 83.1333 | 83.0705 |
| clean | receiver | 7-14 | 24000 | 94.4208 | 94.3850 |
| clean | receiver | 7-7 | 24000 | 87.0833 | 86.2954 |
| clean | receiver | 8-8 | 24000 | 82.0667 | 80.4655 |
| clean | day | 2021_03_01 | 42000 | 82.1119 | 81.6527 |
| clean | day | 2021_03_08 | 42000 | 86.3476 | 86.0289 |
| clean | day | 2021_03_15 | 42000 | 82.7667 | 82.3818 |
| clean | day | 2021_03_23 | 42000 | 89.7571 | 89.6124 |
| clean | transmitter | 0 | 28000 | 97.2500 | 16.4343 |
| clean | transmitter | 1 | 28000 | 64.3500 | 13.0514 |
| clean | transmitter | 2 | 28000 | 78.4607 | 14.6551 |
| clean | transmitter | 3 | 28000 | 72.5214 | 14.0121 |
| clean | transmitter | 4 | 28000 | 99.8250 | 16.6521 |
| clean | transmitter | 5 | 28000 | 99.0679 | 16.5886 |
| practical_high | receiver | 1-1 | 24000 | 79.6208 | 77.3321 |
| practical_high | receiver | 14-7 | 24000 | 71.7875 | 71.2130 |
| practical_high | receiver | 2-1 | 24000 | 85.4833 | 85.0987 |
| practical_high | receiver | 20-1 | 24000 | 78.6958 | 78.5838 |
| practical_high | receiver | 7-14 | 24000 | 88.6500 | 88.3568 |
| practical_high | receiver | 7-7 | 24000 | 84.1333 | 83.2557 |
| practical_high | receiver | 8-8 | 24000 | 80.7167 | 79.7571 |
| practical_high | day | 2021_03_01 | 42000 | 78.1571 | 77.7593 |
| practical_high | day | 2021_03_08 | 42000 | 83.4643 | 83.1689 |
| practical_high | day | 2021_03_15 | 42000 | 80.4667 | 80.1043 |
| practical_high | day | 2021_03_23 | 42000 | 83.1048 | 82.8957 |
| practical_high | transmitter | 0 | 28000 | 87.3929 | 15.5454 |
| practical_high | transmitter | 1 | 28000 | 56.9857 | 12.1000 |
| practical_high | transmitter | 2 | 28000 | 77.8464 | 14.5906 |
| practical_high | transmitter | 3 | 28000 | 70.5893 | 13.7932 |
| practical_high | transmitter | 4 | 28000 | 98.3143 | 16.5250 |
| practical_high | transmitter | 5 | 28000 | 96.6607 | 16.3837 |
| practical_mid | receiver | 1-1 | 24000 | 77.3208 | 75.3503 |
| practical_mid | receiver | 14-7 | 24000 | 69.9917 | 69.5107 |
| practical_mid | receiver | 2-1 | 24000 | 82.5083 | 82.1395 |
| practical_mid | receiver | 20-1 | 24000 | 75.9875 | 75.8833 |
| practical_mid | receiver | 7-14 | 24000 | 85.8875 | 85.6244 |
| practical_mid | receiver | 7-7 | 24000 | 81.4875 | 80.6994 |
| practical_mid | receiver | 8-8 | 24000 | 78.7708 | 77.9418 |
| practical_mid | day | 2021_03_01 | 42000 | 75.9857 | 75.6092 |
| practical_mid | day | 2021_03_08 | 42000 | 80.9405 | 80.6177 |
| practical_mid | day | 2021_03_15 | 42000 | 78.0119 | 77.6498 |
| practical_mid | day | 2021_03_23 | 42000 | 80.4643 | 80.2773 |
| practical_mid | transmitter | 0 | 28000 | 84.7857 | 15.2944 |
| practical_mid | transmitter | 1 | 28000 | 56.1929 | 11.9922 |
| practical_mid | transmitter | 2 | 28000 | 75.4250 | 14.3319 |
| practical_mid | transmitter | 3 | 28000 | 66.0536 | 13.2595 |
| practical_mid | transmitter | 4 | 28000 | 94.7964 | 16.2215 |
| practical_mid | transmitter | 5 | 28000 | 95.8500 | 16.3135 |
| practical_low_suburban | receiver | 1-1 | 24000 | 74.6250 | 72.8246 |
| practical_low_suburban | receiver | 14-7 | 24000 | 67.6792 | 67.1883 |
| practical_low_suburban | receiver | 2-1 | 24000 | 79.9833 | 79.5497 |
| practical_low_suburban | receiver | 20-1 | 24000 | 73.5875 | 73.4718 |
| practical_low_suburban | receiver | 7-14 | 24000 | 82.3125 | 82.0448 |
| practical_low_suburban | receiver | 7-7 | 24000 | 78.7667 | 77.9660 |
| practical_low_suburban | receiver | 8-8 | 24000 | 76.7208 | 75.8773 |
| practical_low_suburban | day | 2021_03_01 | 42000 | 73.4452 | 73.0442 |
| practical_low_suburban | day | 2021_03_08 | 42000 | 77.9810 | 77.6284 |
| practical_low_suburban | day | 2021_03_15 | 42000 | 75.7071 | 75.2910 |
| practical_low_suburban | day | 2021_03_23 | 42000 | 77.8238 | 77.5952 |
| practical_low_suburban | transmitter | 0 | 28000 | 81.6214 | 14.9801 |
| practical_low_suburban | transmitter | 1 | 28000 | 54.8964 | 11.8136 |
| practical_low_suburban | transmitter | 2 | 28000 | 72.9857 | 14.0639 |
| practical_low_suburban | transmitter | 3 | 28000 | 60.9393 | 12.6216 |
| practical_low_suburban | transmitter | 4 | 28000 | 91.8571 | 15.9593 |
| practical_low_suburban | transmitter | 5 | 28000 | 95.1357 | 16.2512 |
| practical_high_urban | receiver | 1-1 | 24000 | 76.5125 | 74.5126 |
| practical_high_urban | receiver | 14-7 | 24000 | 69.3583 | 68.8491 |
| practical_high_urban | receiver | 2-1 | 24000 | 81.5792 | 81.2019 |
| practical_high_urban | receiver | 20-1 | 24000 | 75.2917 | 75.1769 |
| practical_high_urban | receiver | 7-14 | 24000 | 85.0375 | 84.7529 |
| practical_high_urban | receiver | 7-7 | 24000 | 80.2042 | 79.3881 |
| practical_high_urban | receiver | 8-8 | 24000 | 77.5083 | 76.5773 |
| practical_high_urban | day | 2021_03_01 | 42000 | 74.7905 | 74.3954 |
| practical_high_urban | day | 2021_03_08 | 42000 | 80.2048 | 79.9069 |
| practical_high_urban | day | 2021_03_15 | 42000 | 77.1167 | 76.7823 |
| practical_high_urban | day | 2021_03_23 | 42000 | 79.5976 | 79.3757 |
| practical_high_urban | transmitter | 0 | 28000 | 82.8786 | 15.1063 |
| practical_high_urban | transmitter | 1 | 28000 | 55.3750 | 11.8799 |
| practical_high_urban | transmitter | 2 | 28000 | 74.5179 | 14.2331 |
| practical_high_urban | transmitter | 3 | 28000 | 66.2714 | 13.2858 |
| practical_high_urban | transmitter | 4 | 28000 | 92.7893 | 16.0433 |
| practical_high_urban | transmitter | 5 | 28000 | 95.7321 | 16.3033 |
| practical_mid_urban | receiver | 1-1 | 24000 | 62.7583 | 61.5955 |
| practical_mid_urban | receiver | 14-7 | 24000 | 57.4333 | 56.7585 |
| practical_mid_urban | receiver | 2-1 | 24000 | 67.0375 | 66.6388 |
| practical_mid_urban | receiver | 20-1 | 24000 | 60.6750 | 60.4721 |
| practical_mid_urban | receiver | 7-14 | 24000 | 68.9583 | 68.5143 |
| practical_mid_urban | receiver | 7-7 | 24000 | 65.3792 | 64.6178 |
| practical_mid_urban | receiver | 8-8 | 24000 | 63.6750 | 62.7494 |
| practical_mid_urban | day | 2021_03_01 | 42000 | 61.5143 | 61.0037 |
| practical_mid_urban | day | 2021_03_08 | 42000 | 65.9095 | 65.4008 |
| practical_mid_urban | day | 2021_03_15 | 42000 | 62.8190 | 62.3648 |
| practical_mid_urban | day | 2021_03_23 | 42000 | 64.5667 | 64.2713 |
| practical_mid_urban | transmitter | 0 | 28000 | 64.9000 | 13.1191 |
| practical_mid_urban | transmitter | 1 | 28000 | 48.0036 | 10.8114 |
| practical_mid_urban | transmitter | 2 | 28000 | 60.0464 | 12.5060 |
| practical_mid_urban | transmitter | 3 | 28000 | 47.7429 | 10.7716 |
| practical_mid_urban | transmitter | 4 | 28000 | 70.6964 | 13.8055 |
| practical_mid_urban | transmitter | 5 | 28000 | 90.8250 | 15.8653 |
| practical_low_urban | receiver | 1-1 | 24000 | 51.8083 | 50.9794 |
| practical_low_urban | receiver | 14-7 | 24000 | 47.2417 | 46.1562 |
| practical_low_urban | receiver | 2-1 | 24000 | 55.2958 | 54.7351 |
| practical_low_urban | receiver | 20-1 | 24000 | 49.9583 | 49.4564 |
| practical_low_urban | receiver | 7-14 | 24000 | 56.7875 | 56.0606 |
| practical_low_urban | receiver | 7-7 | 24000 | 53.0625 | 52.1054 |
| practical_low_urban | receiver | 8-8 | 24000 | 52.7583 | 51.8306 |
| practical_low_urban | day | 2021_03_01 | 42000 | 50.8167 | 50.1441 |
| practical_low_urban | day | 2021_03_08 | 42000 | 54.2857 | 53.4437 |
| practical_low_urban | day | 2021_03_15 | 42000 | 52.3690 | 51.5936 |
| practical_low_urban | day | 2021_03_23 | 42000 | 52.1929 | 51.7453 |
| practical_low_urban | transmitter | 0 | 28000 | 52.0250 | 11.4071 |
| practical_low_urban | transmitter | 1 | 28000 | 40.1286 | 9.5457 |
| practical_low_urban | transmitter | 2 | 28000 | 48.4679 | 10.8818 |
| practical_low_urban | transmitter | 3 | 28000 | 34.3321 | 8.5192 |
| practical_low_urban | transmitter | 4 | 28000 | 55.5821 | 11.9084 |
| practical_low_urban | transmitter | 5 | 28000 | 83.9607 | 15.2135 |

## 解释与成本

全部七场景的整体accuracy与Macro-F1均较原E30上升，中低城市场景收益更大。clean整体只净增292条正确预测；mid urban净增1827条、low urban净增2080条。这里是正确总数的净差，不是逐样本修复/破坏分解。

收益并不覆盖所有RX或TX。最差目标RX为14-7，clean为77.1292%，比原E30下降1.2958个百分点；其中低城市分别改善0.4750/0.3500个百分点。TX20-19的clean下降2.1464个百分点，low urban下降0.1857个百分点；困难类别混淆未完全解决。

新增E20训练实测39.10分钟，4440次更新；七场景预测共57.39秒（含该预测进程的加载及存盘）。新增L前向样本3409920，As观测284160；继承的E30成本单独保留。源V27000仅作健康诊断，最终accuracy为98.4148%，不替代目标测试。

当前只有一个模型seed，未做多seed显著性或稳定性结论。原T1 E30、E30＋20和scratch50阶段、优化器重置与L/U日历不同；本次结果能说明这条既定续训路径的实际变化，不能单独归因于“加20轮”本身。测试基准已暴露，不声明新独立盲测；不据此修改正在训练的scratch路线、选模或重跑。

原始远端证据：本run/E30PLUS20-s2026092701/source/completion.json、prediction/complete.json、scoring/scores.json、scoring/complete.json及complete.json；原E30来自20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02/T1-s2026092701的同名产物。完整私有读回保留在local_artifacts/cvs_t1_e50/test_results_snapshot.json，不提交原始日志、checkpoint或机器状态。
