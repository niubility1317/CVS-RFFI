# 独立时频学生的源域交叉拟合复滤波诊断

固定两个独立学生和原 DURA，对 672 条已探索源 pilot 使用逐记录源 clean 半窗拟合复标量或 17 抽头复 FIR，在另一半窗评估；再交换两半并平均。评分只使用 8–247 采样点，每半 120 点；滤波器的预测 IQ 窗口可跨半窗边界，但拟合不读取留出半窗的 clean。岭系数为 Gram 矩阵平均对角线的 0.001 倍，设置在诊断前固定。

该校准需要部署时不可得的源 clean 参考，结果不是可部署性能、TX 身份准确率或一般上界；IQ 误差按各留出半窗 clean 能量归一，所以不能与此前整窗归一结果直接比较。未读 F、TX 标签、V 或目标，也未更新模型；真实 checkpoint 的 16 条源 smoke 通过。

| 源组 | 固定信号 | 未校准 NMSE | 交叉拟合标量 | 交叉拟合 FIR | FIR 优于未校准数 / 48 |
| --- | --- | ---: | ---: | ---: | ---: |
| RX4_6/clean | received | 0.000000 | 0.000000 | 0.000076 | 0 |
| RX4_6/clean | student_8192 | 0.053191 | 0.052593 | 0.049339 | 34 |
| RX4_6/clean | student_32768 | 0.031545 | 0.030724 | 0.036238 | 7 |
| RX4_6/clean | dura | 0.000044 | 0.000043 | 0.000118 | 0 |
| RX4_6/practical_high | received | 1.905722 | 0.999849 | 0.596752 | 48 |
| RX4_6/practical_high | student_8192 | 0.227050 | 0.219008 | 0.202386 | 37 |
| RX4_6/practical_high | student_32768 | 0.174348 | 0.166017 | 0.144771 | 45 |
| RX4_6/practical_high | dura | 0.005173 | 0.004297 | 0.004166 | 44 |
| RX4_6/practical_mid | received | 1.892779 | 1.000699 | 0.593786 | 48 |
| RX4_6/practical_mid | student_8192 | 0.244774 | 0.238865 | 0.220921 | 40 |
| RX4_6/practical_mid | student_32768 | 0.176716 | 0.172486 | 0.149380 | 41 |
| RX4_6/practical_mid | dura | 0.007981 | 0.007012 | 0.006856 | 43 |
| RX4_6/practical_low_suburban | received | 1.908749 | 1.001136 | 0.579641 | 48 |
| RX4_6/practical_low_suburban | student_8192 | 0.230781 | 0.225798 | 0.201468 | 43 |
| RX4_6/practical_low_suburban | student_32768 | 0.164899 | 0.160583 | 0.128882 | 45 |
| RX4_6/practical_low_suburban | dura | 0.008704 | 0.007806 | 0.008473 | 35 |
| RX4_6/practical_high_urban | received | 1.891886 | 0.999178 | 0.595845 | 48 |
| RX4_6/practical_high_urban | student_8192 | 0.260047 | 0.258210 | 0.241572 | 39 |
| RX4_6/practical_high_urban | student_32768 | 0.182049 | 0.178995 | 0.152755 | 45 |
| RX4_6/practical_high_urban | dura | 0.009113 | 0.007814 | 0.006863 | 40 |
| RX4_6/practical_mid_urban | received | 1.943564 | 0.999077 | 0.656146 | 48 |
| RX4_6/practical_mid_urban | student_8192 | 0.410548 | 0.424727 | 0.411337 | 31 |
| RX4_6/practical_mid_urban | student_32768 | 0.259723 | 0.257049 | 0.226522 | 39 |
| RX4_6/practical_mid_urban | dura | 0.022914 | 0.020706 | 0.023636 | 31 |
| RX4_6/practical_low_urban | received | 1.954952 | 1.001617 | 0.744280 | 48 |
| RX4_6/practical_low_urban | student_8192 | 0.588849 | 0.604088 | 0.602931 | 23 |
| RX4_6/practical_low_urban | student_32768 | 0.357628 | 0.356824 | 0.318671 | 45 |
| RX4_6/practical_low_urban | dura | 0.036427 | 0.034723 | 0.040413 | 13 |
| RX8/clean | received | 0.000000 | 0.000000 | 0.000067 | 0 |
| RX8/clean | student_8192 | 0.089947 | 0.090476 | 0.091201 | 26 |
| RX8/clean | student_32768 | 0.028004 | 0.028266 | 0.027794 | 28 |
| RX8/clean | dura | 0.000032 | 0.000031 | 0.000090 | 0 |
| RX8/practical_high | received | 2.027095 | 1.000898 | 0.747763 | 48 |
| RX8/practical_high | student_8192 | 0.450499 | 0.449473 | 0.451114 | 29 |
| RX8/practical_high | student_32768 | 0.280032 | 0.276264 | 0.257935 | 40 |
| RX8/practical_high | dura | 0.013428 | 0.012264 | 0.013496 | 25 |
| RX8/practical_mid | received | 2.008095 | 1.000010 | 0.724750 | 48 |
| RX8/practical_mid | student_8192 | 0.432200 | 0.433065 | 0.426826 | 31 |
| RX8/practical_mid | student_32768 | 0.264681 | 0.261059 | 0.240833 | 42 |
| RX8/practical_mid | dura | 0.014190 | 0.012902 | 0.013529 | 30 |
| RX8/practical_low_suburban | received | 2.005733 | 1.000380 | 0.746925 | 48 |
| RX8/practical_low_suburban | student_8192 | 0.472559 | 0.471771 | 0.466790 | 30 |
| RX8/practical_low_suburban | student_32768 | 0.309142 | 0.304952 | 0.288820 | 40 |
| RX8/practical_low_suburban | dura | 0.023647 | 0.022214 | 0.025069 | 24 |
| RX8/practical_high_urban | received | 1.976486 | 1.000606 | 0.729575 | 48 |
| RX8/practical_high_urban | student_8192 | 0.467149 | 0.469104 | 0.470803 | 24 |
| RX8/practical_high_urban | student_32768 | 0.294649 | 0.293320 | 0.282319 | 38 |
| RX8/practical_high_urban | dura | 0.013902 | 0.012855 | 0.013024 | 31 |
| RX8/practical_mid_urban | received | 1.984085 | 1.002373 | 0.770680 | 48 |
| RX8/practical_mid_urban | student_8192 | 0.586734 | 0.589684 | 0.586242 | 23 |
| RX8/practical_mid_urban | student_32768 | 0.390621 | 0.388000 | 0.358005 | 38 |
| RX8/practical_mid_urban | dura | 0.050354 | 0.048741 | 0.052868 | 24 |
| RX8/practical_low_urban | received | 2.003709 | 1.001425 | 0.816577 | 48 |
| RX8/practical_low_urban | student_8192 | 0.658063 | 0.644441 | 0.653549 | 27 |
| RX8/practical_low_urban | student_32768 | 0.478610 | 0.462767 | 0.454161 | 35 |
| RX8/practical_low_urban | dura | 0.099709 | 0.093317 | 0.101735 | 21 |

低城区 RX4/6：32768 学生 0.357628→0.318671，下降 10.89%（45/48 条改善）；RX8 为 0.478610→0.454161，下降 5.11%（35/48 条改善）。DURA 对应两组为 0.036427→0.040413 和 0.099709→0.101735，平均误差上升。8192 学生在 RX4/6 的平均误差也上升。交叉拟合反映这组源 IQ 中可由有 clean 校准的固定 FIR 修正的一部分差异，不能证明无需 clean 的系数估计或身份识别收益。

两个学生的源曝光和优化步数不同；pilot 已被探索，原 DURA 可能见过同源域。大样本学生的七景测试按既定计划继续。
