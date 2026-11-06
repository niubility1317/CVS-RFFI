# 独立时频学生源域相位与复增益诊断

固定 8192 源 ID 学生、32768 源 ID 学生和 DURA，在已探索的 672 条源 pilot 上比较原始 IQ 误差、逐记录理想相位对齐、逐记录理想复增益对齐。后两者均使用同条源 clean IQ，部署时不可获得；未读取 F、TX 标签、V 或目标，未更新模型。学生拟合 ID 与 pilot 互斥，DURA 可能见过同源域；真实 checkpoint 的 16 条源 smoke 通过。

| 源分组 | 固定模型 | 未对齐 NMSE | 理想相位 NMSE | 理想复增益 NMSE |
| --- | --- | ---: | ---: | ---: |
| RX4_6/clean | received | 0.000000 | 0.000000 | 0.000000 |
| RX4_6/clean | student_8192 | 0.052394 | 0.052166 | 0.051297 |
| RX4_6/clean | student_32768 | 0.031506 | 0.031402 | 0.030394 |
| RX4_6/clean | dura | 0.000046 | 0.000045 | 0.000044 |
| RX4_6/practical_high | received | 1.961178 | 1.926057 | 0.997716 |
| RX4_6/practical_high | student_8192 | 0.231577 | 0.228152 | 0.219814 |
| RX4_6/practical_high | student_32768 | 0.177297 | 0.173723 | 0.166489 |
| RX4_6/practical_high | dura | 0.005421 | 0.004495 | 0.004477 |
| RX4_6/practical_mid | received | 1.942067 | 1.913367 | 0.996663 |
| RX4_6/practical_mid | student_8192 | 0.252166 | 0.246420 | 0.239733 |
| RX4_6/practical_mid | student_32768 | 0.182160 | 0.178385 | 0.173705 |
| RX4_6/practical_mid | dura | 0.008028 | 0.006853 | 0.006800 |
| RX4_6/practical_low_suburban | received | 1.955433 | 1.925987 | 0.997803 |
| RX4_6/practical_low_suburban | student_8192 | 0.238616 | 0.233459 | 0.226773 |
| RX4_6/practical_low_suburban | student_32768 | 0.171009 | 0.167116 | 0.162354 |
| RX4_6/practical_low_suburban | dura | 0.008719 | 0.007625 | 0.007569 |
| RX4_6/practical_high_urban | received | 1.941086 | 1.904281 | 0.995820 |
| RX4_6/practical_high_urban | student_8192 | 0.266182 | 0.261637 | 0.255353 |
| RX4_6/practical_high_urban | student_32768 | 0.187794 | 0.184065 | 0.179348 |
| RX4_6/practical_high_urban | dura | 0.009094 | 0.007747 | 0.007704 |
| RX4_6/practical_mid_urban | received | 1.971055 | 1.905097 | 0.996867 |
| RX4_6/practical_mid_urban | student_8192 | 0.415410 | 0.405164 | 0.399017 |
| RX4_6/practical_mid_urban | student_32768 | 0.265770 | 0.253909 | 0.250641 |
| RX4_6/practical_mid_urban | dura | 0.023243 | 0.019571 | 0.019419 |
| RX4_6/practical_low_urban | received | 1.972738 | 1.891124 | 0.995823 |
| RX4_6/practical_low_urban | student_8192 | 0.593585 | 0.584485 | 0.574007 |
| RX4_6/practical_low_urban | student_32768 | 0.366779 | 0.356529 | 0.352714 |
| RX4_6/practical_low_urban | dura | 0.037404 | 0.032675 | 0.032451 |
| RX8/clean | received | 0.000000 | 0.000000 | 0.000000 |
| RX8/clean | student_8192 | 0.093726 | 0.093118 | 0.092575 |
| RX8/clean | student_32768 | 0.029652 | 0.029491 | 0.029380 |
| RX8/clean | dura | 0.000033 | 0.000033 | 0.000032 |
| RX8/practical_high | received | 2.087357 | 1.844536 | 0.992337 |
| RX8/practical_high | student_8192 | 0.477417 | 0.472879 | 0.469848 |
| RX8/practical_high | student_32768 | 0.303923 | 0.300944 | 0.296388 |
| RX8/practical_high | dura | 0.015451 | 0.014213 | 0.014118 |
| RX8/practical_mid | received | 2.069527 | 1.836613 | 0.992280 |
| RX8/practical_mid | student_8192 | 0.454968 | 0.449275 | 0.446924 |
| RX8/practical_mid | student_32768 | 0.282757 | 0.278477 | 0.274636 |
| RX8/practical_mid | dura | 0.015404 | 0.014034 | 0.013944 |
| RX8/practical_low_suburban | received | 2.062427 | 1.854374 | 0.993530 |
| RX8/practical_low_suburban | student_8192 | 0.495349 | 0.490458 | 0.483939 |
| RX8/practical_low_suburban | student_32768 | 0.328242 | 0.322983 | 0.317267 |
| RX8/practical_low_suburban | dura | 0.024758 | 0.023117 | 0.023003 |
| RX8/practical_high_urban | received | 2.040788 | 1.854708 | 0.993562 |
| RX8/practical_high_urban | student_8192 | 0.496789 | 0.492176 | 0.489127 |
| RX8/practical_high_urban | student_32768 | 0.321400 | 0.317831 | 0.313745 |
| RX8/practical_high_urban | dura | 0.015304 | 0.013631 | 0.013529 |
| RX8/practical_mid_urban | received | 2.024338 | 1.864721 | 0.994315 |
| RX8/practical_mid_urban | student_8192 | 0.606460 | 0.600020 | 0.592660 |
| RX8/practical_mid_urban | student_32768 | 0.410355 | 0.403352 | 0.399897 |
| RX8/practical_mid_urban | dura | 0.051656 | 0.049077 | 0.048739 |
| RX8/practical_low_urban | received | 2.015924 | 1.842929 | 0.992664 |
| RX8/practical_low_urban | student_8192 | 0.670919 | 0.655588 | 0.640495 |
| RX8/practical_low_urban | student_32768 | 0.493731 | 0.475525 | 0.466691 |
| RX8/practical_low_urban | dura | 0.100678 | 0.091899 | 0.091476 |

低城区 RX4/6：8192 学生由 0.593585 降到 0.574007、32768 学生由 0.366779 降到 0.352714，DURA 未对齐为 0.037404。RX8：8192 学生由 0.670919 降到 0.640495、32768 学生由 0.493731 降到 0.466691，DURA 未对齐为 0.100678。即使使用不可部署的理想整体复增益，学生仍明显高于 DURA；常数相位或增益不能解释主要波形误差，但剩余差异可能来自多种因素，不能确定具体原因。

两个学生的训练样本与优化步数不同，不能将差异单独归因于某项机制；源波形误差不是 TX 身份准确率，不是所有恢复方法的上界。大样本学生按既定方案继续完整七景测试。
