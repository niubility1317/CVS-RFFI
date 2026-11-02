# 冻结源 V：恢复输出与缓存模拟 Action 的一致性

**源端证据不支持直接加强‘贴近带噪观测’损失。** 低档城区中，新版有1731/1800条IQ更接近clean；其中1247条对接收观测的归一化残差增大。新版残差能量与同条clean参考残差能量的绝对差距均值反而下降32.573363%。更强观测拟合可能重新吸收已经去除的噪声，不能把残差越小直接当作恢复越好。

本分析复用同一已冻结源V的四方法IQ、原clean参考、判决和原R07模拟Action trace。没有新网络前向、训练或目标读取；只做只读CPU计算，没有改写原预测、模型或Main结果。

## 低档城区的五条读出

| 读出 | IQ NMSE | 观测残差 MSE | 残差/缓存噪声方差均值 | 其中位数 | 与同条clean残差能量的绝对差距均值 | Fclean一致率% |
|---|---:|---:|---:|---:|---:|---:|
| received | 2.008676 | 0.936864 | 195.762233 | 12.057104 | 194.762480 | 60.611111 |
| physical | 0.069798 | 0.101286 | 1.001995 | 0.860586 | 0.344762 | 84.722222 |
| guided | 0.072318 | 0.101741 | 1.227678 | 0.891859 | 0.489140 | 85.111111 |
| detail | 0.041880 | 0.106892 | 1.040276 | 0.940954 | 0.232462 | 91.555556 |
| clean_reference | 0.000000 | 0.109071 | 0.999754 | 0.999903 | 0.000000 | 100.000000 |

单看残差比的均值，物理版更接近1；逐记录与clean参考比较，新版却更接近。不同统计量必须分开：均值可能掩盖病例间偏差，噪声方差加权也不等于先对原MSE取均值后再归一化。clean_reference使用真实源clean，无法部署，不是额外训练或oracle性能上限。

## 配对病例：更接近 clean 并不要求更贴近接收噪声

| 新版 IQ NMSE 相对物理版 | 新版观测残差比 | 样本数 | Fclean 修复 | Fclean 损伤 | Fclean 一致性不变 |
|---|---|---:|---:|---:|---:|
| smaller | smaller | 484 | 24 | 3 | 457 |
| smaller | tied | 0 | 0 | 0 | 0 |
| smaller | larger | 1247 | 138 | 33 | 1076 |
| tied | smaller | 0 | 0 | 0 | 0 |
| tied | tied | 0 | 0 | 0 | 0 |
| tied | larger | 0 | 0 | 0 | 0 |
| larger | smaller | 6 | 1 | 0 | 5 |
| larger | tied | 0 | 0 | 0 | 0 |
| larger | larger | 63 | 3 | 7 | 53 |

全体低档城区中，1310/1800条新版观测残差比增加；IQ改善且残差增加组包含138次Fclean修复与33次Fclean损伤。这些是相对F(raw clean)判决的变化，不是真TX纠错：本次没有重新读取TX标签，不能与既有源V真TX的165修复/43损伤混同。

## 全部七景

| 场景 | 读出 | IQ NMSE | 残差比均值 | 与同条clean残差能量的绝对差距均值 | Fclean一致率% |
|---|---|---:|---:|---:|---:|
| clean | physical | 0.000134 | 6724.081256 | 6724.081256 | 99.944444 |
| clean | guided | 0.000262 | 13104.680745 | 13104.680745 | 99.888889 |
| clean | detail | 0.000038 | 1923.454924 | 1923.454924 | 99.888889 |
| clean | clean_reference | 0.000000 | 0.000000 | 0.000000 | 100.000000 |
| practical_high | physical | 0.009708 | 2.582792 | 1.778748 | 99.222222 |
| practical_high | guided | 0.011023 | 3.521555 | 2.579073 | 99.333333 |
| practical_high | detail | 0.006365 | 2.338545 | 1.500019 | 99.277778 |
| practical_high | clean_reference | 0.000000 | 1.000296 | 0.000000 | 100.000000 |
| practical_mid | physical | 0.013807 | 1.333465 | 0.714472 | 98.333333 |
| practical_mid | guided | 0.015485 | 1.744971 | 0.940422 | 98.388889 |
| practical_mid | detail | 0.008938 | 1.313743 | 0.584178 | 98.833333 |
| practical_mid | clean_reference | 0.000000 | 1.001505 | 0.000000 | 100.000000 |
| practical_low_suburban | physical | 0.020007 | 0.983653 | 0.470313 | 96.444444 |
| practical_low_suburban | guided | 0.021674 | 1.190001 | 0.511740 | 96.611111 |
| practical_low_suburban | detail | 0.012731 | 1.028323 | 0.344003 | 98.277778 |
| practical_low_suburban | clean_reference | 0.000000 | 1.002237 | 0.000000 | 100.000000 |
| practical_high_urban | physical | 0.013287 | 2.445876 | 1.651412 | 98.611111 |
| practical_high_urban | guided | 0.014759 | 3.359198 | 2.432728 | 98.666667 |
| practical_high_urban | detail | 0.008587 | 2.181476 | 1.340080 | 99.000000 |
| practical_high_urban | clean_reference | 0.000000 | 0.999131 | 0.000000 | 100.000000 |
| practical_mid_urban | physical | 0.047081 | 1.234264 | 0.565648 | 91.055556 |
| practical_mid_urban | guided | 0.049344 | 1.508137 | 0.716808 | 91.333333 |
| practical_mid_urban | detail | 0.027934 | 1.222711 | 0.434816 | 95.666667 |
| practical_mid_urban | clean_reference | 0.000000 | 0.999932 | 0.000000 | 100.000000 |
| practical_low_urban | physical | 0.069798 | 1.001995 | 0.344762 | 84.722222 |
| practical_low_urban | guided | 0.072318 | 1.227678 | 0.489140 | 85.111111 |
| practical_low_urban | detail | 0.041880 | 1.040276 | 0.232462 | 91.555556 |
| practical_low_urban | clean_reference | 0.000000 | 0.999754 | 0.000000 | 100.000000 |

![源V一致性](E:/type10-7/automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01/measurement_consistency.png)

## 定义、执行与边界

把候选恢复IQ代入缓存的FIR＋时变main/image Action代理，与原received IQ比较。观测残差MSE是两个实分量、256点的平均平方误差；再除以缓存的平均实分量噪声方差，得到逐记录无量纲残差比。“与clean差距”是该比值与同条真实clean代入后的比值之差的绝对值。它只比较残差能量，没有比较完整噪声实现或完整协方差，不能充当正确似然或训练阈值。

Action代理包含原模拟trace中实现后的AGC/接收机系数，不是硬件Jacobian或完整PA模型。缓存噪声方差是标量，main/image后实际I/Q噪声可以非圆对称；本报告不把标量当完整噪声协方差。源clean的残差比约1仅验证本固定源面板的量级，不推出Main上相同，也不是信息恢复极限。

全部840个总体/RX/day/RX×day×方法×场景汇总与168个配对条件组完整输出，病例全部保留，smaller/tied/larger的容差固定为1e-9。共读取52200条保存IQ（29个数组）、52200条保存判决记录及12600条缓存Action trace；0原始IQ数据集、TX标签、目标输入、新神经网络前向或优化。源V接收机在G训练中已出现，不能从这里证明未见RX的泛化机制。

实际分析代码 `62cf25c161afe1410ad6250249931389d9210a1c`，固定预测代码 `f29b31417ce98a280eac25ea2fa78e0c49a5f0ff`。5项针对性检查通过，快速Action与既有稠密实算子一致，覆盖反射边界与I/Q镜像项。252个重叠公开分组的IQ NMSE/Fclean指标复算差0，同条条件计数全覆盖；没有重新评分Main或选择新checkpoint。

Main六景0/6、两个clean保持、17个完成候选及默认DURA均不改变。该结果没有为盲目加入更强观测拟合约束提供直接依据；下一改造仍需依靠合法源端证据。恢复输出的RF特质真实性与理论极限，也不能仅由观测残差、IQ误差或Fclean一致率判定。
