# 独立细节分支：冻结源 V 验证

VERIFIED：同一原始地面完整 Tri-SIG-TIME SSL、原 V 的 1800 条物理包、七景，全部输出固定后由独立 CPU 读取 V 真 TX 评分。四条路径为 received、纯物理 CovarianceMAP、原 CovarianceFeature、当前 CovarianceDetail。前三条路径逐值复用已有缓存；本次只计算最终版 detail。没有训练、选模或目标数据读取。

| 场景 | received TX% | 纯物理 TX% | 原特征指导 TX% | 新细节网络 TX% | 新版相对物理修复/损伤 |
|---|---:|---:|---:|---:|---:|
| clean | 98.611111 | 98.611111 | 98.555556 | 98.666667 | 1/0 |
| practical_high | 95.888889 | 98.388889 | 98.388889 | 98.333333 | 4/5 |
| practical_mid | 91.888889 | 97.444444 | 97.555556 | 98.277778 | 23/8 |
| practical_low_suburban | 90.555556 | 95.944444 | 96.111111 | 97.944444 | 42/6 |
| practical_high_urban | 91.666667 | 97.722222 | 97.722222 | 98.111111 | 13/6 |
| practical_mid_urban | 73.166667 | 90.555556 | 90.833333 | 95.388889 | 104/17 |
| practical_low_urban | 60.277778 | 84.222222 | 84.722222 | 91.000000 | 165/43 |

同一 1800 条物理包在七景重复使用，不能当作 12600 个独立物理样本。下表把波形、特征和识别判决分开：

| 场景 | 方法 | IQ NMSE | F 余弦距离 | 源方差标准化 F 误差 | F(clean) 判决一致率% |
|---|---|---:|---:|---:|---:|
| clean | received | 0.000000 | -0.000000 | 0.000000 | 100.000000 |
| clean | physical | 0.000134 | 0.000023 | 0.000311 | 99.944444 |
| clean | guided | 0.000262 | 0.000067 | 0.000842 | 99.888889 |
| clean | detail | 0.000038 | 0.000025 | 0.000316 | 99.888889 |
| practical_high | received | 2.015465 | 0.043114 | 0.338243 | 96.500000 |
| practical_high | physical | 0.009708 | 0.002377 | 0.028223 | 99.222222 |
| practical_high | guided | 0.011023 | 0.002609 | 0.029540 | 99.333333 |
| practical_high | detail | 0.006365 | 0.001417 | 0.016172 | 99.277778 |
| practical_mid | received | 2.015797 | 0.055759 | 0.421679 | 92.388889 |
| practical_mid | physical | 0.013807 | 0.005209 | 0.057710 | 98.333333 |
| practical_mid | guided | 0.015485 | 0.005114 | 0.057538 | 98.388889 |
| practical_mid | detail | 0.008938 | 0.002467 | 0.028035 | 98.833333 |
| practical_low_suburban | received | 2.014925 | 0.064665 | 0.470011 | 91.111111 |
| practical_low_suburban | physical | 0.020007 | 0.007956 | 0.083022 | 96.444444 |
| practical_low_suburban | guided | 0.021674 | 0.007689 | 0.082843 | 96.611111 |
| practical_low_suburban | detail | 0.012731 | 0.003483 | 0.038347 | 98.277778 |
| practical_high_urban | received | 2.014877 | 0.055326 | 0.407867 | 92.388889 |
| practical_high_urban | physical | 0.013287 | 0.004404 | 0.046133 | 98.611111 |
| practical_high_urban | guided | 0.014759 | 0.004691 | 0.048464 | 98.666667 |
| practical_high_urban | detail | 0.008587 | 0.001896 | 0.022082 | 99.000000 |
| practical_mid_urban | received | 2.011351 | 0.104175 | 0.756976 | 73.555556 |
| practical_mid_urban | physical | 0.047081 | 0.018056 | 0.181580 | 91.055556 |
| practical_mid_urban | guided | 0.049344 | 0.017799 | 0.181564 | 91.333333 |
| practical_mid_urban | detail | 0.027934 | 0.006241 | 0.080045 | 95.666667 |
| practical_low_urban | received | 2.008676 | 0.138138 | 1.030676 | 60.611111 |
| practical_low_urban | physical | 0.069798 | 0.035036 | 0.307742 | 84.722222 |
| practical_low_urban | guided | 0.072318 | 0.034029 | 0.304916 | 85.111111 |
| practical_low_urban | detail | 0.041880 | 0.013077 | 0.143283 | 91.555556 |

特征误差改善按训练中实际连续特征项 .01×（余弦距离＋源方差标准化 MSE）比较，容差固定为 1e-7；条件分组比较新版 detail 与纯物理版，真 TX 修复与损伤均相对纯物理版定义。

| 场景 | 特征改善组 TX 修复 | 特征改善组 TX 损伤 | 特征变差组 TX 修复 | 特征变差组 TX 损伤 |
|---|---:|---:|---:|---:|
| clean | 0 | 0 | 1 | 0 |
| practical_high | 3 | 3 | 1 | 2 |
| practical_mid | 20 | 3 | 3 | 5 |
| practical_low_suburban | 38 | 2 | 4 | 4 |
| practical_high_urban | 11 | 2 | 2 | 4 |
| practical_mid_urban | 101 | 3 | 3 | 14 |
| practical_low_urban | 161 | 9 | 4 | 34 |

低档城区新版源 V 准确率 91.000000%，比源 raw clean 低 7.611111 个百分点。相对纯物理版净纠错 +122/1800，修复 165、损伤 43，描述性配对双侧精确二项 p=4.81122e-18。该值没有多重比较校正，也没有新增随机种子，不构成稳定收益或 Main 泛化证明。

“突出发射机特质”另作冻结特征的描述性分析：使用训练前已固定的源特征方差作尺度，在预测全部固定后计算真实 TX 类间与类内方差。没有拟合新分类器，没有保存或部署类别质心。按 15 个 RX×day 格子分别计算再加权，便于观察接收机与日期混合的影响。

| 场景 | 方法 | TX 类间/类内方差 | RX×day 内 TX 类间/类内方差 |
|---|---|---:|---:|
| clean | received | 6.859278 | 10.983036 |
| clean | physical | 6.853762 | 10.950106 |
| clean | guided | 6.843332 | 10.933056 |
| clean | detail | 6.851001 | 10.999844 |
| practical_high | received | 2.649306 | 4.729107 |
| practical_high | physical | 6.340072 | 9.878422 |
| practical_high | guided | 6.348520 | 9.882544 |
| practical_high | detail | 7.135137 | 11.562328 |
| practical_mid | received | 1.911611 | 2.894669 |
| practical_mid | physical | 5.436834 | 7.886428 |
| practical_mid | guided | 5.521342 | 8.058408 |
| practical_mid | detail | 6.845357 | 10.942787 |
| practical_low_suburban | received | 1.656568 | 2.421955 |
| practical_low_suburban | physical | 4.859450 | 6.708412 |
| practical_low_suburban | guided | 4.888475 | 6.830403 |
| practical_low_suburban | detail | 6.738384 | 10.610992 |
| practical_high_urban | received | 1.956365 | 3.055914 |
| practical_high_urban | physical | 5.756170 | 8.616660 |
| practical_high_urban | guided | 5.698789 | 8.457729 |
| practical_high_urban | detail | 7.012464 | 11.176833 |
| practical_mid_urban | received | 0.657215 | 0.866257 |
| practical_mid_urban | physical | 3.257800 | 4.150175 |
| practical_mid_urban | guided | 3.279141 | 4.187909 |
| practical_mid_urban | detail | 5.631789 | 8.177383 |
| practical_low_urban | received | 0.327038 | 0.429162 |
| practical_low_urban | physical | 2.092424 | 2.511658 |
| practical_low_urban | guided | 2.129654 | 2.557516 |
| practical_low_urban | detail | 4.209074 | 5.657282 |

更大的类间/类内比只能说明这个固定 F 特征空间、固定面板中的描述性可分性；它不证明恢复出了真实硬件成分，不保证默认分类头边界更合适，也不能替代上表真实 TX 准确率。恢复后可能修复 F(raw clean) 的错误，因此 Fclean 一致率不是信息保留上限。IQ NMSE、特征误差、判决一致率与真实 TX 准确率各有不同口径，不能合并为单一“恢复 clean 的几成”。

V 只作冻结验证，原面板有历史曝光。V 与训练物理记录互斥，但源 RX/day 相同；验证接收机实例 seed2026101902、训练 seed2026101911。两者包分布、信道实例与接收机实例同时不同，不能从本诊断分配单一因素的因果份额。所有模型已固定 final30，本次不改变 checkpoint、阈值、训练范围或默认 DURA。

完整 28 个方法×场景单元、420 个 TX/RX/day/总体分组、105 个条件分组和所有保存 IQ/特征/logits/判决在读取 V 真值前核对；缓存基线判决变化 0，最大指标复算误差 1.42e-14。独立复算使用源 V 标签 1800 条（标签文件含 27000 行），读取已保存预测/参考 IQ，0 新神经网络前向、0 原始 IQ 数据集读取、0 目标真值读取。不能把这次真值侧复算描述为零标签读取。

本次新增 12600 条 G 和 12600 条 F 输出，复用 1800 条 clean F 参考和三条完整基线；实际源 Ls 2 条检查另计 2 条 G/F。所有模型与持久状态保持，0 V 训练、身份标签损失、类别原型、额外身份头、OE/novel/目标输入。正式 Main 六景逐个距 raw clean ≤3 个百分点且两个 clean 总体保持，仍由独立 Main 结果判定。
