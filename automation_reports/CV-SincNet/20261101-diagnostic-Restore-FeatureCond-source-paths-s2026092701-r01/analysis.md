# 冻结特征条件恢复器源端路径诊断

**条件分支实际参与恢复；本源面板中，使用 clean 特征参照显著降低身份特征误差。** 这提示继续检验预测连续特征的训练目标，但参照额外知道 source clean，不能证明该收益可由 received 学得，也不能替代已完成 Main 的0/6负结果。

![源端诊断](E:/type10-7/automation_reports/CV-SincNet/20261101-diagnostic-Restore-FeatureCond-source-paths-s2026092701-r01/source_path_diagnostic.png)

## 固定源结果

| 范围 | 路径 | IQ NMSE | F余弦误差 | 标准化F误差 | Fclean判决一致率 | 平均复功率 |
|---|---|---:|---:|---:|---:|---:|
| ALL（n=10240） | 正常条件 | 0.024695 | 0.005728 | 0.062406 | 96.787109% | 1.000000 |
| ALL（n=10240） | 预测特征置零，保留FiLM bias | 0.408318 | 0.187309 | 1.323237 | 37.548828% | 1.000000 |
| ALL（n=10240） | 跳过全部FiLM | 0.435000 | 0.192754 | 1.374564 | 35.302734% | 1.000000 |
| ALL（n=10240） | 源clean特征参照 | 0.023897 | 0.002770 | 0.033722 | 99.091797% | 1.000000 |
| ALL（n=10240） | 投影前raw | 60.774396 | 0.121061 | 0.945440 | 58.593750% | 75.827557 |
| ALL（n=10240） | 预测latent的原默认头 | N/A | 0.008421 | 0.091068 | 96.259766% | N/A |
| ALL（n=10240） | 缓存DURA | 0.022287 | 0.008388 | 0.096272 | 95.146484% | N/A |
| ALL（n=10240） | 缓存FullFGrad | 0.020882 | 0.004869 | 0.057185 | 97.158203% | N/A |
| practical_low_urban（n=1436） | 正常条件 | 0.071576 | 0.018608 | 0.190443 | 88.718663% | 1.000000 |
| practical_low_urban（n=1436） | 预测特征置零，保留FiLM bias | 0.635190 | 0.209883 | 1.501073 | 29.874652% | 1.000000 |
| practical_low_urban（n=1436） | 跳过全部FiLM | 0.664710 | 0.214608 | 1.537704 | 28.899721% | 1.000000 |
| practical_low_urban（n=1436） | 源clean特征参照 | 0.066893 | 0.006744 | 0.083954 | 97.493036% | 1.000000 |
| practical_low_urban（n=1436） | 投影前raw | 71.417485 | 0.151996 | 1.160050 | 48.607242% | 88.494702 |
| practical_low_urban（n=1436） | 预测latent的原默认头 | N/A | 0.020770 | 0.205812 | 86.559889% | N/A |
| practical_low_urban（n=1436） | 缓存DURA | 0.063043 | 0.025007 | 0.273576 | 83.913649% | N/A |
| practical_low_urban（n=1436） | 缓存FullFGrad | 0.061619 | 0.015575 | 0.176007 | 89.902507% | N/A |

## 配对得失

| 范围 | 相对正常路径的读出 | 修复 | 损伤 | 净变化 |
|---|---|---:|---:|---:|
| ALL | 预测特征置零，保留FiLM bias | 54 | 6120 | -6066 |
| ALL | 跳过全部FiLM | 51 | 6347 | -6296 |
| ALL | 源clean特征参照 | 258 | 22 | +236 |
| ALL | 投影前raw | 82 | 3993 | -3911 |
| ALL | 预测latent的原默认头 | 80 | 134 | -54 |
| clean | 预测特征置零，保留FiLM bias | 0 | 1003 | -1003 |
| clean | 跳过全部FiLM | 0 | 1017 | -1017 |
| clean | 源clean特征参照 | 0 | 0 | +0 |
| clean | 投影前raw | 0 | 4 | -4 |
| clean | 预测latent的原默认头 | 0 | 10 | -10 |
| practical_high | 预测特征置零，保留FiLM bias | 5 | 793 | -788 |
| practical_high | 跳过全部FiLM | 5 | 848 | -843 |
| practical_high | 源clean特征参照 | 5 | 10 | -5 |
| practical_high | 投影前raw | 2 | 666 | -664 |
| practical_high | 预测latent的原默认头 | 5 | 5 | +0 |
| practical_mid | 预测特征置零，保留FiLM bias | 5 | 846 | -841 |
| practical_mid | 跳过全部FiLM | 6 | 894 | -888 |
| practical_mid | 源clean特征参照 | 16 | 4 | +12 |
| practical_mid | 投影前raw | 5 | 709 | -704 |
| practical_mid | 预测latent的原默认头 | 10 | 5 | +5 |
| practical_low_suburban | 预测特征置零，保留FiLM bias | 3 | 875 | -872 |
| practical_low_suburban | 跳过全部FiLM | 3 | 907 | -904 |
| practical_low_suburban | 源clean特征参照 | 22 | 4 | +18 |
| practical_low_suburban | 投影前raw | 5 | 706 | -701 |
| practical_low_suburban | 预测latent的原默认头 | 10 | 8 | +2 |
| practical_high_urban | 预测特征置零，保留FiLM bias | 5 | 846 | -841 |
| practical_high_urban | 跳过全部FiLM | 4 | 886 | -882 |
| practical_high_urban | 源clean特征参照 | 16 | 0 | +16 |
| practical_high_urban | 投影前raw | 7 | 670 | -663 |
| practical_high_urban | 预测latent的原默认头 | 6 | 8 | -2 |
| practical_mid_urban | 预测特征置零，保留FiLM bias | 7 | 883 | -876 |
| practical_mid_urban | 跳过全部FiLM | 6 | 909 | -903 |
| practical_mid_urban | 源clean特征参照 | 70 | 1 | +69 |
| practical_mid_urban | 投影前raw | 23 | 622 | -599 |
| practical_mid_urban | 预测latent的原默认头 | 23 | 41 | -18 |
| practical_low_urban | 预测特征置零，保留FiLM bias | 29 | 874 | -845 |
| practical_low_urban | 跳过全部FiLM | 27 | 886 | -859 |
| practical_low_urban | 源clean特征参照 | 129 | 3 | +126 |
| practical_low_urban | 投影前raw | 40 | 616 | -576 |
| practical_low_urban | 预测latent的原默认头 | 26 | 57 | -31 |

## 机制判断与下一项假设

1. 不能把上轮负结果解释为FiLM没有启用。在同一冻结模型中，zero-feature保留已学bias、无FiLM完全跳过调制，两者都使源端特征保持显著下降。它证明该模型已依赖条件分支，不证明其他从零无FiLM模型一定较差。
2. 低档城区正常路径的Fclean一致率88.718663%，预测latent经原默认头为86.559889%；source clean特征参照为97.493036%，修复129、损伤3，净增126/1436。相应IQ NMSE只从.071576降到.066893。对这组冻结反事实，特征误差变化比波形误差更明显；不能将差额解释为可恢复信息比例。
3. 源clean参照从未用于训练时的G前向，也不用于部署。它的输入分布与预测latent不同，decoder和encoder已共同训练；额外clean信息、输入分布及模型耦合都可能影响差异。它不是理想encoder的可达性能界。
4. 直接取消已训练G的单位功率投影没有可比的物理意义。投影前raw平均复功率约75.83，低档城区约88.49；训练只约束投影后的方向，raw尺度没有被独立监督。原实际输出保持单位复功率，不能把raw分支的巨大NMSE说成实际G失效，也不能直接切换该分支部署。
5. 当前正常输出的源IQ/F误差仍略差于既有FullFGrad。综合训练源证据，下一项可证伪假设是提高连续latent回归的相对权重，检验预测特征约束是否不足；先采用与原IQ目标同为1的固定权重，保留相同结构、源顺序、优化器和30轮预算。该假设来自源内诊断，不保证跨RX/Main收益；不用此参照选择Main输出或修改已有模型。

## 范围、成本与核验

固定原R07训练角色L_s128/U_s384共512包×20存量draw=10240个相关案例，0TX/V/OE/novel/target读取、0更新或新统计拟合。原FeatureCond/c8659b15…final30和原Tri-SIG-TIME SSL/3ff8/defaulthead冻结；DURA/FullFGrad仅精确复用source-style/ac801 identity记录，同epoch/索引/scene/Fclean逐值一致。并未重复旧G推理。实际诊断耗时19.268173秒。

正常/zero/noFiLM/clean-oracle共享encoder10240行，四decoder40960行；raw复用正常路径投影前张量。五IQ路径51200行F0特征+clean512参考=51712行，latent另10240次原默认头运算，不重复F0 encoder。source L_s2实际smoke另12条F0特征/10次头运算、4条G encoder/10条decoder。按请求行数计，不等于内部三视图或计算核调用数。

3项针对性测试与一次P0/P1通过。真实源L_s2smoke核actual/latent/raw与原G逐值一致，PASS立即完整矩阵；PID/CWD/命令/CUDA/GPU及新日志独立读回。GPU4准入剩余24110MiB、已有compute0、只新增1条本任务NN，其他owner保持；完整20draw日志、所有预测及配对、源面板和缓存控制对齐、模型状态不变独立核实。最终worker/controller均退出0，独立均值复算误差0，实际单位功率逐记录最大误差2.98e-7。

这是源内机制诊断，Fclean一致率不是TX准确率，512包×20draw不能当作10240条独立物理样本。没有新增Main、V/OE或Phase3结果。默认DURA、所有历史权重和原Main负结果保持；六景逐个距raw clean≤3pp+clean的完整目标仍ACTIVE。
