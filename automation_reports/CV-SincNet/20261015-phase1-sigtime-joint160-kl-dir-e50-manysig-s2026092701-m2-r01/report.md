# SIG_TIME：输出 KL 与 joint160 方向迁移 E50 结果

状态：ANALYZED / VERIFIED。两条新行均从零完成 E50（33 轮 label＋17 轮旧门控半监督），每行 11100 次成功更新、七种正式视图独立 truth-last 评分；冻结后源身份机制分析也已完成。仅 model seed 2026092701，尚无本轮跨 seed 复现证据。

方向迁移没有优于原间隔配方：low urban54.8024%、clean80.4940%，相对普通拼接B0的low提高0.8327个百分点、clean下降0.7988个百分点；相对C3的low低0.1327个百分点。方向对KL仅low+0.0446、urban+0.0413个百分点，urban Macro-F1反而−0.0590个百分点，单seed不足以判断优越性。60%及clean下降不超过0.5个百分点的目标未达。

固定原 SIG_TIME、内部门控、参考响应、默认 cosine6×160/scale30，原 CE 继续联合训练全部参数。无新增部署模块、替换头或独立拟合诊断分类器。B=OUTPUT_KL、C=JOINT_DIR 具有相同网络、数据、原 U 训练、课程、教师前向和 E50 预算，仅额外目标不同。A 为既有 C3，使用学生 clean 间隔／接收质量权重，且少一条 L clean EMA 前向；A 对 B/C 是整套配方比较，不是严格只改一个公式。

| 方法 | urban 均值 (%) | low urban (%) | mid urban (%) | clean (%) | suburban 均值 (%) | urban Macro-F1 (%) | 最差 urban RX (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 65.0927 | 53.9696 | 64.2851 | 81.2929 | 78.0657 | 64.7060 | 59.1569 |
| C3 | 65.4464 | 54.9351 | 64.8137 | 80.4161 | 77.4732 | 65.1370 | 59.6972 |
| OUTPUT_KL | 65.2615 | 54.7577 | 64.6179 | 80.3286 | 77.2722 | 65.0166 | 59.2792 |
| JOINT_DIR | 65.3028 | 54.8024 | 64.6708 | 80.4940 | 77.3151 | 64.9576 | 59.5708 |

| 差值（百分点） | urban | low urban | mid urban | clean | suburban | urban F1 | 最差 RX |
|---|---:|---:|---:|---:|---:|---:|---:|
| OUTPUT_KL − C3 | -0.1849 | -0.1774 | -0.1958 | -0.0875 | -0.2010 | -0.1204 | -0.4181 |
| JOINT_DIR − C3 | -0.1437 | -0.1327 | -0.1429 | +0.0780 | -0.1581 | -0.1794 | -0.1264 |
| JOINT_DIR − OUTPUT_KL | +0.0413 | +0.0446 | +0.0530 | +0.1655 | +0.0429 | -0.0590 | +0.2917 |
| JOINT_DIR − B0 | +0.2101 | +0.8327 | +0.3857 | -0.7988 | -0.7506 | +0.2516 | +0.4139 |

60% 是探索目标，不是已达成结果；实际状态为 NOT_MET_SINGLE_SEED。全部负差保留，目标分数不用于改参、选模、晋级、扩 seed 或选择性重跑。预算耗尽不等于收敛。

完整逐视图／RX／day／TX 表保存在每行 scoring/scores.json，每行126条指标、1176000条预测；两行共252条指标、2352000条预测。原 clean 与六 practical 的每视图168000物理记录保持不变。

| 新行 | 主训练前向样本 | 源面板 | 原 V 验证 | 训练墙钟 (s) | 实际裁剪步数／11100 | 含测试总前向样本 |
|---|---:|---:|---:|---:|---:|---:|
| OUTPUT_KL-s2026092701 | 6190200 | 144000 | 1404000 | 6939.267 | 120 | 8914204 |
| JOINT_DIR-s2026092701 | 6190200 | 144000 | 1404000 | 6613.118 | 133 | 8914204 |

两条新行各父训练0、新增50、累计50 epoch，合计22200更新。每行L clean/satellite/ownEMA clean各1420800前向，原U教师／strong学生各963900。另有共享dispatcher smoke2、一次源校准2250前向／6.498秒／0更新、冻结后源分析23040前向／43.333秒／0更新。计入上述新增训练、验证、测试、smoke、校准和后分析共17853700个前向样本；这不是FLOPs或等硬件成本。
校准依赖既有C3 E50源权重，但两条新行没有加载、恢复或继承其权重／EMA／优化器。既有A的50轮／11100更新／6457.786秒及4769400主训练前向是历史依赖成本，不冒充新行父训练或遗漏该依赖。

全量step日志逐条核实各11100步连续、每epoch222步、更新均实际应用，未见非有限跳步或U真值暴露；完整stdout未见技术错误。原V每epoch27000条、50次；源geometry全部5300次cache hit、0miss，另外final geometry和wrapper健康验证各27000条，合计1404000前向／行。无额外fitted head或目标训练。原clip阈值5.0，报告实际裁剪，不在线追调lambda。

冻结后源分析使用原合法L_s的固定720条（曾参与这组训练），以固定C3 final50 clean正确集合为参考；不是新的内部L_dev或未见源测试。模型与ownEMA状态不变，0更新、0拟合头、无U／query读取。最近错误TX方向是720条合法源clean向量中的最大异类cosine，不是训练出的诊断分类器。

| 方法（source low urban） | 准确率 (%) | 固定C3集合保持率 (%) | 修复C3错误 | 新增C3错误 | clean配对cosine | 最近错TX cosine | 正确−错误cosine | 默认头归一化margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 63.0556 | 63.3194 | 17 | 32 | 0.721944 | 0.888234 | -0.166290 | 0.086505 |
| C3 | 65.1389 | 65.4114 | 0 | 0 | 0.728871 | 0.899138 | -0.170267 | 0.086563 |
| OUTPUT_KL | 64.8611 | 65.1325 | 7 | 9 | 0.720042 | 0.905261 | -0.185219 | 0.086480 |
| JOINT_DIR | 65.1389 | 65.4114 | 14 | 14 | 0.820010 | 0.936687 | -0.116677 | 0.073509 |

固定C3 clean正确集合为717/720。状态／锁定／新增噪声SNR分层按实际交叉单元样本数加权汇总；新增噪声SNR不是总接收SNR。这里只比较相同完整信道抽样的关联，不是去除单项损伤的因果实验；原四行的受控损伤分析另行保留。方向相似度提高本身不代表身份判别或默认头准确率提高。

方向迁移相对B0的源low改善集中在BLOCKED（n417，41.0072%→44.6043%）、degraded（n146，63.0137%→65.7534%）、unlocked（n320，35.9375%→39.0625%）和新增噪声SNR<3dB（n239，33.4728%→37.6569%），LOS/PARTIAL准确率均未变。这些是固定训练见过的源面板关联，不能据此断言独立解决多径、Doppler或噪声，更不能代替目标分层因果证据。
相对C3，方向在固定717集合上修复14个错误、同时新增14个，保持率均65.4114%。clean/satellite配对cosine从.728871升至.820010，最近错误TX cosine也从.899138升至.936687；默认头margin从.086563降至.073509。方向更接近并未带来默认头收益。各坐标方差仍非零，不能仅凭平均方差降低宣称完整塌缩。

| 源机制 | clean joint均方差 | low joint均方差 | PA gain绝对均值 | ref gain绝对均值 | SIG gain绝对均值 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.229717 | 0.150899 | 0.270388 | 0.073101 | 0.084074 |
| C3 | 0.238501 | 0.146318 | 0.271072 | 0.074729 | 0.083401 |
| OUTPUT_KL | 0.238401 | 0.146199 | 0.272888 | 0.078085 | 0.084260 |
| JOINT_DIR | 0.242960 | 0.119463 | 0.276800 | 0.066435 | 0.085549 |

表中gain为实际tanh后的逐坐标向量绝对均值，SIG含原0.2系数；它不是样本动态门控贡献比例。完整逐分支方差／norm／逐坐标gain在源原始JSON中保留。训练辅助项对W／三gain的梯度始终为0，原联合CE／U训练仍使五组特征、W和gain产生非零总更新。

实际数据：L6300／U56700／V27000，源RX1/3/4/6/8、day1/2/3、split/data392005；U标签隐藏。25MHz、IQ256、原native RMS后模拟器输入RMS，输出不再RMS；实际导入experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py，residual/post_sync、EQfalse（G=1）、reference_snr_db30、质量适配／phase tracking／IQcomp true、reflect。训练仅low suburban／mid urban／low urban，完整依赖和RX/day稳定session、跨epoch/view命名随机流保持。

静态系数在读取A目标成绩和新训练前已由source-only校准固定：KL0.04998551146175206，direction0.44114306609656917。两行重放A的源课程eta，不按自身或目标分数推进。一次校准的0.1是五组／源batch的梯度比中位数，不代表后续强度恒定，也不等同新八位置实验的全局校准口径。

实际训练release commit：704a3e917de3bd8f8a739fddb2a21212b2840148；冻结后源release commit：3db2906e49750819a6cb2a8f376325fbcb4f3b7f。一次archive传输比对／远端compile／普通N607账户预检、真实合规source checkpoint no-query smoke和PID/CWD/argv/GPU／日志独立回读均已VERIFIED。训练接线11项检查、源分析3项检查PASS；训练一次限定P0/P1审查PASS，未改变的后分析不重复审查。没有热改、停止或重启健康实验。

训练命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_urban_identity50.dispatch`；CWD PROJECT/releases/cvs_sigtime_joint160_kl_dir_e50_20261015_r01。后分析命令同Python下`-m experiments.cvs_urban_identity50.mechanism`，CWD PROJECT/releases/cvs_sigtime_joint160_identity_source_20261015_r01。精确launcher、不可变run/output、逐row矩阵、GPU、owner、stop rule和预期产物仍见experiment.json及source_identity_plan.md。

图表（实际完成后生成）：
- [complete_training_curves.png](E:/type10-7/local_artifacts/cvs_joint160_identity_20261015/final/figures/complete_training_curves.png)
- [per_seed_formal_accuracy.png](E:/type10-7/local_artifacts/cvs_joint160_identity_20261015/final/figures/per_seed_formal_accuracy.png)
- [source_identity_geometry.png](E:/type10-7/local_artifacts/cvs_joint160_identity_20261015/final/figures/source_identity_geometry.png)
- [source_state_quality_strata.png](E:/type10-7/local_artifacts/cvs_joint160_identity_20261015/final/figures/source_state_quality_strata.png)

原始核实产物：E:\type10-7\local_artifacts\cvs_joint160_identity_20261015\final\training_audit.json；E:\type10-7\local_artifacts\cvs_joint160_identity_20261015\source_identity_results.json；E:\type10-7\local_artifacts\cvs_joint160_identity_20261015\source_delivery_r01\readback.json。IQ、checkpoint、预测、原始日志及生成输出均不入Git。

当前实现不是完整多阶段设计：容差方向、同信道中间视图、分组SIG_TIME局部教学、PA/ref局部目标与额外U特征目标尚未运行；严格A/B/C只改公式的要求也未由这个历史A比较满足。新八位置家族独立按其已登记源规则运行，不把本轮目标结果用于位置排序或扩展。
