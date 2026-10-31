# 冻结PCD相对马氏拒识

ANALYZED，执行 VERIFIED；科学 NO_PROMOTION，完整目标 ACTIVE。

# Phase3 冻结 PCD 的相对马氏拒识（2026-10-31）

Run `20261031-phase3-pcd-relative-metric-s2026092701-r01`。当前 PCD 是研究基线，源 Us 重放未超过它；本行仅检验拒识统计是否过度依赖源 RX/day 的局部几何，不训练新网络，不改身份判决。

## 依据与唯一候选

三种已考察路线为：继续源 Us 伪标签训练、目标无标签偏移/缩放、冻结表征后的背景相对化。前两类已有未过门槛的结果，本轮选第三类。旧 F0 多分支 RMD、硬 RX/day 投影和额外 MetricNet 的结果保留；它们不是当前最终 PCD 单位表征的本轮读出。

方法参考 [Ren 等的 RMD 原论文](https://arxiv.org/abs/2106.09022)。论文说明背景距离可减弱部分无关坐标的影响；本轮将其作为待证伪假设，不能据图像/语言结果推定星地 RFFI 有效。不增加投影、可训练头或逐记录源 bank。

固定原 F0 Tri-SIG-TIME SSL E50/11100/default6、fresh f4b5 的 TARGET_PCD444、DURA v1.0 R69（reuse/graph、非 native、FP32/TF32off）。PCD 已有 U576 祖先，不能把本轮源统计称为整个模型纯源。新身份始终是同 DURA 输出的原 F0 默认头。

## 源统计及分数

source_geometry074 的源 L6300 缓存 PCD unit160 与原 L IDs/y 按物理 ID 对齐。FP32 单位向量转 FP64 后按原 density.unit 重新单位化；六类均值不再单位化。只用 L 拟合六均值、全局均值、类内共享 population 协方差及不分标签的全局 population 协方差。两者固定 `0.9*C+0.1*trace(C)/160*I` 后 Cholesky；不扫描收缩、不使用 OE/U/V/Q 拟合，退化或非有限停止而不加静默回退。

记每类平方马氏距离为 d_c²，全局背景平方距离为 d_0²。四个新分数依次为 `MD_pooled=-min d_c²`、`MD_predicted=-d_F0²`、`RMD_pooled=d_0²-min d_c²`、`RMD_predicted=d_0²-d_F0²`，高分接收。pooled 只用于拒识，不改变身份；两种分数均按原 F0 预测类路由六阈值。不是依真 TX 选择距离。

固定六读出为 OLD_PCD_pooled_ratio、OLD_PCD_predicted_ratio、MD_pooled、MD_predicted、RMD_pooled、RMD_predicted。唯一主是最后一项，MD 是相同新统计下的背景消融，原 PCD 两列/密度/阈值精确复用。主对 OLD_PCD 评价相对收益，不能依目标结果换主或选择收缩。

## 数据、冻结与完整测试

复用原 source392005 L6300/Us56700/V27000 祖先、源 OE6134 训练祖先与固定 G10/DURA。源五 RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、03-08/15/23、旧六 TX14-10/14-7/20-15/20-19/6-15/8-20 不变。新统计仅 L；原 Us/OE 实际新读0。

真实 checkpoint 无 query 合成 smoke PASS 后读 L 特征并冻结四聚合；随后源已知 V27000 仅按原 known12 风险规则校准四组六阈值，30交叉组另诊断。旧两组六τ不变。所有36值先于 NV/Q 冻结，再完成 source NV3150 与 RX3-2/index34 的 clean/star future-Q10238 两路预测，全部三主条件均 DURA。NV19-15/19-16/7-1各1050，仅独立评分。

future-Q 仍为03-15/23的5525/4713条，已知4508、未知5730。六已知各1491/1182/150/406/1186/93，四未知5-5/10-7/2-19/7-11各1454/1439/1427/1410。旧 PCD 的早日 U576 仅为祖先，本行不新读取；7-11整类从该早日U排除，不声称所有祖先均未见。03-23缺旧20-19/8-20时完整宏 AK/CK/H=N/A，不删类认证。clean/star同物理顺序，不增加独立记录，无 clean 学习。

G10 保持源 MIX01 scene2026103001/channel2026104001/receiver2027，future scene2026103133/channel2026103134；600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、一次 input RMS、无 output RMS。本轮不生成 IQ、重抽信道或重新恢复真实数据。

6×3条件全部预测完成后，独立 CPU scorer 才连接 source NV、clean Q、star Q truth 各一次；U truth0。主报逐 TX 宏 AK/CK/RU/H、同 DURA 原 F0 闭集损失、TX/RX/day/六场景、缺类和配对得失。source known12/cal 通过、cleanRU≥95%/starRU≥90%/CK≥90%/原F0闭集−AK≤2pp、全部必需主景覆盖须同时满足；另要求主 starAK 严格高于 OLD_PCD_pred 且 RU 不下降，否则 NO_PROMOTION、不扩种子。MD 只解释相对背景作用，不代替主。

## 成本与执行

新神经优化、真实 F0/signal/G 前向均0；source L6300特征拟合、V27000四新六τ、NV3150与Q20476缓存行评分另计，合成模型 smoke 计数取实际产物。四 FP64 数组为52320值/418560B（含schema字符串另计），四新τ192B，旧τ96B，共288B；单主六τ48B。主部署神经参数仍834665，原F0/r03/PCD/G训练和缓存成本继承；实际进程驻留及批量耗时不当作B1时延。

单seed2026092701/单owner root，默认 N607、新不可覆盖root，单 GPU 预测后独立CPU评分；实际剩余≥4096MiB、每卡新增≤2，不干预其他用户。代码commit、聚焦负测、一次直接P0/P1、单preflight/单archive SHA/compile、实际PID/CWD/argv/GPU/log增长与truth-last收尾。非法数据/祖先、冻结状态改变、非有限、碰撞或无法合法闭合即保留停止；科学低分完成完整矩阵。

本行不是新增目标无标签适应、发现或匿名注册，adaptation_success/A_N/H_OW=N/A。source V是校准角色，NV/Q历史曝光、单开发RX/seed、全项目physical history UNKNOWN、正式RX24-18未访问保持；完整Phase3目标 ACTIVE。


实际代码 `a24c413e5ab727ec283c07405eb9b1b2e68b036b`；环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-pcd-relative-metric-s2026092701-r01-release`；owner `codex-phase3-pcd-relative-metric-20261031`；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.pcd_relative_metric_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-pcd-relative-metric-s2026092701-r01-release/runtime.json`。输入/输出见experiment.json。


# Phase3 冻结 PCD 相对马氏拒识结果（2026-10-31）

**NO_PROMOTION，原 PCD 研究基线保持，完整目标 ACTIVE。** 唯一主 RMD_predicted 的星地A_K为29.201764%、R_U为41.041863%、H为34.123944%；对旧PCD的A_K增加1.250041个百分点，但R_U下降8.265836个百分点、H下降1.554167个百分点。相对背景项没有改善本轮联合表现，不能将增加接受或pooled/MD的局部高值当作成功。

Run `20261031-phase3-pcd-relative-metric-s2026092701-r01`；actual code `a24c413e5ab727ec283c07405eb9b1b2e68b036b`；predict/score均COMPLETE/0，pipeline `ANALYZED_DEVELOPMENT`。设计、公式与权限见[最终计划](PAMF_PHASE3_PCD_RELATIVE_METRIC_PLAN_20261031.md)。

## 方法与固定边界

严格冻结原F0 E50/11100/default6、fresh/f4b5 TARGET_PCD444及DURA v1.0 R69。PCD沿source-only r03初始化并继承U576适应祖先；本轮拟合输入仅source L，不等于整个模型source-only。身份c始终是同DURA原F0默认头预测，pooled分数也不改判身份。

新统计只读source_geometry074缓存的L6300最终PCD unit160及合法L标签，按ID对齐公共元数据。FP32单位向量按既有density.unit转FP64重新单位化；六类均值不再单位化，拟合共享类内population协方差与不分标签的全局population协方差，两者均固定0.9C＋0.1trace(C)/160 I再Cholesky。无收缩扫描、jitter/floor或静默回退。

若d_c²为到类c的平方马氏距离、d_0²为全局背景平方距离，则四新分数依次为MD_pooled=−min d_c²、MD_predicted=−d_F0²、RMD_pooled=d_0²−min d_c²、RMD_predicted=d_0²−d_F0²，高分接收。两种predicted及所有阈值路由均用F0预测类，不用真TX。新背景是源L全局统计，不是OE GMM或目标拟合。

先实际checkpoint无查询合成烟测，再读L并冻结新统计。V27000仅独立校准四新六τ，旧PCD两列及其两组六τ/密度/缓存预测精确复用；全部36τ先于NV/Q冻结。全6×3预测完成后，独立CPU才连接source NV、clean Q、star Q真值各一次，U truth0，无再拟合或阈值扫描。

## 数据与完整18运行点

沿source合同392005：L6300/Us56700/V27000及OE6134训练祖先，源五RX1/3/4/6/8（1-19/18-2/19-2/2-19/3-19）、03-08/15/23、旧六TX保持。新统计仅L；Us/OE/V之外的拟合与新目标U读取均0，source V是额外已知标签校准而非独立已知测试。NV三类19-15/19-16/7-1各1050，合计3150。

目标RX3-2/index34未来Q10238：旧4508、未注册四类5730；旧六类各1491/1182/150/406/1186/93，四未知5-5/10-7/2-19/7-11各1454/1439/1427/1410。03-15/23各5525/4713，clean/star同canonical且均DURA。PCD早日U576仅祖先，本轮不新访问；7-11曾从该早日U整类排除，不宣称所有未注册设备从未无标签曝光。

固定G10源MIX01 scene2026103001/channel2026104001、future scene2026103133/channel2026103134、receiver2027；600 km/25 MHz/2.45 GHz、residual/post_sync/noEQ/refSNR30 dB/reflect，一次input RMS、无output RMS。本轮只消费缓存，不新生成received、重抽信道或恢复真实IQ。

下表为逐TX等权宏百分比。C_K是已知接受率；原F0同DURA闭集依source/clean/star固定90.981481/48.567015/41.721059%，六方法完全相同。

| 方法 | 条件 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|
| OLD_PCD_pooled_ratio | source_DURA | 89.770370 | 97.081481 | 63.619048 | 74.465443 |
| OLD_PCD_pooled_ratio | clean_DURA | 32.123647 | 44.791196 | 60.790839 | 42.034854 |
| OLD_PCD_pooled_ratio | star_DURA | 29.175335 | 50.230381 | 48.454372 | 36.420916 |
| OLD_PCD_predicted_ratio | source_DURA | 89.733333 | 96.748148 | 64.285714 | 74.907247 |
| OLD_PCD_predicted_ratio | clean_DURA | 31.661804 | 43.761391 | 61.979647 | 41.912794 |
| OLD_PCD_predicted_ratio | star_DURA | 27.951722 | 48.582412 | 49.307698 | 35.678111 |
| MD_pooled | source_DURA | 89.977778 | 97.537037 | 47.746032 | 62.386916 |
| MD_pooled | clean_DURA | 32.097418 | 45.536130 | 56.306507 | 40.887178 |
| MD_pooled | star_DURA | 29.506978 | 52.400818 | 42.398531 | 34.797125 |
| MD_predicted | source_DURA | 89.962963 | 97.337037 | 48.761905 | 63.244111 |
| MD_predicted | clean_DURA | 32.390903 | 45.544550 | 56.877771 | 41.275898 |
| MD_predicted | star_DURA | 28.937084 | 51.654685 | 42.655747 | 34.482026 |
| RMD_pooled | source_DURA | 89.988889 | 97.585185 | 44.603175 | 59.643786 |
| RMD_pooled | clean_DURA | 32.538271 | 47.597363 | 52.683208 | 40.229776 |
| RMD_pooled | star_DURA | 29.957922 | 55.114687 | 39.806964 | 34.187225 |
| RMD_predicted | source_DURA | 89.974074 | 97.311111 | 46.158730 | 61.015257 |
| RMD_predicted | clean_DURA | 32.619200 | 46.794910 | 53.900457 | 40.642551 |
| RMD_predicted | star_DURA | 29.201764 | 52.810850 | 41.041863 | 34.123944 |

主对原F0闭集的A_K损伤依source/clean/star为1.007407/15.947815/12.519295个百分点。原F0身份不变，A_K收益来自拒识集合变化而非身份纠错；C_K增加不能证明拒识分离改善。

## 配对：接受增加与未知损失

| RMD_predicted−对照 | 条件 | 正确接收gain/loss/net | 接受net | 未知拒识gain/loss/net | ΔA_K pp | ΔR_U pp |
|---|---|---|---:|---|---:|---:|
| OLD_PCD_pred | source_DURA | 98/33/65 | 152 | 14/585/-571 | 0.240741 | -18.126984 |
| OLD_PCD_pred | clean_DURA | 24/11/13 | 90 | 41/501/-460 | 0.957396 | -8.079191 |
| OLD_PCD_pred | star_DURA | 37/10/27 | 129 | 32/502/-470 | 1.250041 | -8.265836 |
| MD_pred | source_DURA | 17/14/3 | -7 | 23/105/-82 | 0.011111 | -2.603175 |
| MD_pred | clean_DURA | 17/4/13 | 57 | 17/187/-170 | 0.228297 | -2.977314 |
| MD_pred | star_DURA | 13/3/10 | 49 | 16/108/-92 | 0.264680 | -1.613885 |

星地对OLD_PCD接受净＋129、正确净＋27，错误接受因而净＋102；未知拒对净−470。对MD接受净＋49、正确净＋10，未知拒对净−92，H再降0.358082pp。source相对OLD_PCD正确净＋65却少拒未知571，R_U下降18.126984pp，不能用source known保护通过掩盖拒识损失。

MD与RMD共享同一源均值/协方差，区别是背景距离项；四读出各自用固定known12规则校准τ。这里比较的是完整读出运行点，不是相同数值τ下的纯代数效应，也不能据此泛化为一切RMD无效。MD只作机制对照，不替换固定主。

## 分组、覆盖与风险保护

| 主星地场景 | A_K% | C_K% | R_U% | 原F0损伤pp |
|---|---:|---:|---:|---:|
| practical_high | 29.383767 | 45.995890 | 49.247644 | 12.475440 |
| practical_mid | 28.471003 | 43.156403 | 45.864905 | 20.505408 |
| practical_low_suburban | 32.319048 | 52.253443 | 42.495083 | 14.667255 |
| practical_high_urban | 38.886158 | 55.902462 | 44.373768 | 14.367512 |
| practical_mid_urban | 32.718845 | 49.191388 | 40.354367 | 17.080212 |
| practical_low_urban | 24.276409 | 55.223287 | 37.573373 | 8.655782 |

总体及六主景类别覆盖完整，但clean R_U≥95%、star R_U≥90%、C_K≥90%、原F0同DURA闭集−A_K≤2pp未联合通过；唯一主的星地A_K虽高于OLD_PCD，R_U不下降条件失败。门槛保持，不因MD或pooled某项更高改变主方法。

| 星地旧TX | 条数 | 主A_K% | 主C_K% |
|---|---:|---:|---:|
| 14-10 | 1491 | 45.003353 | 65.861838 |
| 14-7 | 1182 | 12.013536 | 69.543147 |
| 20-15 | 150 | 4.666667 | 32.666667 |
| 20-19 | 406 | 2.955665 | 14.778325 |
| 6-15 | 1186 | 60.033727 | 66.273187 |
| 8-20 | 93 | 50.537634 | 67.741935 |

四未知的主星地R_U依5-5/10-7/2-19/7-11为31.155433/56.289090/36.510161/40.212766%；clean为33.356259/71.091035/57.182901/53.971631%。source NV三类R_U为53.714286/47.619048/37.142857%。各类容量不等，所有宏指标按类等权，不能用总条数替代。

03-23缺旧20-19与8-20，完整A_K/C_K/H为N/A，R_U可独立报告clean54.055832%/star40.432788%；03-15主A_K为clean32.985067%/star29.578927%。不得删除缺类后认证。全部TX/RX/day/场景、配对及逐类Wilson95%区间见既有score/results与analysis，区间不补足历史曝光或缺类。

source known12/cal通过，但known42未全过；额外交叉超限为RX3/high（F0损伤3.235961pp）、RX3/mid_urban（3.155797pp）、RX3/low_urban（2.057902pp）、RX4/high_urban（2.074315pp）、RX8/low_urban（2.622069pp）。五RX主A_K范围83.148148%至92.685185%，源日1/2/3为90.111111/89.933333/89.877778%；这些仍是源校准诊断，不是未见RX验证。

主source AUROC93.494522%/OSCR86.774354%，star49.681270%/25.370144%；旧PCD星地50.997190%/27.001261%。曲线与固定运行点均未支持本轮联合收益，不能把接受率或AUROC单项当晋级依据。

## 状态容量、成本与执行

新metric聚合schema为`pamf_phase3_pcd_relative_metric_v1`，五数组仅保存类均值、全局均值和两Cholesky，无逐行源bank；旧density单独保存且不重拟合。统计容量按实际marker与固定shape分列，数值不是新增NN参数。

| 新state成员 | FP64数值个数 | 数值字节 |
|---|---:|---:|
| class_means[6,160] | 960 | 7680 |
| background_mean[160] | 160 | 1280 |
| within/background_cholesky，各[160,160] | 51200 | 409600 |
| 数值合计 | 52320 | 418560 |
| Unicode schema标量 | N/A | 136 |
| 五数组实际合计 | N/A | 418696 |

本轮source_metric_fits=1，V四新六τ24值/192 B，旧两六τ12值/96 B精确复用，共36值/288 B、单主6τ48 B；不能称零统计学习。新增NN优化、真实F0/signal/G前向、IQ生成、Us/OE/目标U拟合均0，缓存特征不增加独立物理样本。

实际checkpoint烟测F0/G各2、signal4（r03/PCD各2）PASS，合成与真实前向分开。主部署神经参数仍F0 232993＋PCD334721＋G266951=834665；继承F011100/r032220/PCD444及U576、G4940与原源特征/GMM/缓存成本，不部署source行bank。
实际predict 14.650344 s、CPUscore 10.262796 s、pipeline 25.026522 s，是缓存评分流程walltime，不能当部署B1时延。4次观测独立核实driver1766648、predictor1766669/CUDA5及CPU1767312的CWD/完整argv；CPU CUDAempty，GPU UUID ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331单次376 MiB非峰值。
三stdout合计4行，driver/predict/score为21/558/49 B，完整扫描warning/error0；predict log0→517→558 B，终态live/owned GPU空、foreign0。全预测先冻结，真值仅CPU三次连接；本报告只读既有JSON/日志，不读NPZ、权重、IQ或原truth，不重评分。

验证11unique＋34sub PASS（core5＋22、eval/score4＋5、launcher2＋7），同case实际producer→launcher交接1＋5/8.29s不另计unique；8 AST/3 CLI/实际config PASS。唯一P0/P1审查无P0/P1，测试1056条旧trapz警告与生产零警告区分，无重复审查或新测试。
包793files/data0，SHA256 `644f9596a33d538dfb5cb344dba34ac70e845d681c3a35d323718d15a554b169` 单次比对、remote compile/runtime VERIFIED，真实pipeline无技术失败。

source V是校准角色，NV/Q已曝光、单开发RX/seed、全项目physical history UNKNOWN、正式RX24-18未读。clean仅测试；无新targetUDA、发现或匿名REG，adaptation_success/A_N/H_OW均N/A。保留原PCD及历史困难集合，不从当前Q选择新收缩/阈值或宣称机制普遍无效。

完整[评分结果](../local_artifacts/pamf_phase3_pcd_relative_metric_20261031_r01/score/results.json)、[分析](../local_artifacts/pamf_phase3_pcd_relative_metric_20261031_r01/analysis.json)、[校准](../local_artifacts/pamf_phase3_pcd_relative_metric_20261031_r01/predict/calibration/complete.json)、[终态](../local_artifacts/pamf_phase3_pcd_relative_metric_20261031_r01/pipeline_state.json)可核查。
