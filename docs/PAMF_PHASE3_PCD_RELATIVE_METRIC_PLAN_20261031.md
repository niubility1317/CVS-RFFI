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
