# F＋跨 RX 同 TX 对比：增强研发复盘

当前基线由用户固定为 F＋跨 RX 同 TX SupCon，scratch E20；默认 cosine 头和骨干联合更新。该覆盖不等于加载历史 E50 权重。以下当前基线实验均为合法源域留出检验，不是正式目标测试；五个源折也不是五个模型 seed。

|受控比较|实际观察|能支持的判断|
|---|---|---|
|普通 practical 配对 CE，五个源 RX|urban 均值 +20.039506 pp、low +14.081481 pp；RX6/RX8 clean 分别 −3.648148/−2.129630 pp|真实 As 的适应收益明确，但 clean 保护不稳定，固定配方未通过全部源折|
|同 TX/day 不同 RX 真实观察配对，held RX1|CROSS−C0 clean +0.425926 pp；CROSS−SAME clean −0.444444 pp；CROSS 的六 practical 全部下降|增加跨 RX 真实同类观察没有在本配方提供独立收益，不能称为有效数字 Ag|
|精确公共相位轮转，held RX1|clean +2.185185 pp，但 seenV 下降 0.671296 pp；low −2.037037 pp，六 practical 全降|clean 局部改善没有转化为星地改善，并违反预登记源保持条件|
|有界广义线性 RXMIX，held RX1|clean +0.703704 pp，未达到固定 +1 pp；urban −1.395062 pp、low −1.074074 pp，六 practical 全降|源椭圆性统计阳性和数值可逆不足以证明训练时身份保持或有效 Ag|
|可恢复状态课程，held RX6|对 C1：clean −0.277778 pp、urban +0.462963 pp、low +0.055556 pp；对 C0：clean −3.925926 pp|课程已实际执行且日志健康，仍未修复 clean，按固定规则关闭|

当前 F＋跨 RX 的普通配对训练对 LOS/PARTIAL/locked 收益较大，BLOCKED/unlocked 收益较小。五源 pooled low urban 的 C0→C1：LOS 36.6190%→64.1942%、PARTIAL 30.2451%→54.9960%、BLOCKED 20.2689%→24.5205%；locked 35.5215%→63.2356%、degraded 24.4915%→35.7404%、unlocked 19.5826%→22.9340%。这是关联分层，不是噪声、多径或门控的独立物理误差份额。

对固定父模型的控制诊断没有支持单一原因解释。源 RX1 C1 在 low urban 移除新增噪声后的受控增量约 +5.16 pp，但仍无法恢复 clean；拉平残余核、去残余载波或去注入 RX IQ 的单项增量有限。实际 EQ 关闭，G=1，不能把 residual 解释为已均衡。FIR 数值延迟、反射边界和有效观察长度均有记录，不能通过弱化正式信道或提高 SNR 消除困难。

旧基线上的去 PA、WHITE30、观察原点扰动、特征／分支／joint 对齐和跨 RX 教师等负结果继续保留，但不把其数值和当前 F＋跨 RX 的数值相加。独立噪声 HQ 的 64.86%→70.97% 来自源 L_dev720、冻结 EMA、零更新；后续匹配训练的 low urban 增量远小于这一冻结诊断，不能将它写成正式训练收益。

这些结果共同否定“只要输入变得更像另一个 RX、辅助特征更接近，或 clean 单折上涨，就已经形成真正混合增强”的判断。它们没有证明所有轻度 IQ 增强都无效，也没有证明 BLOCKED 必然不可恢复。原普通增强已经包括相位、CFO、相位噪声、AWGN、多径、DC、taper、时间平移、增益与部分 PA；重新命名同一扰动不能算新 Ag 的独立证据。

本轮独立检验的单一假设是：保留至少 50% 原 nominal 支持，在源域训练状态健康时有界增加可恢复 mid/low urban 观察，能否恢复已知失败源 RX6 的 clean 而不损失 C1 的 urban。实际一个 scratch E20 和七种源视图已全部完成，固定判据 FAIL。4L 前向、原 U/EMA/SupCon 不变，无新一致性损失、诊断头或部署分支。568320 监督视图中实际替换原 nominal 92602 个、完整物理提案1046334次；不是未执行课程导致的空比较。关闭固定课程，不扫描 eta、质量区间、warm、提案数或替换 RX/seed。详见[课程完整报告](../automation_reports/CV-SincNet/20261016-phase1-FcrossRX-recoverable-source-RX6-e20-s2026092701-m1-r01/report.md)。

后续真正 B00/B10/B01/B11 四格仍要求 Ag 的独立源域价值成立，并报告交互项。当前没有新联合模型证明 low urban >60%、clean 净正、suburban 接近自身 clean 或跨模型 seed 重现；不能用源单折、梯度健康、门控变化或执行正确替代这些结果。

逐行证据、成本和完整产物索引见[研发记录](cvs_genuine_mixedIQ_next_20261016.md)。

![已完成候选的 clean 与 low urban 增量](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/history/closed_routes_clean_low.png)

上图仅使用已完成 run 的保存结果，四个点均为相同 held 源 RX1、一个 model seed、E20 的匹配 C0 增量。SAME_RX 为真实观察配对中的控制，不是另行晋级候选；这些点不是四个独立 seed，也不估计置信区间。数值见[源结果表](E:/type10-7/local_artifacts/cvs_FcrossRX_recoverable20_20261016/history/closed_routes.csv)，绘图没有新 IQ／checkpoint／网络访问。

新增已核实历史证据（2026-10-16）：另一路 owner 的固定 75/25 practical 使用六 L 前向和 heldL1260，和上表本路线四 L 前向／heldV5400不同，不直接比较绝对分数。两种子同配方的 clean 配对差为 +3.095238／−13.095238 pp，均值 −5.000000 pp；urban 配对均值 +12.460317 pp，但第二种子未通过原 clean 条件，固定比例已关闭。它只证明星地适应有重复收益，没有证明 clean 正收益能跨 seed。参见[二种子报告](E:/type10-7/automation_reports/CV-SincNet/20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01/report.md)。

同一六前向／heldL1260路线的常数增益作用核对通过，E20 clean 仅 +0.555556 pp、六 practical 全部下降，未通过 +1 pp 和保持条件；频偏训练 clean 0 pp、urban −1.084656 pp，也已关闭。不能用实际输入／logits 发生变化代替身份收益。来源：[常数增益](E:/type10-7/automation_reports/CV-SincNet/20261016-phase1-F-crossRX-constantgain-sourceRX1-e20-s2026092701-m2-r01/report.md)、[频偏](E:/type10-7/automation_reports/CV-SincNet/20261016-phase1-F-crossRX-frequency-sourceRX1-e20-s2026092701-m2-r01/report.md)。

缓变包络 G0 虽有跨 TX/day 的 RX 关联，但线性 R² 中位数 0.303317 未达预定 0.8，关闭；本轮重复差相关 G0 也关闭：5040 包／18折独立复算一致，RX3 残差功率比中位数 10.275181% 高于预定 10%。后者的 RX 对比平方风险下降 62.523108% 不是默认头准确率，更不证明残差是纯接收噪声。0 模型／网络前向／更新，CPU 1.689140 s。见[本轮报告](../automation_reports/CV-SincNet/20261016-diagnostic-source-repeatcorr-crossfit-manysig-m1-r01/report.md)。

截至此处，没有通过原定源规则的独立 IQ Ag；不把 RDrop、既有 SupCon、源统计可重复性或同类真实观察重新命名成已验证 Ag。固定 F＋跨 RX 同 TX 基线与 E20 保持，当前正式联合提升及跨 seed 重现均未证明。

局部观察衰减 LA16 的实际 E20 源 RX1 结果补充：clean +2.555556pp、宏 F1 +2.635702pp、seenV +0.041667pp，但六 practical 全降，urban −1.919753pp、low −1.370370pp，原源条件 FAIL。low 的 BLOCKED 仅净+8、unlocked净+18，而 LOS/PARTIAL/locked/degraded 的损失更大；TX2 low −14pp。新修复 clean 的159包，其low正确反而42→28，说明局部clean收益没有转化为受损观察下的同一身份证据收益。原C0 clean正确集low保持率25.943187%→25.011096%，平均间隔略升也未变成准确率收益。保持全部负结果，关闭固定独立配方，不做强度/比例/RX/seed救援。一个sourcefold/modelseed、0As/target，不能据此否定所有未运行联合交互，也不能宣称Ag总体有效或联合完成。[报告](../automation_reports/CV-SincNet/20261016-phase1-FcrossRX-localatten-source-RX1-e20-s2026092701-m1-r01/report.md)。

冻结 current C0/C1 的 ordinary／As 梯度共同支配假设按原规则未获支持：C0 负内积批 42.5%／32.5%，C1 12.5%／17.5%，控制均0；120964 NN、0更新。未检验失败RX6或训练早期，不能据此排除所有梯度机制，但本固定理由关闭，不从层/RX/流/阈值挑阳性晋级。[报告](../automation_reports/CV-SincNet/20261016-diagnostic-FcrossRX-cleanAs-gradients-sourcefit-s2026092701-m2-r01/report.md)。


新增完整匹配结果（2026-10-17）：

独立噪声／反向噪声双视图的三行同预算 E20 检验已完成。ANTI 对 RAW10 的源 held RX1 clean 为 +6.1111 个百分点、urban 均值为 +10.3175 个百分点，但对匹配 INDEP 控制的 urban 为 −0.1323、low 为 −0.3968 个百分点，未达到两项均至少 +1 的预登记条件。收益主要出现在 LOS／PARTIAL／locked；BLOCKED／unlocked 没有获得对应改善。固定反向噪声训练路线关闭，不依据 INDEP 的局部高分追加种子或重新选候选。[完整报告](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-FcrossRX-noisepair-sourceRX1-e20-s2026092701-m3-r01/report.md)。

随后冻结 ANTI 默认头、只在源 fit L5040 上移除新增噪声并固定接收状态及补偿参数，BLOCKED 的 clean→full 错误差仅恢复 17.6018%，unlocked 仅恢复 18.3118%，均未达到原定 50%。这否定了当前噪声遮蔽理由的晋级条件；没有证明其他损伤不可恢复。该诊断为已见训练源、零更新，不能作为 held RX 或正式 low urban 超过 60% 的证据。[受控诊断报告](E:/type10-7/automation_reports/CV-SincNet/20261017-diagnostic-FcrossRX-postAs-noise-control-fitL-s2026092701-m2-r01/report.md)。

有界包内时间栅格收缩 TIMEGRID 的一个 scratch E20 已完成，默认头的源 held RX1 clean 为 80.4762%→80.4762%，宏 F1 下降 0.0179 个百分点；urban 均值下降 0.1587 个百分点，low 为 27.6190%→27.6984%，只净增 1 个正确样本。clean 修复 6 个错误并新增 6 个错误。全 4,440 更新、568,320 次 IQ 变换、L／U 序列、覆盖、损失与梯度健康均核对通过，故不是增强未执行。固定候选未通过预登记 clean 与宏 F1 条件，关闭；不调整 rho、插值、概率、权重、RX、种子、epoch 或门槛救援。有界可逆与轻微输入变化都没有在这次训练产生独立泛化收益，且不能把插值效应唯一解释为 RX 时钟变化。[全量结果、分层图与训练曲线](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-FcrossRX-timegrid-sourceRX1-e20-s2026092701-m1-r01/report.md)。

这三项都是源域证据；原有默认分类头、F＋跨 RX 同 TX SupCon 基线、E20 预算和正式 practical 测试条件保持。当前仍没有经独立确认的新增 Ag，也没有 B00／B10／B01／B11 交互、正式目标联合收益或跨模型 seed 重现。新增假设尚需依据合法源域证据另行确定，未启动新的补救训练。


非 DC 局部谱峰 G0 已完整执行并独立复算：fit L5040、18个跨TX/day留出折，0模型／网络／权重／更新。两窗RX对比风险下降94.8190%／91.5295%，17/18折局部峰条件通过，但精确RX/bin仅14/18一致，未达原15/18；固定候选FAIL，关闭且不放宽频率容差。两种选频均靠近既有20点重复参考谐波，与梳状谱相符，不能当作独立接收杂散或安全Ag方向；此解释不用于重选。统计阳性不是默认头收益，未启动E20。见[完整报告](E:/type10-7/automation_reports/CV-SincNet/20261017-diagnostic-source-nonDC-spur-crossfit-manysig-m1-r01/report.md)。
