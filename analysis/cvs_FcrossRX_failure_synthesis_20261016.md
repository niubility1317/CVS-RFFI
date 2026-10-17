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


## P12：原普通增强的物理 RX 分组条件（2026-10-17）

两组新 scratch E20／seed2026092701／源 held RX1L1260，F＋CROSS_RX／原6L／原U／defaultcos30，唯一改变原 anti-shortcut 与 general-channel 的跨 RX 条件相关性，无新算子／损失／NN／As。训练代码82975411f152e2be7d80ebc79b7caa850d2ee858。

独立完整复算：clean81.6667→81.8254（＋0.1587 pp，13修复／11损坏），低于预登记＋1 pp；macroF1＋0.2974 pp、seenV＋0.0417 pp及训练健康均通过。urban均值32.8042→33.8095（＋1.0053 pp），low28.3333→28.8889（＋0.5556 pp，37修复／30损坏）；所有六种practical均小幅增加，不能称为独立泛化突破或跨seed复现。

low urban：LOS＋2.6042 pp／locked＋2.3555 pp；BLOCKED＋0.1401 pp；PARTIAL−2.4691 pp／degraded−3.5088 pp，困难状态仍无稳定改善。这是固定状态表现分层，非噪声／多径因果消融。

完整8880训练步、40 model-epochs、9373358记录NN（含兼容smoke2）、全部七视图、原L/U序列完全匹配、72cells/5040L各组覆盖，约1268.4 s到矩阵结果；进程自然终止，无target访问，所有输出保留。固定RX分组候选科学FAIL并关闭，不改算子子集／分组／RX／seed／阈值／budget救援。

[完整报告](cvs_FcrossRX_rxgroup20_20261017.md)。run_id：20261017-phase1-FcrossRX-rxgroup-sourceRX1-e20-s2026092701-m2-r01。仍无已验证新Ag／Ag×As因子矩阵／正式target联合证明／跨seed low urban>60%或suburban接近clean证据，活动目标继续。


## 补充：原普通增强的独立控制证据（2026-10-17）

另一路 owner 已完成两个三组新 scratch E20／8L／源 held RX1／seed2026092701 的固定检验，不能把其未选控制事后晋级：

- 原生 AWGN20至36dB 的独立加噪：RAW／复用原ordinary控制／AWGN clean为79.6825%／81.1111%／80.6349%。新AWGN相对RAW＋0.9524pp、相对ordinary−0.4762pp，均不满足预设双控制clean条件。固定范围和full-coverage配方关闭，不扫描SNR／权重／RX／seed救援。[完整报告](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-F-crossRX-isolatedAWGN-sourceRX1-e20-s2026092701-m3-r01/report.md)。
- 有界非仿射相位 sigma.006／peak.1：RAW／ordinary／新phase clean为80.6349%／81.4286%／80.0794%，相对两控制−0.5556／−1.3492pp；固定配方关闭。[完整报告](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-F-crossRX-boundedPhase-sourceRX1-e20-s2026092701-m3-r01/report.md)。

这些ordinary额外对重用同一份已经生成的IQ，没有重新抽普通增强；因此不等于已检验第二个独立观察。其同seed原生raw控制与历史同参数控制存在约1pp级判决变化，来源尚未独立定位；不能只凭相同seed和参数宣称逐包复现，也不能将不同run的局部涨分相加。本轮独立观察采用三个新匹配控制，仍要求后续独立seed确认，当前运行不作参数调整。


## P13：同一步内独立 ordinary IQ 观察（2026-10-17）

三个新匹配 scratch E20／源 held RX1L1260／seed2026092701，F＋CROSS_RX／默认cos30／原13L＋7U保持。完整base raw.75／ordinary.25＋SupCon保留，另加.25同IQ RDrop pair；RAW返回raw，REPLAY复用原prePA缓存，REFRESH独立抽原anti→channel。三组同8L、同额外fresh shadow成本，无新增PA/DAC／头／部署参数／As／U视图，三组原L/U序列和保存的原/fresh统计完全相同。

默认头clean：RAW80.3175%、REPLAY81.4286%、REFRESH82.0635%。REFRESH−RAW＋1.7460pp，但−REPLAY仅＋0.6349pp，低于两控制都≥＋1pp的预登记门槛；F1／seenV和健康条件通过，固定方案科学FAIL并关闭。clean对RAW修复28／损坏6，对REPLAY修复15／损坏7，不能将全部相对raw的收益归因于独立刷新。

REFRESH相对RAW六practical全部下降：urban均值−1.2434pp、low27.9365%→27.2222%（−0.7143pp；37修复／46损坏）。对REPLAY urban仅＋0.0794pp、low−0.0794pp。low的BLOCKED相对RAW＋0.8403pp，但对REPLAY−0.1401pp；unlocked相对RAW＋0.8850pp而对REPLAY−0.1770pp。LOS／PARTIAL／locked相对RAW下降，未证明困难urban退化稳定缓解。suburban与clean仍相差约45至48pp，不能声称联合突破。

全13320更新／60 model-epochs／17469956 NN（含dispatcher兼容smoke2），约1563.07s到矩阵完成；每组4440生成和消费／568320fresh包／5040唯一L／72格，完整原生loss/grad／20轮／全stdout／七视图独立复算PASS，5PNG＋5PDF保留。训练commit c72c359581cee36e9fc70440844803288ba851ef；默认formal practical不改、target0，全部进程自然终止。

[完整结果、分层图与曲线](cvs_FcrossRX_viewrefresh20_20261017.md)。关闭固定独立ordinary观察，不扫强度／概率／权重／RX／seed／预算救援，不将REPLAY控制事后晋级。一次source RX／modelseed不能说明正式target增益或跨seed复现；没有新Ag确认／Ag×As因子证据，low urban>60%及最终联合目标仍未完成。


## P14：有限波形功率代理与接收状态切换（2026-10-17）

CPU源fit L5040、两个物理随机流×六原practical、0模型/权重/NN/更新/target。固定H、rx/day/eq1硬件session和底层噪声抽样，比较原白输入代理、显式重放、实际Hx全长功率、无人工FIR前缀内部功率及固定原status/scale控制。完整60480记录／302400原生物理process／864TX×RX×day格独立复算PASS，运行448.956941s，4PNG＋4PDF保留。

mid urban绝对质量偏差中位数0.1728/0.1751dB、lock切换2.1032%/2.6389%；low为0.2337/0.2355dB、2.4206%/2.8770%，四组均未达到原≥1dB及≥5%。边界差约0.135至0.140dB、切换在内部保留68.85%至83.46%，内部至少195点，边界条件均通过。原生重放和固定接收状态控制IQ MSE均0。失败是整体偏差及切换不够大；不能据此断言每包代理都准确，mid有−8.887dB尾部。BLOCKED局部切换较多仍不事后挑子集晋级。

正式测试仍residual/post_sync/EQoff G1/reference30/reflect；新增噪声SNR不是总SNR。G1只排除EQ FIR着色，原RX广义线性IQ镜像与旋转仍作用于噪声；共享底层抽样不等于所有总效应条件的处理后噪声波形相同。完整接收状态转移、LOS/PARTIAL/BLOCKED、RX、CFO、AGC和边界均保留。固定代理/状态切换假设FAIL关闭，无自动E20或默认头试验，不改阈值／状态／流／seed／场景／RX／窗口救援。

本轮未测默认头准确率或clean代价；两个物理流不是模型seed。没有新Ag确认／Ag×As因子矩阵／正式联合收益或跨模型seed证明，活动目标继续。代码4f7968ec32ea52ff56f63c42c1654fb4507573d6，离线f78551c40afa82f8d6b4acb921a0099d641098f4。[完整报告](../automation_reports/CV-SincNet/20261017-diagnostic-FcrossRX-qualityproxy-fitL-manysig-m1-r01/report.md)。


## 补充：逐包场景循环与归一化机制（2026-10-17）

另一路固定逐包六scene循环已完成：B00/C_IID/B01的heldRX1 clean80.1587/84.4444/84.6825%，low27.7778/33.8889/34.0476%；B01对IID的low仅＋0.1587pp、urban均值＋0.5026pp，低于原各＋1pp，固定路线关闭，不改cycle或边际概率救跑。low修复20/新增18，两组clean均正确1061包中low净减少1，均衡覆盖本身不是身份收益。[报告](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01/report.md)。

固定F实际实例检查：206个state张量、900722参数，BatchNorm=0，采用GroupNorm/LayerNorm；故BN运行统计污染不适用。0真实IQ/权重/NN/优化更新的架构检查只排除该实现假设，不说明任何IQ增强有效。

新独立问题是固定E20内的等剂量As时序，不是Ag：RAW/ALL/EARLY7/LATE7四个新scratchE20/8L匹配，完整原base+原13L7U/oldgate/ownEMA保持，额外.25RD对按epoch使用raw或原singleAs(raw)，早晚各198912监督As包。四组全程生成同样568320原物理shadow，禁止source/target结果回流改日历；正式测试不变。它也改变As所在LR/SSL阶段，不冒称单独隔离身份先学因果；实际激活物理ID/状态集合仍需全量说明。代码f788309f5f9e27689ec1e98ae99a6bd83237e1f9，本轮已VERIFIED启动，尚无训练/准确率结果。[固定报告](../automation_reports/CV-SincNet/20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01/report.md)。


## P15：等剂量早晚 As 时序（2026-10-17）

四个新scratch E20／8L／源heldRX1L1260／seed2026092701，F＋跨RX同TX和原13L＋7U保持；原完整base外加.25同IQ RD对，RAW／全20轮As／前7轮As／后7轮As。四行同568320完整物理shadow，早晚均198912消费／全部5040唯一L／72物理格，非同一draw多重集合交换；LR和SSL日历共同改变，不能隔离先学身份因果。

默认头clean为79.841270%／83.730159%／80.000000%／81.269841%；urban均值32.222222%／43.095238%／39.735450%／42.354497%；low26.984127%／33.571429%／32.539683%／34.206349%。LATE对EARLY clean＋1.269841／urban＋2.619048／low＋1.666667pp，但对ALL clean−2.460317／urban−0.740741pp，违反原固定保持条件，科学FAIL关闭。不改启用轮次／剂量／系数／RX／seed／epoch／阈值救援，不将控制事后晋级。

对RAW收益主要在LOS/PARTIAL/locked，BLOCKED18.347339%→20.028011%、unlocked18.230088%→18.761062%，质量<0dB还下降；未解决最困难urban。全部80model-epochs／17760更新／23293274NN、七视图35280预测及完整曲线独立复算PASS，约2380.459s矩阵墙钟，5PNG＋5PDF已实际查看；全原PID自然终止，0target／另训头。

[完整结果、分层与曲线](cvs_FcrossRX_AsTiming20_20261017.md)。训练commit f788309f5f9e27689ec1e98ae99a6bd83237e1f9。一次源RX／modelseed不能证明正式target或跨seed收益；没有新Ag或Ag×As证明，low>60%、suburban接近clean和联合目标仍未完成。


## P16：注入 RX session 覆盖源反证（2026-10-17）

冻结P15 ALL原默认头，仅父已见fitL5040／RX3/4/6/8／一个modelseed；同H/底层噪声比较原12硬件session与三个固定virtual bank、原session逐点replay。原RX补偿/质量/传播边界保持，0新训练或target。完整45360判决／10080物理记录／50400原生调用／48session独立复算PASS，主体100.299359s，3PNG＋3PDF实际查看。

mid原73.253968%，三bank73.273810/73.273810/73.313492%；low原57.738095%，三bank57.698413/57.777778/57.817460%。原session较三bank平均优势−0.033069/−0.026455pp，平均hard变化0.449735/0.978836%，均未达预定+1pp与3%，固定覆盖理由FAIL关闭。不换bank/seed/场景/RX/阈值救跑，不据此启动session-bank训练。BLOCKED/unlocked仍约30.4/27.3%，有限新session并未改变困难状态大差距。

这不是新Ag或训练收益，fitL不是held/target，三个硬件bank不是三个modelseed。父历史E20/4440/5823318NN另列，本轮45362NN含smoke2，0更新/另训头。固定F＋跨RX、E20和正式测试保持；源荷取数ID为opaque，来源/角色已按实际缓存修正并通过审查。总体联合及跨seed目标仍未完成。[完整报告与图](cvs_FcrossRX_RXsession_source_20261017.md)。


## P17：五个名义 raw 控制的描述性差异（2026-10-17）

固定五个历史同 modelseed2026092701／held源RX1L1260／E20／8L／1.25 控制，0新训练/NN/IQ/target。完整44100源判决、66600步、29970L/U记录和100epoch独立复算；206随机初始tensor十对完全相同，有效role/完整顺序/归一化共同配方/存储ordinaryMSE逐项相同。最终clean79.682540%至80.634921%，跨度.952381pp；low26.984127%至27.936508%，跨度.952381pp。十对hard变化clean.555556%至1.587302%、low3.015873%至6.031746%。第1步已有约1e−8及以下梯度差异，但不证明CUDA或某实现的因果作用。

五控制不是五seed或新随机复制；相同MSE不证明逐点IQ相同。只有refresh/timing保存且匹配六景逐包physics，其余三控制缺失，不把未知写成已证实信道差异。历史四native counter＋held原始和29116600，扣除5行EMA deepcopy继承的非执行smoke计数10后实际29116590；本轮5个随机未训练CP/0训练后CP/0NN；3PNG＋3PDF实际生成查看。没有候选晋级、既有失败路线复活或当前SupCon-view判据调整，不把跨度当新阈值。仍无新增Ag／正式联合与跨seed证明。详见[完整描述报告](cvs_FcrossRX_RAWcontrol_audit_20261017.md)。


## 补充：原 SupCon 的视图组织（2026-10-17，原 owner 终态）

四行sourceheldRX1L1260／一个modelseed／固定F＋跨RX＋8L＋1.25／E20，新Q_sat/Q_mix和复用Q0/Q_ord已完成七源视图35280判决。固定Q_mix clean83.809524%、low33.730159%、urban均值42.592593%；对Q_ord为−.634921／−.158730／−1.322751pp，mid−2.063492pp；对Q_sat clean＋.873016、low＋1.111111而urban仅＋.555556pp，违反原各＋1pp。科学FAIL_FIXED_VIEW_ROUTING_CLOSED，不改比例/层位/weight/scene/RX/seed/预算，不选控制替代primary。新40modelE/8880更新与旧40modelE复用成本分开，0正式target／新部署头。

对Q0虽clean＋3.650794、low＋5.952381、urban＋9.523810pp，不能归因为新Ag或超越已有As；三suburban距自身clean仍25.95至32.70pp。原owner全量独立分析见[终态报告](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01/report.md)。本owner只读原报告和聚合产物，未重复收集/评分、改变健康任务或科学门槛。该As组织假设关闭，新增Ag、正式联合与跨seed目标均未完成。


## P18：源eq1观测的共同轴向格点（2026-10-17）

固定fitL5040／RX3/4/6/8／6TX／day1/2/3，0模型/CP/NN/更新/IQ变换/target。新测度在原eq1/RMS表示上，全部5040包均无满足固定候选及float32分辨界的共同轴向affine格点，72cell均0/70，未达逐格95%；因此0crossfit折、RX风险与格距/SNR代理均null，不对阳性子集重选。独立全量逐包/角色/72格复算PASS，2PNG＋2PDF实际生成查看；主体420.515955s。

当前固定格点作为接收增强依据的理由FAIL_CLOSED，不调格距/容差/窗口/RX/seed/阈值或自动dither/E20。没有证明原ADC无量化或任意旋转/非均匀格点不存在；当前表示经过处理，不能假定ADC位数加噪或套用subtractive-dither独立误差。该轮无默认头新准确率/clean代价，未获新Ag或联合/跨seed证据，完整目标仍ACTIVE。代码9b8eda74e12c03f385518c139d89f539c98163fa，离线5187b5dce47e954b98352cb40c7bca7e5686f8fc。[完整反证与图](cvs_source_quantgrid_20261017.md)。
