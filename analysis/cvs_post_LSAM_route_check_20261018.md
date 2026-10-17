# SAM 后续路线核查：As 专属 KL 移除已有负结果

2026-10-18。状态：ANALYZED_EXISTING_EVIDENCE_ONLY。当前HEAD为f07ab91fe4537c98e6e869ee5d8e9552cd6a7663，上一轮完成三行E30训练、七场景truth-last与独立复算，属于实际进展。当前保持F＋跨RX同TX、全部源RX[1,3,4,6,8]、day[1,2,3]、完整L6300/U56700隐藏TX/V27000、split392005、256点、默认cosine头共同训练、原U规则与每模型E30；本核查没有改变任何训练或测试配置。

## 本轮具体假设与实际核查

在加入任何新训练前，核对“只移除As对的随机子网络KL、保留ordinary RDrop和同一As的两次CE，能否是未检验的独立训练组织路线”。已读取当前模型hook、完整AS_REPLAY十轮损失分解，以及历史cvs_as_pair_ce20的实际design、完整结果和固定放弃条件。发现该假设已经有直接匹配负结果，故未实现或发布新候选。不是将没有检索到的路线当作新方法。

当前AS_REPLAY的2220步/10轮记录加权均值为common CE 0.266713892903、common KL 0.029639509355、As CE 0.659351153591、As KL 0.042068351654。KL非零只说明目标中存在此项，不能说明它有害，更不能预测去掉它有识别收益。

[历史As专属KL移除完整报告](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-AsPairCE-sourceRX1-e20-s2026092701-m3-r01/report.md)已ANALYZED／VERIFIED，FAIL_FIXED_AS_KL_REMOVAL_CLOSED：匹配8L前向/总对权重1.25、原base RDrop和SupCon/U保持，新C_CE只移除额外As对KL。相对As＋RDrop，clean−2.539683pp、lowurban+.714286pp、urban均值−.555556pp，midurban−1.190476pp；固定clean/low/mid/urban/suburban条件失败。该run为从零E20、fitRX3/4/6/8的L5040/U45360/V21600，RX1源读出L1260、seed2026092701，无target；不能将其数值当当前全源E30成绩，亦不能据此宣称所有KL移除都普遍无效。其固定关闭约定明确不改系数、场景、epoch、RX或seed救援。本轮不以新预算或完整源域为借口重复该关闭路线。

## 场景覆盖与关联对话

当前fullsource原Practical训练只抽取原三种TRAIN_SCENES；这项事实本身不证明扩场景能提高识别。实际仓库已有[六场景IID与逐物理包循环的匹配源实验](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01/report.md)及其完整关闭记录；不能把“六场景循环”重新命名为尚未探索的增强，不根据当前正式分数修改其边际概率救跑。旧4RX/E20结果与当前5RX/E30不同，历史边界需保留。

已重新读取关联对话“开发 CVS Phase1 星地增强训练”的完成内容，并核对其当前本地策略。该对话的[跨视图SupCon覆盖E30报告](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01/report.md)记录固定SAT192源资格失败、正式urban各场景下降，三行完整七视图评分结束。此处不复制其视图覆盖或新增关系成员路线；其scratch自身20+10与本路线原E20父续10分开，不相加预测收益，不把其他对话的结果当本路线配对控制。其本地策略另有polarity源检验关闭状态，本轮未复算该源结果，不援引其具体指标。

## IQ 候选来源边界

定向核查窄带干扰没有形成可发布的新IQ Ag。已完成的[receiverBandNoise完整L6300源检验](../automation_reports/CV-SincNet/20261017-diagnostic-F-receiverBandNoise-fullL-s2026092701-m1-r01/report.md)在固定带限Gaussian配方下失败并关闭，不调整带宽/SNR/窗口/抽样救跑。窄带连续波和带限Gaussian并非同一个算子，但本轮没有取得当前源数据存在可分离RX窄带杂散、以及其强度/身份保持的证据。

[WiSig官方说明](https://cores.ee.ucla.edu/wisig/overview/)说明各TX逐一发送，识别信号是前导首256点及其均衡版本；它没有给本候选提供逐包背景干扰或长上下文映射。[Analog Devices原始技术说明](https://www.analog.com/en/resources/technical-articles/use-selectivity-to-improve-receiver-intercept-point.html)讨论接收机对干扰及互调杂散的响应，这是物理动机，不能证明当前ManySig中有相同因素，也不能给已均衡256点数字加音提供身份保持保证。因此未生成CW IQ或新增源模型读出；这不是证明所有干扰增强均无效。

## 当前判定和成本

本轮最小动作是读取已有实际配置、全部当前十轮记录和历史已完成匹配实验。重复假设的放弃条件已由既有识别结果触发，固定As KL移除及六场景循环继续关闭；没有新增候选、幅度、权重、父模型、RX、seed或训练预算。当前不存在足够源依据支持的新IQ Ag假设，不能为满足“每轮新假设”格式而编造一项。下一真实训练仍须基于不同的合法源依据，不能复用本页统计宣称泛化、星地或联合成功。

本轮新增IQ读取/生成、checkpoint加载、NN前向、教师前向、梯度、优化更新、源/目标预测、truth读取、远端任务均0；仅历史文件和已完成同row汇总的只读复算。没有重复烟测、审查、审批、发布或数据验证。未修改历史实验或其他健康任务。完整目标保持ACTIVE，仍缺有效新Ag、B10 clean独立收益、B11联合clean净正、lowurban>60、suburban保持和跨seed证据。
