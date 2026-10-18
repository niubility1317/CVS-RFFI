# L 普通增强输出换 raw：完整 practical 的匹配 E30

状态：PLANNED，已登记、实现中，未发布或训练。实际代码提交在冻结后填写。

唯一假设：完整 practical 已提供传播和接收变化时，末10轮的有标签clean监督视图使用原raw，能否在不改变监督份额的情况下改善low/mid urban，并让clean对匹配NoAs下降不超过0.1个绝对百分点。本轮研究训练输入组织，不新增损失、门控或部署分支；不预判收益。

## 四条件、唯一变化和历史边界

旧NOAS30/AS30冻结读出复用；新LRAW_NOAS30/LRAW_AS30对应原B00/B01，各从自己的原P43 FINAL20学生及ownEMA追加10轮，总E30/6660updates。seed2026092701；父run20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01，发布commit19aedc3604fed9c193237dd92366164178af0983，stage20均为scratch20/4440updates/固定FINAL20。严格核对对应row来源和原数据角色，不使用NoU/weighted/MIX/目标选出的父模型。

唯一改变为L专用ordinary augmentor的返回值：保留调用前raw clone，原完整ordinary在另一个clone上照常执行一次，弃用增强输出，返回原raw。原计算和该call的RNG消耗保留，不能宣称普通增强计算已省掉或整个训练随机轨迹已经一致。只包装native唯一ordinary工厂的实际L128、可见标签和labels/no_pa签名；不改全局DataAugmentation、U weak教师或独立U strong。

返回的是pre-call raw，不是旧PA前general-channel prefix，也不是去PA但保留其它变换。它同时旁路DataAugmentation的随机变换及其归一化尾部；不再额外归一化、裁窗、循环移位或合成新IQ。该raw与原生labeled_inputs和既有raw128控制同源，模型内部预处理和必要补偿保持。原增强概率/范围/标签依赖/次序实际记录为“仍计算但输出弃用”，原used-input与raw的MSE应为0，另记discarded ordinary MSE和call数，不能把输出生成数量当实际施加了某个扰动的包数。

前20轮保留原ordinary，故不是全程NoAug。PAwaveform-only的当前匹配E30已完成；本轮明确旁路整条ordinary输出，不能把它换名当首次PA研究。原As本来就是complete practical(raw64)，不是practical(ordinary)，不宣称修复As重复噪声或双信道。

网络保持F三路SIG_TIME/统一Sinc/真实joint160/900722参数/206状态、原6×160 cosine scale30默认头，与骨干正常联合训练。每步6L/768NN、L总尺度1.25不变：.5RD(primary128)+.5RD(mixed128)+.25RD(raw128)+原.1CROSS_RX。新As的clean raw系数1.0、As系数.25；新NoAs全部raw系数1.25。RDrop两前向使用同一已生成IQ，不另抽增强。ORD128名称和128位置/原层位time_down,t2.pool/tau.2不改，但实际anchor输入变为raw128；没有追加anchors/前向/参数或新loss。RDrop不是IQ泛化增强。

原U/EMA/门控与弱强增强保持，包括ownEMA严格恢复、nativephase label0/pseudo10、use_unlabeled=True/use_ema_teacher=True、原lambda_u/ent；不是去U实验。ordinary取父20实际args冻结epoch13，SupCon累计+20，channel累计+20/+4440。原practical三scene IID、refSNR30、session硬件、quality/lock/tracking/AGC、噪声与补偿依赖完整保留，一包只处理一次。NoExplicitCFO保持，未增加PA/DAC/CFO标签保持假设。

## 数据与固定判定

源全RX[1,3,4,6,8]/day[1,2,3]、L6300/U56700 TX隐藏/V27000、split392005、eq1、IQ256保持，无RX留出或L5580/Ldev720。复用现有VALIDATED_ONCE载体和原source_contract.role_ids，不新增验证/签章/哈希链。V仅冻结验证、不更新模型。

固定FINAL10/累计30，不挑epoch。源晋级仅两条件合取：newAs fullV27000 clean−newNoAs≥−.1绝对pp；固定V1800两draw mid/low等权均值newAs−原As≥+.5pp。任一失败关闭固定LRaw，不扫开关、归一化、parent、seed或额外预算救跑；全部all7正式仍必须收尾。通过只具有后续总E50研究资格，不自动延长。主线程先取回并保存源决定，再读本次正式分数；正式结果不回流设计/选模/晋级/seed或选择性重跑。

每新row完成即自动固定源读出、clean＋六practical预测，独立exec CPU scorer仅在该row七视图预测固定后连接truth；不等待另一健康row，不因源科学失败跳过正式。不重新预测或评分旧两控制。固定正式每视图168000，targetRX[0,2,5,7,9,10,11]/day[0,1,2,3]，clean/practical_high/practical_mid/practical_low_suburban/practical_high_urban/practical_mid_urban/practical_low_urban；原receivedIQ/SNR/信道/难包全保留。

报告四条件all7 accuracy/macroF1/worstRX、RX/day/TX、修复/新增错误、urban三场均值、原As增量与匹配NoAs总效应、交互(新As−新NoAs)−(旧As−旧NoAs)、suburban绝对率及距自身clean差。正式阶段界限newAs−newNoAs clean≥−.1pp；另报新As对原As/原NoAs的clean，不靠降低clean缩小suburban差。2pp仅最初探索建议，不新增每cell门槛。n1/microgain/源健康不是可重复双提升。

## 执行、成本和停止规则

实际新增两条各E10/2220updates，共20model-epochs/4440updates；每模型累计E30/6660。父20与旧E30读出成本复用、分别报告。每新条L NN1704960、old U学生和教师各预期567000、source12600、formal1176002；native eval和所有计数按实际记录。As行142080As包，NoAs0；每行284160个discarded ordinary输出而非新模型视图。extraNN0不代表clone/原增强/物理信道/日志CPU开销0。首步两个真实父各2NN和两次真实6L合成1536NN，共1540NN/0更新/0realIQ/query；验证真实factory/raw返回、原U路径、RDsameIQ、默认头/Sinc/骨干梯度，PASS直接训练。

普通N607既有CVS-RFFI/PyTorch/fullFP32，每进程CPU2，无安装或管理员操作。CWD=PROJECT/releases/cvs_FcrossRX_LRaw_E30_20261018_r01；命令CVS-RFFI/python -u experiments/cvs_lraw_tail30/runner.py；不可覆盖输出PROJECT/runs/20261018-phase1-FcrossRX-LRaw-e30-s2026092701-m2-r01；唯一owner=codex/root/practical-LRaw。共享capacity queue每GPU全owner compute进程≤3且空闲≥12000MB，健康旧任务不动。

仅本run数据/父来源/数值/默认头/输出冲突/非法预测或系统技术失败停止新调度并保留产物，健康行可收尾，不自动重试；不以性能早停。一个普通preflight、一次release归档比较/远端compile、launcher首步smoke及一次PID/CWD/argv/GPU/log增长核实；一次直接P0/P1审查及必要问题定点复核，不加报告形式审批、seal或多层hash。

预期每row source checkpoint/completion/source_frozen/initialization/actualargs/ordinaryactual/EMA和optimizer说明、完整10epoch/2220step/原loss分解/实际L/U序列/包装输出记录、source、七view predictions/score/complete；0NN分析4logic/504CM/84分区、源决定/同row报告/成本/登记。数据/权重/原prediction/log/私有runtime不入Git。既有暴露基准背景保留，不声称新盲测或真实在轨验证。
