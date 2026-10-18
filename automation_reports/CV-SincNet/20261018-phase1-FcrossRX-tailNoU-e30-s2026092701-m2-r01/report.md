# 末段 NoU × 完整 practical：单种子 E30 最小比较

状态：PLANNED；已登记、实现中，未发布或训练。实际代码 commit 在冻结实现后填写，不将准备写成运行。

唯一假设：原 F＋CROSS_RX 的有标签完整 practical 已在前20轮建立适应后，末10轮移除原 U 伪标签训练，能否提高固定 practical low/mid urban，同时让 clean 对匹配 NoAs 的下降不超过0.1个绝对百分点。现有尾段 U 目标标量约占原As总目标0.481%；这不是梯度归因或识别收益，也不预判去U有效。

## 固定矩阵和来源

|逻辑条件|实际新增训练|父来源|末段 U|末段 As|
|---|---|---|---|---|
|NOAS30|0，复用冻结读出|原P43 ownNoAs20→原NoAs10|原U|无|
|AS30|0，复用冻结读出|原P43 ownAs20→原As10|原U|原完整As64/step|
|TAIL_NO_U_NOAS30|新增10轮/2220更新|原P43 ownNoAs20学生|0|无，selected为raw64|
|TAIL_NO_U_AS30|新增10轮/2220更新|原P43 ownAs20学生|0|原完整As(raw64)/step|

仅seed2026092701。原run 20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01，实际发布commit19aedc3604fed9c193237dd92366164178af0983；各自stage20为scratch20/4440updates/固定FINAL20。严格核对原source roles和继承来源；父student/ownEMA均核实，子只加载student，child EMA=None。旧NoAs30/As30只作冻结读出控制，不作新初始化。不加载weighted/MIX/target-selected来源。

这是tail-NoU，不是全程NoSSL：父20轮仍有原13L+7U，有标签L全程存在。新段10轮保持原native label_epochs=0/pseudo_epochs=10字段，但真实use_unlabeled=False/use_ema_teacher=False；无U iterator、U增强、教师或学生U前向、EMA构建/更新。原生a1_ema_successful_updates即使无EMA仍累计成功optimizer步，不误报为EMA更新数。U56700的物理角色和TX隐藏权限保持。

网络保持F三路SIG_TIME/统一Sinc/真实joint160、默认6×160 cosine scale30，随骨干联合训练，不新增部署分支/头或拟合诊断头。每步原6L/768样本前向、总L尺度1.25：.5 RD(ordinary128)+.5 RD(mixed128)+.25 RD(raw128)+原.1 ORD128 CROSS_RX；第二对比视图不重抽IQ。As行原mixed64ordinary+64completeAs(raw)，NoAs选中位置rawclone。time_down/t2.pool、tau.2、SupCon累计+20保持。

普通增强沿用各自原E20实际args并固定其epoch13；信道仍累计E21..30/step4441..6660。训练As为原三场IID、referenceSNR30、RX/day session硬件、quality/lock/tracking/AGC依赖与内部补偿保持。不叠As、不新增PA/DAC/CFO或噪声。NoExplicitCFO保持。

移除U会改变总前向及全局dropout/ordinary随机数轨迹；本实验衡量完整训练配方，不声称隔离纯伪标签loss、逐步IQ完全相同或成本匹配。新NoAs与As共享相同末段NoU规则和L前向预算；与旧两控制另报As×tailNoU交互，不能将其原有As收益或计算差全部归给新机制。

## 数据、冻结与放弃条件

全部原源RX[1,3,4,6,8]/day[1,2,3]，L6300/U56700 TX隐藏/V27000，split392005、dataset eq1、IQ256；不留出训练RX，不改为L5580/Ldev720。沿用现有VALIDATED_ONCE句柄，不新增数据重验/哈希链。V不训练或更新模型。

只固定FINAL10/累计30，不选epoch。源资格只有两条：新As full V27000 clean−新NoAs≥−0.1pp；固定V1800两draw的mid/low等权均值，新As−原AS30≥+0.5pp。两条同时通过才具有后续总E50研究资格；没有自动扩预算。任一失败关闭固定tail-NoU配方，不扫U权重、phase、父、seed或追加预算救跑。源决定在root首次读取新正式分数前保存。单种子源通过不等于跨RX正式通过或重复双提升。

所有新row训练后立即执行固定源读出、clean＋六practical全部预测，独立CPU truth-last scorer在该row七视图prediction完成后才接truth。科学负结果仍完成all7，旧两row只复用原prediction/score，不重跑。原固定168000/view、target RX[0,2,5,7,9,10,11]/day[0,1,2,3]、IQ/SNR/信道/难包均保持。全部七视图为clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。

正式阶段目标为新As−新NoAs clean≥−0.1绝对pp；同时报告与原NoAs和原As配对变化、全部accuracy/F1/worstRX/RX-day-TX、urban三场均值、错误修复/新增、suburban绝对率及距自身clean差。报告交互(新As−新NoAs)−(原As−原NoAs)。最初2pp suburban建议仅作探索性读出，不新增晋级硬门槛；不靠clean下降缩小差距。n1/microgain不宣称可重复双提升，正式结果不回流方案/晋级/seed/选择性重跑。

## 成本、命令和停止规则

新增两条各E10/2220updates，共20model-epochs/4440updates；每条累计E30/6660。两原E20父和旧两E30读出成本复用、分别报告。每新条L前向1704960，As条142080As包，NoAs0；新U学生/教师NN及EMA更新预期均0，原教师未构建。source12600NN/formal1176002NN另计；native eval按实际计数，不把关闭U当全部额外成本为0。首步真实两父各2NN及两6L合成1536NN，共1540NN/0更新/无真实IQ或query，PASS即继续训练。CPU物理信道、验证和完整wall另记。

代码/config身份为实际Git提交。环境为既有N607普通账户CVS-RFFI/PyTorch/fullFP32/每进程CPU2，无安装或管理员操作；CWD=PROJECT/releases/cvs_FcrossRX_tailNoU_E30_20261018_r01；命令=CVS-RFFI/python -u experiments/cvs_tail_no_u30/runner.py。不可覆盖输出根=PROJECT/runs/20261018-phase1-FcrossRX-tailNoU-e30-s2026092701-m2-r01。唯一owner=codex/root/practical-tailNoU，动态queue遵守每GPU全owner compute进程≤3且空闲≥12000MB，健康旧任务不动。

只在本run源数据/父来源/数值/默认头/输出冲突/非法预测或系统技术失败时按预登记停止新调度，保留产物及健康行，不自动重试；不以性能早停。一次普通preflight、一个release archive单次SHA比较、remote compile、首步smoke后PID/CWD/argv/GPU/log增长核实；只有一次直接P0/P1审查及必要问题定点复核，不增加报告审批/签章/额外哈希门槛。

预期每新row含source checkpoint/completion/source_frozen/initialization/resolved args/optimizer/teacher初始化说明、完整10epoch/2220step/原损失分解/L物理曝光/零U计数，固定source、全部7prediction及独立score/complete；0NN同row分析覆盖4logic/504CM/84partitions、成本/源决定/正式报告及登记。数据、权重、原始prediction/log与私有runtime不入Git。当前没有性能结果。

本地实现109项聚焦负例/19py+2remoteAST、分析4AST/10toy PASS；唯一合并审查P0=0/P1=0。真实1540NN烟测及远端发布尚未执行，无新增识别成绩。
