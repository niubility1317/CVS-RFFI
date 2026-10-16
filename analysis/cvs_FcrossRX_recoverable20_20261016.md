# 可恢复状态课程：先在源 RX6 检验 clean 代价能否消除

当前基线固定 F＋跨 RX 同 TX SupCon／joint160／默认cosine30联合训练，scratch E20=13L＋7U／4440更新／FINAL20。RXMIX已完整失败：源RX1 clean+.703704pp但六practical全降，不继续调幅度。当前唯一新假设是：把新增课程的困难部分限定为中等新增噪声且locked/degraded的urban完整观测，能否修复普通paired-As在RX6的clean退化，同时保持其urban收益。

旧单SIG_TIME E30课程的实际增量仅low+.619pp、urban+.336pp，不能预称新课程有效；其hard条件是quality_snr_db在3至18且lock!=locked，会纳入unlocked并排除locked。新F＋SupCon下paired-As的源证据以LOS/PARTIAL/locked收益为主，BLOCKED/unlocked改善较弱。新候选改变的是完整观测的抽样组织，不改变任何传播、噪声、质量适配、补偿、EQ或边界参数，不声称新物理规律或优于KL。

用有界混合 `p_t=(1-eta_t)*p_nominal+eta_t*Q4`，eta≤.5，nominal支持至少.5。nominal是原C1的三环境等先验完整residual，第一draw的IQ与C1逐值一致。Q4的精确定义：先产生该nominal；若hard分量被选中且它是mid/low urban、locked或degraded、实际signal_to_added_noise_db在3至15则接受；否则最多4个独立的原mid/low urban完整draw中取首个满足条件者；全部失败返回最初nominal，不能返回最后一个失败提案。有限fallback使Q4并非无限条件分布，报告实际接受／fallback／提案数。新增噪声比不是总SNR，接收状态保持原引擎原质量代理计算，不能用选择条件重算成想要的状态。

每个真实L物理样本仍产生一个卫星视图并参加完整CE，无学生置信度筛选。拒绝的是额外模拟提案，不删除原IQ、TX类别或困难测试样本；nominal分量保留全部场景和状态训练支持。硬件仍按RX/day/eq session跨TX稳定，提案每epoch/step刷新，使用独立命名私有种子20261016651，不推进全局RNG；每个提案都从同一原IQ独立进入完整practical，不串接信道、不经ordinary的PA/AWGN/多径。

首5轮eta=0。之后仅使用上一轮fit L的真TX×physicalRX×day原普通视图默认头统计与已有native梯度推进：总体accuracy≥.7、cell accuracy的p10≥.6、cell margin/30的p10≥.02、CE≤1、平均梯度≤5且≤前3轮中位数的2倍；CE≤前3轮中位数×1.05、p10margin不低于前3轮中位数−.01。满足时下轮eta增加.1至.5，否则保持。缺cell不推进；source V、held V、U真值及target不参与课程，监测不新增网络前向、损失或诊断头。所有阈值在实际C2训练前固定，不据结果修改。

模型、默认头、原U／ownEMA／旧门控／cos／SupCon不改。仍是.5RD共同oldordinary＋.5RD原IQ的selected practical，LS=.01、RDrop alpha1；每对同IQ两dropout，四L前向、一次ordinary、原L/U物理采样及总损失尺度匹配C0/C1。SupCon只读第一对已有前向；不做跨信道全特征对齐，不加入间隔／SSL损失或新的Ag。

只选一个已失败源折RX6作首轮证伪，明确是因既有source失败而选择，不是新盲测或挑最好折。fit RX1/3/4/8，L5040／隐藏U45360／seenV21600；heldRX6 L/U不训练，heldV5400仅FINAL20的clean＋六原practical。C0/C1均复用原同合同同seed E20结果／metadata，不加载其checkpoint。C0 clean81.129630%，C1 clean77.481481%、low47.796296%；新模型从零，不继承E50/E200或旧EMA。

固定判据全部满足：C2相对C0的held clean及seen V各下降≤.5pp；urban均值、low及urban macroF1均不低于C1；课程实际启用；完整健康E20和TX×源RX×三场景／状态覆盖。即clean至少恢复至80.629630%，同时不能牺牲C1的urban识别。FAIL关闭固定课程，不调eta／范围／warm／提案数／比例、换RX/seed或阈值救援。PASS仅支持另登记其余源折确认，不自动扩target、多seed或Ag混合；最终clean正收益／low>60%及suburban接近自身clean仍须真正联合模型证明。

新增预算1个scratch E20／4440更新，预期3579018网络包（以实际重算为准），C0/C1既有40模型轮分列，父权重0。提案数与物理模拟／trace耗时额外实测，不把相同网络前向当作相同墙钟。训练后自动完成七源视图，固定fs25MHz／IQ256／residual／post_sync／EQoff G1／reference30／quality adapt／phase tracking／IQcomp／AGC／reflect；正式received IQ和信道不变，0target。一个owner／不可变root，遵守所有owner每GPU最多3实验进程；技术错误保留产物不自动重试，不动健康任务。

精确row／commit／命令／环境／CWD／路径／GPU／stop与artifact见experiment.json。必要验证仅聚焦负例和原网络合成前向、一次限定P0/P1、单archive比较／remotecompile、worker首步own scratchCP no-query smoke及独立运行读回。固定E20结束不代表收敛，一个源折／一个modelseed不代表跨seed复现。

实际运行：代码`a47c7302eefc485e75703a78c749f24144e5b77b`已发布，唯一C2-RX6 worker ownCP no-query smoke及首轮真实PID/CWD/argv/GPU/log增长、E20/13L/7U、Fclip/4L/defaulthead联合更新VERIFIED。首轮eta0／28416 nominal输入／全部72fitL单元，0target。独立读取器24负例和512合成记录PASS；尚无FINAL，不宣称clean修复或urban提升。

可恢复状态课程实际完成：20261016-phase1-FcrossRX-recoverable-source-RX6-e20-s2026092701-m1-r01，E20/七源视图及独立完整回放，固定源规则FAIL；相对C0 clean -3.9259259259259216pp，相对C1 urban 0.4629629629629613pp、low 0.05555555555555314pp。新增20模型轮/4440/3579018网络包，矩阵2892.487446784973s。0target，单modelseed/已暴露源RX6；关闭固定课程，不扫eta/范围/warm/proposals/RX/seed救援。 联合与跨seed目标未完成。[同run报告](../automation_reports/CV-SincNet/20261016-phase1-FcrossRX-recoverable-source-RX6-e20-s2026092701-m1-r01/report.md)。
