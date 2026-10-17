## 2026-10-18：保留 256 点；幅谱 donor 草案不启动

用户确认没有长 IQ 映射，继续现有256点、F＋跨RX同TX、E30与全部五源RX。已核对原文实际可读范围和旧同身份插值／PSD仿射／FIR负结果；保留相位的同TX跨RX幅谱借用不构成已支持的独立新路线，不生成IQ、不新增源预测或训练，不改旧停止条件。本轮新增IQ/CP/NN/更新/teacher/N607作业均0。最近真实完成仍为批次组成三行E30，固定方案小增量但未达预设条件并关闭；完整联合目标ACTIVE，当前无可发布新Ag候选。[核查结论与证据边界](cvs_amplitude_donor_route_check_20261018.md)。

## 2026-10-18：批次组成 E30 完整评分，小幅改善但固定条件失败

三行同父 E20＋10、全部5源RX/L6300/U56700隐藏TX/V27000，均已完成E30、3528000正式决策及独立truth-last。重组clean82.954167%、midurban64.629762%、lowurban53.631548%；相对原顺序+.201786／+.247024／+.190476pp，相对随机+.176190／+.157738／+.188690pp。七视图accuracy/F1均正增量，但low未达预定+.5pp，固定6池方案关闭，不据target扩seed或改pool/父/预算救跑。三个suburban距clean1.013095／3.330357／5.754762pp；部分RX/TX退步，非联合成功。全部370输入池/2220U序列及857参数匹配，378CM/63分区/21配对复算通过，新增NN11311934；没有新IQ A_g或额外NN/loss。全部worker/dispatcher自然结束。当前无已支持新Ag候选，完整联合目标ACTIVE。[完整报告与曲线](../automation_reports/CV-SincNet/20261018-phase1-F-crossRX-batchStratify-e20plus10-s2026092701-m3-r01/report.md)。

## 2026-10-18：完整 practical 的 L 批次组成 E30 三行候选

已发布并核实三行运行，新增首轮完成（累计 E21），真实 CP smoke／进程／GPU／日志增长通过；训练与正式评分尚未完成。原4440步568320 L元数据回放发现所有锚点都有正例，TX×RX组成有波动；6批池重组保持逐物理曝光，平方偏差4.029→.772，随机重组4.054。这不是识别收益。新三行同父ordinaryAs E20＋10至E30，仅L批次组成不同，原F/defaulthead/As/U/4L前向及5源RX不变；无新增IQ A_g或loss，不与关联对话教师U路线重复。若不能分别相对原顺序/随机重组同时保护clean、midurban并使lowurban至少+.5pp，则关闭固定方案。所有正式结果固定报告，不据target晋级/扩种子或救跑。[预登记](../automation_reports/CV-SincNet/20261018-phase1-F-crossRX-batchStratify-e20plus10-s2026092701-m3-r01/report.md)。

## 2026-10-18：有限窗酉相位源保护失败，固定候选关闭

真实检查已完成：全部 5 RX／V1800，2 原默认头 × raw/同一 ordinary × 0／±.1，共21604 NN、0训练／teacher／target。8组端点准确率下降1.00至2.39个百分点，32项身份条件均失败，12项数值条件通过；1260CM／8配对／44判据独立复算一致。不能把保能量或可逆性当身份保护，按固定反证关闭，不缩幅度或换seed救跑，无自动训练。当前没有可晋级新候选；E30、F＋跨RX同TX、256点与不留出训练RX保持。[完整报告](../automation_reports/CV-SincNet/20261018-diagnostic-F-unitaryDispersion-sourceV-s2026092701-m1-r01/report.md)。

## 2026-10-18：256 点有限窗酉相位观测待源域检验

本轮只检查源 V 身份保护，不训练、不访问 target。两个原完整源 E20 默认头、原 V 全 5 RX 的固定 1800 样本，raw 与同一份 ordinary 比较 0／±0.1 rad。合成检查已通过，ordinary 的 Python／Torch RNG 同时固定并恢复；一次 P0/P1 定点修复已通过。不读取窗外 IQ，不留出训练 RX。任一固定身份保护条件失败即关闭；通过也只说明必要保护，不能声称 A_g 或联合目标有效。

[登记报告](../automation_reports/CV-SincNet/20261018-diagnostic-F-unitaryDispersion-sourceV-s2026092701-m1-r01/report.md)；E30 授权与 F＋跨 RX 同 TX 基线保持。

# 联合增强活动基线：F＋跨 RX 同 TX 对比学习

最新完成：[ordinary／As监督份额只读核对](cvs_clean_supervision_mass_audit_20261018.md)，ANALYZED_CODE_AND_CLOSED_SOURCE_LOGS_ONLY。FULL_AS从REPEAT的ordinary总系数1改为ordinary.5／As.5，B01为ordinary.75／As.25；这证实重分配，不证明clean代价的唯一原因。原生CE被hook correction抵消后只加一次，native loss_labeled仍是首ordinary CE而非完整L目标；复用两行各2220完整源步记录核对。实际E13 scale.2506003、PA概率.035084而非base.14、shift最大8而非32；旧ordinary不是未增强x，也不是PA身份保持证据。旧additiveAs固定seed02在旧4RX折clean失败，关闭不救跑；另一对话已完成全源R25及V1800×2draw，不在本对话重复，不混合其scratch30与本E20＋10结果。本次0新IQ／CP／NN／teacher／更新／训练，无新Ag或联合成功，完整目标ACTIVE，当前无获新源依据支持的训练候选。以下旧当前／下一项保留为历史。

最新完成：[跨ordinary／As dropout相关性E30](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-dropoutCoupling-e20plus10-s2026092701-m2-r01/report.md)，ANALYZED／VERIFIED，固定配方失败并关闭。两行同全源普通As父E20＋10，各累计E30／6660更新，全部2352000正式决策后独立评分；252CM、42分区、4440完整L/U序列和每行2220目标／掩码记录核对一致。反射掩码clean83.020238%、mid urban64.392857%、low urban53.414286%，相对独立控制+.022024／−.067857／−.056548pp；六practical准确率全部下降，low修复855／新增950，clean与low同时正确净少140。suburban距自身clean1.298214／3.573214／5.972024pp；旧同预算B00仅作背景，clean净−2.248214pp，low<60，完整目标ACTIVE。本项仅训练组织，不是Ag；不换mask key/p/seed/父/预算救跑、不扩种子。全部5源RX/L6300/U56700隐藏TX/V27000、256点IQ及默认cos头联合训练保持，无训练域留出。当前无已获证据支持的新训练候选；下一项仅只读核对已有common/extra视图是否改变原始clean监督份额，不从本次目标分数选择新系数。以下旧运行／当前均为历史。

当前唯一固定比较已发布运行（RUNNING／VERIFIED）：[跨 ordinary／As dropout 相关性 E30](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-dropoutCoupling-e20plus10-s2026092701-m2-r01/report.md)。两行同原始全源As父E20＋10、E30各自封顶，仅改变共有身份dropout的跨视图相关性，保留两对RDrop和原As/U。属于训练组织，不是新Ag；执行commit a56e02cada3f11db002115afedafb4c024151348；两行已完成追加第1轮／累计E21／各222新更新，单archive/compile/真实CP无query smoke及PID-CWD-argv-GPU-loggrowth均通过，尚未完成训练或正式评分。旧失败路线不改判；全部5源RX、L6300/U56700隐藏TX/V27000、256点IQ、默认cos头联合训练保留，无源训练域留出。当前状态以此run实际证据为准，以下均为历史。

最新完成：[完整接收先验虚拟会话覆盖 E30](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-RXpopulation-e20plus10-s2026092701-m1-r01/report.md)，ANALYZED／VERIFIED，固定配方关闭。全部5源RX/L6300/U56700隐藏TX/V27000保持，同一个普通As父E20＋10至累计E30／6660更新，七视图1176000正式决策固定后独立评分。150虚拟会话全部生效，clean82.923214%、mid urban64.364286%、low urban53.422024%；相对普通As−.040476／−.007143／−.025000pp，七视图准确率均降，预定low+.5、clean/mid保护失败。三个suburban差1.314881／3.594643／5.963095pp，略缩伴随clean下降，不算联合成功；clean相对B00净−2.345238pp。252CM／42分区／七视图配对、4440L/U序列、857实际参数和2220步目标独立核对，low修复331/新增373，clean与low联合正确净少62。0新头／loss／U／额外NN；不换key/间隔/先验/seed/父/预算救跑。用户确认没有可靠更长上下文映射，后续仅现有256点IQ；E30上限与无训练RX留出继续有效。当前无获合法源证据支持的新训练候选，完整联合目标ACTIVE；以下旧运行/下一项均为历史。

最新完成：[原噪声分布正负配对源检验](../automation_reports/CV-SincNet/20261017-diagnostic-F-antitheticNoise-sourceL-s2026092701-m1-r02/report.md)，ANALYZED／VERIFIED_SOURCE_ONLY；固定 coupling 未过预定条件，关闭，不进入 E30 训练或扩 seed。整体原坐标梯度方差比0.888305、默认W比0.951560，均未达≤0.8；mid/low比0.922326／0.855713，仅局部源估计器变化，不是身份收益。全部5源RX／180合法L、25922NN／432autograd／12960完整物理处理／0更新／0target；18组Gram独立复算误差≤8.89e−16，模型状态不变。r01转换技术失败保留，额外2NN／90物理处理单列；r02同科学配置自然结束。用户E30上限、全源L6300/U56700隐藏TX/V27000和无训练RX留出保持。当前无获证据支持的新训练候选，完整联合目标ACTIVE。以下旧“当前／运行／下一项”均为历史。

最新完成：[固定原 FIR 首次训练效用 E30](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01/report.md)，ANALYZED／VERIFIED；本固定训练配方失败并关闭。全部五源 RX、L6300／U56700 隐藏 TX／V27000 无留出，两新增行各匹配 E20＋10、累计 E30／6660更新，七视图全部固定后独立评分。B10 clean−B00=−0.039286 pp；B11 clean85.466667%、mid urban62.910714%、low urban51.547619%，相对 B01 分别−0.055357／−0.020238／−0.028571 pp；七视图交互均负。suburban 距自身 clean4.122619／6.905952／9.517262 pp，微缩伴随 clean下降，不能算联合成功。630CM／105分区／全部配对计数、857实际参数／4440条L/U序列／2220步监督尺度／142080 FIR选择每行均独立核对。旧有限窗PSD保真失败不改判；不改核／幅度／父／seed／预算救跑，不扩种子或构造组合视图。当前无获源证据支持的新训练候选，完整联合目标仍 ACTIVE；本段以下“运行／下一项”均是历史。

最新唯一候选：固定原 12 个有界平滑 FIR 的首次训练效用，已发布运行，执行 commit `97f260f469a47a000eed8ab84e8507a3cd0ca37d`、传输／编译／源输入身份保护／PID-CWD-argv-CUDA-GPU 均已核实。见[完整源 E30 两新增行](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01/report.md)。旧有限窗 PSD 保真失败保持关闭，不调整核／幅度／边界／阈值；本次不继承旧四 RX 模型，全部五 RX／L6300／U56700 隐藏 TX／V27000 保持，无训练域留出。B10/B11 分别从匹配 nonAg B00/B01 全源 E20 父＋自身 EMA 追加 10 轮，累计 E30；新增 Ag 仅在 E21 至 E30，原 As 与 U 保持。复用匹配 E30 B00/B01，四 L 前向与监督尺度相同；没有组合视图或新损失。源输入身份保护通过后同 dispatcher 自动训练与七视图 truth-last，不将源通过记为 Ag 有效。固定识别条件失败即关闭配方，不按目标调核／换父／扩 seed／续预算。完整联合目标 ACTIVE，两条行已开始训练，当前尚无终轮或评分结果。

以下为已完成结果与历史过程，当前入口以本段和当前 run 独立证据为准。

最新状态（E30、全部5源RX，无留训练域）：[MMSE混合E30](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-MMSEmix-e20plus10-s2026092701-m1-r01/report.md) 已 ANALYZED／VERIFIED，六 practical 均下降，固定方案关闭。下一唯一源反证为[等平均 N0 变化](cvs_N0_prior_source_plan_20261017.md)，ANALYZED／VERIFIED_SOURCE_ONLY，固定范围未过 mid＋0.5 pp 条件，关闭；100802NN／0更新／0target，不是新 A_g 或联合证明。以下较旧“运行／当前”语句保留为历史，以这一段及实际 run 状态为准。

新唯一 A_s 假设已实际 RUNNING／VERIFIED：[20261017-phase1-F-crossRX-PDPshape-e20plus10-s2026092701-m1-r01](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-PDPshape-e20plus10-s2026092701-m1-r01/report.md)。在原 total-ensemble RMS/径功率及完整依赖下，等概率使用 [0,1,3]／[0,1,2] 时延形状；不是新 Ag、RDrop 修改或弱化信道。原全源 L6300/U56700隐藏TX/V27000／5RX、同父E20＋10至E30、4L前向和原U保持，复用严格匹配nominal E30。执行476f350914ca72c753365c1b25d9deebf4920701；定向检查／一次P0P1与修复复核／真实CP smoke／单archive/compile/PID-CWD-argv-CUDA-GPU-loggrowth均PASS，原worker已完整追加E1／累计E21／113664LNN／28416As、当前298更新，无终轮评分。固定low≥+.5pp、clean/mid不降、每sub≥−.25pp；失败关闭固定配方，不调形状/比例/seed/预算救跑；全部7视图truth-last，完整目标ACTIVE。

最新预算保持 **E30**；全源五 RX／L6300／U56700 隐藏 TX／V27000，无训练域留出。公开增强实现的后续核查见[文献适用边界](cvs_receiver_augmentation_literature_limits_20261016.md)：AA-FAL 当前代码的 1200 点局部替换不能原样用于 256 点 IQ，I/Q 分别修改的身份保持未验证；ASA 操作中间特征。本轮未执行第三方代码，0 真实 IQ／CP／NN／更新，没有新候选晋级或实验启动。DC 旧负结果保持关闭，完整目标 ACTIVE。

后续入口核查已完成：[共享传播与batch轴核查](cvs_as_route_audit_20261017.md)。共享传播H机制已在旧PROP_GROUP按原条件失败关闭，改变组键不作为新方向，本轮撤回且0新训练。实际F随机初始化合成检查12NN／0CP／0真实IQ／0更新：活动GroupNorm10、LayerNorm4、BatchNorm0、持久buffer更新0，eval合并/拆分默认头最大差1.43e−6。否定顺序前向污染BN状态的具体解释，不宣称整个SupCon训练batch无关或性能提升；未发布新候选，整体goal ACTIVE。

最新完成：[原分布校准刷新 A_s E30](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-calibration-refresh-e20plus10-s2026092701-m1-r01/report.md)，**ANALYZED／VERIFIED；固定收益假设失败，关闭该配方**。严格同一个普通As source-final20＋10，累计30轮／6660更新，全部5源RX／L6300／U56700隐藏TX／V27000保持，无训练域留出。默认cos头联合训练／F／三路SIG_TIME／统一Sinc／原U不变。

全部七视图1176000正式决策固定后独立CPU truth-last，252个CM／42个分区／七视图配对重计一致。候选clean82.909524%、mid urban64.361310%、low urban53.442857%；相对普通As为−.054167／−.010119／−.004167 pp，低urban增益和clean/mid保护条件均失败。low修复381／新增388，clean修复127／新增218；联合clean＋low仅净多5，不能替代绝对识别条件。

三个suburban距自身clean1.293452／3.576190／5.935714 pp，差略缩小伴随clean下降；clean相对同预算B00净−2.358929 pp。low未超过60%，新Ag或完整联合目标未证明，研究仍ACTIVE；当前没有运行或晋级的新训练候选。

15组true接收硬件不变、150组epoch-session校准估计全部实际生效。两行4440源序列逐条相同，68实际参数一致，2220监督目标重算最大差1.19e−7。新增10modelE／2220更新／3770646NN，0额外物理滤波或完整信道预跑，旧nominal源eval瞬态计数缺失保留。仅证实覆盖变化，不是身份收益；不调校准尺度／间隔／seed／父／预算救跑，不自动扩seed。后续假设仍须由独立合法源证据提出，不能据目标差值选参数或候选。

此前完成：[有限窗接收质量 A_s E30](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-practical-power-e20plus10-s2026092701-m1-r01/report.md) 为 **ANALYZED／VERIFIED，固定识别假设失败并关闭**。同一个普通 As source-final20追加10轮，累计30轮／6660更新；复用精确同父nominal E30，不重训或重测旧控制。全部源RX1/3/4/6/8、day1/2/3、L6300／U56700隐藏TX／V27000、split392005保留，无训练域留出。F三路SIG_TIME／统一Sinc／joint160／默认cos头联合训练与原U不变。

七视图1176000个正式决策全部固定后独立CPU truth-last评分，252个CM、全部RX／TX／day与配对计数重计一致。新方法clean82.979167%、mid urban64.372619%、low urban53.417262%；相对普通As的差为+0.015476／+0.001190／−0.029762 pp。low修复380／新增430错；clean与low同时正确净少59。low增量未达到预定+0.5 pp，其他保护条件通过仍不晋级，不调参数／seed／预算救跑。

三个suburban距自身clean1.369048／3.657143／5.989286 pp，仅high达到首轮2 pp探索目标；相对同预算B00 clean85.268452%，新方法clean净−2.289286 pp。low未超过60%，Ag与完整联合目标仍未证明，研究保持ACTIVE。有限窗及本次校准候选均已关闭；完整联合目标保持ACTIVE。

[源物理18900配对](../automation_reports/CV-SincNet/20261017-diagnostic-F-practical-quality-fullL-manysig-m1-r01/report.md)仍只支持处理差异：1010有效处理变化／355锁定变化；实际训练284160 As中15358有效变化／5332锁定变化。物理差异不等于身份收益，本次已完成直接检验并得到负结果。新增仅10modelE／2220更新，总新增NN3770646，额外纯物理预滤波284160次、完整信道预跑0；旧nominal完整源eval瞬态计数缺失如实保留。

此前[完整E30四因子](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01/report.md)也已ANALYZED：B00 clean85.268452%；B10 clean85.338690%，Ag独立仅+0.070238 pp；B01 clean85.522024%／low51.576190%；B11 clean85.174405%／low51.526786%；FULL_AS clean82.963690%／low53.447024%。七视图交互均负，固定时钟Ag联合配方关闭。保留原训练PARTIAL和零更新评分恢复记录，不调ppm、换父或续预算救跑。

[完整源普通拼接E20参照](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02/report.md)和[完整源E20四因子](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-clock-factorial-e20-s2026092701-m3-r01/report.md)全部保留。A_s普通训练已有实际适应收益但付出clean代价；没有获得可联合保留各项收益的新Ag。下一项必须来自合法源域或传播／接收处理证据，不使用已读目标分数选择幅度、候选、父、seed或重跑。

以下是按时间保留的历史过程，旧的“当前／运行／下一项”表述以本页开头的最新状态为准。

## 当前状态（2026-10-17，按用户最新纠偏）

主线恢复为直接验证星地识别收益。后续新训练使用完整L6300／U56700／V27000、全部源RX1/3/4/6/8，不再留出一个源训练域，也不要求先通过源留出折。F＋跨RX同TX、scratch E20、默认头、原U和正式IQ保持。

下一项是建立完整源的普通拼接参照：NATIVE原底座、REPEAT同ordinary额外前向控制、CONCAT普通源/practical等权配对，均固定final20并自动clean＋六practical、独立truth-last评分。三行已实际发布运行 `20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02`，执行commit `b94076b191648103daeb8a9cb1c17bdcc9c0e231`，全源数据角色和PID/CWD/argv/GPU/日志增长VERIFIED；最近轮端点为NATIVE/REPEAT E3、CONCAT E1。尚未完成训练、七视图预测或评分，不把已有源统计负结果当作星地性能。见[研究纠偏与最小矩阵](cvs_full_source_practical_correction_20261017.md)。

历史源留出及统计实验原样保留、准确标注契约；新方案必须直接比较同完整源普通拼接参照。A_g不是放弃，而是不再以其统计入口阻塞A_s直接性能；完整联合目标仍ACTIVE，尚未达成。

## 历史记录与固定底座


当前没有获得跨RX泛化证明的新增Ag。固定轻度频偏候选两组scratchE20/七源视图已完整独立分析，未通过源门槛，固定配方关闭，不改幅度/seed/RX救跑。clean差+0.000000pp，lowurban差-0.079365pp。真实CP输入作用PASS只属于工程证据。原ordinary含PA、默认头、原U保持；尚无target/联合成功结果。下一唯一拟审计旧ordinary循环移位的同draw身份影响：仅fitL5040冻结CP、13个真实强度配置/full与shift旁路、136082NN/0更新；净修复<2pp或不足3/4源RX改善则关闭。尚未实现、登记或启动，不把普通增强修正冒称Ag。
用户于 2026-10-16 指定，增强前参照更换为 **F＋跨 RX 同 TX 对比学习**。该选择覆盖本对话旧的 F-only 参照；训练预算仍为 E20，没有延长到 E50 的新授权。唯一活动配方见 [配置](../configs/cvs_joint_augmentation_F_crossrx_e20.json)。两种子源 B00/B01 已完整分析；独立seed02因clean下降13.095238pp未通过，固定75/25候选关闭。正式完整源四因子尚未训练。

已核实的来源是 `20261016-phase1-F-supcon-e50-clean-manysig-m9-r02` 的 `CROSS_RX`，执行 commit `4df7b94d0fd8ce29056bd8236364e45afa92e07f`。已读取完整报告、最终分析、实际 design/model/mechanism/source/observer，确认实际原生 extra 格式为 `(domain, metadata)`；不复用 r01 的错误读取方式。

保留 F 三路 SIG_TIME、统一完整 Sinc 修正、真实 joint160、默认 cosine 分类头联合训练、RDrop alpha1、LS0.01、已有普通增强和原隐藏 U 规则。浅中层 `time_down`、`t2.pool` 的同 TX 异 RX 正对只作用于原 L 的两次既有 RDrop 前向；同 TX 同 RX 不作负对，不增加网络前向或部署参数。温度 0.2，最大权重 0.1；E1 至 E5 为零、E6 至 E15 线性启用，E16 至 E20 保持 0.1。不把对比学习或 RDrop 称为 IQ 泛化增强 A_g。

E20 是从零训练 13 轮 L＋7 轮原 U，4440 次实际更新；预期对比调用 4440 次、非零权重 3330 次、层位／前向目标 13320 项。所有匹配候选用同 seed 初始化规则和同真实物理数据契约。原 E50 权重不用于新 E20 或内部开发折初始化。

正式 B00/B10/B01/B11 均包含该基线。B00 的“无新增 A_g”仍保留四组共有的旧普通增强，不称为无增强。对比目标保持在共有原 L RDrop 对，不因卫星视图或新增 A_g 隐式扩展到其他视图或 U；所有额外视图均在发布前固定损失尺度和对应前向控制。

正式完整源矩阵使用 L6300/U56700/V27000。若先做 RX1 源留出证伪，则单独登记 L5040/U45360 训练、RX1 原 L1260 检验及已见 V21600 健康诊断；该契约不能冒充 L5580/Ldev720 或完整 L6300。旧 F-only 的 RX1 B00 不作为新 CROSS_RX 的匹配参照，各 cell 从零建立，不继承看过开发包的父权重。

历史 E50 的目标 clean 已暴露，只能说明用户指定配方的来源；不在此据其目标分数重选温度、权重、seed 或启用日程，不宣称新独立盲测。E20 clean 和六 practical 的实际收益都待冻结后的独立评分。正式预测、received IQ、测试信道和真值隔离保持原协议。

以下按实施顺序保留历史候选与当时状态，当前状态以上述开头和末尾已完成结果为准。

本轮包内缓变增益源检验已完成，固定平滑斜坡假设失败：RX 风险降低 63.164130%，15/18 折改善，但包内线性 R² 中位数只有 0.303317，未达到预定 0.8。源统计不依赖模型权重，结果保留；未开展增益增强训练，不调整阈值救跑。

当前唯一候选是 A_s 的监督份额组织：75% 原 L 监督、25% 完整 practical。已实现并登记 `20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01`，B00/B01 均为新的 F＋CROSS_RX/E20，旧 F-only B00 不复用。每组 raw.5＋共有旧普通增强.25＋S槽.25；S 槽为 raw(B00) 或完整 practical(raw)(B01)，共有对比只作用于旧普通增强对。固定 6 次 L 前向和真实 raw.5 份额，防止普通 AWGN/multipath 在 practical 前重复叠加。

最小实验为既定 RX1 源留出契约、两组同 seed 从零 20 轮。源阈值已固定：low urban≥+5pp、urban均值≥+3pp；clean、各 suburban、seen V 均≥−0.5pp，完整有限预算及最终 LCE≤0.5。失败关闭此比例；目标分数不参与。实现、定向检查和一次 P0/P1 审查已通过，实际远端状态以该 run 产物／报告为准。该源证伪不等于正式全源四因子结果，也不是联合目标完成。

联合目标仍完整保留：A_g 跨 RX clean 净收益、A_s practical 适应、二者联合保留各自收益，low urban >60%，suburban 各场景接近本模型 clean（建议探索差距 2 个百分点，未冒称已达到）。目前 A_g 无通过候选；A_s 固定 75/25 方案仅通过一次源配对，联合目标未完成。


源 run 已自然完成两组 E20 和 17640 个七视图默认头决策，独立完整重算 PASS。B00/B01 clean81.666667%→84.761905%，low urban28.571429%→36.111111%，urban均值+13.095238pp；所有原科学条件通过。两组训练物理序列相同，每模型实际NN前向4686678，合计含公共smoke2为9373358。没有目标接触。suburban gap仍22.222222/26.746032/28.809524pp，TX2/TX3在全部卫星场景均下降，不能宣称普遍身份保持。详见该run最终报告与source_analysis.json。

下一唯一假设是该固定组织可在独立seed复现。最小确认拟为seed2026092702同RX1、同B00/B01各从零E20、原阈值和全部数据/视图/前向控制不变；尚未登记或启动。独立确认不通过原联合源条件即不晋级，不通过换seed、RX、比例救跑。正式矩阵仍缺Ag、B10/B11及交互项，当前不启动target。


独立seed02确认已登记为 `20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01`，状态PLANNED，尚未启动。固定runtime profile只改模型seed/run/release/output，底层F与原生seed均为2026092702；split/channel/receiver/eval、六前向、损失/阈值/U不变。两个模型各从零E20。10个负例与种子传递检查PASS，新入口一次限定P0/P1审查PASS；不重复审旧机制。源确认失败不扩seed03、不改RX或比例救跑，联合目标仍未完成。


上述seed02确认已实际发布，执行commit `d71531d53c1d9ec46e3915963e05bff96760e80e`；两组native/config种子均为2026092702，真实smoke、archive单次比对、编译及PID/CWD/argv/GPU/log增长VERIFIED。当前RUNNING，B00 E2/20、B01 E1/20是已观察进度，尚未评分或确认收益。不重新提交同row。


## 当前结论：75/25源确认失败并关闭

seed02两行均自然完成E20/4440更新与全部七视图，独立重算VERIFIED。clean83.095238%→70.000000%（−13.095238pp），lowurban28.492063%→36.031746%（+7.539683pp），urban均值+11.825397pp。原clean保持条件失败，其余条件通过；不追加seed03、换RX/比例或延长预算。两个seed的clean差分别+3.095238/−13.095238pp，均值−5.000000pp；urban均值差+12.460317±0.897913pp，lowurban差+7.539683pp。As适应净收益被重复观察到，稳定clean保持没有成立。seed02差距缩小部分来自clean下跌，不能记为联合成功。四模型累计80模型轮/17760更新/18746716实际NN包前向，全部结果与CM、代价保留。

下一唯一源假设为当前F+CROSS_RX对有界共同相位变化仍有默认头识别缺口。最小拟用自有B00/seed01/final20冻结模型，在fitL5040上做identity/+0.37rad/−0.37rad共15120决策、0更新；不读heldRX1、U、V、target或训练新头。两个方向均需至少2pp整体下降，且各自至少3/4 fitRX下降1pp，才支持另行考虑B00/B10；否则关闭相位方向，不扫描幅度。仅为待实现/登记/启动假设，敏感性不等于Ag有效。历史CNN相位和旧E200诊断不被当作当前F成绩。


当前相位源检验已实现并登记为 `20261016-diagnostic-F-crossRX-commonphase-sourceL-s2026092701-m1-r01`。三固定视图、fit L5040、15122包前向（含合成smoke2）、0更新/0头拟合/0教师/0As；10个负例和一次P0/P1审查PASS。尚未启动或取得识别结果，原正负双向阈值与失败关闭规则不变。


公共相位冻结源检验现已完整完成并独立复算：fitL identity99.841270%，+0.37rad75.337302%（−24.503968pp），−0.37rad63.015873%（−36.825397pp），各4/4源RX超过1pp下降。0更新/0头拟合/0教师/0As，15122实际NN包前向，功率误差≤2.981e−7，全部15120决策/logits/216视图单元核对通过。真实smoke/发布/进程/日志/自然退出VERIFIED。只证实当前F+CROSS_RX的公共相位脆弱性；未证明Ag有效。下一唯一拟检验独立相位L视图的匹配B00/B10 scratchE20，仍须单独登记，尚未训练，不访问target。


独立相位训练候选现已实现/登记 `20261016-phase1-F-crossRX-commonphase-sourceRX1-e20-s2026092701-m2-r01`，匹配B00/B10各scratchE20/同六L前向/.5raw+.25ordinary+.25槽；B10槽为每包uniform[−.37,.37]rad私有随机流。CROSS_RX仅共有ordinary对，U/ownEMA不变，0As训练。11负例与实际F合成配对控制检查PASS、一次限定P0/P1审查PASS。源clean≥+1pp/F1不降/seenV最多−.5pp，失败关闭固定配方无angle/seed/RX救跑；尚未启动或取得训练结果。


上述公共相位B00/B10已实际发布并运行，执行commit `39d95f79d91ccf2a570f1c1349adacfb409c0833`；archive/compile/真实CPsmoke/两worker PID-CWD-argv-GPU及日志增长VERIFIED。实际seed2026092701、20/13/7和无训练父权重均核对；当前未完成训练/源预测/独立评分，不能称Ag有效。


相位E20源run现已ANALYZED/VERIFIED：两行完整20epoch/4440更新及七视图，全部17640决策/完整step/相位角/物理序列独立核对。clean差-3.9682539682539653pp，clean F1差-4.4209738408452885pp，seenV差-0.07407407407407085pp，原源条件FAIL，固定相位配方关闭，无角度/seed/RX救跑。仅源证据，尚未证明Ag目标泛化或联合目标；实际总NN包前向9373358。


频谱IQ可实现性候选已实现并登记 `20261016-diagnostic-F-crossRX-spectralIQ-fitL-s2026092701-m1-r01`：fit4RX L5040重估固定rectangular/DCT1to8，33tap/.5dB、6RX对双向与匹配control共25视图；126002NN包前向/0更新，不读held/U/V/target，不复用旧fullL系数。9个算子/权限负例和5个门槛负例PASS、一次独立P0/P1审查PASS。原阈值失败即关闭固定滤波，尚未启动或取得真实源身份结果。


频谱IQ零更新候选已完整ANALYZED/VERIFIED：18折MSE比0.577122、18/18优于零，源身份保护/核响应/能量对照通过，但有限PSD相对RMSE最高0.601925超原0.35，固定滤波关闭。全部126000决策独立复算，126002NN/0更新，未用held/U/V/target。下一唯一拟诊断旧ordinary的PA-normal实际触发是否导致身份错误：同原生draw对比full/旁路PA/raw，尚未实现/登记/启动，不把源健康当Ag收益。


已实现并登记旧ordinary PA同抽样反事实 `20261016-diagnostic-F-crossRX-ordinaryPA-fitL-s2026092701-m1-r01`：固定fitL5040/20原日程/raw与full/捕获PA前prefix，206642NN/0更新，11负例及canonical原生输出和RNG终态一致检查PASS，一次P0/P1审查PASS；尚未启动。配对无恢复则不以此启动训练，源恢复也不证明Ag。


普通PA同抽样源诊断完整复算VERIFIED，原判定PASS；实际变动3054包观测，净恢复20.072036673215454pp，206642NN/0更新，无held/U/V/target。只说明冻结fit源的PA因果贡献，不证明Ag/DG/As；具体下一假设见活动配置，未启动新训练。


已实现登记PA旁路E20源配对 `20261016-phase1-F-crossRX-PAbypass-sourceRX1-e20-s2026092701-m2-r01`，两组同seed scratch20/原U/六L前向，B00原ordinary/B10同draw捕获PA前prefix，完整PA生成两组相同。实际native增强日程截13，已修正前探针E14至20外推范围声明。clean>=+1pp/F1>=0/各practical及V>=-.5pp，失败关闭固定配方。14负例与默认头/随机流/原生对照PASS，一次P0/P1审查PASS；尚未启动。


上述普通PA旁路B00/B10已实际发布并运行，执行commit `bdb3733de36b2d0e01515c91b7c9df8462b8c39b`；archive/compile/真实CPsmoke/两worker PID-CWD-argv-GPU及日志增长VERIFIED。实际seed2026092701、20/13/7和无训练父权重均核对；当前未完成训练/源预测/独立评分，不能称Ag有效。


PA旁路两组E20及七源视图已完整ANALYZED/VERIFIED，原条件FAIL固定配方关闭；clean差-1.2698412698412653pp、lowurban差1.428571428571429pp，全部17640身份决策及完整训练/PA/成本复算；无target/联合成功声明。


已实现登记常数增益候选 `20261016-phase1-F-crossRX-constantgain-sourceRX1-e20-s2026092701-m2-r01`：先真实匹配CP的162合成包/no-query输入作用核对，工程FAIL则0真实IQ/0训练关闭；PASS直接两组同源折同seed scratchE20，raw.5/oldordinary.25/第三槽raw或gain.25，旧PA/U/六L前向不变，range固定旧.9至1.1。14负例/原参照梯度/输入路径及一次P0/P1审查PASS。尚未发布或取得真实工程/训练收益。


常数增益候选已实际发布，commit `1d29887ef68319fe9a75b2c82fca30a7860dbf8e`。真实匹配CP的162合成NN、5×32 joint/logits/power独立复算PASS；实际Sinc输入与RMS恢复control符合预定，默认函数有非平凡作用。两worker PID/CWD/argv/GPU/log增长VERIFIED，20/13/7、同seed、无训练父权重均核对。只是输入作用与运行状态，不是Ag身份收益或联合目标完成。


常数增益两组E20及七源视图完整ANALYZED/VERIFIED，原条件FAIL固定配方关闭；全部17640身份决策、完整训练/gain/物理序列/成本复算。实际NN 9373518，0target/另训头，联合目标未完成。


当前下一唯一假设及最小实验见配置next_source_hypothesis与常数增益报告末尾：仅有界接收频偏观测，NoExplicitCFO与内部补偿不变；不复用失败公共常相位或gain配方，不进入关联对话状态课程。尚无新实验启动。


频偏输入候选已实现登记 `20261016-diagnostic-F-crossRX-frequency-input-s2026092701-m1-r01`，真实匹配CP＋162固定合成NN/0真实IQ/0更新/0target，原NoExplicitCFO及内部参考补偿保持。10协议负例、4判断反例和实际未训练F路径PASS，一次限定P0/P1审查PASS；尚未发布或取得真实CP工程结果，更无身份收益。


频偏输入候选已实现登记 `20261016-diagnostic-F-crossRX-frequency-input-s2026092701-m1-r02`，真实匹配CP＋162固定合成NN/0真实IQ/0更新/0target，原NoExplicitCFO及内部参考补偿保持。10协议负例、4判断反例和实际未训练F路径PASS，一次限定P0/P1审查PASS；尚未发布或取得真实CP工程结果，更无身份收益。


频偏输入r01标准库同名遮蔽技术失败已保留，0CP/IQ/NN/更新；修复改名并直接脚本回归/一次定点复审PASS，r02实际完成162合成NN及独立5×32数组复算。正负端点单位joint变化中位数约.028114/.030934，默认logits最大差.800132/.686726，逆变换恢复，工程PASS。仅有输入作用，无身份/Ag收益。下一科学验证为固定范围的匹配源B00/B10 scratchE20；尚未登记启动，联合目标保持。


独立轻度频偏L训练候选已实现登记 `20261016-phase1-F-crossRX-frequency-sourceRX1-e20-s2026092701-m2-r01`，两组同源折/seed scratchE20、raw.5/oldordinary.25/槽.25、6L前向，原U/PA/NoExplicitCFO/内部补偿/defaulthead保持，13负例及一次P0/P1审查PASS。尚未发布/训练/评分，不将工程输入作用当身份收益。


频偏匹配源E20已发布RUNNING，commit `bb38d5ef6dfcd3e60e7fb6d00c9c2fca976bfd66`，实际CP smoke2/归档-编译/两worker PID-CWD-argv-GPU-log增长VERIFIED，seed与20/13/7及0训练父权重核对。独立source全量分析已准备；尚无训练后身份分数，不将工程PASS当Ag有效。


轻度频偏两组E20及七源视图完整ANALYZED/VERIFIED，原条件FAIL固定配方关闭；全部17640身份决策、完整训练/frequency/物理序列/成本复算。实际NN 9373358，0target/另训头，联合目标未完成。


频偏E20固定配方FAIL已完整保留，不扩seed/改频偏救跑。下一唯一普通循环移位同draw反事实源诊断的拟定范围、净2pp/3of4RX条件及成本见配置next_source_hypothesis和原报告末尾；尚无新实验启动。


已实现登记旧循环移位完整同抽样反事实 `20261016-diagnostic-F-crossRX-ordinaryshift-fitL-s2026092701-m1-r01`。实际E1..13，fitL5040，136082NN/0更新，132704增强生成；14负例与原生输出/RNG及PA抽样控制PASS，一次P0/P1审查PASS，尚未启动。PASS也只支持独立E20普通配方修正，不称新Ag。


循环移位同抽样源fit诊断全量复算VERIFIED／PASS，非零3946观测，净恢复3.395843892549417pp；136082NN/0更新。只证明本固定源fit默认头贡献，不证明Ag/DG/As/联合收益，未启动后续训练。


已实现登记shift-bypass E20 `20261016-phase1-F-crossRX-shiftbypass-sourceRX1-e20-s2026092701-m2-r01`，固定RX1/seed01两组scratchE20、六L前向/原U/PA保持、自动源七视图；17负例/2312合成NN/2合成更新验证PASS，一次P0/P1review PASS，尚未启动。旧增强修正不冒称新Ag，完整联合目标保留。


上述普通循环移位旁路B00/B10已实际发布并运行，执行commit `5476e2238d6d43408227decf916f7177a02f2129`；archive/compile/真实CPsmoke/两worker PID-CWD-argv-GPU及日志增长VERIFIED。实际seed2026092701、20/13/7和无训练父权重均核对；当前未完成训练/源预测/独立评分，不能称Ag有效。


旧循环移位旁路两组E20及七源视图完成全量复算VERIFIED，原条件FAIL固定配方关闭；clean差0.3968253968253954pp、lowurban差-1.2698412698412709pp，17640默认头决策及所有step/移位/PA/成本复算；无target或联合成功声明。


下一唯一拟检验为保持完整baseline raw/ordinary监督的追加As训练组织，设置8前向同成本与额外raw监督/尺度控制，3row同源折/seed01 scratchE20。依据是实际旧As raw总权重.75->.50的代码事实，非已证实原因；具体原定候选条件和未启动边界见刚完成shift-bypass报告及活动配置。不复活旧shift/PA/gain/frequency失败配方，不访问target。


已实现并登记 additive As 源反证 20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01：固定RX1/seed01三组scratchE20/8L前向，完整保留原raw.75＋ordinary.25；分别追加raw0/raw.25/完整As(raw).25。20负例/3858合成NN/3合成更新PASS，一次P0/P1审查PASS；当前F的trackedBN为0。最后E20预算元数据已更正，尚未启动。须同时胜过两个对照并保护clean/practical；失败关闭固定方案，不宣称新Ag或联合成功。


上述三组additive As源反证已发布并实际运行；执行commit 7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b，Git远端OID/归档/编译/真实CPsmoke及三worker进程、参数、GPU和日志增长VERIFIED。各组固定20/13/7、4440更新、8L前向/保留原base raw.75；尚未完成训练、源预测或独立评分。


additive As三组E20、七源视图及全部轨迹已ANALYZED/VERIFIED；PASS_SOURCE_SINGLE_SEED。相对D00 clean 3.3333333333333326pp、lowurban 5.714285714285716pp；相对D_raw clean 3.0952380952380953pp、lowurban 6.190476190476191pp。实际60modelepochs/13320更新/17469956NN。仅源fold/单seed，不是Ag或联合目标完成；原科学淘汰/确认规则保持。


本轮additive As通过单seed源规则，但TX标签2/3在六practical仍负；lowurban仅33.809524%，suburban自身clean差25.396825/28.809524/32.063492pp，完整目标未达。下一唯一假设是原样seed2026092702三组E20独立确认；失败停止固定候选，不以seed03或参数救跑；尚未登记发布。


固定additive As seed02独立确认已登记 20261016-phase1-F-crossRX-additiveAs-confirm-sourceRX1-e20-s2026092702-m3-r01，三组原样scratchE20/8L前向，原条件和U保持，实际native与F modelconfig seed均02；24负例/初始state验证PASS、0IQ/0NN/0优化；一次新增profile P0/P1定向审查PASS，未复审旧机制。尚未启动。首轮PASS及TX2/3负面保留，确认失败不做seed03等救跑；目标仍未达。


上述三组additive As源反证已发布并实际运行；执行commit 05984a1229ebac7025a95081dc5fae4108276b28，Git远端OID/归档/编译/真实CPsmoke及三worker进程、参数、GPU和日志增长VERIFIED。各组固定20/13/7、4440更新、8L前向/保留原base raw.75；尚未完成训练、源预测或独立评分。


首种子固定预测补算：首种子的 low urban 在双方 clean 都正确的 982 个样本中修复 188／新错 70，净增 118 个；其他三组净减 46 个，总体净增 72 个（+5.714286 个百分点）。因此，这个源折上的 practical 收益并非只伴随 clean 新修复样本增加；它仍包含明显错误转移，不能推断各 TX 普遍受益或传播因果机制。 0 新 IQ／前向／更新；当前唯一下一假设仍为已运行的 seed02 固定确认，不新增判据。完整分解见该 run 的 error_transitions.md/json。


固定 additive As seed02 三组 E20 与七源视图完成：FAIL_FIXED_ADDITIVE_AS_CLOSED_AFTER_CONFIRMATION。无 target；source_analysis 与两种子汇总分别记录，不把 source PASS 当联合目标完成。


固定 additive As seed02 完整确认失败：D_As clean 74.365079%，相对 D00 −8.253968pp、D_raw −6.825397pp；low urban 36.111111% 虽提高，仍未达 60%。两种子相对 D00 clean 均值差 −2.460317pp、low urban +7.222222pp，原逐 seed 条件失败，关闭此固定候选扩展。绝对 raw 系数保留不足以保护 clean，健康通过不等于联合收益。六模型120 modelE／26640更新／34939912实测NN，无target或额外头。完整结果见确认run的two_seed_source_confirmation.md/json。未产生有效新Ag、正式target或联合目标证明。


下一单一问题已登记 20261016-diagnostic-F-IQ-circularity-source-crossfit-manysig-m1-r01：源RX关联的中心化circularity幅值能否跨TX/day复现，L5040／72单元／18折／9wrongRX／10TXsplit，0模型/NN/更新。固定As候选仍关闭；本轮尚未启动，不构造或声称新的标签保持Ag。


20261016-diagnostic-F-IQ-circularity-source-crossfit-manysig-m1-r01 已完成并独立复算：FAIL_FIXED_CIRCULARITY_RX_BASIS_CLOSED。sameRX 风险减少73.0123%，最佳wrongRX为77.0565%，未达到预定指派特异性；其他条件通过不覆盖此失败。5040源L／0NN／0更新／0target；不改量、窗口、阈值或RX补救，不启动IQ镜像Ag。联合目标仍未完成。


下一独立泛化方向 20261016-diagnostic-F-crossRX-sameTX-IQmix-fitL-manysig-m2-r01 已登记：同 TX/day 的实际跨源 RX IQ 以固定0.1组合，raw／同RX／跨RX对照，两冻结原默认头 source-fit 身份保护检查。L5040／预计50402NN／0新epoch／0更新／0target；旧circularity和As失败仍关闭。尚未启动，PASS也不是Ag/DG收益。


20261016-diagnostic-F-crossRX-sameTX-IQmix-fitL-manysig-m2-r01 双冻结模型完整源fit检查及50400预测复算完成：PASS_SOURCE_IDENTITY_GUARD_ONLY。crossRX−raw accuracy为−0.105820/−0.191799pp，raw正确保留99.887387/99.794648%，IQ相对RMS变化median0.078128。50402NN／0更新／0target。只支持下一匹配E20三组raw-slot/sameRX/crossRX；未有Ag/DG收益或联合成功。


同身份实测 IQ 混合源 E20 20261016-phase1-F-crossRX-sameTX-IQmix-sourceRX1-e20-s2026092701-m3-r01 已登记未启动：B00 raw／C_same 同 RX／B10 跨 RX，完整原 raw.75＋ordinary.25＋SupCon，再加匹配 .25 视图；均 8 次 L 前向、scratchE20／4440 更新。实际 L5040/U45360/V21600、RX1 原 L1260 final-only，固定旧 U 规则。B10 必须相对双方控制 clean≥+1pp、F1≥0、每 practical≥−.5pp、seenV≥−.5pp，失败关闭固定配方。G0 只证身份保持；正式目标和联合目标仍未完成。


上述同身份实测 IQmix 三组 E20 已实际运行；执行 commit 192e6a84d72ca89d29489ce2750a20eb36bc765d。归档、编译、匹配 checkpoint 烟测及三 worker 的进程、实际参数、GPU、日志增长 VERIFIED；固定 20/13/7、4440 更新、8 次 L 前向，As 训练关闭。尚未完成训练与源七视图评分，泛化和联合目标均未证明。


并行分工核对：最新可用《开发 CVS Phase1 星地增强训练》对话也已转向同 TX/day 跨 RX 实测 IQ 插值。与本路相对频率／相位对齐的固定 .1 组合细节不同，但属同一类，不能作为两条独立方向或直接合并跨 seed。当前已运行 IQmix 三组 E20、原判据与七视图源评分不变；后续新候选避开同身份插值类。不据聊天状态推断对方远端作业，不修改或消息干预其任务。


同 TX 实测 IQmix 三组 E20 和源七视图已完整 ANALYZED/VERIFIED：FAIL_FIXED_SAMETX_IQMIX_CLOSED。B10 相对 B00 clean 0.4761904761904745pp、low urban 0.23809523809523725pp；相对 C_same clean -0.4761904761904745pp。真实60modelE／13320更新／17469956NN，0 target。原判据不变；非完整联合证明，后续新方向避开同身份插值类。


下一不同方向仅为草案、未登记／实现／启动：固定完整 practical 分布上的 MC2 损失期望组织，比较同一个 As IQ 重复与两个独立 As IQ，另设同前向 raw 控制；均10 L 前向、原监督系数不变，不平均IQ、不减弱信道、不新增损失或分支、原U不变。检验clean保持与urban增益，失败不扩M/改系数或预算救跑。旧IQmix已关闭。


完整practical MC2源E20 20261017-phase1-F-crossRX-practicalMC2-sourceRX1-e20-s2026092701-m3-r01 已登记未启动：M00 raw/raw、M_repeat As0/As0、M_independent As0/As1，完整原raw.75/ordinary.25/SupCon加两对各.125，均10 L前向／1.25总权重；scratch20/13L7U/4440更新。只平均loss不平均IQ，原practical分布／receiver session／U保持。固定双方源判据已写入report，失败不扩M/调系数或预算；旧IQmix关闭，联合目标未完成。


practical MC2 三组E20已实际 RUNNING/VERIFIED；执行commit 8c55f5f542a6a28b699156859f31345b9d15f76a，归档／编译／匹配CP smoke与worker进程、实际参数、GPU、日志增长均已核实。固定10 L前向／原base .75raw/.25ordinary／extra两对各.125；原U不变。当前无冻结源评分，As采样组织有效性和联合目标均未证明。


MC2首行M00已source-only ANALYZED/VERIFIED：E20/4440更新/7视图8820预测，clean79.841270%、lowurban27.063492%，NN6959958。另两As行仍健康运行，未作科学资格判断，参数/预算/门槛保持。整体和联合目标均未完成。


MC2第二行M_repeat已source-only ANALYZED/VERIFIED：E20/4440更新/7视图8820预测；相对M00 clean 4.761904761904756pp、lowurban 6.190476190476191pp。第三行仍健康运行，科学资格待定，参数／预算／门槛不变，完整联合目标未完成。


practical MC2三组E20与源七视图完整ANALYZED/VERIFIED：FAIL_FIXED_MC2_CLOSED。独立组相对M00 clean 4.6031746031745975pp、lowurban 5.952380952380953pp；相对repeat clean -0.15873015873015817pp。真实60modelE／13320更新／20879876NN；0新增Ag／0target。完整联合目标未完成，固定失败不扩M／调系数／RX／seed／预算。


MC2完整负结果已保留：independent相对repeat clean−0.158730pp，六practical accuracy／F1均负，FAIL_FIXED_MC2_CLOSED；相对raw clean+4.603175pp／lowurban+5.952381pp不能补偿增量失败。lowurban TX2/TX3分别净−32/−23包，全部TX/day与固定clean正确集合错误转移已报告。控制M_repeat不事后晋级，无后继已登记候选；新Ag与完整联合证据仍缺。


下一唯一源依据问题：周期差异残差lag1至4形状是否有跨TX/day可迁移的RX分量；文献与实际EQ1/25Msps/256点口径已核查，尚未实现、登记或运行。固定窗口20:180、8个20点周期、18折与coherence/风险/错RX条件见cycle_residual_covariance_source_basis及研究说明。0拟新增NN/更新/模型/target；只有源统计问题，尚未选择IQ算子或幅度，不宣称Ag或联合收益。MC2及旧失败路线继续关闭。


周期差异残差相关形状源检验 20261017-diagnostic-F-crossRX-cyclecov-fitL-manysig-m1-r01 已实现登记未启动：L5040/72格/18折/3wrongRX控制，窗口20:180、8个20点周期、lag1至4；22本地反例检查及一次独立P0/P1审查PASS。0模型/CP/NN/更新/target，U只有角色ID、无IQ/标签。只检验源RX关联依据；未构造Ag或宣称身份收益，固定失败不改窗口/lag/RX/seed救跑。


周期差异残差源依据 20261017-diagnostic-F-crossRX-cyclecov-fitL-manysig-m1-r01 已完整ANALYZED/VERIFIED：FAIL_FIXED_CYCLE_RESIDUAL_COVARIANCE_BASIS_CLOSED。合并形状风险改善35.211296%/17正折不足以覆盖50/72中位数、43/72q10失败和RX6风险+17.668361%；原固定窗口/cycle/lag/RX/阈值/seed关闭。L5040/0模型/NN/更新/target，全量逐包统计复算与图完成；没有新Ag或后继已登记候选，联合目标ACTIVE。


下一单一候选端部可见性 20261017-diagnostic-F-crossRX-edgevisibility-fitL-manysig-m2-r01 已实现登记未启动：16点支持/外8归零内8半余弦，raw/left/right，两既有合法E20默认头fitL5040操作身份检验。19输入/配置负例和7科学反例/120合成IQ窗PASS，一次P0/P1审查PASS；预计30242NN、0新训练/更新/target。原条件失败不缩窗/换边/改阈值/RX/seed；PASS也不是Ag/DG或联合收益。与另一对话practical正负噪声方向不同，全部旧失败路线继续关闭。


端部可见性 20261017-diagnostic-F-crossRX-edgevisibility-fitL-manysig-m2-r01 完整ANALYZED/VERIFIED：FAIL_FIXED_EDGE_VISIBILITY_IDENTITY_GUARD_CLOSED。两冻结模型/三视图全部30240判决、CE/F1/分层/窗统计复算，实际30242NN/0更新/target。seed02左/右accuracy−1.388889/−.654762pp，左端11包IQchange>.35；参考完全保持仍不足以保真。固定16/8zero8taper关闭，不缩长度/换边/阈值/RX/seed救跑，不启动E20；没有新有效Ag，联合目标ACTIVE。


下一单一候选普通增强输出RMS恢复 20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r01 已实现登记未启动：同一实际E13 ordinary IQ，仅正实数恢复原包功率；四命名抽样、两既有合法E20默认头fitL5040，预计90722NN，0新训练/更新/target。19输入负例＋8科学反例/96合成RMS PASS；原审查两P1修复定向PASS。固定完整ordinary后置RMS/E13/四抽样失败不换位置/部分系数/clip/禁用旧增强/抽样/RX/seed。PASS仅源尺度身份依据，须另登记三匹配E20控制，未证明Ag或联合收益。区别于另一对话practical正负噪声方向，旧失败全部保持关闭。


普通增强输出RMS r01 20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r01 FAILED_TECHNICAL_VERIFIED：SatViewStage日志序列化失败，2合成NN/1CP加载/64合成ordinary，0源NN/训练/target；seed02未启动，全部own进程自然退出。源假设尚未检验，不把技术失败作科学FAIL，不覆盖或自动重试旧根。


普通增强输出RMS日志技术修复r02 20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r02 已实现登记未启动：同一实际E13 ordinary IQ，仅正实数恢复原包功率；四命名抽样、两既有合法E20默认头fitL5040，预计90722NN，0新训练/更新/target。19输入负例＋8科学反例/96合成RMS PASS；原审查两P1修复定向PASS。固定完整ordinary后置RMS/E13/四抽样失败不换位置/部分系数/clip/禁用旧增强/抽样/RX/seed。PASS仅源尺度身份依据，须另登记三匹配E20控制，未证明Ag或联合收益。区别于另一对话practical正负噪声方向，旧失败全部保持关闭。


普通增强RMS恢复 20261017-diagnostic-F-crossRX-ordinaryRMS-fitL-manysig-m2-r02 完整ANALYZED/VERIFIED：FAIL_FIXED_ORDINARY_POST_RMS_E13_FOURDRAW_CLOSED。90720源判决/320batch全复算，90722NN/0训练更新/target；连r01技术失败共90724NN。RMS相对ordinary accuracy−.431548/−.386905pp，四抽样全部下降；修复82/97而新错169/175。正确尺度恢复仍损害身份，关闭固定完整ordinary后置RMS/E13/四抽样，不做PA条件/系数/clip/禁用旧操作/抽样/RX/seed救跑，不启动E20。下一独立普通通用AWGN/multipath审计尚未实现登记启动，不能冒称新Ag；联合目标ACTIVE。


普通AWGN/multipath源审计 20261017-diagnostic-F-crossRX-ordinaryChannel-fitL-manysig-m2-r01 已实现登记未启动：full/no_AWGN/no_MP/no_both，primary固定no_both，单项仅归因不可fallback。全native参数/RNG/PA掩码匹配，移除MP后AWGN功率依赖重算，是总链效应；As原本读raw，没有同视图重复叠加。17负例/7科学反例/480合成ordinary PASS，预检缺字段P1修复定向PASS；171362 NN/0新训练/target，旧RMS等失败保持，不称新增Ag或泛化有效。


普通AWGN/multipath源审计 20261017-diagnostic-F-crossRX-ordinaryChannel-fitL-manysig-m2-r01 完整ANALYZED/VERIFIED：FAIL_FIXED_JOINT_ORDINARY_CHANNEL_REMOVAL_RAW_IDENTITY_GUARD_CLOSED；171360源判决/320逻辑batch全部复算，171362NN/0更新/target。联合移除比full+1.522817/+1.488095pp，但比raw−5.868056/−5.962302pp、raw正确保留94.102742/94.023251%，固定raw保护FAIL，关闭不启动E20，不选单项fallback。每模型MP/AWGN/双门batch=6/17/0，interaction不证明无物理交互；full和raw全部logits与既有RMS源研究对应原视图完全相同。As本来读raw，无同视图普通信道再叠As。下一孤立轻度noise(raw)候选尚未实现登记启动，不能称新增Ag有效；联合目标ACTIVE。


孤立轻度AWGN(raw) 20261017-diagnostic-F-crossRX-isolatedAWGN-fitL-manysig-m2-r01 已实现登记未启动：既有native propercomplex噪声20至36dB、全覆盖四命名流，无MP/PA/时移/后RMS/clip。双冻结原默认头fitL5040，每draw与pooled身份保护，全部尾部保留。14负例/9科学反例/1152合成noise PASS，一次P0/P1审查PASS；预计50402NN/0更新/target。保留公共旧ordinary的未来三匹配E20未实现登记启动；源PASS不称Ag/DG有效，旧RMS/联合移除等固定失败保持关闭。


孤立AWGN源检验 20261017-diagnostic-F-crossRX-isolatedAWGN-fitL-manysig-m2-r01 VERIFIED／PASS_SOURCE_IDENTITY_ONLY：50400完整源判决／50402NN独立复算，20至36dB全覆盖；相对raw准确率−0.143849/−0.138889pp，raw正确保留99.850954/99.845986%，四draw与RX/TX/day全部条件PASS，IQ变化median.039368733/max.106857479；0训练/target。仅支持三匹配scratchE20，不证明Ag或联合目标；既有固定失败保持关闭。


孤立AWGN三匹配E20 20261017-phase1-F-crossRX-isolatedAWGN-sourceRX1-e20-s2026092701-m3-r01 已实现登记未启动：G0sourceidentityPASS后仅检验Ag，公共raw.75/ordinary.25/SupCon/U保持，extra.25 raw/sameordinary/noise，8L前向/pairweight1.25，同scratchseed01/E20；B10须同时胜过两个控制clean≥1pp/F1不降/每practical与seenV≥−.5pp。28负例/4114合成NN/3合成更新PASS，一次P0/P1审查PASS；实际新训练尚为0，联合目标ACTIVE。


孤立AWGN三匹配E20 20261017-phase1-F-crossRX-isolatedAWGN-sourceRX1-e20-s2026092701-m3-r01 VERIFIED RUNNING，commit 06ebf579c87c834d0ebc14856212e3fa97619d30，真实CP smoke/归档传输/compile PASS，GPU0/1/2 actualPID/CWD/argv/CUDA与存活loggrowth核实；scratchE20已开始、尚无完整识别读出和科学结论，不增加预算、改配方或使用target。


孤立AWGN三匹配E20 20261017-phase1-F-crossRX-isolatedAWGN-sourceRX1-e20-s2026092701-m3-r01 ANALYZED/VERIFIED：FAIL_FIXED_NATIVE_AWGN_20_36_FULL_COVERAGE_TRAINING_CLOSED；60modelE/13320更新/17469956NN，完整26460源预测/全部steps/noise统计复算；B10相对raw clean 0.9523809523809601pp，相对ordinary clean -0.4761904761904745pp。仅sourceRX1/seed01，无target；原晋级条件不变，联合目标ACTIVE。


当前唯一下一source假设：原IQ的常相位／线性CFO不变相位曲率统计，是否有跨TX/day共享RX位移；只拟用fitL5040、0CP/NN/新视图/更新/target，按固定交叉TX/day误差验证后才考虑增强；尚未实现登记启动。详见analysis/cvs_phase_curvature_source_basis_20261017.md。旧commonphase训练／phase_curvature结构／cyclecovariance等失败保持关闭，联合目标ACTIVE。


非恒定相位曲率源依据 20261017-diagnostic-F-crossRX-phasecurvature-fitL-manysig-m1-r01 已实现登记未启动：固定80:160全78相邻三点/Complex128eps1e-12，156RI，fitL5040/72格/18组同时留TX与day；pooledMSE减少≥10%/每RX不恶化/至少15正组。29负例+7反例/266合成IQ PASS，一次P0/P1审查PASS；0模型/CP/NN/增强/更新/target。仅源RX关联依据，不重开commonphase/featurecurvature/cyclecov/AWGN失败路线，联合目标ACTIVE。


非恒定相位曲率源依据 20261017-diagnostic-F-crossRX-phasecurvature-fitL-manysig-m1-r01 已完整ANALYZED／VERIFIED：PASS_SOURCE_RX_ASSOCIATION_ONLY，pooledMSE减少91.368334%，每RX正、18/18heldTX/day正；5040×156观察/72格均值方差/18全向量风险独立复算一致。0CP/NN/更新/新IQ/target。描述性最佳逐点相位只能解释76.452086%源均值差，主值积分的大差异RX对RMS约1.28至1.38rad，尚不支持轻度相位模板回放；不是全局幅度不可能性证明。没有固定新IQ算子/范围、身份保护或E20，旧commonphase/featurecurvature/cyclecov/AWGN失败保持关闭。F＋跨RX同TX底座不变；联合目标ACTIVE。


有界非恒定相位源身份检验 20261017-diagnostic-F-crossRX-boundedPhase-fitL-manysig-m2-r01 已实现登记未启动：固定sigma.006、去常数/线性、整路径峰值.1rad、全256单位模，无RX模板/尾部删除/重抽。两个原合法scratchE20 B00默认头、fitL5040、四draw，预计50402NN/40320sourcephase+64synthetic/0更新/target。28负例+18科学/解析反例/388合成phase PASS，一次P0/P1 PASS。旧ordinary本有Wiener，此为独立全覆盖有界视图变体，非新增强原理；identitypass也非DG。共同F＋跨RX同TX/E20/原U保持，联合目标ACTIVE。


有界非恒定相位源身份检验 20261017-diagnostic-F-crossRX-boundedPhase-fitL-manysig-m2-r01 完整ANALYZED／VERIFIED：PASS_SOURCE_IDENTITY_ONLY，两个默认头accuracy相对raw−.019841/−.029762pp，raw正确保留99.965223/99.970191%；四draw和每RX/TX/day通过。50400源预测/10321920相位坐标/320batch全部独立复算，实际50402NN/40320sourcephase+64synthetic/2CP/0更新/target；全IQ变化median.022373/max.058648，312/20160路径整包收缩，全部尾部保留。raw全logits与旧AWGN源检查一致。支持下一三匹配scratchE20，不是DG/As有效；当前尚未实现登记E20。F＋跨RX同TX底座保持，联合目标ACTIVE。


有界非恒定相位三匹配E20 20261017-phase1-F-crossRX-boundedPhase-sourceRX1-e20-s2026092701-m3-r01 已实现登记未启动：G0sourceidentityPASS后仅检验Ag，公共raw.75/ordinary.25/SupCon/U保持，extra.25 raw/sameordinary/phase，8L前向/pairweight1.25，同scratchseed01/E20；B10须同时胜过两个控制clean≥1pp/F1不降/每practical与seenV≥−.5pp。28负例/4114合成NN/3合成更新PASS，一次P0/P1审查PASS；实际新训练尚为0，联合目标ACTIVE。


有界非恒定相位三匹配E20 20261017-phase1-F-crossRX-boundedPhase-sourceRX1-e20-s2026092701-m3-r01 VERIFIED RUNNING，commit 51fc25d6862b7773e8d67fcc95a852c2e52ecc86，真实CP smoke/归档传输/compile PASS，GPU0/1/2 actualPID/CWD/argv/CUDA与存活loggrowth核实；scratchE20已开始、尚无完整识别读出和科学结论，不增加预算、改配方或使用target。


有界非恒定相位三匹配E20 20261017-phase1-F-crossRX-boundedPhase-sourceRX1-e20-s2026092701-m3-r01 ANALYZED/VERIFIED：FAIL_FIXED_NATIVE_BOUNDED_NONAFFINE_PHASE_006_01_FULL_COVERAGE_TRAINING_CLOSED；60modelE/13320更新/17469956NN，完整26460源预测/全部steps/phase统计复算；B10相对raw clean -0.5555555555555536pp，相对ordinary clean -1.3492063492063444pp。仅sourceRX1/seed01，无target；原晋级条件不变，联合目标ACTIVE。


下一唯一来源假设为同一原L物理包的raw/equalized接收预处理观察，见analysis/cvs_paired_preprocessing_basis_20261017.md。官方FullWiSig说明与loader eq维度不能证明compact数组逐行配对；先查构造索引/原角色，不假定同sig_i，不先读新IQ，无新模型/更新/target。配对或权限不可证即放弃该入口，不扩大L或改为equalized=both；不复活已关闭相位/噪声/其他固定失败。


raw／EQ 接收预处理配对入口已完成代码及既有元数据审计，状态 UNVERIFIED_SAME_PACKET_PAIRING_CURRENT_INPUT_CLOSED：官方均衡可跳包且不存原包映射，compact 分别截前缀；原契约、准备记录和实际依赖加载器没有补足 raw↔EQ 对应表。真实配对未知，不按同 sig_i／相似度／相邻包猜配对，不改 equalized=both 或扩大 L。新增真实 IQ／CP／NN／更新／target 均为 0。详见 analysis/cvs_paired_preprocessing_basis_20261017.md。下一唯一候选为与信号斜率相关的采样时刻扰动，目前仅 4 模板／80 合成点／5 项代数检查通过，单音等不可辨认反例保留；先检验短前导估计能否区分白噪声、增益和相位，没有源实验登记或训练。详见 analysis/cvs_sampling_jitter_source_basis_20261017.md。已关闭固定路线保持，F＋跨 RX 同 TX／E20／原 U 不变，联合目标 ACTIVE。


采样抖动短前导估计器 20261017-diagnostic-F-crossRX-jitterEstimator-synthetic-m1-r01 已实现登记未启动：5模板／9情况／3SNR／32抽样／70包／4周期／20点，共4320格及24192000合成复观察；两周期差协方差与另外两周期交叉方向，不提供真模板；全部3配对、非负4方差矩求解，固定5倍标准误可辨认条件。22输入负例＋11公式/退化检查通过，一次P0/P1通过；0真实IQ／CP／NN／更新／target。固定任一组失败关闭入口，不改周期/窗口/包数/频带/不确定度/幅度/阈值/seed救跑，不进入源数据或E20。F＋跨RX同TX/E20/原U保持，联合目标ACTIVE。


采样抖动短前导估计器 20261017-diagnostic-F-crossRX-jitterEstimator-synthetic-m1-r01 已完整 ANALYZED／VERIFIED，FAIL_FIXED_FOURCYCLE_JITTER_COVARIANCE_ENTRANCE_CLOSED；4320格/135组充分统计及方差/判据全部独立复算一致。134组通过，唯一20dB gain_white组可辨认30/32＜95%，按原规则关闭；正例误差、误归因及退化拒绝条件全部通过，不等于物理抖动不存在。24192000合成观察/12.301378秒，0真实IQ/CP/NN/更新/target，不进入源数据或E20、不调判据救跑。F＋跨RX同TX/E20/原U保持，联合目标ACTIVE。


下一唯一问题转向不增加视图的 A_s 训练组织：按物理 L 包均衡六场景覆盖，见 analysis/cvs_physical_scene_coverage_hypothesis_20261017.md。已确认原 PairedChannel 独立均匀场景及每step刷新，既有训练序列与scene_counts可供零IQ历史复算；实际每包覆盖尚未复算，无新E20实现/登记/启动。不能把计数均匀当识别增益，若匹配成本/clean/suburban保护下不胜随机As的urban识别则放弃。旧多视图/相位/噪声/四周期抖动失败均保持关闭，F＋跨RX同TX/E20/原U不变，联合目标ACTIVE。


逐物理包六scene循环 20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01 已实现登记未启动：历史全部4440步精确replay，所有5040包都见过6scene、无缺失，次数差中位11/max27；只以减少曝光不均匀为新As训练组织假设，不是新Ag。三匹配scratchE20/8L/总weight1.25，完整raw.75+ordinary.25/SupCon+原U保留，extra.25raw/IIDfullAs/balancedsinglefullAs，边际sceneuniform/传播噪声补偿不改。35负例/4114合成NN/3合成更新/520fullpractical PASS，一次P0/P1 PASS；0实际新训练/target。低urban与urbanmean须同时胜过随机As各1pp、clean及F1不降等原固定条件，不靠覆盖指标宣称有效。联合目标ACTIVE。


逐物理包六场景均衡三匹配E20 20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01 VERIFIED RUNNING，commit c9a74172e5f81bb4f04ce12cd4386ad4dd953ee9，真实CP smoke/归档传输/compile PASS，GPU0/1/2 actualPID/CWD/argv/CUDA与存活loggrowth核实；scratchE20已开始、尚无完整识别读出和科学结论，不增加预算、改配方或使用target。


20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01 B00首行SOURCE_ROW_ANALYZED：完整E20/4440更新/7view独立复算，sourceRX1 clean80.158730%／lowurban27.777778%，5823318NN；两As组仍健康运行，未形成组间科学判定，不改参数，0target。


逐物理包六scene循环 20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01 ANALYZED/VERIFIED：FAIL_FIXED_PER_PHYSICAL_PACKET_SIX_SCENE_SCHEDULING_CLOSED；三匹配scratchE20/60modelE/13320更新/17469956NN，26460源预测与两As行1136640scene选择全量复算。B01−CIID clean 0.23809523809523725pp／lowurban 0.15873015873015817pp；0target。联合目标ACTIVE，当前无新Ag证明。


下一唯一问题转向真实同包连续窗口可用性，见 analysis/cvs_contiguous_window_availability_20261017.md。实际冻结后端支持T>256中心裁剪，当前out_len256/EQ1；这不证明现有合法L有长上下文，未读取新IQ或启动训练。若仅256点或缺原包／参考对应，关闭入口，不以roll/插值/拼接相邻包/换eq救援。0新增NN/更新/target；闭合的scene均衡失败保持不变，联合目标ACTIVE。


真实同包连续窗口入口NO_TRACEABLE_LONG_CONTEXT_CURRENT_INPUT_CLOSED：官方FullWiSig/PKL保留256点，compact仅选择包行，现存元记录无raw capture／参考定位；未逐包扫描本地数组，不声称实测长度。0新IQ/CP/NN/更新/target，不拼接邻包或换eq救跑。下一唯一依据问题为短窗频域非仿射相位的可辨识RX关联，见analysis/cvs_frequency_phase_observability_basis_20261017.md；先固定有限窗口代数/合成辨识问题，尚未登记或执行新实验，不能以旧幅度谱或时域相位结果作为成功证据。联合目标ACTIVE。


短窗频域相位合成诊断 20261017-diagnostic-F-crossRX-frequencyPhase-synthetic-m1-r01 已实现登记未启动：固定80:160／lag20自身CFO补偿（数值不作身份）／12谐波相位闭合量，3模板×7情况×3SNR×4抽样＝252格／1270080合成包／325140480复点；正共享RX、无RX、TX曲率、RX幅度、交互、窗外边界及缺频退化均保留。19输入负例＋8代数检查及一次P0/P1通过。0真实IQ／CP／NN／更新／target。全部固定标准须通过；任一失败关闭此估计器，不改窗口/载波/补偿/阈值/SNR/seed救跑，不进入源数据或E20。通过仅支持另行源观测登记。基线F＋跨RX同TX／E20／原U保持，联合目标ACTIVE。


频域相位合成诊断 20261017-diagnostic-F-crossRX-frequencyPhase-synthetic-m1-r01 已完整ANALYZED／VERIFIED，PASS_FIXED_SYNTHETIC_MEASUREMENT_ONLY：252/252格／63/63组全部原判据通过；正例最大相对RMSE3.999642%、最小交叉风险改善98.092207%，全部负例不误检、退化全拒绝。325140480合成复点／50.409012秒，0真实IQ/CP/NN/更新/target。此为测度合成基础，不是RX因果、标签保持或Ag/As性能证据；下一唯一问题是另行预登记fitL源观测，尚未启动。F＋跨RX同TX/E20/原U保持，联合目标ACTIVE。


频域相位源观测 20261017-diagnostic-F-crossRX-frequencyPhase-fitL-manysig-m1-r01 已实现登记未启动：沿用合成通过的80:160／lag20／12谐波／8闭合量及数值条件，fitL5040 RX3/4/6/8、day1/2/3／六TX／72格全包保留，任一必要包不可辨识即FAIL；不使用U/V/RX1/target。21协议负例＋4科学用例PASS，一次P0/P1 PASS；0真实IQ已读／CP／NN／更新／增强。源关联不是RX因果或Ag性能，FAIL关闭当前入口，不改测度/质量/阈值/包/seed救跑、不进E20。F＋跨RX同TX/E20/原U保持，联合目标ACTIVE。


频域相位源观测 20261017-diagnostic-F-crossRX-frequencyPhase-fitL-manysig-m1-r01 已完整ANALYZED／VERIFIED，FAIL_FIXED_SOURCE_PHASE_OBSERVABILITY_CLOSED：5040包全部观测，2492通过必要频点、2548未通过；RX3/4/6/8分别通过0/1248/1243/1。72格全留，22全通过／35全拒绝／15部分；共享RX折实际0，不筛包或只取RX4/6。全部5040观察质量与72格均值方差独立复算一致。1次cache／40320闭合量／1.396791秒，0CP/NN/更新/新视图/target。关闭当前源依据，不改测度/窗口/频点/质量/阈值/子集/seed救跑，不进增强/E20；不等于真实RX相位不存在。F＋跨RX同TX/E20/原U保持，联合目标ACTIVE。


当前唯一待检验问题转为既有跨RX同TX对比项的视图覆盖：已读实际model.hook/supcon及scene_cycle extra路径，SupCon只看ordinary两次RDrop的浅/中map，raw与As均未进入该项。拟用相同As/CE/8L前向/总权重，固定ordinary-only／As-only／64:64混合的原项输入控制，复用原loss而不加项；这是As训练组织，不是新Ag，未登记或启动。见analysis/cvs_existing_supcon_view_coverage_20261017.md。频域相位源入口保持关闭，联合目标ACTIVE。


原SupCon视图路由 20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01 已实现登记未启动：旧Q0/Q_ord实际全配置/完整LUV物理ID/seed/20轮4440步/8L/1.25权重/默认头核对，复用40旧modelE，不载入权重；新增Q_sat全Asmaps、Q_mix64/64出现位置maps，2scratchE20/40新modelE/8880更新。完整raw.75/ordinary.25+extra.25fullAs与原浅中层SupCon总权重、4项/schedule/温度保持，0新增NN/loss项/U/teacher。29负例/4108合成NN/2合成更新/512fullAs PASS，旧ordinary-route loss exact/74参数grad接近、同As IQ/RNG、默认头更新/export160一致，一次P0/P1 PASS。Q_mix须同时胜过Q_ord/Q_sat low/urbanmean各1pp并保护clean/F1/suburban，vsQ0 low5pp/mean3pp；失败关闭固定比例/层位/系数/概率/seed，不救跑。这是As训练组织，不是Ag，联合目标ACTIVE。


原SupCon视图组织 20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01 已实际RUNNING／VERIFIED，执行ed0ca91481f8c11a28cb8f4d4726a7c3d488720e；两旧控制复用、两新scratchE20实际GPU0/1 live，realCP2NN无querysmoke/归档传输/remotecompile/PID-CWD-argv-GPU-log增长均PASS。两行至少完整E1/222更新、227328LNN/28416fullAs/全部5040L覆盖，原E6启用对比项，当前warmup routing0，无新增NN/loss项。新训练/预测/正式评分未完成，无性能结论，不热改/重启。F+CROSS_RX/E20/默认头/原U保持，完整联合目标ACTIVE。


20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01：原进程至少完整 E10，E6 路由实际启用并符合全 As/64:64 配置；无新增网络前向/损失，未热改。离线分析 34 反例 PASS，新识别结果待 final20/七视图；完整目标 ACTIVE。


20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01：ANALYZED／VERIFIED，FAIL_FIXED_VIEW_ROUTING_CLOSED；四行35280源判决、每行E20/4440步、完整loss/As场景/路由/U组织/预算独立复核。Q_mix固定primary，0target，完整Ag＋As联合目标ACTIVE。


当前下一唯一源问题：共享微弱加性IQ方向的跨RX/day迁移敏感性，见analysis/cvs_shared_iq_direction_basis_20261017.md。算子/固定12源单元/15+19反例及真实F合成检查已实现通过，合计84合成NN/36输入梯度贡献，0真实IQ/CP/更新/target。不是旧U-VAT一致性训练；无新训练loss/teacher/部署模块。尚未登记/发布/读取新源数据，E20未启动，敏感性不等于Ag收益。另一对话当前量化残留观测不在此任务中修改；原所有失败路线保持关闭，完整目标ACTIVE。


共享微弱加性IQ方向 20261017-diagnostic-F-crossRX-sharedDirection-fitL-s2026092701-m1-r01 ANALYZED／VERIFIED：FAIL_FIXED_SHARED_DIRECTION_CLOSED。原fitL5040/12heldRXday/三视图15120判决完整复算；raw99.841270%、direction99.801587%、permuted99.841270%，差0.039683pp、0/4RX≥1pp；修复1破坏3。75602新NN／30240输入梯度贡献／240调用，0更新/As/target，未进入E20。固定所有knobs关闭；无新Ag候选晋级。F＋跨RX同TX／E20／原U/默认头保持，完整联合目标ACTIVE。


下一唯一待检验问题为额外As对的RDrop KL是否抑制原监督收益：20261017-phase1-F-crossRX-AsPairCE-sourceRX1-e20-s2026092701-m3-r01，新C_CE从零E20/4440，复用Q0/Q_ord两个匹配8L/1.25控制结果，不载其权重。只移除额外As的KL，原baseRDrop/SupCon/U和singlefullAs保持，纯两CE同IQ。34反例＋真实F2054合成NN/1更新验证PASS，74参数梯度差仅旧.25KL；一次P0/P1 PASS。已登记未发布，无新真实IQ/CP/target。此为As配方而非Ag；失败关闭系数/CE权重/场景/日历/RX/seed救跑，完整目标ACTIVE。


20261017-phase1-F-crossRX-AsPairCE-sourceRX1-e20-s2026092701-m3-r01 已RUNNING／VERIFIED，执行89505b966a853e9126f372c89581f4c2b1c8eb16：单新scratch行实际GPU0，smoke/控制复用/归档/remotecompile/原PID-CWD-argv-GPU-log增长PASS。已完整E1/222更新/227328LNN/28416fullAs/全部5040L，原U/SupCon启用时间保持，尚未完成训练、源预测或性能判定。只移除As对KL，真实meanCE两前向；无新Ag，完整目标ACTIVE，健康原进程继续不重启。


2026-10-17：20261017-phase1-F-crossRX-AsPairCE-sourceRX1-e20-s2026092701-m3-r01 已完整 E20／4440 更新／源七视图独立复算 ANALYZED、VERIFIED，FAIL_FIXED_AS_KL_REMOVAL_CLOSED。C_CE clean81.904762%、low urban34.603175%；相对普通As＋RDrop clean−2.539683pp、low＋.714286pp、urban均值−.555556pp，关闭仅移除额外As对KL，不扫中间权重/场景/RX/seed救援；无新Ag或target证据，F＋跨RX同TX对比基线保持、完整goal ACTIVE。报告 automation_reports/CV-SincNet/20261017-phase1-F-crossRX-AsPairCE-sourceRX1-e20-s2026092701-m3-r01/report.md。


2026-10-17：下一唯一训练组织假设为六步L重排，详见analysis/cvs_L_batch_order_basis_20261017.md。完整4440原L批／568320观察／13320对比统计复算，0正对缺失更新，故缺失解释否定；固定重排保留每块及全程物理多重集、将positive_min全程下限4→10。已实现元数据/合成tensor原型，并在实际N607 CPU通过222步/37块、逐包IQ/y/domain/metadata与原全局RNG/U路径检查，0真实IQ/CP/NN/更新/target。本地RandomSampler NumPy桥接失败单独保留，未改环境。尚未完成observer/launcher、未登记或启动E20；将需两条共同预取/私有L流的新匹配E20，不能复用旧控制，也不把该训练组织当IQ Ag；固定识别门槛失败关闭，不调块长/seed/RX/权重救跑，联合目标ACTIVE。


2026-10-17：20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01 已补齐两新匹配E20的design/model/source/runner/dispatch/publish/登记入口，REGISTERED尚未发布。真实冻结native iterator及222步wrapper已只读核对；原型N607 CPU接口PASS，本地26反例及两新模型与原F全部206状态/初始logits逐值一致、6合成NN/0更新/真实IQ/CP/target。唯一P0/P1审查进行中。O0/O1同buffer6/私有L/原8L-1.25/U/defaulthead，仅重排不同；不称IQ Ag，不复用旧控制，完整联合目标ACTIVE。


2026-10-17：20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01 已 VERIFIED RUNNING，执行commit9059b58502f207331bf9643d9d1399d308a2c2da，单archive/compile/真实匹配CP no-query smoke PASS。两原worker PID/CWD/argv/CUDA/GPU4/5与日志增长独立核实，两行均完整E1/222更新/37块，O0 moved0、O1 moved28367；实际L5040/U45360/V21600，0target，无final20/七视图新成绩。保持每行E20，不热改/重启；当前只是训练组织，不是IQ Ag或联合收益，完整goal ACTIVE。


2026-10-17：20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01 已完整 ANALYZED／VERIFIED，FAIL_FIXED_SIX_STEP_L_ORDER_CLOSED。两条新 scratch E20／各4440更新，17640源判决、每行740块物理多重集及全部实际交付输入、13320 SupCon mask term、20轮loss/U/教师预算独立复算。O0→O1 clean80.238095%→78.412698%（−1.825397pp），Macro-F1−2.012662pp，low urban27.936508%→27.380952%（−.555556pp）；mid suburban−.555556pp。原正对统计改善但身份收益失败，关闭固定6步重排，不改块/分配/私有Lseed/SupCon/CE/RX/modelseed/预算救跑。40新modelE／8880更新／11646638实际NN含dispatcher2；0target/新IQAg/As训练/父权重。原进程自然完成，4图PNG/PDF已查看，完整goal ACTIVE。报告 automation_reports/CV-SincNet/20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01/report.md。


全频段 PSD 正仿射：`20261017-diagnostic-F-crossRX-PSDaffine-fitL-aggregate-m1-r01`，执行commit `1091fc97568605cd1115b6fa14754f781df197cc`，432拟合／432独立复算VERIFIED，0/6 RX对通过，固定模型关闭；主拟合1.494705 s、复算0.963868 s；0新IQ/CP/模型/NN/更新/target。此结果不能证明A_g／A_s或联合成功，完整目标ACTIVE。报告见[本行结果](../automation_reports/CV-SincNet/20261017-diagnostic-F-crossRX-PSDaffine-fitL-aggregate-m1-r01/report.md)。


最新全源普通拼接参照已RUNNING／VERIFIED：`20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02`，全L6300/U56700/V27000，无源RX训练留出。r01仅合成测试桩失败、无真实训练，已保留；r02只修测试桩，科学配方不变。三组各scratchE20，自动逐行clean＋六practical/独立truth-last已随队列发布。完整结果待实际返回，Ag与联合目标未证明。

2026-10-17：最新有效预算E30，全部5源RX/L6300/U56700-hidden/V27000，无训练RX留出。固定N0源输入研究仍FAILED；冻结输入即时准确率不等于训练效用，先前据此一概不训练的停止条件过强，保留原失败与阈值后单独登记首次训练反证：20261017-phase1-F-crossRX-N0prior-e20plus10-s2026092701-m1-r01，同普通As E20父/ownEMA追加10、4L/defaulthead/原U，N0倍率.5/1.5等prior、原quality/lock/tracking/AGC重算，0新loss/U/NN。执行commit a47ffaf90888347fc3474d43ed365c7e51b1c1a9，运行PID/CWD/argv/GPU/log增长VERIFIED，已见累计E23/至少716新更新，尚无正式成绩。不是新Ag，无目标反馈/自动扩种；固定失败即结束配方，不改幅度/概率/seed/父/预算救跑。详见本行report。

2026-10-17：20261017-phase1-F-crossRX-N0prior-e20plus10-s2026092701-m1-r01 已 ANALYZED／VERIFIED，固定N0配方首次真实训练＋全部7×168000 predictions/logp＋独立CPUtruth-last完成，原进程自然结束。普通拼接式→N0：clean82.963690%→82.867857%（−.095833pp）、midurban64.371429%→64.438690%（+.067262pp）、lowurban53.447024%→53.475000%（+.027976pp）；low修复934/新增887，clean且low共同正确净少41包。suburban距ownclean=1.172619/3.478571/5.852976pp，clean相对B00−2.400595pp。新2220/累计6660更新，1136640LNN、284160As，训练1320.279301秒；N0计数142311/141849、经验均值.000999187078，未声称经验均值精确相等。4440采样记录/68共有参数一致、252CM独立复算。CLOSED_FAILED_PREREGISTERED_IDENTITY_BENEFIT，不扩种/改倍率/概率/seed/父/预算救跑；源输入失败保留，As学习效用现已实测无足够净收益，Ag和完整联合目标仍未证明。当前无新的有源依据训练候选，不重复已失败路线。

2026-10-17：全源慢变包络新依据 20261017-diagnostic-F-envelopeProfile-fullL-manysig-m1-r01 已 ANALYZED／VERIFIED，执行commit 9aea3ef017db7491a903e460d59caf0642cd5e72。与旧常数gain/平稳谱FIR不同，固定全256→16×16点功率→包归一化/中心log→DCT1/2；全L6300/90格70包、18折排TXANDday但全部5RX保持。prediction/zeroMSE比1.163239792，只有3/18折改善；RX1/3/4/6/8=1.100670617/1.140431345/.765527377/1.349024850/1.186332689。6300包12600系数/90格/18折/5RX独立复算一致，固定测度失败关闭，不挑RX4或改mode/window/floor/阈值救跑。主体.746861秒，1cache/0CP/NN/新IQ/头/教师/更新/target，原进程自然结束。只否定当前稳定RX包络依据，不证明AGC不存在；0新身份成绩，Ag/As联合目标ACTIVE且尚无新Ag训练候选。
