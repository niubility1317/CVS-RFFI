# RXMIX源首折预登记

REGISTERED_NOT_RUN。精确row／command／environment／CWD／paths／owner／GPU／stop见experiment.json。

# RXMIX：只检验有界接收 I/Q 变化的独立 clean 价值

基线固定 F＋跨 RX 同 TX SupCon，SIG_TIME_X3／CENTER_25K／CENTER_UNITS、joint160、默认 cosine30 联合训练，从头E20=13L＋7U、4440更新、FINAL20。当前没有合格独立Ag；本候选不重开关闭的 DC、C4、频率响应 bank 或真实 partner 配方。源椭圆性统计仅通过先验资格：聚合风险降低27.486642%、12/18正折、RX对差p95=.032791289171，TX index1/3的全部6折为负；它不是硬件因果或身份保持证据。

唯一新增 IQ 算子为 `y=(x+b*conj(x))/sqrt(1+abs(b)**2)`，作用于同一批已生成 ordinary 接收 IQ 的第二对 RDrop；第一对保持原 ordinary，权重各.5，各部分独立归一化。每对内部两次前向用完全相同IQ，只改变dropout。四L前向、原L/U物理采样、一次ordinary生成、总损失尺度、默认头／F／U／ownEMA／旧门控／cos／原L-only SupCon保持匹配C0；SupCon只读第一对两次已有前向。此处不加入As、额外一致性、投影头或新部署模块。

六个幅度项由 fit RX3/4/6/8 的源椭圆性均值对差绝对值除2、上限截断为.02产生。只读已登记同合同统计的私有RX均值profile，不读取逐包统计、RX1 IQ或目标；源profile不提交Git。该换算是待验证的小扰动尺度，不是校准真实RX硬件的估计。虚拟session由physicalRX/day定义，所有TX共用同一复数b；每epoch刷新，幅度项按各session私有offset循环六项，方向由命名私有种子20261016921／epoch／RX／day生成，范围[-π,π)。不读取TX、loss、quality或预测来选参数，不推进全局RNG。训练、验证和测试流分离。

这个算子前向添加有界广义线性响应，不对原包拟合、扣除或反演未知TX失真，不等于替换成真实RX。|b|≤.02使已声明响应可逆，det>0；逐点能量倍率在(1-|b|)^2/(1+|b|²)至(1+|b|)^2/(1+|b|²)内。执行检查本算子的已知逆与能量界；这只能确认数值实现，不能证明TX身份标签保持。分母是响应系数的常数功率标度，不是重新做逐包RMS归一化。采样率25MHz、IQ256、原native RMS位置、边界、时延不改，不新加CFO／PA／DAC／噪声／多径。

只做一个源heldRX1候选：fit L5040／hidden U45360／seenV21600，heldRX1的L/U不训练，heldV5400仅FINAL20 clean＋六原practical。所有合法有标签样本参与原默认头监督。C0复用同seed／同合同／同前向的既有scratch E20结果，不加载其权重；新模型、optimizer和ownEMA从零。比较固定C0 clean83.444444%、low25.685185%，不和历史不同流、E50或E200混算。源V曾被多候选使用，如实保留探索背景，不能称独立盲测。

固定源判据：held clean提升≥1个百分点、macroF1不降、seenV下降≤.5个百分点、完整E20及健康检查通过。FAIL关闭固定RXMIX，不扫幅度／概率／比重／坐标或换RX/seed救援；PASS仅支持另登记其余源折确认，不自动进入目标、跨seed或Ag+As。六practical完整报告，不能用局部保持率或辅助loss代替整体识别。特别保留TX index1/3的表现，不能删类。

预算1新scratch E20／4440更新／3579018网络包（含ownCP无query smoke、原EMA和七源视图）；568320新增IQ算子曝光、240个epoch×RX/day虚拟session；C0原E20成本单列、父权重0。一个owner／不可变root、GPU遵守所有owner每卡最多3进程与≥12000MiB空闲；技术错误保留产物不自动重试，不动健康外部任务。训练后自动完成源clean＋六原practical，固定原fs/residual/post_sync/EQoff即G1／reference30／接收处理／正式IQ，不访问target。

精确row／commit／command／environment／CWD／paths／GPU／stop／artifacts见本run experiment.json。发布前仅聚焦协议负例和真实默认头合成前向验证、一次限定P0/P1、一个archive本远比较与remotecompile；own scratch实际checkpoint无query smoke是worker第一步，PASS立即继续。当前为实现候选，不是识别增益或完成正式实验的声明。

发布前 LOCAL_VERIFIED_NOT_RUN：1024合成网络包／2更新、16负例、206初态匹配、5120合成变换、240个epoch/RX/day session、公式／已知逆／能量界、第一对IQ逐值C0匹配、原头三路Sinc梯度、首对SupCon／四L前向／原EMA和全局RNG均PASS；隔离实际E20参数、真实C0和源统计metadata／污染拒绝／源角色隔离与私有profile上界fixture均PASS。唯一P0/P1只读审查PASS。0真实IQ／checkpoint／目标。N607发布时一次preflight／单archive比较／remotecompile；worker第一步own scratch checkpoint无query smoke，PASS继续，不需token。

实际发布 VERIFIED/RUNNING，训练代码772046d5758bbecf0a98d12542350f9862852eb6。唯一worker own scratchCP无query smoke PASS；E1的PID/CWD/argv/GPU/log增长、实际5040/45360/21600/5400源角色、原头/SIG3/Sinc更新、F物理clip、四L前向、原SupCon均VERIFIED。RXMIX实际幅度最大0.01642912927936461，按RX/day/epoch稳定，无TX参数选择、边界变化、逐包RMS、额外噪声或未知TX反演；运行时已知逆最大误差4.76837158203125e-07、能量界浮点误差2.86102294921875e-06，无执行failure。一个archive比较／remotecompile通过；原U/ownEMA不改，0target/As，FINAL性能待定。

离线读取器已实现：只读保存的source结果／metadata／私有源profile（仅4个fitRX统计值）、逐条重放session幅度与方向、全4440L／5994L-U／568320变换、全部20轮／F clip／SupCon／七视图原头分层。16元数据负例与384合成trace包PASS，0IQ／checkpoint／网络，未导入torch；运行release仍固定772046d5758bbecf0a98d12542350f9862852eb6，未热改。实际读取时仍需等待FINAL及全部七源视图，不把读取器合成验证作为识别结果。
