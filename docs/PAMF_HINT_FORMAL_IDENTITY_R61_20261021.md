# R61：两个特征约束压缩版的固定无 F 身份诊断

状态：**VERIFIED / ANALYZED**。同168000物理记录×7场景，4候选加raw的140特征单元全部先于本诊断truth固定且有限。2016组TX/RX/day统计、336条件阴性对照、140配对差均独立NumPy复算；R46父及R53同宽控制1008条记录与R56完全一致；所属CPU控制器、提取与评分进程自然退出。

R60两行的全部七场景预测和独立评分完成后才读取保存IQ。复用R56raw/指定350521父/R53同宽控制84单元，新算两新版56单元；同原R34源LU8192固定校准，0F权重/embedding/logits、checkpoint、额外神经预测、训练、分类头拟合、新校准或GPU。不改变任何训练、权重、预测、排序或父交付；识别率和推理成本分别见R60原run报告。

表中margin为同TX余弦相似度减其他五TX均值，**不是识别准确率或逐细节真实性证明**。参考clean每TX/RX/day最多2条，按opaque physicalID排序；参考RX和day均不同，完整六TX单元等权。所有四解析组和全部场景完整保留，包含clean下降及个别TX弱/负关联。

|场景|候选|raw combined margin|恢复 combined margin|原clean参考|
|---|---|---:|---:|---:|
|clean|r46_350521|0.124015|0.109981|0.124015|
|clean|r53_prune_decay_control|0.124015|0.111223|0.124015|
|clean|r60_context_hint|0.124015|0.108534|0.124015|
|clean|r60_all_hint|0.124015|0.110381|0.124015|
|practical_high|r46_350521|0.005703|0.099789|0.124015|
|practical_high|r53_prune_decay_control|0.005703|0.103692|0.124015|
|practical_high|r60_context_hint|0.005703|0.102618|0.124015|
|practical_high|r60_all_hint|0.005703|0.101037|0.124015|
|practical_mid|r46_350521|0.003276|0.092008|0.124015|
|practical_mid|r53_prune_decay_control|0.003276|0.096168|0.124015|
|practical_mid|r60_context_hint|0.003276|0.095780|0.124015|
|practical_mid|r60_all_hint|0.003276|0.094029|0.124015|
|practical_low_suburban|r46_350521|0.002559|0.085436|0.124015|
|practical_low_suburban|r53_prune_decay_control|0.002559|0.090035|0.124015|
|practical_low_suburban|r60_context_hint|0.002559|0.090367|0.124015|
|practical_low_suburban|r60_all_hint|0.002559|0.088048|0.124015|
|practical_high_urban|r46_350521|0.005492|0.096866|0.124015|
|practical_high_urban|r53_prune_decay_control|0.005492|0.100641|0.124015|
|practical_high_urban|r60_context_hint|0.005492|0.099851|0.124015|
|practical_high_urban|r60_all_hint|0.005492|0.098208|0.124015|
|practical_mid_urban|r46_350521|0.002913|0.079533|0.124015|
|practical_mid_urban|r53_prune_decay_control|0.002913|0.083247|0.124015|
|practical_mid_urban|r60_context_hint|0.002913|0.083934|0.124015|
|practical_mid_urban|r60_all_hint|0.002913|0.081963|0.124015|
|practical_low_urban|r46_350521|0.002280|0.066143|0.124015|
|practical_low_urban|r53_prune_decay_control|0.002280|0.069657|0.124015|
|practical_low_urban|r60_context_hint|0.002280|0.071337|0.124015|
|practical_low_urban|r60_all_hint|0.002280|0.069157|0.124015|

mirror64、spectral4、reference24、combined92使用同一固定源均值/std下限0.05；combined按sqrt(3×组维数)缩放使平方误差三组等权。以下逐候选列出lowurban的四解析组，不以某一组涨分代替整体性能或TX真实性。

|lowurban候选|mirror margin|spectral margin|reference margin|combined margin|combined负或零margin的TX|
|---|---:|---:|---:|---:|---|
|r46_350521|0.081266|0.076193|0.006108|0.066143|2|
|r53_prune_decay_control|0.085995|0.078548|0.006213|0.069657|2|
|r60_context_hint|0.087254|0.083648|0.005805|0.071337|2|
|r60_all_hint|0.085243|0.080198|0.005388|0.069157|2|

逐TX、RX、day的完整正/负结果、有效/零向量/缺参考计数与配对clean描述量误差保留在JSON/CSV。以下置换对照不改变参考标签；199次固定RX/day内query标签置换为描述性阴性对照，不是p值或独立新样本。

|lowurban候选|combined实际margin|query标签置换均值|置换5%至95%|高于95%分位|
|---|---:|---:|---|---|
|r46_350521|0.066143|0.000019|-0.000321至0.000342|True|
|r53_prune_decay_control|0.069657|0.000018|-0.000286至0.000334|True|
|r60_context_hint|0.071337|0.000023|-0.000282至0.000336|True|
|r60_all_hint|0.069157|0.000017|-0.000298至0.000362|True|

|lowurban配对比较|共同有效数|combined margin变化|配对描述量误差变化|
|---|---:|---:|---:|
|r60_context_hint − r46_350521|168000|0.005194|-0.094823|
|r60_all_hint − r46_350521|168000|0.003014|-0.083651|
|r60_context_hint − r53_prune_decay_control|168000|0.001680|-0.024765|
|r60_all_hint − r53_prune_decay_control|168000|-0.000500|-0.013593|
|r60_all_hint − r60_context_hint|168000|-0.002180|0.011172|

解析提取阶段106.222s，评分阶段552.079s，控制器总墙钟2189.976s（含父完成等待）。2352000次新解析输入与84个旧引用来自同168000记录的相关场景/读出，不称新增独立样本。

真实2条Ls解析no-query smoke、36项聚焦检查、CLIhelp、一次限P0/P1审查，以及普通账户一次预检/归档SHA/compile/实际PID/CWD/argv/空CUDA/2线程/日志增长均核实。等待父时不持R41score.lock；父全部正式评分后提取与评分才使用唯一CPU评分槽。

固定描述符在 clean TX2 已有区分不足，恢复后负 margin 不能单独解释为输入身份错误。该类不足与各 TX 的全部结果一并保留。

本研究使用已暴露工程基准，且事后评价明确使用target clean参考和truth，不称新的独立盲测或完全无标签评价。跨RX/day与置换只提供固定坐标下有限同TX关联，仍可能含载荷、接收链和日期因素，不能证明每个生成细节来源、未知TX泛化或真实在轨有效性。原lowurban70%、其他五景距rawclean≤5pp与更强输入TX细节目标继续ACTIVE。

完整2016组记录、336条件对照和140配对差见[汇总](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-feature-hint-formal-analytic-TX-s2026092701-m2-r61/summary.json)与[逐组CSV](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-feature-hint-formal-analytic-TX-s2026092701-m2-r61/row_metrics.csv)；固定输入/方法/矩阵/命令/预算/停止规则见[登记](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-feature-hint-formal-analytic-TX-s2026092701-m2-r61/experiment.json)。
