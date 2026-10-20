# R50：八个压缩优化新版的固定无 F 身份诊断

描述量边界补充：lowurban 的 TX 标签2，其原clean参考combined margin本身为-0.00130792，27.38万恢复后为-0.01134678。因此，这个解析坐标对该TX的clean可分性也有限；不能把负margin单独归因为恢复器生成了错误身份，也不能用整体正margin证明每个TX、每条记录或每个细节真实。固定四解析组/阈值/父/训练均不据此调整，所有原结果保留。

状态：**VERIFIED / ANALYZED**。同168000物理记录×7场景，9候选加raw的280特征单元全部先于本诊断truth固定且有限。4536组TX/RX/day统计、756条件阴性对照、420配对差均独立NumPy复算；R46父504条记录与R47完全一致；所属CPU控制器、提取与评分进程自然退出。

R48两版与R49六版的全部七场景预测和独立评分完成后才读取保存IQ。复用R47raw/指定350521父56单元，新算八新版224单元；同原R34源LU8192固定校准，0F权重/embedding/logits、checkpoint、额外神经预测、训练、分类头拟合、新校准或GPU。不改变任何训练、权重、预测、排序或父交付；识别率和推理成本分别见R48/R49原run报告。

表中margin为同TX余弦相似度减其他五TX均值，**不是识别准确率或逐细节真实性证明**。参考clean每TX/RX/day最多2条，按opaque physicalID排序；参考RX和day均不同，完整六TX单元等权。所有四解析组和全部场景完整保留，包含clean下降及个别TX弱/负关联。

|场景|候选|raw combined margin|恢复 combined margin|原clean参考|
|---|---|---:|---:|---:|
|clean|r46_350521|0.124015|0.109981|0.124015|
|clean|r48_retain_refine|0.124015|0.111506|0.124015|
|clean|r48_prune_refine|0.124015|0.111208|0.124015|
|clean|r49_retain_refine|0.124015|0.111972|0.124015|
|clean|r49_prune_near|0.124015|0.112731|0.124015|
|clean|r49_prune_balanced|0.124015|0.112073|0.124015|
|clean|r49_prune_small|0.124015|0.104534|0.124015|
|clean|r49_context_slim|0.124015|0.108852|0.124015|
|clean|r49_prior_slim|0.124015|0.112406|0.124015|
|practical_high|r46_350521|0.005703|0.099789|0.124015|
|practical_high|r48_retain_refine|0.005703|0.100945|0.124015|
|practical_high|r48_prune_refine|0.005703|0.104496|0.124015|
|practical_high|r49_retain_refine|0.005703|0.101455|0.124015|
|practical_high|r49_prune_near|0.005703|0.098143|0.124015|
|practical_high|r49_prune_balanced|0.005703|0.101584|0.124015|
|practical_high|r49_prune_small|0.005703|0.097079|0.124015|
|practical_high|r49_context_slim|0.005703|0.099003|0.124015|
|practical_high|r49_prior_slim|0.005703|0.103272|0.124015|
|practical_mid|r46_350521|0.003276|0.092008|0.124015|
|practical_mid|r48_retain_refine|0.003276|0.095073|0.124015|
|practical_mid|r48_prune_refine|0.003276|0.096443|0.124015|
|practical_mid|r49_retain_refine|0.003276|0.095441|0.124015|
|practical_mid|r49_prune_near|0.003276|0.092212|0.124015|
|practical_mid|r49_prune_balanced|0.003276|0.094282|0.124015|
|practical_mid|r49_prune_small|0.003276|0.090061|0.124015|
|practical_mid|r49_context_slim|0.003276|0.091948|0.124015|
|practical_mid|r49_prior_slim|0.003276|0.095415|0.124015|
|practical_low_suburban|r46_350521|0.002559|0.085436|0.124015|
|practical_low_suburban|r48_retain_refine|0.002559|0.090118|0.124015|
|practical_low_suburban|r48_prune_refine|0.002559|0.089897|0.124015|
|practical_low_suburban|r49_retain_refine|0.002559|0.090404|0.124015|
|practical_low_suburban|r49_prune_near|0.002559|0.087209|0.124015|
|practical_low_suburban|r49_prune_balanced|0.002559|0.088259|0.124015|
|practical_low_suburban|r49_prune_small|0.002559|0.084024|0.124015|
|practical_low_suburban|r49_context_slim|0.002559|0.085788|0.124015|
|practical_low_suburban|r49_prior_slim|0.002559|0.089138|0.124015|
|practical_high_urban|r46_350521|0.005492|0.096866|0.124015|
|practical_high_urban|r48_retain_refine|0.005492|0.098569|0.124015|
|practical_high_urban|r48_prune_refine|0.005492|0.101351|0.124015|
|practical_high_urban|r49_retain_refine|0.005492|0.098998|0.124015|
|practical_high_urban|r49_prune_near|0.005492|0.095805|0.124015|
|practical_high_urban|r49_prune_balanced|0.005492|0.098959|0.124015|
|practical_high_urban|r49_prune_small|0.005492|0.094538|0.124015|
|practical_high_urban|r49_context_slim|0.005492|0.096378|0.124015|
|practical_high_urban|r49_prior_slim|0.005492|0.100116|0.124015|
|practical_mid_urban|r46_350521|0.002913|0.079533|0.124015|
|practical_mid_urban|r48_retain_refine|0.002913|0.084826|0.124015|
|practical_mid_urban|r48_prune_refine|0.002913|0.083264|0.124015|
|practical_mid_urban|r49_retain_refine|0.002913|0.084843|0.124015|
|practical_mid_urban|r49_prune_near|0.002913|0.082152|0.124015|
|practical_mid_urban|r49_prune_balanced|0.002913|0.083101|0.124015|
|practical_mid_urban|r49_prune_small|0.002913|0.079582|0.124015|
|practical_mid_urban|r49_context_slim|0.002913|0.081078|0.124015|
|practical_mid_urban|r49_prior_slim|0.002913|0.083154|0.124015|
|practical_low_urban|r46_350521|0.002280|0.066143|0.124015|
|practical_low_urban|r48_retain_refine|0.002280|0.072945|0.124015|
|practical_low_urban|r48_prune_refine|0.002280|0.069414|0.124015|
|practical_low_urban|r49_retain_refine|0.002280|0.072836|0.124015|
|practical_low_urban|r49_prune_near|0.002280|0.070475|0.124015|
|practical_low_urban|r49_prune_balanced|0.002280|0.070225|0.124015|
|practical_low_urban|r49_prune_small|0.002280|0.067651|0.124015|
|practical_low_urban|r49_context_slim|0.002280|0.068582|0.124015|
|practical_low_urban|r49_prior_slim|0.002280|0.070278|0.124015|

mirror64、spectral4、reference24、combined92使用同一固定源均值/std下限0.05；combined按sqrt(3×组维数)缩放使平方误差三组等权。以下逐候选列出lowurban的四解析组，不以某一组涨分代替整体性能或TX真实性。

|lowurban候选|mirror margin|spectral margin|reference margin|combined margin|combined负或零margin的TX|
|---|---:|---:|---:|---:|---|
|r46_350521|0.081266|0.076193|0.006108|0.066143|2|
|r48_retain_refine|0.090401|0.076930|0.006597|0.072945|2|
|r48_prune_refine|0.085541|0.079903|0.006075|0.069414|2|
|r49_retain_refine|0.090010|0.077143|0.006538|0.072836|2|
|r49_prune_near|0.086848|0.080347|0.006919|0.070475|2|
|r49_prune_balanced|0.086203|0.078218|0.006704|0.070225|2|
|r49_prune_small|0.082681|0.080799|0.005428|0.067651|2|
|r49_context_slim|0.084916|0.076876|0.006628|0.068582|2|
|r49_prior_slim|0.087944|0.070166|0.006607|0.070278|2|

逐TX、RX、day的完整正/负结果、有效/零向量/缺参考计数与配对clean描述量误差保留在JSON/CSV。以下置换对照不改变参考标签；199次固定RX/day内query标签置换为描述性阴性对照，不是p值或独立新样本。

|lowurban候选|combined实际margin|query标签置换均值|置换5%至95%|高于95%分位|
|---|---:|---:|---|---|
|r46_350521|0.066143|0.000019|-0.000321至0.000342|True|
|r48_retain_refine|0.072945|0.000019|-0.000324至0.000350|True|
|r48_prune_refine|0.069414|0.000021|-0.000275至0.000339|True|
|r49_retain_refine|0.072836|0.000021|-0.000307至0.000360|True|
|r49_prune_near|0.070475|0.000023|-0.000312至0.000373|True|
|r49_prune_balanced|0.070225|0.000022|-0.000289至0.000356|True|
|r49_prune_small|0.067651|0.000024|-0.000284至0.000337|True|
|r49_context_slim|0.068582|0.000020|-0.000315至0.000354|True|
|r49_prior_slim|0.070278|0.000025|-0.000275至0.000359|True|

|lowurban配对比较|共同有效数|combined margin变化|配对描述量误差变化|
|---|---:|---:|---:|
|r48_retain_refine − r46_350521|168000|0.006802|-0.120709|
|r48_prune_refine − r46_350521|168000|0.003271|-0.093340|
|r49_retain_refine − r46_350521|168000|0.006693|-0.123350|
|r49_prune_near − r46_350521|168000|0.004331|-0.113726|
|r49_prune_balanced − r46_350521|168000|0.004082|-0.105828|
|r49_prune_small − r46_350521|168000|0.001508|-0.038052|
|r49_context_slim − r46_350521|168000|0.002438|-0.103363|
|r49_prior_slim − r46_350521|168000|0.004135|-0.098638|
|r48_prune_refine − r48_retain_refine|168000|-0.003531|0.027368|
|r49_prune_near − r49_retain_refine|168000|-0.002361|0.009623|
|r49_prune_balanced − r49_retain_refine|168000|-0.002611|0.017522|
|r49_prune_small − r49_retain_refine|168000|-0.005185|0.085297|
|r49_context_slim − r49_retain_refine|168000|-0.004255|0.019986|
|r49_prior_slim − r49_retain_refine|168000|-0.002558|0.024711|
|r49_retain_refine − r48_retain_refine|168000|-0.000109|-0.002641|

解析提取阶段404.917s，评分阶段1142.196s，控制器总墙钟3619.869s（含父完成等待）。9408000次新解析输入与56个旧引用来自同168000记录的相关场景/读出，不称新增独立样本。

真实2条Ls解析no-query smoke、22项聚焦检查、CLIhelp、一次限P0/P1审查，以及普通账户一次预检/归档SHA/compile/实际PID/CWD/argv/空CUDA/2线程/日志增长均核实。等待父时不持R41score.lock；父全部正式评分后提取与评分才使用唯一CPU评分槽。

本研究使用已暴露工程基准，且事后评价明确使用target clean参考和truth，不称新的独立盲测或完全无标签评价。跨RX/day与置换只提供固定坐标下有限同TX关联，仍可能含载荷、接收链和日期因素，不能证明每个生成细节来源、未知TX泛化或真实在轨有效性。原lowurban70%、其他五景距rawclean≤5pp与更强输入TX细节目标继续ACTIVE。

完整4536组记录、756条件对照和420配对差见[汇总](../automation_reports/CV-SincNet/20261020-diagnostic-restoration-multipoint-formal-analytic-TX-s2026092701-m9-r50/summary.json)与[逐组CSV](../automation_reports/CV-SincNet/20261020-diagnostic-restoration-multipoint-formal-analytic-TX-s2026092701-m9-r50/row_metrics.csv)；固定输入/方法/矩阵/命令/预算/停止规则见[登记](../automation_reports/CV-SincNet/20261020-diagnostic-restoration-multipoint-formal-analytic-TX-s2026092701-m9-r50/experiment.json)。
