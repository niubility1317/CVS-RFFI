# R56：七个压缩优化新版的固定无 F 身份诊断

状态：**VERIFIED / ANALYZED**。同168000物理记录×7场景，9候选加raw的280特征单元全部先于本诊断truth固定且有限。4536组TX/RX/day统计、756条件阴性对照、392配对差均独立NumPy复算；R46父及R49同宽控制1008条记录与R50完全一致；所属CPU控制器、提取与评分进程自然退出。

R53三版、R54两版与R55两版的全部七场景预测和独立评分完成后才读取保存IQ。复用R50raw/指定350521父/R49同宽控制84单元，新算七新版196单元；同原R34源LU8192固定校准，0F权重/embedding/logits、checkpoint、额外神经预测、训练、分类头拟合、新校准或GPU。不改变任何训练、权重、预测、排序或父交付；识别率和推理成本分别见R53/R54/R55原run报告。

表中margin为同TX余弦相似度减其他五TX均值，**不是识别准确率或逐细节真实性证明**。参考clean每TX/RX/day最多2条，按opaque physicalID排序；参考RX和day均不同，完整六TX单元等权。所有四解析组和全部场景完整保留，包含clean下降及个别TX弱/负关联。

|场景|候选|raw combined margin|恢复 combined margin|原clean参考|
|---|---|---:|---:|---:|
|clean|r46_350521|0.124015|0.109981|0.124015|
|clean|r49_retain_control|0.124015|0.111972|0.124015|
|clean|r53_prune_decay|0.124015|0.111223|0.124015|
|clean|r53_retain_anchor|0.124015|0.108700|0.124015|
|clean|r53_prune_anchor|0.124015|0.110354|0.124015|
|clean|r54_activation_energy|0.124015|0.110684|0.124015|
|clean|r54_projection_sensitivity|0.124015|0.110889|0.124015|
|clean|r55_fold_full|0.124015|0.112317|0.124015|
|clean|r55_fold_half|0.124015|0.109942|0.124015|
|practical_high|r46_350521|0.005703|0.099789|0.124015|
|practical_high|r49_retain_control|0.005703|0.101455|0.124015|
|practical_high|r53_prune_decay|0.005703|0.103692|0.124015|
|practical_high|r53_retain_anchor|0.005703|0.102556|0.124015|
|practical_high|r53_prune_anchor|0.005703|0.104142|0.124015|
|practical_high|r54_activation_energy|0.005703|0.094823|0.124015|
|practical_high|r54_projection_sensitivity|0.005703|0.101398|0.124015|
|practical_high|r55_fold_full|0.005703|0.097877|0.124015|
|practical_high|r55_fold_half|0.005703|0.098284|0.124015|
|practical_mid|r46_350521|0.003276|0.092008|0.124015|
|practical_mid|r49_retain_control|0.003276|0.095441|0.124015|
|practical_mid|r53_prune_decay|0.003276|0.096168|0.124015|
|practical_mid|r53_retain_anchor|0.003276|0.095862|0.124015|
|practical_mid|r53_prune_anchor|0.003276|0.095995|0.124015|
|practical_mid|r54_activation_energy|0.003276|0.088492|0.124015|
|practical_mid|r54_projection_sensitivity|0.003276|0.094752|0.124015|
|practical_mid|r55_fold_full|0.003276|0.090452|0.124015|
|practical_mid|r55_fold_half|0.003276|0.091193|0.124015|
|practical_low_suburban|r46_350521|0.002559|0.085436|0.124015|
|practical_low_suburban|r49_retain_control|0.002559|0.090404|0.124015|
|practical_low_suburban|r53_prune_decay|0.002559|0.090035|0.124015|
|practical_low_suburban|r53_retain_anchor|0.002559|0.090311|0.124015|
|practical_low_suburban|r53_prune_anchor|0.002559|0.089565|0.124015|
|practical_low_suburban|r54_activation_energy|0.002559|0.083320|0.124015|
|practical_low_suburban|r54_projection_sensitivity|0.002559|0.089665|0.124015|
|practical_low_suburban|r55_fold_full|0.002559|0.084139|0.124015|
|practical_low_suburban|r55_fold_half|0.002559|0.085795|0.124015|
|practical_high_urban|r46_350521|0.005492|0.096866|0.124015|
|practical_high_urban|r49_retain_control|0.005492|0.098998|0.124015|
|practical_high_urban|r53_prune_decay|0.005492|0.100641|0.124015|
|practical_high_urban|r53_retain_anchor|0.005492|0.100142|0.124015|
|practical_high_urban|r53_prune_anchor|0.005492|0.101277|0.124015|
|practical_high_urban|r54_activation_energy|0.005492|0.091298|0.124015|
|practical_high_urban|r54_projection_sensitivity|0.005492|0.098842|0.124015|
|practical_high_urban|r55_fold_full|0.005492|0.095009|0.124015|
|practical_high_urban|r55_fold_half|0.005492|0.095703|0.124015|
|practical_mid_urban|r46_350521|0.002913|0.079533|0.124015|
|practical_mid_urban|r49_retain_control|0.002913|0.084843|0.124015|
|practical_mid_urban|r53_prune_decay|0.002913|0.083247|0.124015|
|practical_mid_urban|r53_retain_anchor|0.002913|0.085282|0.124015|
|practical_mid_urban|r53_prune_anchor|0.002913|0.083907|0.124015|
|practical_mid_urban|r54_activation_energy|0.002913|0.074889|0.124015|
|practical_mid_urban|r54_projection_sensitivity|0.002913|0.083618|0.124015|
|practical_mid_urban|r55_fold_full|0.002913|0.078421|0.124015|
|practical_mid_urban|r55_fold_half|0.002913|0.080555|0.124015|
|practical_low_urban|r46_350521|0.002280|0.066143|0.124015|
|practical_low_urban|r49_retain_control|0.002280|0.072836|0.124015|
|practical_low_urban|r53_prune_decay|0.002280|0.069657|0.124015|
|practical_low_urban|r53_retain_anchor|0.002280|0.072679|0.124015|
|practical_low_urban|r53_prune_anchor|0.002280|0.070362|0.124015|
|practical_low_urban|r54_activation_energy|0.002280|0.062617|0.124015|
|practical_low_urban|r54_projection_sensitivity|0.002280|0.071118|0.124015|
|practical_low_urban|r55_fold_full|0.002280|0.065384|0.124015|
|practical_low_urban|r55_fold_half|0.002280|0.068739|0.124015|

mirror64、spectral4、reference24、combined92使用同一固定源均值/std下限0.05；combined按sqrt(3×组维数)缩放使平方误差三组等权。以下逐候选列出lowurban的四解析组，不以某一组涨分代替整体性能或TX真实性。

|lowurban候选|mirror margin|spectral margin|reference margin|combined margin|combined负或零margin的TX|
|---|---:|---:|---:|---:|---|
|r46_350521|0.081266|0.076193|0.006108|0.066143|2|
|r49_retain_control|0.090010|0.077143|0.006538|0.072836|2|
|r53_prune_decay|0.085995|0.078548|0.006213|0.069657|2|
|r53_retain_anchor|0.089820|0.074681|0.006473|0.072679|2|
|r53_prune_anchor|0.086938|0.079011|0.005715|0.070362|2|
|r54_activation_energy|0.078118|0.081350|0.005500|0.062617|2|
|r54_projection_sensitivity|0.086735|0.077186|0.005948|0.071118|2|
|r55_fold_full|0.080877|0.073526|0.009223|0.065384|2|
|r55_fold_half|0.084868|0.083706|0.007263|0.068739|2|

逐TX、RX、day的完整正/负结果、有效/零向量/缺参考计数与配对clean描述量误差保留在JSON/CSV。以下置换对照不改变参考标签；199次固定RX/day内query标签置换为描述性阴性对照，不是p值或独立新样本。

|lowurban候选|combined实际margin|query标签置换均值|置换5%至95%|高于95%分位|
|---|---:|---:|---|---|
|r46_350521|0.066143|0.000019|-0.000321至0.000342|True|
|r49_retain_control|0.072836|0.000021|-0.000307至0.000360|True|
|r53_prune_decay|0.069657|0.000018|-0.000286至0.000334|True|
|r53_retain_anchor|0.072679|0.000022|-0.000315至0.000378|True|
|r53_prune_anchor|0.070362|0.000023|-0.000308至0.000349|True|
|r54_activation_energy|0.062617|0.000012|-0.000271至0.000277|True|
|r54_projection_sensitivity|0.071118|0.000018|-0.000296至0.000343|True|
|r55_fold_full|0.065384|0.000027|-0.000307至0.000341|True|
|r55_fold_half|0.068739|0.000024|-0.000283至0.000332|True|

|lowurban配对比较|共同有效数|combined margin变化|配对描述量误差变化|
|---|---:|---:|---:|
|r53_prune_decay − r46_350521|168000|0.003514|-0.070059|
|r53_retain_anchor − r46_350521|168000|0.006536|-0.107146|
|r53_prune_anchor − r46_350521|168000|0.004219|-0.117019|
|r54_activation_energy − r46_350521|168000|-0.003526|-0.064467|
|r54_projection_sensitivity − r46_350521|168000|0.004975|-0.105786|
|r55_fold_full − r46_350521|168000|-0.000759|-0.089576|
|r55_fold_half − r46_350521|168000|0.002595|-0.067962|
|r53_retain_anchor − r49_retain_control|168000|-0.000157|0.016204|
|r53_prune_anchor − r53_prune_decay|168000|0.000706|-0.046960|
|r54_activation_energy − r53_prune_decay|168000|-0.007039|0.005592|
|r54_projection_sensitivity − r53_prune_decay|168000|0.001461|-0.035727|
|r55_fold_full − r53_prune_decay|168000|-0.004273|-0.019517|
|r55_fold_half − r53_prune_decay|168000|-0.000918|0.002097|
|r55_fold_full − r55_fold_half|168000|-0.003355|-0.021614|

解析提取阶段360.351s，评分阶段1331.131s，控制器总墙钟2653.148s（含父完成等待）。8232000次新解析输入与84个旧引用来自同168000记录的相关场景/读出，不称新增独立样本。

真实2条Ls解析no-query smoke、22项聚焦检查、CLIhelp、一次限P0/P1审查，以及普通账户一次预检/归档SHA/compile/实际PID/CWD/argv/空CUDA/2线程/日志增长均核实。等待父时不持R41score.lock；父全部正式评分后提取与评分才使用唯一CPU评分槽。

固定描述符在 clean TX2 已有区分不足，恢复后负 margin 不能单独解释为输入身份错误。该类不足与各 TX 的全部结果一并保留。

本研究使用已暴露工程基准，且事后评价明确使用target clean参考和truth，不称新的独立盲测或完全无标签评价。跨RX/day与置换只提供固定坐标下有限同TX关联，仍可能含载荷、接收链和日期因素，不能证明每个生成细节来源、未知TX泛化或真实在轨有效性。原lowurban70%、其他五景距rawclean≤5pp与更强输入TX细节目标继续ACTIVE。

完整4536组记录、756条件对照和392配对差见[汇总](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-compression-formal-analytic-TX-s2026092701-m7-r56/summary.json)与[逐组CSV](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-compression-formal-analytic-TX-s2026092701-m7-r56/row_metrics.csv)；固定输入/方法/矩阵/命令/预算/停止规则见[登记](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-compression-formal-analytic-TX-s2026092701-m7-r56/experiment.json)。
