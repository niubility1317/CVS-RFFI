# R63：整组 GN3/GN1 压缩版的固定无 F 身份诊断

状态：**VERIFIED / ANALYZED**。同168000物理记录×7场景，4候选加raw的140特征单元全部先于本诊断truth固定且有限。2016组TX/RX/day统计、336条件阴性对照、140配对差均独立NumPy复算；R46父504条记录与R56完全一致；所属CPU控制器、提取与评分进程自然退出。

R62两行的全部七场景预测和独立评分完成后才读取保存IQ。复用R56raw/指定350521父56单元，新算R49同增量预算控制及两新版84单元；同原R34源LU8192固定校准，0F权重/embedding/logits、checkpoint、额外神经预测、训练、分类头拟合、新校准或GPU。不改变任何训练、权重、预测、排序或父交付；识别率和推理成本分别见R62原run报告。

表中margin为同TX余弦相似度减其他五TX均值，**不是识别准确率或逐细节真实性证明**。参考clean每TX/RX/day最多2条，按opaque physicalID排序；参考RX和day均不同，完整六TX单元等权。所有四解析组和全部场景完整保留，包含clean下降及个别TX弱/负关联。

|场景|候选|raw combined margin|恢复 combined margin|原clean参考|
|---|---|---:|---:|---:|
|clean|r46_350521|0.124015|0.109981|0.124015|
|clean|r49_retain_control|0.124015|0.111972|0.124015|
|clean|r62_group_preserve|0.124015|0.102967|0.124015|
|clean|r62_regroup_control|0.124015|0.101259|0.124015|
|practical_high|r46_350521|0.005703|0.099789|0.124015|
|practical_high|r49_retain_control|0.005703|0.101455|0.124015|
|practical_high|r62_group_preserve|0.005703|0.094528|0.124015|
|practical_high|r62_regroup_control|0.005703|0.093323|0.124015|
|practical_mid|r46_350521|0.003276|0.092008|0.124015|
|practical_mid|r49_retain_control|0.003276|0.095441|0.124015|
|practical_mid|r62_group_preserve|0.003276|0.088163|0.124015|
|practical_mid|r62_regroup_control|0.003276|0.086951|0.124015|
|practical_low_suburban|r46_350521|0.002559|0.085436|0.124015|
|practical_low_suburban|r49_retain_control|0.002559|0.090404|0.124015|
|practical_low_suburban|r62_group_preserve|0.002559|0.082626|0.124015|
|practical_low_suburban|r62_regroup_control|0.002559|0.081237|0.124015|
|practical_high_urban|r46_350521|0.005492|0.096866|0.124015|
|practical_high_urban|r49_retain_control|0.005492|0.098998|0.124015|
|practical_high_urban|r62_group_preserve|0.005492|0.092684|0.124015|
|practical_high_urban|r62_regroup_control|0.005492|0.091135|0.124015|
|practical_mid_urban|r46_350521|0.002913|0.079533|0.124015|
|practical_mid_urban|r49_retain_control|0.002913|0.084843|0.124015|
|practical_mid_urban|r62_group_preserve|0.002913|0.079570|0.124015|
|practical_mid_urban|r62_regroup_control|0.002913|0.077420|0.124015|
|practical_low_urban|r46_350521|0.002280|0.066143|0.124015|
|practical_low_urban|r49_retain_control|0.002280|0.072836|0.124015|
|practical_low_urban|r62_group_preserve|0.002280|0.067693|0.124015|
|practical_low_urban|r62_regroup_control|0.002280|0.065654|0.124015|

mirror64、spectral4、reference24、combined92使用同一固定源均值/std下限0.05；combined按sqrt(3×组维数)缩放使平方误差三组等权。以下逐候选列出lowurban的四解析组，不以某一组涨分代替整体性能或TX真实性。

|lowurban候选|mirror margin|spectral margin|reference margin|combined margin|combined负或零margin的TX|
|---|---:|---:|---:|---:|---|
|r46_350521|0.081266|0.076193|0.006108|0.066143|2|
|r49_retain_control|0.090010|0.077143|0.006538|0.072836|2|
|r62_group_preserve|0.084828|0.074048|0.006107|0.067693|2|
|r62_regroup_control|0.082410|0.075594|0.005971|0.065654|2|

逐TX、RX、day的完整正/负结果、有效/零向量/缺参考计数与配对clean描述量误差保留在JSON/CSV。以下置换对照不改变参考标签；199次固定RX/day内query标签置换为描述性阴性对照，不是p值或独立新样本。

|lowurban候选|combined实际margin|query标签置换均值|置换5%至95%|高于95%分位|
|---|---:|---:|---|---|
|r46_350521|0.066143|0.000019|-0.000321至0.000342|True|
|r49_retain_control|0.072836|0.000021|-0.000307至0.000360|True|
|r62_group_preserve|0.067693|0.000026|-0.000282至0.000374|True|
|r62_regroup_control|0.065654|0.000023|-0.000291至0.000357|True|

|lowurban配对比较|共同有效数|combined margin变化|配对描述量误差变化|
|---|---:|---:|---:|
|r62_group_preserve − r46_350521|168000|0.001549|-0.062731|
|r62_regroup_control − r46_350521|168000|-0.000489|-0.062075|
|r62_group_preserve − r49_retain_control|168000|-0.005144|0.060619|
|r62_regroup_control − r49_retain_control|168000|-0.007182|0.061274|
|r62_group_preserve − r62_regroup_control|168000|0.002039|-0.000656|

解析提取阶段153.518s，评分阶段570.688s，控制器总墙钟2496.198s（含父完成等待）。3528000次新解析输入与56个旧引用来自同168000记录的相关场景/读出，不称新增独立样本。

真实2条Ls解析no-query smoke、30项聚焦检查、CLIhelp、一次限P0/P1审查，以及普通账户一次预检/归档SHA/compile/实际PID/CWD/argv/空CUDA/2线程/日志增长均核实。等待父时不持R41score.lock；父全部正式评分后提取与评分才使用唯一CPU评分槽。

固定描述符在 clean TX2 已有区分不足，恢复后负 margin 不能单独解释为输入身份错误。该类不足与各 TX 的全部结果一并保留。

本研究使用已暴露工程基准，且事后评价明确使用target clean参考和truth，不称新的独立盲测或完全无标签评价。跨RX/day与置换只提供固定坐标下有限同TX关联，仍可能含载荷、接收链和日期因素，不能证明每个生成细节来源、未知TX泛化或真实在轨有效性。原lowurban70%、其他五景距rawclean≤5pp与更强输入TX细节目标继续ACTIVE。

完整2016组记录、336条件对照和140配对差见[汇总](summary.json)与[逐组CSV](row_metrics.csv)；固定输入/方法/矩阵/命令/预算/停止规则见[登记](experiment.json)。
