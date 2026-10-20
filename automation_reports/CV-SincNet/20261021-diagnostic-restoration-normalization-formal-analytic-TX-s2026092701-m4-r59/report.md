# R59：四个归一化压缩版的固定无 F 身份诊断

状态：**VERIFIED / ANALYZED**。同168000物理记录×7场景，6候选加raw的196特征单元全部先于本诊断truth固定且有限。3024组TX/RX/day统计、504条件阴性对照、280配对差均独立NumPy复算；R46父及R55旧归一化参考1008条记录与R56完全一致；所属CPU控制器、提取与评分进程自然退出。

R57两行与R58两行的全部七场景预测和独立评分完成后才读取保存IQ。复用R56raw/指定350521父/R55旧归一化参考84单元，新算四新版112单元；同原R34源LU8192固定校准，0F权重/embedding/logits、checkpoint、额外神经预测、训练、分类头拟合、新校准或GPU。不改变任何训练、权重、预测、排序或父交付；识别率和推理成本分别见R57/R58原run报告。

表中margin为同TX余弦相似度减其他五TX均值，**不是识别准确率或逐细节真实性证明**。参考clean每TX/RX/day最多2条，按opaque physicalID排序；参考RX和day均不同，完整六TX单元等权。所有四解析组和全部场景完整保留，包含clean下降及个别TX弱/负关联。

|场景|候选|raw combined margin|恢复 combined margin|原clean参考|
|---|---|---:|---:|---:|
|clean|r46_350521|0.124015|0.109981|0.124015|
|clean|r55_fold_full_reference|0.124015|0.112317|0.124015|
|clean|r57_gn_full|0.124015|0.109267|0.124015|
|clean|r57_gn_half|0.124015|0.110780|0.124015|
|clean|r58_calibrated|0.124015|0.110768|0.124015|
|clean|r58_identity_control|0.124015|0.110786|0.124015|
|practical_high|r46_350521|0.005703|0.099789|0.124015|
|practical_high|r55_fold_full_reference|0.005703|0.097877|0.124015|
|practical_high|r57_gn_full|0.005703|0.105087|0.124015|
|practical_high|r57_gn_half|0.005703|0.099437|0.124015|
|practical_high|r58_calibrated|0.005703|0.104170|0.124015|
|practical_high|r58_identity_control|0.005703|0.097102|0.124015|
|practical_mid|r46_350521|0.003276|0.092008|0.124015|
|practical_mid|r55_fold_full_reference|0.003276|0.090452|0.124015|
|practical_mid|r57_gn_full|0.003276|0.098789|0.124015|
|practical_mid|r57_gn_half|0.003276|0.092179|0.124015|
|practical_mid|r58_calibrated|0.003276|0.094670|0.124015|
|practical_mid|r58_identity_control|0.003276|0.089517|0.124015|
|practical_low_suburban|r46_350521|0.002559|0.085436|0.124015|
|practical_low_suburban|r55_fold_full_reference|0.002559|0.084139|0.124015|
|practical_low_suburban|r57_gn_full|0.002559|0.093409|0.124015|
|practical_low_suburban|r57_gn_half|0.002559|0.086164|0.124015|
|practical_low_suburban|r58_calibrated|0.002559|0.087717|0.124015|
|practical_low_suburban|r58_identity_control|0.002559|0.083265|0.124015|
|practical_high_urban|r46_350521|0.005492|0.096866|0.124015|
|practical_high_urban|r55_fold_full_reference|0.005492|0.095009|0.124015|
|practical_high_urban|r57_gn_full|0.005492|0.102539|0.124015|
|practical_high_urban|r57_gn_half|0.005492|0.096932|0.124015|
|practical_high_urban|r58_calibrated|0.005492|0.101395|0.124015|
|practical_high_urban|r58_identity_control|0.005492|0.094323|0.124015|
|practical_mid_urban|r46_350521|0.002913|0.079533|0.124015|
|practical_mid_urban|r55_fold_full_reference|0.002913|0.078421|0.124015|
|practical_mid_urban|r57_gn_full|0.002913|0.087003|0.124015|
|practical_mid_urban|r57_gn_half|0.002913|0.081271|0.124015|
|practical_mid_urban|r58_calibrated|0.002913|0.083602|0.124015|
|practical_mid_urban|r58_identity_control|0.002913|0.077923|0.124015|
|practical_low_urban|r46_350521|0.002280|0.066143|0.124015|
|practical_low_urban|r55_fold_full_reference|0.002280|0.065384|0.124015|
|practical_low_urban|r57_gn_full|0.002280|0.073711|0.124015|
|practical_low_urban|r57_gn_half|0.002280|0.068439|0.124015|
|practical_low_urban|r58_calibrated|0.002280|0.070331|0.124015|
|practical_low_urban|r58_identity_control|0.002280|0.065194|0.124015|

mirror64、spectral4、reference24、combined92使用同一固定源均值/std下限0.05；combined按sqrt(3×组维数)缩放使平方误差三组等权。以下逐候选列出lowurban的四解析组，不以某一组涨分代替整体性能或TX真实性。

|lowurban候选|mirror margin|spectral margin|reference margin|combined margin|combined负或零margin的TX|
|---|---:|---:|---:|---:|---|
|r46_350521|0.081266|0.076193|0.006108|0.066143|2|
|r55_fold_full_reference|0.080877|0.073526|0.009223|0.065384|2|
|r57_gn_full|0.090929|0.078389|0.007407|0.073711|2|
|r57_gn_half|0.083834|0.079266|0.007833|0.068439|2|
|r58_calibrated|0.084826|0.087304|0.006460|0.070331|2|
|r58_identity_control|0.080832|0.074185|0.009048|0.065194|2|

逐TX、RX、day的完整正/负结果、有效/零向量/缺参考计数与配对clean描述量误差保留在JSON/CSV。以下置换对照不改变参考标签；199次固定RX/day内query标签置换为描述性阴性对照，不是p值或独立新样本。

|lowurban候选|combined实际margin|query标签置换均值|置换5%至95%|高于95%分位|
|---|---:|---:|---|---|
|r46_350521|0.066143|0.000019|-0.000321至0.000342|True|
|r55_fold_full_reference|0.065384|0.000027|-0.000307至0.000341|True|
|r57_gn_full|0.073711|0.000024|-0.000289至0.000356|True|
|r57_gn_half|0.068439|0.000029|-0.000295至0.000328|True|
|r58_calibrated|0.070331|0.000021|-0.000308至0.000343|True|
|r58_identity_control|0.065194|0.000029|-0.000290至0.000339|True|

|lowurban配对比较|共同有效数|combined margin变化|配对描述量误差变化|
|---|---:|---:|---:|
|r57_gn_full − r46_350521|168000|0.007567|-0.185848|
|r57_gn_half − r46_350521|168000|0.002296|-0.114584|
|r58_calibrated − r46_350521|168000|0.004187|-0.024801|
|r58_identity_control − r46_350521|168000|-0.000950|-0.069808|
|r57_gn_full − r55_fold_full_reference|168000|0.008327|-0.096272|
|r57_gn_half − r55_fold_full_reference|168000|0.003055|-0.025008|
|r58_calibrated − r55_fold_full_reference|168000|0.004947|0.064775|
|r58_identity_control − r55_fold_full_reference|168000|-0.000190|0.019768|
|r57_gn_full − r57_gn_half|168000|0.005272|-0.071264|
|r58_calibrated − r58_identity_control|168000|0.005137|0.045007|

解析提取阶段199.677s，评分阶段902.279s，控制器总墙钟4405.513s（含父完成等待）。4704000次新解析输入与84个旧引用来自同168000记录的相关场景/读出，不称新增独立样本。

真实2条Ls解析no-query smoke、26项聚焦检查、CLIhelp、一次限P0/P1审查，以及普通账户一次预检/归档SHA/compile/实际PID/CWD/argv/空CUDA/2线程/日志增长均核实。等待父时不持R41score.lock；父全部正式评分后提取与评分才使用唯一CPU评分槽。

固定描述符在 clean TX2 已有区分不足，恢复后负 margin 不能单独解释为输入身份错误。该类不足与各 TX 的全部结果一并保留。

本研究使用已暴露工程基准，且事后评价明确使用target clean参考和truth，不称新的独立盲测或完全无标签评价。跨RX/day与置换只提供固定坐标下有限同TX关联，仍可能含载荷、接收链和日期因素，不能证明每个生成细节来源、未知TX泛化或真实在轨有效性。原lowurban70%、其他五景距rawclean≤5pp与更强输入TX细节目标继续ACTIVE。

完整3024组记录、504条件对照和280配对差见[汇总](summary.json)与[逐组CSV](row_metrics.csv)；固定输入/方法/矩阵/命令/预算/停止规则见[登记](experiment.json)。
