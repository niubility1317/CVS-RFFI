# R67：折叠、延迟裁剪及同轨迹平均版的固定无 F 身份诊断

状态：**VERIFIED / ANALYZED**。同168000物理记录×7场景，9候选加raw的280特征单元全部先于本诊断truth固定且有限。4536组TX/RX/day统计、756条件阴性对照、504配对差均独立NumPy复算；M35/R53共1008条记录与R56完全一致；所属CPU控制器、提取与评分进程自然退出。

R64/R65/R66七个固定读出的全部七场景预测和独立评分完成后才读取保存IQ。复用R56raw/M35/R53共84单元，新算七新版196单元；同原R34源LU8192固定校准，0F权重/embedding/logits、checkpoint、额外神经预测、训练、分类头拟合、新校准或GPU。不改变任何训练、权重、预测、排序或父交付；识别率和推理成本见各父run报告。R66三读出共享一次源轨迹，不能当作三次训练；本诊断无新增训练成本。

表中margin为同TX余弦相似度减其他五TX均值，**不是识别准确率或逐细节真实性证明**。参考clean每TX/RX/day最多2条，按opaque physicalID排序；参考RX和day均不同，完整六TX单元等权。所有四解析组和全部场景完整保留，包含clean下降及个别TX弱/负关联。

|场景|候选|raw combined margin|恢复 combined margin|原clean参考|
|---|---|---:|---:|---:|
|clean|r46_350521|0.124015|0.109981|0.124015|
|clean|r53_prune_decay|0.124015|0.111223|0.124015|
|clean|r64_fold_full|0.124015|0.104315|0.124015|
|clean|r64_fold_half|0.124015|0.105514|0.124015|
|clean|r65_prune_after5|0.124015|0.110906|0.124015|
|clean|r65_prune_after10|0.124015|0.111669|0.124015|
|clean|r66_final|0.124015|0.111529|0.124015|
|clean|r66_late_mean|0.124015|0.111049|0.124015|
|clean|r66_step_ema|0.124015|0.111329|0.124015|
|practical_high|r46_350521|0.005703|0.099789|0.124015|
|practical_high|r53_prune_decay|0.005703|0.103692|0.124015|
|practical_high|r64_fold_full|0.005703|0.094055|0.124015|
|practical_high|r64_fold_half|0.005703|0.098332|0.124015|
|practical_high|r65_prune_after5|0.005703|0.100487|0.124015|
|practical_high|r65_prune_after10|0.005703|0.096349|0.124015|
|practical_high|r66_final|0.005703|0.105690|0.124015|
|practical_high|r66_late_mean|0.005703|0.105086|0.124015|
|practical_high|r66_step_ema|0.005703|0.104930|0.124015|
|practical_mid|r46_350521|0.003276|0.092008|0.124015|
|practical_mid|r53_prune_decay|0.003276|0.096168|0.124015|
|practical_mid|r64_fold_full|0.003276|0.087965|0.124015|
|practical_mid|r64_fold_half|0.003276|0.091873|0.124015|
|practical_mid|r65_prune_after5|0.003276|0.091761|0.124015|
|practical_mid|r65_prune_after10|0.003276|0.088110|0.124015|
|practical_mid|r66_final|0.003276|0.097584|0.124015|
|practical_mid|r66_late_mean|0.003276|0.097100|0.124015|
|practical_mid|r66_step_ema|0.003276|0.096809|0.124015|
|practical_low_suburban|r46_350521|0.002559|0.085436|0.124015|
|practical_low_suburban|r53_prune_decay|0.002559|0.090035|0.124015|
|practical_low_suburban|r64_fold_full|0.002559|0.082925|0.124015|
|practical_low_suburban|r64_fold_half|0.002559|0.086033|0.124015|
|practical_low_suburban|r65_prune_after5|0.002559|0.084596|0.124015|
|practical_low_suburban|r65_prune_after10|0.002559|0.081136|0.124015|
|practical_low_suburban|r66_final|0.002559|0.091026|0.124015|
|practical_low_suburban|r66_late_mean|0.002559|0.090635|0.124015|
|practical_low_suburban|r66_step_ema|0.002559|0.090225|0.124015|
|practical_high_urban|r46_350521|0.005492|0.096866|0.124015|
|practical_high_urban|r53_prune_decay|0.005492|0.100641|0.124015|
|practical_high_urban|r64_fold_full|0.005492|0.092438|0.124015|
|practical_high_urban|r64_fold_half|0.005492|0.096382|0.124015|
|practical_high_urban|r65_prune_after5|0.005492|0.096768|0.124015|
|practical_high_urban|r65_prune_after10|0.005492|0.092933|0.124015|
|practical_high_urban|r66_final|0.005492|0.102588|0.124015|
|practical_high_urban|r66_late_mean|0.005492|0.102021|0.124015|
|practical_high_urban|r66_step_ema|0.005492|0.101804|0.124015|
|practical_mid_urban|r46_350521|0.002913|0.079533|0.124015|
|practical_mid_urban|r53_prune_decay|0.002913|0.083247|0.124015|
|practical_mid_urban|r64_fold_full|0.002913|0.080017|0.124015|
|practical_mid_urban|r64_fold_half|0.002913|0.082842|0.124015|
|practical_mid_urban|r65_prune_after5|0.002913|0.077243|0.124015|
|practical_mid_urban|r65_prune_after10|0.002913|0.074360|0.124015|
|practical_mid_urban|r66_final|0.002913|0.084653|0.124015|
|practical_mid_urban|r66_late_mean|0.002913|0.084307|0.124015|
|practical_mid_urban|r66_step_ema|0.002913|0.083875|0.124015|
|practical_low_urban|r46_350521|0.002280|0.066143|0.124015|
|practical_low_urban|r53_prune_decay|0.002280|0.069657|0.124015|
|practical_low_urban|r64_fold_full|0.002280|0.068882|0.124015|
|practical_low_urban|r64_fold_half|0.002280|0.070507|0.124015|
|practical_low_urban|r65_prune_after5|0.002280|0.062903|0.124015|
|practical_low_urban|r65_prune_after10|0.002280|0.060137|0.124015|
|practical_low_urban|r66_final|0.002280|0.070741|0.124015|
|practical_low_urban|r66_late_mean|0.002280|0.070608|0.124015|
|practical_low_urban|r66_step_ema|0.002280|0.070099|0.124015|

mirror64、spectral4、reference24、combined92使用同一固定源均值/std下限0.05；combined按sqrt(3×组维数)缩放使平方误差三组等权。以下逐候选列出lowurban的四解析组，不以某一组涨分代替整体性能或TX真实性。

|lowurban候选|mirror margin|spectral margin|reference margin|combined margin|combined负或零margin的TX|
|---|---:|---:|---:|---:|---|
|r46_350521|0.081266|0.076193|0.006108|0.066143|2|
|r53_prune_decay|0.085995|0.078548|0.006213|0.069657|2|
|r64_fold_full|0.085480|0.080105|0.006067|0.068882|2|
|r64_fold_half|0.087826|0.076183|0.006383|0.070507|2|
|r65_prune_after5|0.078021|0.082812|0.006718|0.062903|2|
|r65_prune_after10|0.074214|0.079499|0.007104|0.060137|2|
|r66_final|0.087089|0.078917|0.006163|0.070741|2|
|r66_late_mean|0.087061|0.081925|0.006377|0.070608|2|
|r66_step_ema|0.086538|0.080143|0.006701|0.070099|2|

逐TX、RX、day的完整正/负结果、有效/零向量/缺参考计数与配对clean描述量误差保留在JSON/CSV。以下置换对照不改变参考标签；199次固定RX/day内query标签置换为描述性阴性对照，不是p值或独立新样本。

|lowurban候选|combined实际margin|query标签置换均值|置换5%至95%|高于95%分位|
|---|---:|---:|---|---|
|r46_350521|0.066143|0.000019|-0.000321至0.000342|True|
|r53_prune_decay|0.069657|0.000018|-0.000286至0.000334|True|
|r64_fold_full|0.068882|0.000024|-0.000287至0.000364|True|
|r64_fold_half|0.070507|0.000025|-0.000288至0.000359|True|
|r65_prune_after5|0.062903|0.000020|-0.000269至0.000352|True|
|r65_prune_after10|0.060137|0.000014|-0.000257至0.000334|True|
|r66_final|0.070741|0.000019|-0.000296至0.000363|True|
|r66_late_mean|0.070608|0.000020|-0.000297至0.000359|True|
|r66_step_ema|0.070099|0.000019|-0.000289至0.000367|True|

|lowurban配对比较|共同有效数|combined margin变化|配对描述量误差变化|
|---|---:|---:|---:|
|r64_fold_full − r46_350521|168000|0.002739|-0.077084|
|r64_fold_half − r46_350521|168000|0.004364|-0.087263|
|r65_prune_after5 − r46_350521|168000|-0.003240|-0.057143|
|r65_prune_after10 − r46_350521|168000|-0.006006|0.037491|
|r66_final − r46_350521|168000|0.004598|-0.091859|
|r66_late_mean − r46_350521|168000|0.004464|-0.127295|
|r66_step_ema − r46_350521|168000|0.003955|-0.111270|
|r64_fold_full − r53_prune_decay|168000|-0.000774|-0.007025|
|r64_fold_half − r53_prune_decay|168000|0.000851|-0.017204|
|r65_prune_after5 − r53_prune_decay|168000|-0.006753|0.012916|
|r65_prune_after10 − r53_prune_decay|168000|-0.009520|0.107550|
|r66_final − r53_prune_decay|168000|0.001084|-0.021801|
|r66_late_mean − r53_prune_decay|168000|0.000951|-0.057236|
|r66_step_ema − r53_prune_decay|168000|0.000442|-0.041211|
|r66_late_mean − r66_final|168000|-0.000133|-0.035436|
|r66_step_ema − r66_final|168000|-0.000642|-0.019411|
|r65_prune_after5 − r65_prune_after10|168000|0.002767|-0.094634|
|r64_fold_full − r64_fold_half|168000|-0.001625|0.010180|

解析提取阶段360.773s，评分阶段1216.543s，控制器总墙钟3979.976s（含父完成等待）。8232000次新解析输入与84个旧引用来自同168000记录的相关场景/读出，不称新增独立样本。

真实2条Ls解析no-query smoke、33项聚焦检查、CLIhelp、一次限P0/P1审查，以及普通账户一次预检/归档SHA/compile/实际PID/CWD/argv/空CUDA/2线程/日志增长均核实。等待父时不持R41score.lock；父全部正式评分后提取与评分才使用唯一CPU评分槽。

固定描述符在 clean TX2 已有区分不足，恢复后负 margin 不能单独解释为输入身份错误。该类不足与各 TX 的全部结果一并保留。

本研究使用已暴露工程基准，且事后评价明确使用target clean参考和truth，不称新的独立盲测或完全无标签评价。跨RX/day与置换只提供固定坐标下有限同TX关联，仍可能含载荷、接收链和日期因素，不能证明每个生成细节来源、未知TX泛化或真实在轨有效性。原lowurban70%、其他五景距rawclean≤5pp与更强输入TX细节目标继续ACTIVE。

完整4536组记录、756条件对照和504配对差见[汇总](summary.json)与[逐组CSV](row_metrics.csv)；固定输入/方法/矩阵/命令/预算/停止规则见[登记](experiment.json)。
