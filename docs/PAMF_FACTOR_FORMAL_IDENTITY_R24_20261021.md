# R24：低秩压缩全部八版的固定无 F 身份关联

状态：VERIFIED / ANALYZED。四个父run的全部八版先完成七景正式预测与独立评分，之后本诊断固定280解析特征单元，再独立连接truth统计4536组、756条件对照和364配对差。所属CPU进程已退出。

复用R50raw/指定350521父的56单元，八新版新增224单元。0F权重/输出/embedding、checkpoint、额外神经预测、训练、头拟合或新源校准；不改变父模型、预测、选择或训练。下列margin是同TX余弦减其他五TX均值，不是分类准确率或逐细节真实性证明。

|场景|候选|raw combined margin|恢复 combined margin|clean参照 margin|
|---|---|---:|---:|---:|
|clean|r46_350521|0.124015|0.109981|0.124015|
|clean|r18_factor625_stage|0.124015|0.108301|0.124015|
|clean|r18_factor625_cond|0.124015|0.112710|0.124015|
|clean|r19_prior625_cond|0.124015|0.109533|0.124015|
|clean|r19_prior750_cond|0.124015|0.109769|0.124015|
|clean|r22_complex625|0.124015|0.114783|0.124015|
|clean|r22_complex625_identity|0.124015|0.102963|0.124015|
|clean|r23_activation625_identity|0.124015|0.122264|0.124015|
|clean|r23_activation500_identity|0.124015|0.119607|0.124015|
|practical_high|r46_350521|0.005703|0.099789|0.124015|
|practical_high|r18_factor625_stage|0.005703|0.100795|0.124015|
|practical_high|r18_factor625_cond|0.005703|0.102605|0.124015|
|practical_high|r19_prior625_cond|0.005703|0.098846|0.124015|
|practical_high|r19_prior750_cond|0.005703|0.104692|0.124015|
|practical_high|r22_complex625|0.005703|0.114780|0.124015|
|practical_high|r22_complex625_identity|0.005703|0.116599|0.124015|
|practical_high|r23_activation625_identity|0.005703|0.117153|0.124015|
|practical_high|r23_activation500_identity|0.005703|0.118958|0.124015|
|practical_mid|r46_350521|0.003276|0.092008|0.124015|
|practical_mid|r18_factor625_stage|0.003276|0.092997|0.124015|
|practical_mid|r18_factor625_cond|0.003276|0.095767|0.124015|
|practical_mid|r19_prior625_cond|0.003276|0.088724|0.124015|
|practical_mid|r19_prior750_cond|0.003276|0.097220|0.124015|
|practical_mid|r22_complex625|0.003276|0.109566|0.124015|
|practical_mid|r22_complex625_identity|0.003276|0.112183|0.124015|
|practical_mid|r23_activation625_identity|0.003276|0.113123|0.124015|
|practical_mid|r23_activation500_identity|0.003276|0.114004|0.124015|
|practical_low_suburban|r46_350521|0.002559|0.085436|0.124015|
|practical_low_suburban|r18_factor625_stage|0.002559|0.087670|0.124015|
|practical_low_suburban|r18_factor625_cond|0.002559|0.090162|0.124015|
|practical_low_suburban|r19_prior625_cond|0.002559|0.080266|0.124015|
|practical_low_suburban|r19_prior750_cond|0.002559|0.091142|0.124015|
|practical_low_suburban|r22_complex625|0.002559|0.105569|0.124015|
|practical_low_suburban|r22_complex625_identity|0.002559|0.108651|0.124015|
|practical_low_suburban|r23_activation625_identity|0.002559|0.109557|0.124015|
|practical_low_suburban|r23_activation500_identity|0.002559|0.109614|0.124015|
|practical_high_urban|r46_350521|0.005492|0.096866|0.124015|
|practical_high_urban|r18_factor625_stage|0.005492|0.098719|0.124015|
|practical_high_urban|r18_factor625_cond|0.005492|0.100653|0.124015|
|practical_high_urban|r19_prior625_cond|0.005492|0.095534|0.124015|
|practical_high_urban|r19_prior750_cond|0.005492|0.102049|0.124015|
|practical_high_urban|r22_complex625|0.005492|0.112886|0.124015|
|practical_high_urban|r22_complex625_identity|0.005492|0.114652|0.124015|
|practical_high_urban|r23_activation625_identity|0.005492|0.115244|0.124015|
|practical_high_urban|r23_activation500_identity|0.005492|0.116562|0.124015|
|practical_mid_urban|r46_350521|0.002913|0.079533|0.124015|
|practical_mid_urban|r18_factor625_stage|0.002913|0.084373|0.124015|
|practical_mid_urban|r18_factor625_cond|0.002913|0.086233|0.124015|
|practical_mid_urban|r19_prior625_cond|0.002913|0.074269|0.124015|
|practical_mid_urban|r19_prior750_cond|0.002913|0.085055|0.124015|
|practical_mid_urban|r22_complex625|0.002913|0.100768|0.124015|
|practical_mid_urban|r22_complex625_identity|0.002913|0.103005|0.124015|
|practical_mid_urban|r23_activation625_identity|0.002913|0.103715|0.124015|
|practical_mid_urban|r23_activation500_identity|0.002913|0.102757|0.124015|
|practical_low_urban|r46_350521|0.002280|0.066143|0.124015|
|practical_low_urban|r18_factor625_stage|0.002280|0.073285|0.124015|
|practical_low_urban|r18_factor625_cond|0.002280|0.073453|0.124015|
|practical_low_urban|r19_prior625_cond|0.002280|0.058559|0.124015|
|practical_low_urban|r19_prior750_cond|0.002280|0.071198|0.124015|
|practical_low_urban|r22_complex625|0.002280|0.089200|0.124015|
|practical_low_urban|r22_complex625_identity|0.002280|0.091553|0.124015|
|practical_low_urban|r23_activation625_identity|0.002280|0.092249|0.124015|
|practical_low_urban|r23_activation500_identity|0.002280|0.089171|0.124015|

|lowurban候选|TX|恢复combined margin|clean参照margin|
|---|---|---:|---:|
|r46_350521|0|0.117089|0.189362|
|r46_350521|1|0.091073|0.164325|
|r46_350521|2|-0.009089|-0.001308|
|r46_350521|3|0.064235|0.134620|
|r46_350521|4|0.077040|0.158697|
|r46_350521|5|0.056510|0.098392|
|r18_factor625_stage|0|0.138349|0.189362|
|r18_factor625_stage|1|0.094338|0.164325|
|r18_factor625_stage|2|-0.006470|-0.001308|
|r18_factor625_stage|3|0.065799|0.134620|
|r18_factor625_stage|4|0.091288|0.158697|
|r18_factor625_stage|5|0.056407|0.098392|
|r18_factor625_cond|0|0.141867|0.189362|
|r18_factor625_cond|1|0.109853|0.164325|
|r18_factor625_cond|2|-0.014131|-0.001308|
|r18_factor625_cond|3|0.077232|0.134620|
|r18_factor625_cond|4|0.088127|0.158697|
|r18_factor625_cond|5|0.037767|0.098392|
|r19_prior625_cond|0|0.103534|0.189362|
|r19_prior625_cond|1|0.071881|0.164325|
|r19_prior625_cond|2|-0.005904|-0.001308|
|r19_prior625_cond|3|0.053960|0.134620|
|r19_prior625_cond|4|0.083026|0.158697|
|r19_prior625_cond|5|0.044855|0.098392|
|r19_prior750_cond|0|0.123132|0.189362|
|r19_prior750_cond|1|0.096370|0.164325|
|r19_prior750_cond|2|-0.011772|-0.001308|
|r19_prior750_cond|3|0.067160|0.134620|
|r19_prior750_cond|4|0.096100|0.158697|
|r19_prior750_cond|5|0.056199|0.098392|
|r22_complex625|0|0.175054|0.189362|
|r22_complex625|1|0.135482|0.164325|
|r22_complex625|2|-0.011743|-0.001308|
|r22_complex625|3|0.094925|0.134620|
|r22_complex625|4|0.103719|0.158697|
|r22_complex625|5|0.037763|0.098392|
|r22_complex625_identity|0|0.178226|0.189362|
|r22_complex625_identity|1|0.137729|0.164325|
|r22_complex625_identity|2|-0.010692|-0.001308|
|r22_complex625_identity|3|0.095330|0.134620|
|r22_complex625_identity|4|0.109355|0.158697|
|r22_complex625_identity|5|0.039371|0.098392|
|r23_activation625_identity|0|0.173863|0.189362|
|r23_activation625_identity|1|0.145702|0.164325|
|r23_activation625_identity|2|-0.014546|-0.001308|
|r23_activation625_identity|3|0.101314|0.134620|
|r23_activation625_identity|4|0.118661|0.158697|
|r23_activation625_identity|5|0.028498|0.098392|
|r23_activation500_identity|0|0.161376|0.189362|
|r23_activation500_identity|1|0.130216|0.164325|
|r23_activation500_identity|2|-0.003343|-0.001308|
|r23_activation500_identity|3|0.091103|0.134620|
|r23_activation500_identity|4|0.108022|0.158697|
|r23_activation500_identity|5|0.047650|0.098392|

|lowurban配对比较|共同有效数|combined margin变化|配对描述误差变化|
|---|---:|---:|---:|
|r18_factor625_stage − r46_350521|168000|0.007142|-0.221728|
|r18_factor625_cond − r46_350521|168000|0.007309|-0.242657|
|r19_prior625_cond − r46_350521|168000|-0.007585|0.056420|
|r19_prior750_cond − r46_350521|168000|0.005055|-0.102334|
|r22_complex625 − r46_350521|168000|0.023057|-0.367595|
|r22_complex625_identity − r46_350521|168000|0.025410|-0.367497|
|r23_activation625_identity − r46_350521|168000|0.026106|-0.354946|
|r23_activation500_identity − r46_350521|168000|0.023028|-0.305491|
|r18_factor625_cond − r18_factor625_stage|168000|0.000167|-0.020929|
|r19_prior750_cond − r19_prior625_cond|168000|0.012640|-0.158753|
|r22_complex625_identity − r22_complex625|168000|0.002353|0.000098|
|r23_activation625_identity − r22_complex625_identity|168000|0.000696|0.012551|
|r23_activation500_identity − r23_activation625_identity|168000|-0.003078|0.049455|

解析提取405.788s，独立评分进程1263.528s，总墙钟13187.140s（含等待父完成）。9408000解析输入来自同168000物理记录的相关读出/场景，不增加独立样本数。

全部四解析组、七景、TX/RX/day、有效/零向量/缺参考计数、配对误差和199次RX/day内query标签置换控制均保留。参照RX与day同时不同；原R34源LU8192统计/std下限0.05保持，0新增拟合。置换控制不是独立p值。

clean TX2已有描述符区分不足；恢复后负margin不能单独解释为错误TX。本研究只证明固定坐标下的有限同TX关联，不证明每个细节的来源、未知TX泛化或真实在轨有效性。原七景准确率、clean保持、lowurban70%及其他五景距rawclean5pp仍单列，整体目标未完成。

完整统计见[结果](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-factor-formal-analytic-TX-s2026092701-m8-r24/summary.json)和[逐组CSV](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-factor-formal-analytic-TX-s2026092701-m8-r24/row_metrics.csv)，方法/输入/预算/命令见[登记](../automation_reports/CV-SincNet/20261021-diagnostic-restoration-factor-formal-analytic-TX-s2026092701-m8-r24/experiment.json)。
