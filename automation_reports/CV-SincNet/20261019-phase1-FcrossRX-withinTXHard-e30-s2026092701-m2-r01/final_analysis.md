# 类内有界困难 As CE：E30 完整结果

本轮完成两条原As20＋10尾段及全部七环境独立truth-last评分。唯一机制是As CE类内有界权重与当前困难度的关联；所有As仍监督，原完整practical／clean/raw／同IQ RDropKL／ORD128 CROSS_RX／U及默认cosine头不变。

源预登记判定：未通过，关闭固定配方，不扩E50或种子，不扫描gamma／top比例。正式分数未改变源决定。只有1个模型种子，没有可重复双提升证据。

困难排序相对匹配NoAs：clean 0.055952pp，mid urban 27.951786pp，low urban 21.440476pp，urban均值 28.628175pp。clean下降≤0.1pp阶段约束：满足。

相对原As的机制增量：clean -0.037500pp，mid 0.277976pp，low 0.336310pp，urban均值 0.259722pp。相对相同权重随机分配：clean -0.083333pp，mid 0.277381pp，low 0.288095pp。不能把原As相对NoAs的大收益当新增机制收益。

## 所有逐行指标

|行|视图|准确率 %|macro F1 %|最差RX %|
|---|---|---:|---:|---:|
|NOAS30|clean|85.247619|84.979636|78.450000|
|NOAS30|practical_high|42.685119|39.311416|34.304167|
|NOAS30|practical_mid|40.995238|37.954813|33.520833|
|NOAS30|practical_low_suburban|39.472619|36.758731|32.054167|
|NOAS30|practical_high_urban|41.439881|38.481030|34.000000|
|NOAS30|practical_mid_urban|35.132738|33.446603|29.495833|
|NOAS30|practical_low_urban|30.139881|29.184589|25.750000|
|AS30|clean|85.341071|85.044474|78.362500|
|AS30|practical_high|81.405357|81.084230|71.908333|
|AS30|practical_mid|78.661310|78.359706|70.212500|
|AS30|practical_low_suburban|75.861905|75.572741|67.741667|
|AS30|practical_high_urban|77.767262|77.423208|69.108333|
|AS30|practical_mid_urban|62.806548|62.450463|56.458333|
|AS30|practical_low_urban|51.244048|50.887035|46.379167|
|HARD30|clean|85.303571|85.006388|78.887500|
|HARD30|practical_high|81.562500|81.253844|72.033333|
|HARD30|practical_mid|78.862500|78.562350|70.362500|
|HARD30|practical_low_suburban|76.055952|75.762124|67.866667|
|HARD30|practical_high_urban|77.932143|77.596220|69.108333|
|HARD30|practical_mid_urban|63.084524|62.699931|56.708333|
|HARD30|practical_low_urban|51.580357|51.148234|46.741667|
|RANDOM30|clean|85.386905|85.096013|78.995833|
|RANDOM30|practical_high|81.437500|81.123441|72.087500|
|RANDOM30|practical_mid|78.654762|78.356590|70.254167|
|RANDOM30|practical_low_suburban|75.866667|75.579805|67.812500|
|RANDOM30|practical_high_urban|77.794643|77.456562|69.162500|
|RANDOM30|practical_mid_urban|62.807143|62.448514|56.516667|
|RANDOM30|practical_low_urban|51.292262|50.923213|46.437500|

|行|urban均值 %|high gap pp|mid gap pp|low suburban gap pp|
|---|---:|---:|---:|---:|
|NOAS30|35.570833|42.562500|44.252381|45.775000|
|AS30|63.939286|3.935714|6.679762|9.479167|
|HARD30|64.199008|3.741071|6.441071|9.247619|
|RANDOM30|63.964683|3.949405|6.732143|9.520238|

差距为各行clean−对应suburban，不能用降低clean缩小差距。全部RX/day/TX分层、CM、macroF1及最差RX配对差在results_summary.json。

## 错误修复和新增错误

|配对|视图|Δaccuracy pp|修复|新增错误|净修复|
|---|---|---:|---:|---:|---:|
|HARD30_minus_NOAS30|clean|0.055952|3899|3805|94|
|HARD30_minus_NOAS30|practical_high|38.877381|68687|3373|65314|
|HARD30_minus_NOAS30|practical_mid|37.867262|67560|3943|63617|
|HARD30_minus_NOAS30|practical_low_suburban|36.583333|65898|4438|61460|
|HARD30_minus_NOAS30|practical_high_urban|36.492262|65400|4093|61307|
|HARD30_minus_NOAS30|practical_mid_urban|27.951786|53715|6756|46959|
|HARD30_minus_NOAS30|practical_low_urban|21.440476|44881|8861|36020|
|HARD30_minus_AS30|clean|-0.037500|361|424|-63|
|HARD30_minus_AS30|practical_high|0.157143|584|320|264|
|HARD30_minus_AS30|practical_mid|0.201190|702|364|338|
|HARD30_minus_AS30|practical_low_suburban|0.194048|747|421|326|
|HARD30_minus_AS30|practical_high_urban|0.164881|641|364|277|
|HARD30_minus_AS30|practical_mid_urban|0.277976|981|514|467|
|HARD30_minus_AS30|practical_low_urban|0.336310|1170|605|565|
|HARD30_minus_RANDOM30|clean|-0.083333|185|325|-140|
|HARD30_minus_RANDOM30|practical_high|0.125000|573|363|210|
|HARD30_minus_RANDOM30|practical_mid|0.207738|699|350|349|
|HARD30_minus_RANDOM30|practical_low_suburban|0.189286|751|433|318|
|HARD30_minus_RANDOM30|practical_high_urban|0.137500|628|397|231|
|HARD30_minus_RANDOM30|practical_mid_urban|0.277381|981|515|466|
|HARD30_minus_RANDOM30|practical_low_urban|0.288095|1127|643|484|
|RANDOM30_minus_AS30|clean|0.045833|322|245|77|
|RANDOM30_minus_AS30|practical_high|0.032143|279|225|54|
|RANDOM30_minus_AS30|practical_mid|-0.006548|270|281|-11|
|RANDOM30_minus_AS30|practical_low_suburban|0.004762|319|311|8|
|RANDOM30_minus_AS30|practical_high_urban|0.027381|282|236|46|
|RANDOM30_minus_AS30|practical_mid_urban|0.000595|383|382|1|
|RANDOM30_minus_AS30|practical_low_urban|0.048214|506|425|81|
|AS30_minus_NOAS30|clean|0.093452|3868|3711|157|
|AS30_minus_NOAS30|practical_high|38.720238|68469|3419|65050|
|AS30_minus_NOAS30|practical_mid|37.666071|67279|4000|63279|
|AS30_minus_NOAS30|practical_low_suburban|36.389286|65618|4484|61134|
|AS30_minus_NOAS30|practical_high_urban|36.327381|65141|4111|61030|
|AS30_minus_NOAS30|practical_mid_urban|27.673810|53261|6769|46492|
|AS30_minus_NOAS30|practical_low_urban|21.104167|44297|8842|35455|

## 源判定和完整训练日志

{"passed":false,"candidate_minus_random_midlow_pp":0.26388888888888573,"candidate_minus_matchedNoAs_fullV_clean_pp":-0.12592592592592933,"candidate_minus_AsMeasure_midlow_pp":0.3055555555555478,"clean_min_pp":-0.1,"midlow_min_pp":0.5,"source_failure_skips_formal":false,"automatic_extension":false,"no_gamma_topfraction_seed_rescue":true}

H01：五份逐步日志计数 2220/2220/2220/2220/2220，epoch 10，源观察检查不一致 []；完整曲线由逐步日志计算，见机器可读汇总。

R01：五份逐步日志计数 2220/2220/2220/2220/2220，epoch 10，源观察检查不一致 []；完整曲线由逐步日志计算，见机器可读汇总。

物理加载序列相同：true；As物理序列相同：true。这些只是执行证据，不能替代识别增益。

## 实际成本与边界

新增20模型轮、4440更新、8704668样本前向；相对原每步配方的额外模型前向／VJP均0，另有8880次CE算子、568320个额外logit样本处理。CPU真实父模型与两hook smoke1540NN／0更新已计入，原NoAs／As完整E30及两个原E20父模型为既有成本，不重复收费。

|新增行|学生训练NN|学生验证NN|教师NN|native墙钟秒|正式墙钟秒|
|---|---:|---:|---:|---:|---:|
|HARD30|2271960|324002|567000|1181.135820|54.865153|
|RANDOM30|2271960|324002|567000|1178.123174|54.771172|

worker墙钟之和不是并行历时；源读出和并行历时未独立归档，不填0。排序CE/标量化主机计时保留在risk_compute，不声称GPU纯计算耗时。无源RX留出，完整L6300/U56700 TXhidden/V27000、默认头、固定receivedIQ不变。两draw不是模型种子，未证明跨seed稳定性。训练与全部all7技术缺项为0，最终lowurban>60%与可重复clean正收益仍按真实结果判断，目标保持ACTIVE。


## 结果含义与限制

本轮在单种子上观察到真实的小幅机制增量：相对原As，low修复1170／新增605，净+565；mid修复981／新增514，净+467。clean相对原As修复361／新增424，净−63；相对NoAs则修复3899／新增3805，净+94（+0.055952pp）。这不是大范围稳定clean提升。low相对原As在全部7个RX和4个day均为正，但TX2与TX5贡献净+287／+299，TX1与TX3为−12／−62；不能称全部身份一致改善。

HARD的low macroF1为51.148234%、最差RX为46.741667%；相对原As分别+0.261199pp／+0.362500pp。mid macroF1为62.699931%、最差RX56.708333%，分别+0.249468pp／+0.250000pp。suburban实际准确率提高0.157143／0.201190／0.194048pp，但仍与自身clean差3.741071／6.441071／9.247619pp；相对原As的gap缩小也含clean−0.0375pp，不能全部归为suburban提升。

源midlow增量仅+0.305556pp且fullVclean−0.125926pp，不满足预定+0.5／−0.1边界，因此固定gamma.25／tophalf不晋级。正式阶段约束通过不改变源决定；low仍51.580357%，最终60%目标、suburban接近clean及可重复双提升均未完成。没有否定所有困难风险方法，只结束本次固定版本；不根据正式TX/RX分层另调权重。

旧NoAs／As result.json的U_steps=0属于旧遥测字段，不能解释成未训练U。本轮另完整读取两组各2220条原native日志，均有567000U观测、EMA末计数2220、U真值可见性最大0；学生总前向减L前向及教师567000也一致。机器汇总将原字段单列U_steps_reported_legacy，实际U计数来自完整日志。该核算NN／IQ／CP／truth新增读取均0。

本地对已固定聚合的504个CM与35组整体配对／分区算术核对一致；这不替代最初独立truth-last评分。新增训练代码8d7628f3fed309a068f50ee403afcf4ac736d367，所有所属进程自然退出，技术缺项0。

原对照的历史墙钟来自不同运行时资源条件，不能把与本次1181.136／1178.123秒的差异全部归因于新方法；只以两新增行的同NN与额外CE计数说明预算。原benchmark已在历史实验中暴露，不声称新的独立盲测。
