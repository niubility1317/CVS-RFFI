# MIXED128 E30：单种子四条件最终分析

**固定 E30 的 MIXED128 配方不保留：源资格已失败并先于正式读取关闭；正式结果也未满足 clean 下降不超过 0.1 个百分点的阶段目标。** 新 As 相对原 As 的六种 practical 准确率全部下降，没有观察到新增机制带来的星地与 clean 双提升。

新 As 的 clean 为 **84.493452%**，匹配新 NoAs 为85.158929%，变化 **−0.665476 pp**（净少1118个正确样本），阶段判定FAIL。新 As 的low/mid urban为51.187500%/62.749405%；相对匹配新NoAs提高+21.134524/+27.692262 pp，这是完整As训练的总体收益。相对原As，low/mid urban反而为−0.056548/−0.057143 pp，clean为−0.847619 pp，不能将前述大幅As收益归给MIXED128。

源预定比较MIXED_AS30−AS30的fullV clean为+0.033333 pp，mid/low两draw等权均值仅+0.069444 pp，未达到+0.5 pp。source_read_before_formal已由root先保存，固定配方关闭、不跑E50、不扩种子；正式结果仅描述，不改变该决定，也不用于新设计、调参或救跑。

本轮只有固定模型种子2026092701（n=1），不报告模型seed标准差或置信区间，不声称稳定双提升。
## 唯一变量与已完成范围

本轮唯一假设是：将已有跨RX同TX SupCon的输入从ORD128改为现有mixed128对应RDrop前向的特征图，可以在不增加NN前向的情况下改善practical识别并控制clean代价。本次观测不支持保留该固定配方；这不等同于证明所有mixed对比方案都无效。

原 .5 primary ordinary＋.5 mixed＋.25 raw 的L权重不变，mixed内ordinary64与selected64各.25，总L尺度1.25，已有SupCon系数.1、6L前向及U规则不变。NoAs selected64实际为raw64；As selected64为完整practical(raw64)。没有使用其它路线的.125/.375权重，没有新增loss、分支、头或部署参数。

P41 SAT192已源失败；本次保留该背景，关键变化是沿用原L batch的128个位置，每位置每term一个anchor，不额外拼接selected64。原sampler保持，允许batch内物理ID重复，不去重、不要求128个物理ID全唯一。dataset eq1是索引，practical接收均衡EQ=false是另一配置，二者不混用。

新两行各自严格复用对应原E20 student/own EMA，再增加E10，累计E30/6660updates；原NOAS30/AS30的冻结E30作为只读参照。保持F＋跨RX同TX对比学习、SIG_TIME/真实joint160、默认cosine头联合训练、全部5源RX、L6300/U56700（TX真值隐藏）/V27000、IQ256与固定测试IQ/SNR/信道/困难样本。

## 源资格先读与正式判断分开

|源条件：新As−原As|实际变化（pp）|固定门槛|结果|
|---|---|---:|---:|
|fullV clean|+0.033333|≥ −0.1|PASS|
|V1800两draw mid/low等权均值|+0.069444|≥ +0.5|FAIL|

两项合取FAIL，E50 source eligible=false、fixed_recipe_closed=true、automatic_extension=false。没有追加逐cell、置信度或身份保持门槛；NoAs的源变化与交互只报告。源科学失败没有跳过正式收尾。

|条件|fullV clean（%）|fullV最差RX（%）|panel clean（%）|low suburban（%）|mid urban（%）|low urban（%）|mid/low均值（%）|
|---|---|---:|---:|---:|---:|---:|---:|
|NOAS30|98.359259|95.611111|98.444444|49.861111|43.555556|35.611111|39.583333|
|AS30|98.255556|95.333333|98.333333|88.666667|72.805556|59.083333|65.944444|
|MIXED_NOAS30|98.355556|95.611111|98.388889|49.333333|43.222222|35.611111|39.416667|
|MIXED_AS30|98.288889|95.351852|98.277778|89.027778|73.361111|58.666667|66.013889|

|条件|draw|mid urban（%）|low urban（%）|mid/low等权（%）|
|---|---|---:|---:|---:|
|NOAS30|1|42.944444|35.666667|39.305556|
|NOAS30|2|44.166667|35.555556|39.861111|
|AS30|1|72.555556|60.111111|66.333333|
|AS30|2|73.055556|58.055556|65.555556|
|MIXED_NOAS30|1|42.944444|35.777778|39.361111|
|MIXED_NOAS30|2|43.500000|35.444444|39.472222|
|MIXED_AS30|1|73.111111|59.722222|66.416667|
|MIXED_AS30|2|73.611111|57.611111|65.611111|

新As相对原As的源panel mid urban提高+0.555556 pp，但low urban下降−0.416667 pp，等权净增不足门槛。fullV27000仅使用原生冻结accuracy/RX，没有完整V逐包重算F1或错误修复；不将其与V1800面板混用。面板clean1800次，三场practical各2draw×1800=3600次观测，两次draw不是两个模型seed。源无high urban；源panel fc=2.45GHz与训练/正式2.462GHz的既有差异保留。

## 四条件全部七视图准确率

单位为%；每个正式视图固定168000个样本。urban_mean只为三个urban准确率等权均值，不是新增评分视图。

|view|原NoAs30|原As30|MIXED NoAs30|MIXED As30|
|---|---|---:|---:|---:|
|clean|85.247619|85.341071|85.158929|84.493452|
|practical_high|42.685119|81.405357|42.667262|81.250595|
|practical_mid|40.995238|78.661310|40.893452|78.505357|
|practical_low_suburban|39.472619|75.861905|39.389286|75.793452|
|practical_high_urban|41.439881|77.767262|41.345833|77.625000|
|practical_mid_urban|35.132738|62.806548|35.057143|62.749405|
|practical_low_urban|30.139881|51.244048|30.052976|51.187500|
|urban_mean|35.570833|63.939286|35.485317|63.853968|

## 五项配对与交互

以下为准确率绝对百分点。普通As总效果、新机制在As上的增量、NoAs控制变化分别报告；不把NoAs退化记作纯As收益。

|view|新As−新NoAs|新As−原As|新NoAs−原NoAs|原As−原NoAs|新As−原NoAs|交互|
|---|---|---:|---:|---:|---:|---:|
|clean|-0.665476|-0.847619|-0.088690|+0.093452|-0.754167|-0.758929|
|practical_high|+38.583333|-0.154762|-0.017857|+38.720238|+38.565476|-0.136905|
|practical_mid|+37.611905|-0.155952|-0.101786|+37.666071|+37.510119|-0.054167|
|practical_low_suburban|+36.404167|-0.068452|-0.083333|+36.389286|+36.320833|+0.014881|
|practical_high_urban|+36.279167|-0.142262|-0.094048|+36.327381|+36.185119|-0.048214|
|practical_mid_urban|+27.692262|-0.057143|-0.075595|+27.673810|+27.616667|+0.018452|
|practical_low_urban|+21.134524|-0.056548|-0.086905|+21.104167|+21.047619|+0.030357|
|urban_mean|+28.368651|-0.085317|-0.085516|+28.368452|+28.283135|+0.000198|

交互=(新As−原As)−(新NoAs−原NoAs)。urban三场交互仅+0.000198 pp，接近零；mid/low交互的小正数并不改变新As自身均低于原As的事实。clean交互为−0.758929 pp。新NoAs的clean也下降−0.088690 pp，因此新As相对原NoAs的clean下降为−0.754167 pp，另列保留。

原As−原NoAs clean为+0.093452 pp，只是已有单seed的微小观察，不能称稳定双提升。

## accuracy、macro-F1与最差RX完整指标

|条件|view|accuracy（%）|macro-F1（%）|最差RX accuracy（%）|
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
|MIXED_NOAS30|clean|85.158929|84.882125|78.841667|
|MIXED_NOAS30|practical_high|42.667262|39.334820|34.350000|
|MIXED_NOAS30|practical_mid|40.893452|37.919956|33.387500|
|MIXED_NOAS30|practical_low_suburban|39.389286|36.734487|31.945833|
|MIXED_NOAS30|practical_high_urban|41.345833|38.439371|33.962500|
|MIXED_NOAS30|practical_mid_urban|35.057143|33.429953|29.416667|
|MIXED_NOAS30|practical_low_urban|30.052976|29.181450|25.650000|
|MIXED_AS30|clean|84.493452|84.136117|76.241667|
|MIXED_AS30|practical_high|81.250595|80.911269|71.737500|
|MIXED_AS30|practical_mid|78.505357|78.201655|70.079167|
|MIXED_AS30|practical_low_suburban|75.793452|75.510773|67.804167|
|MIXED_AS30|practical_high_urban|77.625000|77.268023|68.979167|
|MIXED_AS30|practical_mid_urban|62.749405|62.403343|56.266667|
|MIXED_AS30|practical_low_urban|51.187500|50.839035|46.654167|

最差RX是在该条件该视图的7个正式RX中取最低准确率。新As相对原As的low urban最差RX局部提高+0.275000 pp，但该视图整体accuracy/F1分别下降−0.056548/−0.048000 pp；不能用局部改善替代完整识别收益。

## 五项配对的修复与新增错误

以下直接使用已完成独立aggregate，不重新读取逐包预测或truth。修复=control错而candidate对；新增错误=control对而candidate错；净修复=修复−新增错误。全部35个overall配对视图、595条RX/day/TX分层配对及F1/最差RX差值保存在精简results.json。

|配对|view|修复|新增错误|净修复|ΔAcc（pp）|
|---|---|---:|---:|---:|---:|
|MIXED_AS30__MIXED_NOAS30|clean|3219|4337|-1118|-0.665476|
|MIXED_AS30__MIXED_NOAS30|practical_high|68311|3491|64820|+38.583333|
|MIXED_AS30__MIXED_NOAS30|practical_mid|67245|4057|63188|+37.611905|
|MIXED_AS30__MIXED_NOAS30|practical_low_suburban|65709|4550|61159|+36.404167|
|MIXED_AS30__MIXED_NOAS30|practical_high_urban|65146|4197|60949|+36.279167|
|MIXED_AS30__MIXED_NOAS30|practical_mid_urban|53425|6902|46523|+27.692262|
|MIXED_AS30__MIXED_NOAS30|practical_low_urban|44586|9080|35506|+21.134524|
|MIXED_AS30__AS30|clean|895|2319|-1424|-0.847619|
|MIXED_AS30__AS30|practical_high|1720|1980|-260|-0.154762|
|MIXED_AS30__AS30|practical_mid|2020|2282|-262|-0.155952|
|MIXED_AS30__AS30|practical_low_suburban|2421|2536|-115|-0.068452|
|MIXED_AS30__AS30|practical_high_urban|1851|2090|-239|-0.142262|
|MIXED_AS30__AS30|practical_mid_urban|2873|2969|-96|-0.057143|
|MIXED_AS30__AS30|practical_low_urban|3542|3637|-95|-0.056548|
|MIXED_NOAS30__NOAS30|clean|360|509|-149|-0.088690|
|MIXED_NOAS30__NOAS30|practical_high|1307|1337|-30|-0.017857|
|MIXED_NOAS30__NOAS30|practical_mid|1252|1423|-171|-0.101786|
|MIXED_NOAS30__NOAS30|practical_low_suburban|1296|1436|-140|-0.083333|
|MIXED_NOAS30__NOAS30|practical_high_urban|1242|1400|-158|-0.094048|
|MIXED_NOAS30__NOAS30|practical_mid_urban|1221|1348|-127|-0.075595|
|MIXED_NOAS30__NOAS30|practical_low_urban|1249|1395|-146|-0.086905|
|AS30__NOAS30|clean|3868|3711|157|+0.093452|
|AS30__NOAS30|practical_high|68469|3419|65050|+38.720238|
|AS30__NOAS30|practical_mid|67279|4000|63279|+37.666071|
|AS30__NOAS30|practical_low_suburban|65618|4484|61134|+36.389286|
|AS30__NOAS30|practical_high_urban|65141|4111|61030|+36.327381|
|AS30__NOAS30|practical_mid_urban|53261|6769|46492|+27.673810|
|AS30__NOAS30|practical_low_urban|44297|8842|35455|+21.104167|
|MIXED_AS30__NOAS30|clean|3130|4397|-1267|-0.754167|
|MIXED_AS30__NOAS30|practical_high|68235|3445|64790|+38.565476|
|MIXED_AS30__NOAS30|practical_mid|67076|4059|63017|+37.510119|
|MIXED_AS30__NOAS30|practical_low_suburban|65536|4517|61019|+36.320833|
|MIXED_AS30__NOAS30|practical_high_urban|64997|4206|60791|+36.185119|
|MIXED_AS30__NOAS30|practical_mid_urban|53286|6890|46396|+27.616667|
|MIXED_AS30__NOAS30|practical_low_urban|44427|9067|35360|+21.047619|

相对原As，clean修复895、新增错误2319，净损失1424；mid urban净损失96，low urban净损失95。相对匹配新NoAs，clean修复3219、新增4337，净损失1118，超过阶段容许的168个样本损失。补充的固定NOAS-clean正确cohort统计完整保留，不替代overall判定。

## Suburban绝对性能与clean下降造成的差距缩小

差距定义为同一模型clean−suburban（pp）。

|条件|suburban view|绝对准确率（%）|与自身clean差距（pp）|
|---|---|---:|---:|
|NOAS30|practical_high|42.685119|42.562500|
|NOAS30|practical_mid|40.995238|44.252381|
|NOAS30|practical_low_suburban|39.472619|45.775000|
|AS30|practical_high|81.405357|3.935714|
|AS30|practical_mid|78.661310|6.679762|
|AS30|practical_low_suburban|75.861905|9.479167|
|MIXED_NOAS30|practical_high|42.667262|42.491667|
|MIXED_NOAS30|practical_mid|40.893452|44.265476|
|MIXED_NOAS30|practical_low_suburban|39.389286|45.769643|
|MIXED_AS30|practical_high|81.250595|3.242857|
|MIXED_AS30|practical_mid|78.505357|5.988095|
|MIXED_AS30|practical_low_suburban|75.793452|8.700000|

新As三个suburban差距为3.242857/5.988095/8.700000 pp，仍超过早期2pp探索建议；该建议不新增为本轮资格门槛。相对原As，三个差距虽然缩小，但三个suburban准确率本身均下降，差距缩小全部由更大的clean下降造成：

|view|suburban实际变化（pp）|clean下降贡献（pp）|总差距缩小（pp）|
|---|---|---:|---:|
|practical_high|-0.154762|+0.847619|+0.692857|
|practical_mid|-0.155952|+0.847619|+0.691667|
|practical_low_suburban|-0.068452|+0.847619|+0.779167|

不能以降低clean所得的更小差距宣称suburban保护成功。

## 完整训练解析与可比范围

两个新行各完整解析2220step/10epoch，共4440updates/20 model-epochs；旧原E30完整审计直接复用，没有重扫旧step日志。每新行L NN=1704960，L loader packets=284160，实际6L；U student/teacher各567000。SupCon mode=MIXED128、128输入位置/term、每位置1anchor、4个对应RDrop pass maps、extra anchor positions=0、extra NN=0，原权重与前向顺序保持。

新NoAs/As之间4440条L/U sequence记录mismatch=0，ordinary MSE差异记录0；这是新两行之间的比较，不能扩写成与旧参考的逐step IQ完全相同。各新行对原同条件E30 tail的四counter相等；旧10个epoch均值比较按已有审计保留，不声称未记录的逐点IQ或逐ID×scene相等。

本次两个新行实际batch内重复物理ID计数均为0；这只是观测，仍保留原sampler并允许原有重复。每个L物理样本的selected曝光统计已保存；NoAs的142080次是raw identity control，As的142080次才是真实As。实际U selected分别8964/8725，规则不变不要求选中数量相等。日志健康与损失分解正确只证明执行完整，不证明性能有效。

## 实际成本：新增、复用、student/teacher与读出分列

NN单位为样本前向次数。旧行成本是完整E30复用，新行成本是新增E10段，不能直接按wall比较速度。

|条件|范围|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|
|---|---|---:|---:|---:|---:|---:|
|NOAS30|旧E30复用|6078780|918004|0|963900|7960684|
|AS30|旧E30复用|6078780|918004|0|963900|7960684|
|MIXED_NOAS30|新E10增量|2271960|324002|0|567000|3162962|
|MIXED_AS30|新E10增量|2271960|324002|0|567000|3162962|

|条件|source readout NN|formal prediction NN|本段＋自身读出 NN|native wall（s）|formal wall（s）|
|---|---|---:|---:|---:|---:|
|NOAS30|12600|1176002|9149286|2146.190179|48.089353|
|AS30|12600|1176002|9149286|2860.909118|49.649733|
|MIXED_NOAS30|12600|1176002|4351564|802.031533|58.136710|
|MIXED_AS30|12600|1176002|4351564|1061.257673|47.197470|

|项目|实际值|
|---|---|
|新增native NN|6325924|
|新增source readout NN|25200|
|新增formal prediction NN|2352004|
|新增训练＋读出NN|8703128|
|本run共享smoke NN|1540|
|本run总新增NN|8704668|
|MIXED NoAs / As各自复用E20 native NN|4797722 / 4797722|
|MIXED NoAs / As各自累计E30 native NN|7960684 / 7960684|
|旧两个E30及读出NN（复用，不重计新增）|18298572|

新增native model-wall合计1863.289206 s，formal prediction model-wall合计105.334180 s。wall为逐模型阶段秒数，不是并行总耗时或加速证据；source inference wall未单独记录，不能拿source_summary的聚合分析耗时代替。own E20父训练和旧E30已单列复用，不再加到8704668新增NN。报告整理本身0 NN/CP/IQ/updates，没有新预测或评分。

## 完整性、缺项与最终判定

source、formal与完整日志均完成，complete.status=ANALYZED。全部4×7固定预测与独立评分完成后才连接truth；独立重算504个CM、84次分区合并，与既有scorer一致，independent_recount=VERIFIED。预期11个最终aggregate文件均存在且非空，缺项0。CM及原日志保留在原最终产物，不复制进精简results.json；逐样本ID、raw预测、checkpoint、IQ和私有runtime同样不复制。

源fullV逐包F1/配对错误、source high urban、正式channel-state accuracy和单独source inference wall没有提供，按可用指标边界披露，不伪造。

最终回答：本模型在完整As训练下相对匹配无As对照的low/mid urban提高21.134524/27.692262 pp，但clean付出0.665476 pp，**不满足0.1pp阶段目标**。新增MIXED128相对原As没有提高六种practical准确率，clean额外下降0.847619 pp。low urban绝对准确率51.187500%，未达到60%。**尚无星地增强同时改善clean的可重复证据。**

固定配方已按先行源规则关闭，不E50、不扩seed；这不是根据正式结果选择性停止或救跑。本报告只描述已冻结结果，不据正式成绩提出新设计或候选晋级。n=1限制与已暴露基准背景保留，practical模拟结果不称真实在轨验证。



## 科研图与可复核汇总

[绝对表现](figures/mixedcontrast_single_seed_absolute.png)、[配对变化和交互](figures/mixedcontrast_single_seed_paired.png)、[clean护栏与F1/最差RX](figures/mixedcontrast_single_seed_clean_and_quality.png)。每组另有同名PDF，130个图点与汇总逐值核对，最大误差1.42e−14pp。图中n=1，无模型种子SD或CI。完整分层数值见[results.json](results.json)，先于正式读取的源决定见[source_decision.json](source_decision.json)。
