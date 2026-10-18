# As-only label smoothing .01→.05：E30三条件结果

来源双规则未通过，关闭固定epsilon .05，不E50、不扫epsilon或种子救跑。来源新As fullVclean−原NoAs为-0.107407个绝对百分点（要求≥−0.1），两draw mid/low等权新As−原As为-0.138889个百分点（要求≥+0.5）。来源决定由root在正式结果读取前保存；正式结果不反向改变它。

正式AsLS clean为85.423810%，相对原NoAs +0.176190个百分点，通过clean下降不超过0.1个绝对百分点的阶段约束；相对原As的clean变化为+0.082738个百分点。AsLS low/mid urban为50.901786%/62.435714%，相对原As变化-0.342262/-0.370833个百分点。只有模型seed2026092701，不能据此声称可重复双提升。

20261018-phase1-FcrossRX-AsLS-e30-s2026092701-m1-r01。新1个B01、旧2个冻结条件：NOAS30、AS30、SATLS_AS30。新增行从原P43 ownAs20学生与ownEMA追加10轮，累计E30/6660更新；不是LRaw、CE温度或其他变体父。默认cosine头随骨干联合训练。

唯一变量是现有mixed128中selected64 As两次CE的epsilon从已有0.01改为固定0.05。六类true target从0.9916667改为0.9583333、各其他类从0.0016667改为0.0083333；不是hard/onehot改soft。common ordinary、mixed另一64、raw shape、U仍0.01，mixed128对称KL、ORD128 SupCon、全部合法As真标签监督、6L/总L1.25和As总体0.25份额不变。RDrop两pass复用同一已生成IQ，不重抽practical。没有置信筛选、类专属规则、新头、门控、分支或新增NN前向。

完整 RX／日期／TX 配对结果见[分层分析](strata_analysis.md)。mid／low urban在全部7个RX下降；clean微增主要来自TX3抵消部分TX损失，不能称所有身份均受益。

## 七视图绝对结果

每视图n=168000，以下accuracy、macro-F1、最差RX均直接取已有独立重计汇总。urban均值仅三场urban accuracy等权。

|条件|视图|accuracy %|macro-F1 %|最差RX %|
|---|---|---|---|---|
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
|SATLS_AS30|clean|85.423810|85.130727|78.504167|
|SATLS_AS30|practical_high|81.076190|80.752195|70.400000|
|SATLS_AS30|practical_mid|78.306548|77.993946|68.941667|
|SATLS_AS30|practical_low_suburban|75.498214|75.194455|66.587500|
|SATLS_AS30|practical_high_urban|77.476786|77.116139|67.825000|
|SATLS_AS30|practical_mid_urban|62.435714|62.038704|55.454167|
|SATLS_AS30|practical_low_urban|50.901786|50.513573|45.583333|

|条件|urban三场均值 %|
|---|---|
|NOAS30|35.570833|
|AS30|63.939286|
|SATLS_AS30|63.604762|

## 三组配对与错误转移

新As−原NoAs是当前配方As总体效应；新As−原As才是新增label smoothing的同初始化/同NN方法增量；原As−原NoAs作既有As参照。无As时该处理恒等，因此没有新NoAs行，不构造假四因子或交互。

|比较|视图|accuracy Δpp|macro-F1 Δpp|最差RX Δpp|修复|新增错误|净修复|
|---|---|---|---|---|---|---|---|
|SATLS_AS30−NOAS30|clean|+0.176190|+0.151092|+0.054167|3899|3603|296|
|SATLS_AS30−NOAS30|practical_high|+38.391071|+41.440779|+36.095833|67900|3403|64497|
|SATLS_AS30−NOAS30|practical_mid|+37.311310|+40.039133|+35.420833|66661|3978|62683|
|SATLS_AS30−NOAS30|practical_low_suburban|+36.025595|+38.435724|+34.533333|64951|4428|60523|
|SATLS_AS30−NOAS30|practical_high_urban|+36.036905|+38.635109|+33.825000|64640|4098|60542|
|SATLS_AS30−NOAS30|practical_mid_urban|+27.302976|+28.592101|+25.958333|52555|6686|45869|
|SATLS_AS30−NOAS30|practical_low_urban|+20.761905|+21.328984|+19.833333|43601|8721|34880|
|SATLS_AS30−AS30|clean|+0.082738|+0.086253|+0.141667|416|277|139|
|SATLS_AS30−AS30|practical_high|-0.329167|-0.332035|-1.508333|609|1162|-553|
|SATLS_AS30−AS30|practical_mid|-0.354762|-0.365760|-1.270833|665|1261|-596|
|SATLS_AS30−AS30|practical_low_suburban|-0.363690|-0.378286|-1.154167|701|1312|-611|
|SATLS_AS30−AS30|practical_high_urban|-0.290476|-0.307069|-1.283333|643|1131|-488|
|SATLS_AS30−AS30|practical_mid_urban|-0.370833|-0.411759|-1.004167|780|1403|-623|
|SATLS_AS30−AS30|practical_low_urban|-0.342262|-0.373462|-0.795833|930|1505|-575|
|AS30−NOAS30|clean|+0.093452|+0.064839|-0.087500|3868|3711|157|
|AS30−NOAS30|practical_high|+38.720238|+41.772814|+37.604167|68469|3419|65050|
|AS30−NOAS30|practical_mid|+37.666071|+40.404893|+36.691667|67279|4000|63279|
|AS30−NOAS30|practical_low_suburban|+36.389286|+38.814010|+35.687500|65618|4484|61134|
|AS30−NOAS30|practical_high_urban|+36.327381|+38.942178|+35.108333|65141|4111|61030|
|AS30−NOAS30|practical_mid_urban|+27.673810|+29.003860|+26.962500|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|+21.104167|+21.702446|+20.629167|44297|8842|35455|

|比较|urban均值 Δpp|
|---|---|
|SATLS_AS30−NOAS30|+28.033929|
|SATLS_AS30−AS30|-0.334524|
|AS30−NOAS30|+28.368452|

## 来源证据与正式结论分开

只应用既定两条来源规则，不以熵、置信度、有效秩或CE下降替代身份识别收益；来源失败仍完成既定七视图正式收尾。

|条件|fullV clean %|V1800 clean %|mid/low两draw等权 %|
|---|---|---|---|
|NOAS30|98.359259|98.444444|39.583333|
|AS30|98.255556|98.333333|65.944444|
|SATLS_AS30|98.251852|98.388889|65.805556|

|条件|draw|mid urban %|low urban %|
|---|---|---|---|
|NOAS30|1|42.944444|35.666667|
|NOAS30|2|44.166667|35.555556|
|AS30|1|72.555556|60.111111|
|AS30|2|73.055556|58.055556|
|SATLS_AS30|1|72.500000|60.111111|
|SATLS_AS30|2|72.666667|57.944444|

fullV27000使用冻结原生accuracy/RX，不虚构fullV逐包F1/修复。V1800三场两draw不是两个模型seed，源没有high urban；panel2.45GHz与训练/正式2.462GHz差异保留。全5源RX、L6300/U56700真值隐藏/V27000、256点、split392005和物理权限不变，不留出一个训练RX。

## suburban绝对值与自身clean差距

|条件|视图|accuracy %|自身clean−suburban pp|
|---|---|---|---|
|NOAS30|practical_high|42.685119|42.562500|
|NOAS30|practical_mid|40.995238|44.252381|
|NOAS30|practical_low_suburban|39.472619|45.775000|
|AS30|practical_high|81.405357|3.935714|
|AS30|practical_mid|78.661310|6.679762|
|AS30|practical_low_suburban|75.861905|9.479167|
|SATLS_AS30|practical_high|81.076190|4.347619|
|SATLS_AS30|practical_mid|78.306548|7.117262|
|SATLS_AS30|practical_low_suburban|75.498214|9.925595|

|新As−原As|clean Δpp|suburban Δpp|差距 Δpp|
|---|---|---|---|
|practical_high|+0.082738|-0.329167|+0.411905|
|practical_mid|+0.082738|-0.354762|+0.437500|
|practical_low_suburban|+0.082738|-0.363690|+0.446429|

差距变化=clean变化−suburban变化。clean下降本身会缩小差距，不能自动当作suburban保护。

## 实际训练与额外CE计算成本

原两行native为累计E30，新As行native为新增末10轮。四种模型NN分别来自实际计数器；父训练和旧读出不再计入本轮新增。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source读出NN|formal预测NN|native秒|
|---|---|---|---|---|---|---|---|---|
|NOAS30|6078780|918004|0|963900|7960684|12600|1176002|2146.190|
|AS30|6078780|918004|0|963900|7960684|12600|1176002|2860.909|
|SATLS_AS30|2271960|324002|0|567000|3162962|12600|1176002|1064.360|

本轮实际新增10modelEpoch/2220更新；训练及source/formal读出4351564NN，共享smoke770NN，总新增4352334NN。预登记参考为4352334NN（含770smoke）；实际与参考差+0NN，以真实计数为准，不把参考数预填成实测。父20 native 4797722NN复用，候选累计30 native 7960684NN。旧两E30及读出18298572NN单列复用。

|项目|实际记录|
|---|---|
|新增CE operator次数|8880|
|新增CE logit包|568320|
|原mixed CE宽度|[128, 128]|
|额外CE宽度|[64, 64, 64, 64]|
|实际mixed CE全部宽度|[128, 128, 64, 64, 64, 64]|
|operator CPU wall秒|2.9181240575853735|
|scalarization CPU wall秒|0.502464838558808|
|计时语义|perf_counter CPUwall,notCUDAevent;no explicitCUDA synchronization;scalar reads may synchronize|

固定实现每step额外4次selected64 CE，完整末10轮应记录8880次、568320个logit包；原mixed两次128包CE均值复用。新增loss与autograd标量计算不是免费。perf_counter记录CPU wall，不是CUDA event；无显式CUDA同步，scalar读取可能同步；额外GPU反向耗时未独立隔离，不能由零新增模型NN推断零计算成本或提速。

|条件|source预测wall秒|formal预测wall秒|
|---|---|---|
|NOAS30|未单列记录|48.08935279189609|
|AS30|未单列记录|49.649732606019825|
|SATLS_AS30|未单列记录|48.94721947587095|

source_summary.wall_seconds为离线分析耗时，不能冒充source模型预测wall；未记录的计时明确缺失。native wall是模型阶段秒，不据此推断并行速度。

新行完整2220step/10epoch、selected64两pass old/new CE、mixed128差值/原KL/完整L/U流均解析。实际U消费567000包，EMA本段2220次、累计6660次。旧original30完整audit直接复用，不回扫旧stepstream。

## 完整性与解释边界

最终ANALYZED；全部3×7固定预测与独立truth-last评分齐后，既有分析重核378CM/63次RX/day/TX分区合并。本公开JSON不复制CM、逐ID、原预测、logit、原日志或私有runtime路径。

理论动机仅是更强的已有uniform label smoothing可能改变As CE的梯度；它不是传播质量门控或已证实的标签噪声处理。不能把CE_T2/CE_HALF旧失败或clean-only平滑论文直接当本方法收益。来源与正式结论分别报告；单种子、微小clean上涨或阶段PASS都不等于可重复的星地增强同时改善clean。正式不反馈选模、epsilon、场景、seed或选择性重跑。

缺项：0（11个最终汇总文件齐备）；未单列的extra计时字段：无。本整理新增NN/CP/IQ/预测/评分/训练更新/远程调用均0。
