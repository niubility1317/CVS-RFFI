# 末10轮停用U与EMA：E30结果分析

源两项合取未通过，按固定规则关闭本配方，不E50、不改U/phase或选种子救跑。源 fullV clean 的新As−新NoAs为 -0.133333 个百分点（要求≥−0.1）；两draw mid/low等权的新As−旧As为 +0.097222 个百分点（要求≥+0.5）。该源决定在正式结果读取前保存，正式分数不改变它。

正式测试的新As clean为 85.330357%，相对新NoAs +0.216071 个百分点，通过 clean下降不超过0.1个绝对百分点的阶段约束。low/mid urban为 51.437500%/63.033333%；相对旧As的变化为 +0.193452/+0.226786 个百分点，clean -0.010714 个百分点。只有模型种子2026092701，未获得可重复的星地与clean双提升证据。

20261018-phase1-FcrossRX-tailNoU-e30-s2026092701-m2-r01。四逻辑条件为旧NOAS30、旧AS30、TAIL_NO_U_NOAS30、TAIL_NO_U_AS30；每模型总E30。仅两个新行各从原P43 own20学生追加10轮，父EMA只验证来源、子不恢复EMA，默认cosine头随骨干联合训练，6L/总L尺度1.25、ORD128 SupCon、As(raw64)/NoAs raw64及物理信道保持。

## 正式七视图

每视图n=168000。绝对准确率、macro-F1和最差RX如下；urban均值仅指三种urban准确率等权均值。

|条件|视图|accuracy %|macro-F1 %|最差RX %|
|---|---|---|---|---|
|旧NoAs|clean|85.247619|84.979636|78.450000|
|旧NoAs|practical_high|42.685119|39.311416|34.304167|
|旧NoAs|practical_mid|40.995238|37.954813|33.520833|
|旧NoAs|practical_low_suburban|39.472619|36.758731|32.054167|
|旧NoAs|practical_high_urban|41.439881|38.481030|34.000000|
|旧NoAs|practical_mid_urban|35.132738|33.446603|29.495833|
|旧NoAs|practical_low_urban|30.139881|29.184589|25.750000|
|旧As|clean|85.341071|85.044474|78.362500|
|旧As|practical_high|81.405357|81.084230|71.908333|
|旧As|practical_mid|78.661310|78.359706|70.212500|
|旧As|practical_low_suburban|75.861905|75.572741|67.741667|
|旧As|practical_high_urban|77.767262|77.423208|69.108333|
|旧As|practical_mid_urban|62.806548|62.450463|56.458333|
|旧As|practical_low_urban|51.244048|50.887035|46.379167|
|tailNoU NoAs|clean|85.114286|84.839196|77.708333|
|tailNoU NoAs|practical_high|42.220238|38.576706|35.600000|
|tailNoU NoAs|practical_mid|40.723810|37.338108|34.700000|
|tailNoU NoAs|practical_low_suburban|39.242857|36.166995|32.941667|
|tailNoU NoAs|practical_high_urban|41.064881|37.812279|35.354167|
|tailNoU NoAs|practical_mid_urban|35.030952|32.972330|30.237500|
|tailNoU NoAs|practical_low_urban|30.158929|28.912511|26.200000|
|tailNoU As|clean|85.330357|85.052154|78.666667|
|tailNoU As|practical_high|81.641667|81.312725|72.608333|
|tailNoU As|practical_mid|78.769048|78.458260|70.508333|
|tailNoU As|practical_low_suburban|76.018452|75.709909|68.075000|
|tailNoU As|practical_high_urban|77.951190|77.612224|69.433333|
|tailNoU As|practical_mid_urban|63.033333|62.627333|56.837500|
|tailNoU As|practical_low_urban|51.437500|50.956783|46.791667|

|条件|urban三场景均值 %|
|---|---|
|旧NoAs|35.570833|
|旧As|63.939286|
|tailNoU NoAs|35.418254|
|tailNoU As|64.140675|

## 配对变化与错误转移

新As−新NoAs衡量新配方的As总效应；新As−旧As、新NoAs−旧NoAs分别衡量tail-NoU对两种训练的增量。不能把NoAs退化计作纯As收益。交互=(新As−新NoAs)−(旧As−旧NoAs)，不将不同实验增益相加预测结果。

|比较|视图|accuracy Δpp|macro-F1 Δpp|最差RX Δpp|修复|新增错误|净修复|
|---|---|---|---|---|---|---|---|
|tailNoU As−tailNoU NoAs|clean|+0.216071|+0.212958|+0.958333|3972|3609|363|
|tailNoU As−tailNoU NoAs|practical_high|+39.421429|+42.736018|+37.008333|69556|3328|66228|
|tailNoU As−tailNoU NoAs|practical_mid|+38.045238|+41.120152|+35.808333|67855|3939|63916|
|tailNoU As−tailNoU NoAs|practical_low_suburban|+36.775595|+39.542914|+35.133333|66162|4379|61783|
|tailNoU As−tailNoU NoAs|practical_high_urban|+36.886310|+39.799945|+34.079167|66056|4087|61969|
|tailNoU As−tailNoU NoAs|practical_mid_urban|+28.002381|+29.655003|+26.600000|53800|6756|47044|
|tailNoU As−tailNoU NoAs|practical_low_urban|+21.278571|+22.044271|+20.591667|44604|8856|35748|
|tailNoU As−旧As|clean|-0.010714|+0.007680|+0.304167|736|754|-18|
|tailNoU As−旧As|practical_high|+0.236310|+0.228495|+0.700000|1095|698|397|
|tailNoU As−旧As|practical_mid|+0.107738|+0.098555|+0.295833|1209|1028|181|
|tailNoU As−旧As|practical_low_suburban|+0.156548|+0.137168|+0.333333|1428|1165|263|
|tailNoU As−旧As|practical_high_urban|+0.183929|+0.189016|+0.325000|1235|926|309|
|tailNoU As−旧As|practical_mid_urban|+0.226786|+0.176870|+0.379167|2085|1704|381|
|tailNoU As−旧As|practical_low_urban|+0.193452|+0.069748|+0.412500|2547|2222|325|
|tailNoU NoAs−旧NoAs|clean|-0.133333|-0.140440|-0.741667|780|1004|-224|
|tailNoU NoAs−旧NoAs|practical_high|-0.464881|-0.734710|+1.295833|2586|3367|-781|
|tailNoU NoAs−旧NoAs|practical_mid|-0.271429|-0.616705|+1.179167|2642|3098|-456|
|tailNoU NoAs−旧NoAs|practical_low_suburban|-0.229762|-0.591736|+0.887500|2599|2985|-386|
|tailNoU NoAs−旧NoAs|practical_high_urban|-0.375000|-0.668750|+1.354167|2649|3279|-630|
|tailNoU NoAs−旧NoAs|practical_mid_urban|-0.101786|-0.474273|+0.741667|2690|2861|-171|
|tailNoU NoAs−旧NoAs|practical_low_urban|+0.019048|-0.272078|+0.450000|2828|2796|32|
|旧As−旧NoAs|clean|+0.093452|+0.064839|-0.087500|3868|3711|157|
|旧As−旧NoAs|practical_high|+38.720238|+41.772814|+37.604167|68469|3419|65050|
|旧As−旧NoAs|practical_mid|+37.666071|+40.404893|+36.691667|67279|4000|63279|
|旧As−旧NoAs|practical_low_suburban|+36.389286|+38.814010|+35.687500|65618|4484|61134|
|旧As−旧NoAs|practical_high_urban|+36.327381|+38.942178|+35.108333|65141|4111|61030|
|旧As−旧NoAs|practical_mid_urban|+27.673810|+29.003860|+26.962500|53261|6769|46492|
|旧As−旧NoAs|practical_low_urban|+21.104167|+21.702446|+20.629167|44297|8842|35455|
|tailNoU As−旧NoAs|clean|+0.082738|+0.072518|+0.216667|3802|3663|139|
|tailNoU As−旧NoAs|practical_high|+38.956548|+42.001309|+38.304167|68720|3273|65447|
|tailNoU As−旧NoAs|practical_mid|+37.773810|+40.503447|+36.987500|67386|3926|63460|
|tailNoU As−旧NoAs|practical_low_suburban|+36.545833|+38.951178|+36.020833|65788|4391|61397|
|tailNoU As−旧NoAs|practical_high_urban|+36.511310|+39.131194|+35.433333|65377|4038|61339|
|tailNoU As−旧NoAs|practical_mid_urban|+27.900595|+29.180730|+27.341667|53647|6774|46873|
|tailNoU As−旧NoAs|practical_low_urban|+21.297619|+21.772193|+21.041667|44582|8802|35780|

|比较|urban均值 Δpp|
|---|---|
|tailNoU As−tailNoU NoAs|+28.722421|
|tailNoU As−旧As|+0.201389|
|tailNoU NoAs−旧NoAs|-0.152579|
|旧As−旧NoAs|+28.368452|
|tailNoU As−旧NoAs|+28.569841|

|视图|交互 Δpp|
|---|---|
|clean|+0.122619|
|practical_high|+0.701190|
|practical_mid|+0.379167|
|practical_low_suburban|+0.386310|
|practical_high_urban|+0.558929|
|practical_mid_urban|+0.328571|
|practical_low_urban|+0.174405|
|urban_mean|+0.353968|

## 源域结果与资格

固定规则只有上述两项；其余源RX/日期分层、交互及NoAs变化均报告，不添加每cell或逐包置信度门槛。源失败仍完成全部正式预测和独立评分。

|条件|fullV clean %|panel clean %|mid/low等权 %|
|---|---|---|---|
|旧NoAs|98.359259|98.444444|39.583333|
|旧As|98.255556|98.333333|65.944444|
|tailNoU NoAs|98.333333|98.444444|38.986111|
|tailNoU As|98.200000|98.277778|66.041667|

|条件|draw|mid urban %|low urban %|
|---|---|---|---|
|旧NoAs|1|42.944444|35.666667|
|旧NoAs|2|44.166667|35.555556|
|旧As|1|72.555556|60.111111|
|旧As|2|73.055556|58.055556|
|tailNoU NoAs|1|42.055556|35.500000|
|tailNoU NoAs|2|43.166667|35.222222|
|tailNoU As|1|72.611111|59.888889|
|tailNoU As|2|73.111111|58.555556|

fullV27000使用冻结原生accuracy/RX，不虚构fullV逐包F1或错误修复。源panel V1800的两传播draw不是两个模型seed，源三场没有high urban。保留panel2.45GHz与训练/正式2.462GHz的既有差异。全5源RX、L6300/U56700隐藏TX/V27000、256点IQ、split392005与物理ID权限保持。

## suburban绝对值与差距

|条件|视图|accuracy %|自身clean−suburban pp|
|---|---|---|---|
|旧NoAs|practical_high|42.685119|42.562500|
|旧NoAs|practical_mid|40.995238|44.252381|
|旧NoAs|practical_low_suburban|39.472619|45.775000|
|旧As|practical_high|81.405357|3.935714|
|旧As|practical_mid|78.661310|6.679762|
|旧As|practical_low_suburban|75.861905|9.479167|
|tailNoU NoAs|practical_high|42.220238|42.894048|
|tailNoU NoAs|practical_mid|40.723810|44.390476|
|tailNoU NoAs|practical_low_suburban|39.242857|45.871429|
|tailNoU As|practical_high|81.641667|3.688690|
|tailNoU As|practical_mid|78.769048|6.561310|
|tailNoU As|practical_low_suburban|76.018452|9.311905|

|新As−旧As|clean变化 pp|suburban变化 pp|自身差距变化 pp|
|---|---|---|---|
|practical_high|-0.010714|+0.236310|-0.247024|
|practical_mid|-0.010714|+0.107738|-0.118452|
|practical_low_suburban|-0.010714|+0.156548|-0.167262|

差距变化=clean变化−suburban变化。负clean变化本身会缩小差距，不能将此归为suburban保护。

## 实际训练成本与历史复用

旧行native列为累计E30；新行为本轮末10轮。四种NN计数直接来自实际计数器，teacher未构建时不虚构offset。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source读出NN|formal预测NN|native秒|
|---|---|---|---|---|---|---|---|---|
|旧NoAs|6078780|918004|0|963900|7960684|12600|1176002|2146.190|
|旧As|6078780|918004|0|963900|7960684|12600|1176002|2860.909|
|tailNoU NoAs|1704960|324002|0|0|2028962|12600|1176002|593.149|
|tailNoU As|1704960|324002|0|0|2028962|12600|1176002|855.627|

|新条件|父20 native NN（复用）|新增末10 native NN|累计30 native NN|
|---|---|---|---|
|tailNoU NoAs|4797722|2028962|6826684|
|tailNoU As|4797722|2028962|6826684|

本轮新增20 modelEpoch/4440更新；训练及source/formal读出6435128NN、共享smoke1540NN，总新增6436668NN。旧两E30及其读出共18298572NN复用，不重计新增；每新行累计E30/6660更新。wall为模型阶段秒，不能据此推断并行加速。

|新条件|末段L包|U消费包|child EMA更新|parent EMA历史更新|teacher train/eval NN|
|---|---:|---:|---:|---:|---|
|tailNoU NoAs|284160|0|0|4440|0/0|
|tailNoU As|284160|0|0|4440|0/0|

## 结论边界与完整性

前20轮仍有7轮原U训练和own EMA历史；父EMA只核实来源，不在子训练恢复。末10轮同时取消U遍历、全部U目标、U学生前向、教师构建/前向与EMA更新，不能称全程NoSSL，也不能隔离归因给某个伪标签系数或EMA。停止U会改变dropout、ordinary及loader随机流和实际计算量，不声称新旧逐步IQ/随机轨迹一致。

原生a1_ema_successful_updates虽保留成功optimizer步数语义，不能当作实际EMA更新；实际子EMA更新为0、child ema_model=None，父EMA历史4440单列。phase标签仅披露，真实停U以flags、零消费、零前向和完整L序列为依据。

最终状态ANALYZED；全部4×7固定预测和独立truth-last评分后，既有聚合重算504张CM/84次RX/day/TX分区合并。新两行完整2220step/10epoch各自解析；旧审计复用，不回扫历史stepstream。完整分层错误保存在原聚合，本精简JSON不复制CM、逐ID、原预测或日志。

只有一个模型seed。阶段clean基本不降、单seed局部提高或不到0.1个百分点的正值，均不是可重复双提升证据；既有暴露基准背景保留，正式分数不用于新设计、晋级或选择性重跑。

缺项：0（既有11个最终汇总文件齐备）。本整理新增NN/CP/IQ/预测/评分/更新/远程调用均为0。

实际训练/配置/分析代码提交：17cc611299091815ea71f744a32dcac96818ca4b。源汇总先取回并由主线程保存判定，随后才取回/读取正式汇总；源归档比较和最终11文件归档比较均 VERIFIED。最终进程核实 dispatcher、两worker和0NN分析均已正常结束，failure为空；没有重训、重复预测或评分。

原生训练秒包含其自身验证与信道处理，正式预测秒另存于精简结果；source inference wall未单独记录，不伪造。技术声明产物缺项为0；单seed和上述度量可用性边界仍保留。
