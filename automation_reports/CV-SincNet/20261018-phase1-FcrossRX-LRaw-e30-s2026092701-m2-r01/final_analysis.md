# 末10轮L ordinary输入改raw：E30结果分析

源两项合取通过，仅获得E50来源资格，不代表已经执行或自动扩预算。源 fullV clean 的新As−新NoAs为 -0.029630 个百分点（要求≥−0.1）；两draw mid/low等权的新As−旧As为 +1.138889 个百分点（要求≥+0.5）。该源决定在正式结果读取前保存，正式分数不改变它。

正式测试的新As clean为 84.352381%，相对新NoAs +1.633929 个百分点，通过 clean下降不超过0.1个绝对百分点的阶段约束。low/mid urban为 51.597619%/62.907143%；相对旧As的变化为 +0.353571/+0.100595 个百分点，clean -0.988690 个百分点。只有模型种子2026092701，未获得可重复的星地与clean双提升证据。

20261018-phase1-FcrossRX-LRaw-e30-s2026092701-m2-r01。四逻辑条件为旧NOAS30、旧AS30、LRAW_NOAS30、LRAW_AS30；每模型总E30。仅两个新行各从原P43 own20学生/EMA合法来源追加10轮。默认cosine头随骨干联合训练，6L/总L尺度1.25、ORD128 SupCon成员、As(raw64)/NoAs raw64及物理信道保持。末10轮普通L输入返回raw128，因此既有ORD128锚实际也来自raw128；Uweak/strong、教师和EMA均保留。

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
|LRaw NoAs|clean|82.718452|82.337475|75.050000|
|LRaw NoAs|practical_high|41.388690|37.473265|30.070833|
|LRaw NoAs|practical_mid|39.658929|36.409801|29.370833|
|LRaw NoAs|practical_low_suburban|37.877381|35.118040|28.483333|
|LRaw NoAs|practical_high_urban|39.981548|36.630720|29.745833|
|LRaw NoAs|practical_mid_urban|33.397024|31.932659|25.962500|
|LRaw NoAs|practical_low_urban|28.450595|27.924007|22.520833|
|LRaw As|clean|84.352381|83.962034|77.091667|
|LRaw As|practical_high|81.120238|80.799680|70.254167|
|LRaw As|practical_mid|78.575595|78.282313|69.000000|
|LRaw As|practical_low_suburban|75.966667|75.700869|66.775000|
|LRaw As|practical_high_urban|77.566667|77.260552|67.687500|
|LRaw As|practical_mid_urban|62.907143|62.672125|55.991667|
|LRaw As|practical_low_urban|51.597619|51.275459|46.458333|

|条件|urban三场景均值 %|
|---|---|
|旧NoAs|35.570833|
|旧As|63.939286|
|LRaw NoAs|33.943056|
|LRaw As|64.023810|

## 配对变化与错误转移

新As−新NoAs衡量新配方的As总效应；新As−旧As、新NoAs−旧NoAs分别衡量LRaw对两种训练的增量。不能把NoAs退化计作纯As收益。交互=(新As−新NoAs)−(旧As−旧NoAs)，不将不同实验增益相加预测结果。

|比较|视图|accuracy Δpp|macro-F1 Δpp|最差RX Δpp|修复|新增错误|净修复|
|---|---|---|---|---|---|---|---|
|LRaw As−LRaw NoAs|clean|+1.633929|+1.624560|+2.041667|6724|3979|2745|
|LRaw As−LRaw NoAs|practical_high|+39.731548|+43.326416|+40.183333|71136|4387|66749|
|LRaw As−LRaw NoAs|practical_mid|+38.916667|+41.872512|+39.629167|70438|5058|65380|
|LRaw As−LRaw NoAs|practical_low_suburban|+38.089286|+40.582828|+38.291667|69702|5712|63990|
|LRaw As−LRaw NoAs|practical_high_urban|+37.585119|+40.629832|+37.941667|68269|5126|63143|
|LRaw As−LRaw NoAs|practical_mid_urban|+29.510119|+30.739466|+30.029167|57793|8216|49577|
|LRaw As−LRaw NoAs|practical_low_urban|+23.147024|+23.351452|+23.937500|49377|10490|38887|
|LRaw As−旧As|clean|-0.988690|-1.082440|-1.270833|1719|3380|-1661|
|LRaw As−旧As|practical_high|-0.285119|-0.284550|-1.654167|1964|2443|-479|
|LRaw As−旧As|practical_mid|-0.085714|-0.077393|-1.212500|2670|2814|-144|
|LRaw As−旧As|practical_low_suburban|+0.104762|+0.128128|-0.966667|3200|3024|176|
|LRaw As−旧As|practical_high_urban|-0.200595|-0.162655|-1.420833|2505|2842|-337|
|LRaw As−旧As|practical_mid_urban|+0.100595|+0.221662|-0.466667|4695|4526|169|
|LRaw As−旧As|practical_low_urban|+0.353571|+0.388424|+0.079167|5983|5389|594|
|LRaw NoAs−旧NoAs|clean|-2.529167|-2.642161|-3.400000|1661|5910|-4249|
|LRaw NoAs−旧NoAs|practical_high|-1.296429|-1.838152|-4.233333|13213|15391|-2178|
|LRaw NoAs−旧NoAs|practical_mid|-1.336310|-1.545012|-4.150000|13204|15449|-2245|
|LRaw NoAs−旧NoAs|practical_low_suburban|-1.595238|-1.640690|-3.570833|12725|15405|-2680|
|LRaw NoAs−旧NoAs|practical_high_urban|-1.458333|-1.850310|-4.254167|12990|15440|-2450|
|LRaw NoAs−旧NoAs|practical_mid_urban|-1.735714|-1.513944|-3.533333|11721|14637|-2916|
|LRaw NoAs−旧NoAs|practical_low_urban|-1.689286|-1.260582|-3.229167|10828|13666|-2838|
|旧As−旧NoAs|clean|+0.093452|+0.064839|-0.087500|3868|3711|157|
|旧As−旧NoAs|practical_high|+38.720238|+41.772814|+37.604167|68469|3419|65050|
|旧As−旧NoAs|practical_mid|+37.666071|+40.404893|+36.691667|67279|4000|63279|
|旧As−旧NoAs|practical_low_suburban|+36.389286|+38.814010|+35.687500|65618|4484|61134|
|旧As−旧NoAs|practical_high_urban|+36.327381|+38.942178|+35.108333|65141|4111|61030|
|旧As−旧NoAs|practical_mid_urban|+27.673810|+29.003860|+26.962500|53261|6769|46492|
|旧As−旧NoAs|practical_low_urban|+21.104167|+21.702446|+20.629167|44297|8842|35455|
|LRaw As−旧NoAs|clean|-0.895238|-1.017601|-1.358333|4101|5605|-1504|
|LRaw As−旧NoAs|practical_high|+38.435119|+41.488264|+35.950000|68205|3634|64571|
|LRaw As−旧NoAs|practical_mid|+37.580357|+40.327500|+35.479167|67447|4312|63135|
|LRaw As−旧NoAs|practical_low_suburban|+36.494048|+38.942138|+34.720833|66238|4928|61310|
|LRaw As−旧NoAs|practical_high_urban|+36.126786|+38.779523|+33.687500|65190|4497|60693|
|LRaw As−旧NoAs|practical_mid_urban|+27.774405|+29.225522|+26.495833|54465|7804|46661|
|LRaw As−旧NoAs|practical_low_urban|+21.457738|+22.090870|+20.708333|46249|10200|36049|

|比较|urban均值 Δpp|
|---|---|
|LRaw As−LRaw NoAs|+30.080754|
|LRaw As−旧As|+0.084524|
|LRaw NoAs−旧NoAs|-1.627778|
|旧As−旧NoAs|+28.368452|
|LRaw As−旧NoAs|+28.452976|

|视图|交互 Δpp|
|---|---|
|clean|+1.540476|
|practical_high|+1.011310|
|practical_mid|+1.250595|
|practical_low_suburban|+1.700000|
|practical_high_urban|+1.257738|
|practical_mid_urban|+1.836310|
|practical_low_urban|+2.042857|
|urban_mean|+1.712302|

## 源域结果与资格

固定规则只有上述两项；其余源RX/日期分层、交互及NoAs变化均报告，不添加每cell或逐包置信度门槛。源失败仍完成全部正式预测和独立评分。

|条件|fullV clean %|panel clean %|mid/low等权 %|
|---|---|---|---|
|旧NoAs|98.359259|98.444444|39.583333|
|旧As|98.255556|98.333333|65.944444|
|LRaw NoAs|98.348148|98.555556|39.708333|
|LRaw As|98.318519|98.333333|67.083333|

|条件|draw|mid urban %|low urban %|
|---|---|---|---|
|旧NoAs|1|42.944444|35.666667|
|旧NoAs|2|44.166667|35.555556|
|旧As|1|72.555556|60.111111|
|旧As|2|73.055556|58.055556|
|LRaw NoAs|1|43.166667|36.444444|
|LRaw NoAs|2|43.888889|35.333333|
|LRaw As|1|73.555556|60.833333|
|LRaw As|2|74.000000|59.944444|

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
|LRaw NoAs|practical_high|41.388690|41.329762|
|LRaw NoAs|practical_mid|39.658929|43.059524|
|LRaw NoAs|practical_low_suburban|37.877381|44.841071|
|LRaw As|practical_high|81.120238|3.232143|
|LRaw As|practical_mid|78.575595|5.776786|
|LRaw As|practical_low_suburban|75.966667|8.385714|

|新As−旧As|clean变化 pp|suburban变化 pp|自身差距变化 pp|
|---|---|---|---|
|practical_high|-0.988690|-0.285119|-0.703571|
|practical_mid|-0.988690|-0.085714|-0.902976|
|practical_low_suburban|-0.988690|+0.104762|-1.093452|

差距变化=clean变化−suburban变化。负clean变化本身会缩小差距，不能将此归为suburban保护。

## 实际训练成本与历史复用

旧行native列为累计E30；新行为本轮末10轮。四种NN计数直接来自实际计数器，教师eval使用实际计数器减原计数offset；U与EMA均真实继续。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source读出NN|formal预测NN|native秒|
|---|---|---|---|---|---|---|---|---|
|旧NoAs|6078780|918004|0|963900|7960684|12600|1176002|2146.190|
|旧As|6078780|918004|0|963900|7960684|12600|1176002|2860.909|
|LRaw NoAs|2271960|324002|0|567000|3162962|12600|1176002|815.364|
|LRaw As|2271960|324002|0|567000|3162962|12600|1176002|1044.778|

|新条件|父20 native NN（复用）|新增末10 native NN|累计30 native NN|
|---|---|---|---|
|LRaw NoAs|4797722|3162962|7960684|
|LRaw As|4797722|3162962|7960684|

本轮新增20 modelEpoch/4440更新；训练及source/formal读出8703128NN、共享smoke1540NN，总新增8704668NN。旧两E30及其读出共18298572NN复用，不重计新增；每新行累计E30/6660更新。wall为模型阶段秒，不能据此推断并行加速。

原ordinary增强在每step仍完整执行一次128包，消耗该call原随机数并完成计算，只丢弃其输出。返回值是pre-call raw clone，并由两条独立日志流lraw_steps与ordinary_draw_steps核对。没有省去ordinary计算；其单独CPU/GPU耗时未拆分时仅披露已包含在native wall，不虚构独立节省。

|新条件|末段L包|U消费包|本段EMA更新|累计EMA更新|teacher train/eval NN|
|---|---:|---:|---:|---:|---|
|LRaw NoAs|284160|567000|2220|6660|0/567000|
|LRaw As|284160|567000|2220|6660|0/567000|

|新条件|factory创建数|ordinary调用/包|弃用MSE mean/max|弃用后变化包数|实际used-raw MSE最大值|
|---|---|---|---|---|---|
|LRaw NoAs|1|2220/284160|0.22721489/2.6144626|155228|0.0|
|LRaw As|1|2220/284160|0.22721489/2.6144626|155228|0.0|

|条件|source预测wall秒|formal预测wall秒|
|---|---|---|
|旧NoAs|未单列记录|48.08935279189609|
|旧As|未单列记录|49.649732606019825|
|LRaw NoAs|未单列记录|48.29190647392534|
|LRaw As|未单列记录|47.518481956096366|

source_summary.wall_seconds属于离线分析处理耗时，不能当作source模型预测耗时；source预测wall若未在既有costs中单列，即明确缺失。该耗时缺失不改变已实计的NN计数或预测/评分完成状态。

## 结论边界与完整性

前20轮保留原ordinary增强、7轮原U训练和ownEMA历史，不能称全程NoAug。末10轮只将L ordinary槽的最终返回改为raw；完整oldaug仍运行一次，Uweak/strong、U目标、学生U前向、教师前向与EMA更新均保留。该变动同时影响普通L分类与原ORD128 SupCon所看到的输入，不把效果归因到其中某个单独损失。

保留的是该ordinary call的随机数消耗与计算，不声称完整训练的dropout、教师、伪标签接受和所有随机轨迹逐步相同。discardedordinary_MSE是非负有限观察量，允许为0，不按其大小筛选样本或增加来源门槛。As仍从raw64生成，不能称本机制修复了ordinary与As串联重复损伤。

最终状态ANALYZED；全部4×7固定预测和独立truth-last评分后，既有聚合重算504张CM/84次RX/day/TX分区合并。新两行完整2220step/10epoch各自解析，包含两条LRaw流及完整L/U物理序列；旧审计复用，不回扫历史stepstream。完整分层错误保存在原聚合，本精简JSON不复制CM、逐ID、原预测或日志。

只有一个模型seed。阶段clean基本不降、单seed局部提高或不到0.1个百分点的正值，均不是可重复双提升证据；既有暴露基准背景保留，正式分数不用于新设计、晋级或选择性重跑。源双规则通过只表示E50资格，分析不自动扩预算。

缺项：0（既有11个最终汇总文件齐备）。本整理新增NN/CP/IQ/预测/评分/更新/远程调用均为0。


E50执行决定：源判据两项PASS已于读取本次formal前保存，随后先登记 20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01 两条own30+20；本次formal仅如实报告，不回流决定或参数。E50当前PLANNED，实现中，尚无50结果。
