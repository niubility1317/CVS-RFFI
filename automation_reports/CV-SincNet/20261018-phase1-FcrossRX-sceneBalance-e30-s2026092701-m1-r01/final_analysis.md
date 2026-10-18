# 同步三场景配额 E30：最终分析

固定配额方案在正式评分前已经因两项源规则均失败而关闭：fullV clean 的 BAL−NOAS30 为 -0.118519 个百分点（要求 ≥−0.1），mid／low 两 draw 等权的 BAL−AS30 为 +0.041667 个百分点（要求 ≥+0.5）。不 E50、不换种子救跑；正式结果不改变该决定。

正式结果中，BAL_AS30 的 clean 为 85.395833%，相对同预算 NOAS30 提高 0.148214 个百分点，满足 clean 下降不超过 0.1 个百分点的阶段约束；low／mid urban 为 51.233929%／62.796429%。但相对原 As30，新增配额的 low／mid urban 均为 -0.010119 个百分点，urban 均值仅 +0.001587 个百分点；clean +0.054762 个百分点也是单种子的微小观察。相对 NOAS30 的大幅星地收益主要来自既有普通 As，不能归给新增配额；尚未证明配额同时改善星地和 clean，更没有可重复双提升证据。

运行：20261018-phase1-FcrossRX-sceneBalance-e30-s2026092701-m1-r01。三条件均为总 E30：仅 BAL_AS30 从 own 原 As20 student／EMA 追加 10 轮；NOAS30、AS30 是已冻结的原 P43 E30 参照。新增实验 10 modelEpoch／2220 更新，累计 30 轮／6660 更新；默认分类头、6L、总损失尺度 1.25、ORD128 SupCon 和原 U 规则保持。

## 正式七视图

全部数值来自已完成的固定预测、独立 truth-last 评分及既有独立聚合重算。每个视图 n=168000。以下先列绝对值，再列原 As 效应与新增分层分配效应。

|条件|视图|accuracy %|macro-F1 %|最差 RX %|
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
|BAL_AS30|clean|85.395833|85.104079|78.654167|
|BAL_AS30|practical_high|81.376190|81.061004|71.729167|
|BAL_AS30|practical_mid|78.624405|78.316556|70.095833|
|BAL_AS30|practical_low_suburban|75.802976|75.502409|67.562500|
|BAL_AS30|practical_high_urban|77.792262|77.441906|68.987500|
|BAL_AS30|practical_mid_urban|62.796429|62.419160|56.391667|
|BAL_AS30|practical_low_urban|51.233929|50.870397|46.412500|

|条件|urban 三场景均值 %|
|---|---|
|NOAS30|35.570833|
|AS30|63.939286|
|BAL_AS30|63.940873|

|比较|视图|accuracy Δpp|macro-F1 Δpp|最差 RX Δpp|修复|新增错误|净修复|
|---|---|---|---|---|---|---|---|
|AS30__NOAS30|clean|+0.093452|+0.064839|-0.087500|3868|3711|157|
|AS30__NOAS30|practical_high|+38.720238|+41.772814|+37.604167|68469|3419|65050|
|AS30__NOAS30|practical_mid|+37.666071|+40.404893|+36.691667|67279|4000|63279|
|AS30__NOAS30|practical_low_suburban|+36.389286|+38.814010|+35.687500|65618|4484|61134|
|AS30__NOAS30|practical_high_urban|+36.327381|+38.942178|+35.108333|65141|4111|61030|
|AS30__NOAS30|practical_mid_urban|+27.673810|+29.003860|+26.962500|53261|6769|46492|
|AS30__NOAS30|practical_low_urban|+21.104167|+21.702446|+20.629167|44297|8842|35455|
|BAL_AS30__AS30|clean|+0.054762|+0.059605|+0.291667|356|264|92|
|BAL_AS30__AS30|practical_high|-0.029167|-0.023226|-0.179167|599|648|-49|
|BAL_AS30__AS30|practical_mid|-0.036905|-0.043149|-0.116667|671|733|-62|
|BAL_AS30__AS30|practical_low_suburban|-0.058929|-0.070332|-0.179167|748|847|-99|
|BAL_AS30__AS30|practical_high_urban|+0.025000|+0.018699|-0.120833|705|663|42|
|BAL_AS30__AS30|practical_mid_urban|-0.010119|-0.031303|-0.066667|1066|1083|-17|
|BAL_AS30__AS30|practical_low_urban|-0.010119|-0.016638|+0.033333|1290|1307|-17|
|BAL_AS30__NOAS30|clean|+0.148214|+0.124443|+0.204167|3880|3631|249|
|BAL_AS30__NOAS30|practical_high|+38.691071|+41.749588|+37.425000|68490|3489|65001|
|BAL_AS30__NOAS30|practical_mid|+37.629167|+40.361743|+36.575000|67243|4026|63217|
|BAL_AS30__NOAS30|practical_low_suburban|+36.330357|+38.743679|+35.508333|65505|4470|61035|
|BAL_AS30__NOAS30|practical_high_urban|+36.352381|+38.960877|+34.987500|65226|4154|61072|
|BAL_AS30__NOAS30|practical_mid_urban|+27.663690|+28.972558|+26.895833|53186|6711|46475|
|BAL_AS30__NOAS30|practical_low_urban|+21.094048|+21.685807|+20.662500|44207|8769|35438|

|比较|urban 均值 Δpp|
|---|---|
|AS30__NOAS30|+28.368452|
|BAL_AS30__AS30|+0.001587|
|BAL_AS30__NOAS30|+28.370040|

AS30−NOAS30 是原完整 practical 增强的总效应；BAL_AS30−AS30 才是本次固定分层分配的增量；BAL_AS30−NOAS30 是新配方的总效应。这里只有三个条件，没有新的平衡 NoAs 条件，因此不构造四格因子交互，也不把整个 As 收益归给配额。

## 源域先行决定

源 fullV clean：BAL−NOAS30 -0.118519 个百分点（要求 ≥−0.1）；两 draw 的 mid／low 等权：BAL−AS30 +0.041667 个百分点（要求 ≥+0.5）。两项合取通过=False；未通过固定源规则，关闭本固定配方，不 E50／扫参数／选种子救跑。源失败不跳过正式评分，正式结果不改变该源决定。

|条件|fullV clean %|panel clean %|mid／low 等权 %|
|---|---|---|---|
|NOAS30|98.359259|98.444444|39.583333|
|AS30|98.255556|98.333333|65.944444|
|BAL_AS30|98.240741|98.277778|65.986111|

|条件|源 panel 视图|accuracy %|macro-F1 %|最差 RX %|
|---|---|---|---|---|
|NOAS30|clean|98.444444|98.441469|95.555556|
|NOAS30|practical_low_suburban|49.861111|46.937084|41.805556|
|NOAS30|practical_mid_urban|43.555556|41.595346|36.111111|
|NOAS30|practical_low_urban|35.611111|34.525219|30.277778|
|AS30|clean|98.333333|98.329950|95.833333|
|AS30|practical_low_suburban|88.666667|88.571599|82.361111|
|AS30|practical_mid_urban|72.805556|72.719826|68.472222|
|AS30|practical_low_urban|59.083333|58.744394|55.555556|
|BAL_AS30|clean|98.277778|98.274555|95.555556|
|BAL_AS30|practical_low_suburban|88.666667|88.565108|82.222222|
|BAL_AS30|practical_mid_urban|72.888889|72.802511|67.777778|
|BAL_AS30|practical_low_urban|59.083333|58.761203|55.138889|

|条件|draw|mid urban %|low urban %|
|---|---|---|---|
|NOAS30|1|42.944444|35.666667|
|NOAS30|2|44.166667|35.555556|
|AS30|1|72.555556|60.111111|
|AS30|2|73.055556|58.055556|
|BAL_AS30|1|72.666667|60.000000|
|BAL_AS30|2|73.111111|58.166667|

fullV 27000 仅使用原生冻结 accuracy／RX 指标，不声称已经得到 fullV 逐包 F1、修复或新增错误。V1800 的两个传播 draw 不是两个模型种子；源 panel 没有 high urban。保留源 panel 2.45 GHz 与训练／正式 2.462 GHz 的既有差异。

## suburban 与 clean 的差距

|条件|suburban 视图|绝对 accuracy %|自身 clean−suburban pp|
|---|---|---|---|
|NOAS30|practical_high|42.685119|42.562500|
|NOAS30|practical_mid|40.995238|44.252381|
|NOAS30|practical_low_suburban|39.472619|45.775000|
|AS30|practical_high|81.405357|3.935714|
|AS30|practical_mid|78.661310|6.679762|
|AS30|practical_low_suburban|75.861905|9.479167|
|BAL_AS30|practical_high|81.376190|4.019643|
|BAL_AS30|practical_mid|78.624405|6.771429|
|BAL_AS30|practical_low_suburban|75.802976|9.592857|

本次相对原 As30，三个 suburban 的绝对准确率均下降，而 clean 提高 0.054762 个百分点，差距分别扩大 0.083929、0.091667、0.113690 个百分点。因此没有 suburban 保护增益，也不存在通过降低 clean 缩小差距的情形。

## 实际成本与完整性

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|源读出 NN|正式预测 NN|native 秒|
|---|---|---|---|---|---|---|---|---|
|NOAS30|6078780|918004|0|963900|7960684|12600|1176002|2146.190|
|AS30|6078780|918004|0|963900|7960684|12600|1176002|2860.909|
|BAL_AS30|2271960|324002|0|567000|3162962|12600|1176002|1037.987|

本轮新增训练及读出 4351564 NN；共享 smoke 770 NN；总新增 4352334 NN。own As20 父 native 成本为 4797722 NN，新增尾段 native 成本为 3162962 NN，累计候选 native NN 为 7960684；原两个 E30 的训练与读出复用，不重复计入本轮新增。wall 是模型阶段秒，不当作并行加速证据。

最终状态 ANALYZED；独立重算 378 张 CM、63 次 RX／day／TX 分区合并。新尾段 2220 step／10 epoch 完整解析；旧原始 E30 完整 audit 复用，不回扫旧 step stream。完整分层指标、配对错误和曲线继续保留在原聚合中。

## 归因和结论边界

当前普通 As 已经逐物理样本独立抽取三场景；新增机制只是原 As64 输入位置内精确 21／21／22 配额、额外位置按累计 step 轮换、私有 permutation 分配。保留的是 nominal 1／3，并非与原 IID 每包、每物理 ID 或全尾段实际场景次数逐一匹配。配额使尾段每场景固定 47360 次，原 IID 实际次数可能略偏。因此结果归于整套固定分层分配，不能声称已隔离纯梯度方差。

不按 TX、RX、day、质量、置信度或 PID 唯一性分配；原采样器可能重复物理 ID，继续保留并报告，不去重。原物理 channel seed／state／session、received IQ、正式 SNR、补偿和全部困难样本保持。n=1 的局部正收益、不到 0.1 个百分点的上涨、源检查通过都不等于可重复双提升。当前已完成结果仅用于描述，不据正式分数设计或晋级新方案。

缺项：0（既有 11 个最终汇总文件齐备）。本整理新增 NN／CP／IQ／预测／评分／更新均为 0；不读取真值、原始预测、物理 ID 或原训练日志。


源读出 wall 未单独记录；新正式预测 wall 为47.651990秒，native训练阶段为1037.987181秒，不能用聚合分析耗时补作source inference计时。源high urban、fullV逐包F1/错误配对和正式channel-state分层未提供，属于可用指标边界；预登记产物缺项仍为0。尚未完成多模型种子重复验证；low urban低于60%，suburban与clean差仍明显，整体目标未完成。

## 科研图与汇总

[绝对表现](figures/scenebalance_single_seed_absolute.png)、[三项配对](figures/scenebalance_single_seed_paired.png)、[clean护栏与F1/最差RX](figures/scenebalance_single_seed_clean_and_quality.png)。同名PDF可导出；95个图点逐值核对，最大误差1.42e−14pp。数值见[results.json](results.json)，源先读关闭决定见[source_decision.json](source_decision.json)。
