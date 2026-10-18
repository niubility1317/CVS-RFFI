# As64 TX×RX 分层选择：E30 三条件结果

源域两条件未通过，关闭固定 waterfill64 配方，不 E50、不扫分组或种子救跑。新 As fullV clean−原 NoAs 为 -0.114815 个绝对百分点（要求 ≥−0.1），两 draw mid/low 等权新 As−原 As 为 -0.013889 个百分点（要求 ≥+0.5）。根任务先保存源决定，再读取正式结果；正式结果仅描述，不反向改变资格或关闭决定。

正式 clean 为 85.281548%，相对原 NoAs +0.033929 个绝对百分点，0.1 pp 阶段约束=通过；相对原 As -0.059524 pp。low/mid urban 为 51.322619%/62.822024%，相对原 As +0.078571/+0.015476 pp。模型 seed 仅 2026092701；阶段通过或局部微增不能证明可重复双提升。

20261018-phase1-FcrossRX-AsAlloc-e30-s2026092701-m1-r01。新增一个 B01，由原 P43 ownAs20 学生与 ownEMA 追加 10 轮，累计 E30/6660 更新；原 NOAS30 与 AS30 固定读出复用。默认 cosine 头随骨干联合训练。

唯一变量是从原 L128 位置按可用 TX×RX cell 做可行 waterfill 配额，无放回选择 As64。原 L128 loader、ordinary128、ORD128 SupCon、raw shape、global label_smoothing=.01、mixed128 对称 KL、6L/768NN、总 L 尺度 1.25、As .25 份额及原 U／教师／EMA 均保持。mixed 中 As 位置和 ordinary 补集均改变，每个输入位置恰有一个 mixed 视图，所有位置仍保留 clean 监督。无置信度筛选、标签白名单、U 真值读取、新分支或新增神经前向。

选择器改变有限 TX/RX/day/physical 的 As 曝光，不能称已隔离纯梯度方差；同轨迹旧 parity mask 只是元数据反事实，不是原 As30 的逐步重放。输入物理 ID 沿原 sampler，如有重复不去重。不能声称逐步 IQ、dropout、U 或完整优化轨迹相同。

完整跨 RX／日期／TX 配对表见[分层分析](strata_analysis.md)。mid仅3/7个RX提高，low为5/7；两场最差RX分别下降.333333/.125000pp。局部微增并非普遍改善。

## 七视图绝对结果

每视图 n=168000。accuracy、macro-F1、最差 RX 取现有独立 truth-last 重计汇总；urban 均值为三种 urban accuracy 等权。

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
|ALLOC_AS30|clean|85.281548|84.988245|78.308333|
|ALLOC_AS30|practical_high|81.361905|81.051862|71.270833|
|ALLOC_AS30|practical_mid|78.623214|78.329245|69.708333|
|ALLOC_AS30|practical_low_suburban|75.877381|75.590190|67.295833|
|ALLOC_AS30|practical_high_urban|77.767262|77.430416|68.475000|
|ALLOC_AS30|practical_mid_urban|62.822024|62.449036|56.125000|
|ALLOC_AS30|practical_low_urban|51.322619|50.904235|46.254167|

|条件|urban 三场均值 %|
|---|---|
|NOAS30|35.570833|
|AS30|63.939286|
|ALLOC_AS30|63.970635|

## 三项配对与错误转移

新 As−原 NoAs 是当前配方总体 As 效应；新 As−原 As 才是新增选择机制增量；原 As−原 NoAs 为原普通 practical 效应。无 As 时本操作恒等，没有新 NoAs 行，不构造假四行因子或交互。

|比较|视图|accuracy Δpp|F1 Δpp|最差 RX Δpp|修复|新增错误|净修复|
|---|---|---|---|---|---|---|---|
|ALLOC_AS30−NOAS30|clean|+0.033929|+0.008610|-0.141667|3822|3765|57|
|ALLOC_AS30−NOAS30|practical_high|+38.676786|+41.740446|+36.966667|68486|3509|64977|
|ALLOC_AS30−NOAS30|practical_mid|+37.627976|+40.374432|+36.187500|67270|4055|63215|
|ALLOC_AS30−NOAS30|practical_low_suburban|+36.404762|+38.831459|+35.241667|65663|4503|61160|
|ALLOC_AS30−NOAS30|practical_high_urban|+36.327381|+38.949386|+34.475000|65209|4179|61030|
|ALLOC_AS30−NOAS30|practical_mid_urban|+27.689286|+29.002433|+26.629167|53342|6824|46518|
|ALLOC_AS30−NOAS30|practical_low_urban|+21.182738|+21.719645|+20.504167|44485|8898|35587|
|ALLOC_AS30−AS30|clean|-0.059524|-0.056229|-0.054167|335|435|-100|
|ALLOC_AS30−AS30|practical_high|-0.043452|-0.032368|-0.637500|718|791|-73|
|ALLOC_AS30−AS30|practical_mid|-0.038095|-0.030461|-0.504167|781|845|-64|
|ALLOC_AS30−AS30|practical_low_suburban|+0.015476|+0.017449|-0.445833|943|917|26|
|ALLOC_AS30−AS30|practical_high_urban|+0.000000|+0.007208|-0.633333|844|844|0|
|ALLOC_AS30−AS30|practical_mid_urban|+0.015476|-0.001427|-0.333333|1200|1174|26|
|ALLOC_AS30−AS30|practical_low_urban|+0.078571|+0.017200|-0.125000|1516|1384|132|
|AS30−NOAS30|clean|+0.093452|+0.064839|-0.087500|3868|3711|157|
|AS30−NOAS30|practical_high|+38.720238|+41.772814|+37.604167|68469|3419|65050|
|AS30−NOAS30|practical_mid|+37.666071|+40.404893|+36.691667|67279|4000|63279|
|AS30−NOAS30|practical_low_suburban|+36.389286|+38.814010|+35.687500|65618|4484|61134|
|AS30−NOAS30|practical_high_urban|+36.327381|+38.942178|+35.108333|65141|4111|61030|
|AS30−NOAS30|practical_mid_urban|+27.673810|+29.003860|+26.962500|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|+21.104167|+21.702446|+20.629167|44297|8842|35455|

|比较|urban 均值 Δpp|
|---|---|
|ALLOC_AS30−NOAS30|+28.399802|
|ALLOC_AS30−AS30|+0.031349|
|AS30−NOAS30|+28.368452|

## 源域与正式结论分开

|条件|fullV clean %|panel clean %|两 draw mid/low %|
|---|---|---|---|
|NOAS30|98.359259|98.444444|39.583333|
|AS30|98.255556|98.333333|65.944444|
|ALLOC_AS30|98.244444|98.277778|65.930556|

|条件|draw|mid urban %|low urban %|
|---|---|---|---|
|NOAS30|1|42.944444|35.666667|
|NOAS30|2|44.166667|35.555556|
|AS30|1|72.555556|60.111111|
|AS30|2|73.055556|58.055556|
|ALLOC_AS30|1|72.611111|60.055556|
|ALLOC_AS30|2|73.000000|58.055556|

fullV27000 使用冻结原生 accuracy/RX，不能虚构全 V 逐包 F1 或修复。源 V1800 三场两 draw 独立重计，两 draw 不是模型 seed，源没有 high urban。面板 2.45 GHz 与训练/正式 2.462 GHz 差异保留。全 5 源 RX、L6300/U56700 真值隐藏/V27000、256 点、split392005 和物理权限不变，无 RX 留出。

## suburban 绝对值与差距

|条件|视图|accuracy %|自身 clean−suburban pp|
|---|---|---|---|
|NOAS30|practical_high|42.685119|42.562500|
|NOAS30|practical_mid|40.995238|44.252381|
|NOAS30|practical_low_suburban|39.472619|45.775000|
|AS30|practical_high|81.405357|3.935714|
|AS30|practical_mid|78.661310|6.679762|
|AS30|practical_low_suburban|75.861905|9.479167|
|ALLOC_AS30|practical_high|81.361905|3.919643|
|ALLOC_AS30|practical_mid|78.623214|6.658333|
|ALLOC_AS30|practical_low_suburban|75.877381|9.404167|

|新 As−原 As|clean Δpp|suburban Δpp|差距 Δpp|
|---|---|---|---|
|practical_high|-0.059524|-0.043452|-0.016071|
|practical_mid|-0.059524|-0.038095|-0.021429|
|practical_low_suburban|-0.059524|+0.015476|-0.075000|

差距变化=clean 变化−suburban 变化。clean 降低也会缩差，必须同时看绝对准确率。

## 实际训练、曝光与成本

旧两条件 native 为累计 E30，新条件 native 为新增末 10 轮。父 E20、旧模型及历史读出复用单列，不计为新训练。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source NN|formal NN|native 秒|
|---|---|---|---|---|---|---|---|---|
|NOAS30|6078780|918004|0|963900|7960684|12600|1176002|2146.190|
|AS30|6078780|918004|0|963900|7960684|12600|1176002|2860.909|
|ALLOC_AS30|2271960|324002|0|567000|3162962|12600|1176002|1057.828|

本轮新增 10 model-epoch/2220 更新；训练及 source/formal 读出 4351564 NN，共享 smoke 770 NN，总新增 4352334 NN。预登记参考 4352334 NN（含 770 smoke），实际差 +0 NN，以实际计数为准。父 E20 native 4797722 NN，候选累计 E30 native 7960684 NN；旧两 E30 及读出 18298572 NN 复用。

|选择器项目|实际记录|
|---|---|
|As 选择调用|2220|
|As 包|142080|
|改变位置数（相对当前轨迹旧 mask）|142222|
|selector CPU wall 秒|1.6620997083373368|
|hook CPU wall 秒|2.5821187591645867|
|新增 CE operator|0|
|计时语义|perf_counter CPUwall;hook includesselector andcopies,do notaddboth;no explicitCUDA synchronization|

selector 时间已包含在 hook 时间，不能相加；CPU wall 不是 CUDA event，不隔离 GPU 前向/反向耗时。没有额外 NN 不代表选择器零计算。完整每 cell availability/selected/旧 mask 曝光和物理曝光直方图保留在公开 JSON，旧 mask 不等于原训练实际曝光。

|选择曝光维度|stratum|As 包|
|---|---|---|
|TX|2|23750|
|TX|5|23614|
|TX|1|23640|
|TX|0|23723|
|TX|3|23670|
|TX|4|23683|
|RX|6|28454|
|RX|4|28483|
|RX|3|28388|
|RX|8|28451|
|RX|1|28304|
|day|1|47360|
|day|3|47381|
|day|2|47339|

|条件|source 预测 wall 秒|formal 预测 wall 秒|
|---|---|---|
|NOAS30|未单列记录|48.08935279189609|
|AS30|未单列记录|49.649732606019825|
|ALLOC_AS30|未单列记录|52.119795670965686|

新增段完整 2220 步/10 轮、waterfill 配额/位置/补集及完整 L/U 流均解析；实际 U 消费 567000 包，EMA 本段 2220 次、累计 6660 次。旧 original30 完整审计复用，不回扫旧 stepstream。source 模型预测 wall 若未记录即明确缺失，不能用离线分析 wall 替代。

## 完整性与解释边界

全部 3×7 固定预测与独立 truth-last 评分齐后，既有分析重核 378 CM/63 次 RX/day/TX 分区合并。最终状态 ANALYZED；技术缺项：0（11 个最终汇总文件齐备）。source 预测 wall 未单列的条件：NOAS30, AS30, ALLOC_AS30。

n=1；阶段通过、局部增益或不足 0.1 pp 的上涨不能证明可重复 clean 正收益。已暴露基准背景保留，正式不反馈参数、seed、parent 或选择性重跑。固定源规则失败即关闭此配方，源通过仅 eligible，分析不启动扩预算。本整理新增 NN/CP/IQ/更新/预测/评分/远程调用均 0。
