# UQualified E30：冻结结果整理

源域固定两条件为FAIL；正式 Q−N clean -0.475000 个百分点，0.1 个百分点阶段约束未满足。单个模型种子不构成可重复双提升证据。

Q−C 路由增量：clean -0.475595、mid urban +0.276190、low urban +0.499405、urban 三场景均值 +0.264683 个百分点。Q−R 的对应值为 -0.565476／+0.283333／+0.491667／+0.267659。这两个比较的含义不同。

源决定由主线程先保存，再显式启动正式汇总；下面的正式结果只描述冻结模型，不改变源决定，不反向选择阈值、位置、种子或预算。


20261019-phase1-FcrossRX-UQualified-e30-s2026092701-m2-r01；model seed2026092701。新R/Q各original ownAs20+10，旧N/C完整训练、源与正式读出复用。

源两条件PASS=False；fullV Q−N -0.185185pp（≥−.1），两draw等权mid/low Q−C +0.083333pp（≥+.5）。源先保存，正式另一次显式调用；FAIL关闭固定配方，仍完成all7，不自动延长/扫阈值/选种子。

正式Q−N clean -0.475000pp，净修复-798；0.1pp阶段约束=False。n=1，不证明可重复双提升。

Q−C是完整route增量；Q−R只比较资格位置与同本行k规则随机位置。两行各自teacher随训练改变，未来k与累计剂量可不同，不宣称严格同剂量位置因果隔离。Q的As位置等于原conf&temporal资格；R均匀私有随机选取同本行资格数。k=0不调用channel，原Gaussian整批先生成。L6/1.25/ORD128、CE资格均值和全部U熵均不改，0额外NN；U真值始终隐藏。

|条件|视图|ACC %|macro-F1 %|最差RX %|
|---|---|---:|---:|---:|
|MATCHED_NOAS30|clean|85.289286|85.022839|78.812500|
|MATCHED_NOAS30|practical_high|42.454167|39.127962|33.583333|
|MATCHED_NOAS30|practical_mid|40.767857|37.789113|32.941667|
|MATCHED_NOAS30|practical_low_suburban|39.249405|36.620891|31.433333|
|MATCHED_NOAS30|practical_high_urban|41.178571|38.288154|33.312500|
|MATCHED_NOAS30|practical_mid_urban|34.901786|33.309767|28.866667|
|MATCHED_NOAS30|practical_low_urban|30.007738|29.168986|25.500000|
|UTEACHER30|clean|85.289881|84.990137|78.254167|
|UTEACHER30|practical_high|81.442857|81.127155|71.912500|
|UTEACHER30|practical_mid|78.673810|78.373516|70.166667|
|UTEACHER30|practical_low_suburban|75.869048|75.582751|67.708333|
|UTEACHER30|practical_high_urban|77.798810|77.461301|68.995833|
|UTEACHER30|practical_mid_urban|62.811905|62.451998|56.366667|
|UTEACHER30|practical_low_urban|51.276786|50.910140|46.416667|
|URANDOM30|clean|85.379762|85.088819|78.383333|
|URANDOM30|practical_high|81.427381|81.112540|71.733333|
|URANDOM30|practical_mid|78.674405|78.375172|70.150000|
|URANDOM30|practical_low_suburban|75.897024|75.608307|67.625000|
|URANDOM30|practical_high_urban|77.789286|77.452198|68.820833|
|URANDOM30|practical_mid_urban|62.804762|62.449806|56.400000|
|URANDOM30|practical_low_urban|51.284524|50.922322|46.400000|
|UQUALIFIED30|clean|84.814286|84.504302|78.541667|
|UQUALIFIED30|practical_high|81.448214|81.081120|73.333333|
|UQUALIFIED30|practical_mid|78.702381|78.347149|71.545833|
|UQUALIFIED30|practical_low_suburban|76.100000|75.758641|69.237500|
|UQUALIFIED30|practical_high_urban|77.817262|77.442299|70.450000|
|UQUALIFIED30|practical_mid_urban|63.088095|62.698377|57.500000|
|UQUALIFIED30|practical_low_urban|51.776190|51.303161|47.608333|

|配对|视图|ΔACC pp|修复|新增|净修复|
|---|---|---:|---:|---:|---:|
|UQUALIFIED30−UTEACHER30|clean|-0.475595|739|1538|-799|
|UQUALIFIED30−UTEACHER30|practical_high|+0.005357|1561|1552|9|
|UQUALIFIED30−UTEACHER30|practical_mid|+0.028571|1849|1801|48|
|UQUALIFIED30−UTEACHER30|practical_low_suburban|+0.230952|2205|1817|388|
|UQUALIFIED30−UTEACHER30|practical_high_urban|+0.018452|1786|1755|31|
|UQUALIFIED30−UTEACHER30|practical_mid_urban|+0.276190|2822|2358|464|
|UQUALIFIED30−UTEACHER30|practical_low_urban|+0.499405|3560|2721|839|
|UQUALIFIED30−URANDOM30|clean|-0.565476|663|1613|-950|
|UQUALIFIED30−URANDOM30|practical_high|+0.020833|1580|1545|35|
|UQUALIFIED30−URANDOM30|practical_mid|+0.027976|1821|1774|47|
|UQUALIFIED30−URANDOM30|practical_low_suburban|+0.202976|2144|1803|341|
|UQUALIFIED30−URANDOM30|practical_high_urban|+0.027976|1769|1722|47|
|UQUALIFIED30−URANDOM30|practical_mid_urban|+0.283333|2777|2301|476|
|UQUALIFIED30−URANDOM30|practical_low_urban|+0.491667|3509|2683|826|
|UQUALIFIED30−MATCHED_NOAS30|clean|-0.475000|3568|4366|-798|
|UQUALIFIED30−MATCHED_NOAS30|practical_high|+38.994048|69054|3544|65510|
|UQUALIFIED30−MATCHED_NOAS30|practical_mid|+37.934524|67863|4133|63730|
|UQUALIFIED30−MATCHED_NOAS30|practical_low_suburban|+36.850595|66547|4638|61909|
|UQUALIFIED30−MATCHED_NOAS30|practical_high_urban|+36.638690|65878|4325|61553|
|UQUALIFIED30−MATCHED_NOAS30|practical_mid_urban|+28.186310|54529|7176|47353|
|UQUALIFIED30−MATCHED_NOAS30|practical_low_urban|+21.768452|45882|9311|36571|
|URANDOM30−UTEACHER30|clean|+0.089881|248|97|151|
|URANDOM30−UTEACHER30|practical_high|-0.015476|259|285|-26|
|URANDOM30−UTEACHER30|practical_mid|+0.000595|280|279|1|
|URANDOM30−UTEACHER30|practical_low_suburban|+0.027976|313|266|47|
|URANDOM30−UTEACHER30|practical_high_urban|-0.009524|275|291|-16|
|URANDOM30−UTEACHER30|practical_mid_urban|-0.007143|379|391|-12|
|URANDOM30−UTEACHER30|practical_low_urban|+0.007738|467|454|13|
|URANDOM30−MATCHED_NOAS30|clean|+0.090476|3847|3695|152|
|URANDOM30−MATCHED_NOAS30|practical_high|+38.973214|68866|3391|65475|
|URANDOM30−MATCHED_NOAS30|practical_mid|+37.906548|67611|3928|63683|
|URANDOM30−MATCHED_NOAS30|practical_low_suburban|+36.647619|65983|4415|61568|
|URANDOM30−MATCHED_NOAS30|practical_high_urban|+36.610714|65603|4097|61506|
|URANDOM30−MATCHED_NOAS30|practical_mid_urban|+27.902976|53665|6788|46877|
|URANDOM30−MATCHED_NOAS30|practical_low_urban|+21.276786|44609|8864|35745|
|UTEACHER30−MATCHED_NOAS30|clean|+0.000595|3769|3768|1|
|UTEACHER30−MATCHED_NOAS30|practical_high|+38.988690|68900|3399|65501|
|UTEACHER30−MATCHED_NOAS30|practical_mid|+37.905952|67611|3929|63682|
|UTEACHER30−MATCHED_NOAS30|practical_low_suburban|+36.619643|65959|4438|61521|
|UTEACHER30−MATCHED_NOAS30|practical_high_urban|+36.620238|65612|4090|61522|
|UTEACHER30−MATCHED_NOAS30|practical_mid_urban|+27.910119|53637|6748|46889|
|UTEACHER30−MATCHED_NOAS30|practical_low_urban|+21.269048|44585|8853|35732|

|配对|urban三场景均值Δpp|
|---|---:|
|UQUALIFIED30−UTEACHER30|+0.264683|
|UQUALIFIED30−URANDOM30|+0.267659|
|UQUALIFIED30−MATCHED_NOAS30|+28.864484|
|URANDOM30−UTEACHER30|-0.002976|
|URANDOM30−MATCHED_NOAS30|+28.596825|
|UTEACHER30−MATCHED_NOAS30|+28.599802|

|条件|fullV clean %|V1800 clean %|两draw mid/low %|
|---|---:|---:|---:|
|MATCHED_NOAS30|98.355556|98.444444|39.208333|
|UTEACHER30|98.233333|98.277778|66.055556|
|URANDOM30|98.240741|98.333333|66.041667|
|UQUALIFIED30|98.170370|98.277778|66.138889|

源CSV保留完整draw/RX/day/TX/state/lock/实际新增噪声分层与现有logits margin，无新NN或附加资格门槛。fullV27000只有原生accuracy/RX，不伪造逐包F1；V1800两draw不是模型seed，源无highurban，2.45GHz与训练/正式2.462GHz边界保留。

|新行|资格总数|As总数|资格As|资格Gaussian|channel调用|空As step|
|---|---:|---:|---:|---:|---:|---:|
|URANDOM30|8713|8713|189|8524|1950|270|
|UQUALIFIED30|8668|8668|8668|0|1948|272|

完整u_steps重核原teacher资格、所选位置、按资格count平均的.16CE、全U平均的.01entropy；资格伪类计数/RXdayroute及As scene-state-lock-noise曝光是训练聚合，不是U真值准确率。Q全部资格As，R可能有无资格As参与熵；不按包数推断梯度强弱。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source NN|formal NN|
|---|---:|---:|---:|---:|---:|---:|---:|
|MATCHED_NOAS30|2271960|324002|0|567000|3162962|12600|1176002|
|UTEACHER30|2271960|324002|0|567000|3162962|12600|1176002|
|URANDOM30|2271960|324002|0|567000|3162962|12600|1176002|
|UQUALIFIED30|2271960|324002|0|567000|3162962|12600|1176002|

新增2尾段20modelEpoch/4440updates；实际训练+读出8703128NN，单parent共享smoke1570NN，新增总8704698NN。旧N/C及其原770NN失败均不重复计；共同As20父成本去重单列。每新行完整2220step/10epoch、L6300和hiddenU56700流；Ustudent/teacher各567000，As动态模拟不是零CPU成本。route wall含selection/simulation/copy，不相加、不当CUDAevent；source wall缺失保持null。

|条件|suburban|绝对ACC %|自身clean−suburban pp|
|---|---|---:|---:|
|MATCHED_NOAS30|practical_high|42.454167|42.835119|
|MATCHED_NOAS30|practical_mid|40.767857|44.521429|
|MATCHED_NOAS30|practical_low_suburban|39.249405|46.039881|
|UTEACHER30|practical_high|81.442857|3.847024|
|UTEACHER30|practical_mid|78.673810|6.616071|
|UTEACHER30|practical_low_suburban|75.869048|9.420833|
|URANDOM30|practical_high|81.427381|3.952381|
|URANDOM30|practical_mid|78.674405|6.705357|
|URANDOM30|practical_low_suburban|75.897024|9.482738|
|UQUALIFIED30|practical_high|81.448214|3.366071|
|UQUALIFIED30|practical_mid|78.702381|6.111905|
|UQUALIFIED30|practical_low_suburban|76.100000|8.714286|

clean降低也可缩小gap，应同时看绝对ACC。全部4×7固定预测与独立评分齐后truth join，重核504CM/84分区；完整6配对/RXdayTX错误分层保留。默认头联合训练，无留出RX/新诊断头/新部署结构；n1微增不证明稳定双提升，正式结果不反馈机制/参数/种子或救跑。分析NN/CP/IQ/updates均0。

## 动态资格、路由和损失范围

Q 将原 conf_mask & temporal_mask 的全部资格位置变成完整 practical(raw U)；R 私有均匀无放回选择同本行当前资格数 k 个位置。两行会各自更新模型和 EMA，未来资格数及累计 As 剂量可以不同，Q−R 不能表述为每一步严格同剂量的位置单因素因果隔离。资格政策、按资格数平均的 0.16 CE 与按全部 U 平均的 0.01 熵不变。

|新行|资格总次数|selected mask 次数|As 次数|资格 As|资格 Gaussian|非空 channel 调用|空 As step|零 As epoch|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|URANDOM30|8713|8713|8713|189|8524|1950|270|无|
|UQUALIFIED30|8668|8668|8668|8668|0|1948|272|无|

k=0 不调用 channel、不复制 practical 视图，返回原 Gaussian strong；它仍可通过全 U 熵项训练。Q 中资格槽均为 As，R 可有无资格 As 只参与熵。次数不能直接换算成梯度强弱；学生分歧和伪标签类别分布不是 U 真值准确率。现有聚合保留总数、空步骤和逐 epoch 均值，没有逐 batch k 的完整直方图，不臆造分位数或逐包曲线。

|条件|路由／资格组|伪类 0|伪类 1|伪类 2|伪类 3|伪类 4|伪类 5|
|---|---|---:|---:|---:|---:|---:|---:|
|MATCHED_NOAS30|Gaussian|94307|95542|95290|92764|94625|94472|
|MATCHED_NOAS30|Gaussian_qualified|1466|1411|1633|1304|1475|1678|
|UTEACHER30|Gaussian|94410|95997|94850|91944|95047|94752|
|UTEACHER30|Gaussian_qualified|1422|1350|1605|1208|1462|1675|
|URANDOM30|As|1466|1464|1454|1456|1387|1486|
|URANDOM30|As_qualified|37|27|35|22|26|42|
|URANDOM30|Gaussian|92944|94555|93395|90448|93667|93278|
|URANDOM30|Gaussian_qualified|1386|1323|1564|1183|1436|1632|
|UQUALIFIED30|As|1417|1335|1599|1184|1459|1674|
|UQUALIFIED30|As_qualified|1417|1335|1599|1184|1459|1674|
|UQUALIFIED30|Gaussian|92957|94674|93255|90654|93666|93126|

公开 JSON 保留全部 RX/day/route 的曝光及资格量，以及 As 的 scene/state/lock/实际新增噪声区间计数。这里的数字是合法隐藏真值 U 的训练覆盖，不是正式分层准确率。原 N/C 的审计与预测复用，不重新扫描它们的 raw 日志或进行模型前向。

## RX/day 配对修复与新增错误

以下来自已完成独立评分后的既有聚合。receiver 标签按维度使用，不由字符串推断 TX/RX 角色。RX 与 day 是同一批观测的不同分区，不能把它们相加成额外样本。

|配对|视图|维度|分层|ΔACC pp|修复|新增错误|净修复|
|---|---|---|---|---:|---:|---:|---:|
|UQUALIFIED30__UTEACHER30|clean|receiver|1-1|+0.337500|173|92|81|
|UQUALIFIED30__UTEACHER30|clean|receiver|14-7|-0.829167|245|444|-199|
|UQUALIFIED30__UTEACHER30|clean|receiver|2-1|-0.433333|63|167|-104|
|UQUALIFIED30__UTEACHER30|clean|receiver|20-1|-0.533333|55|183|-128|
|UQUALIFIED30__UTEACHER30|clean|receiver|7-14|-0.183333|72|116|-44|
|UQUALIFIED30__UTEACHER30|clean|receiver|7-7|+0.179167|127|84|43|
|UQUALIFIED30__UTEACHER30|clean|receiver|8-8|-1.866667|4|452|-448|
|UQUALIFIED30__UTEACHER30|clean|day|2021_03_01|+0.152381|324|260|64|
|UQUALIFIED30__UTEACHER30|clean|day|2021_03_08|-1.121429|104|575|-471|
|UQUALIFIED30__UTEACHER30|clean|day|2021_03_15|-0.383333|170|331|-161|
|UQUALIFIED30__UTEACHER30|clean|day|2021_03_23|-0.550000|141|372|-231|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|receiver|1-1|+0.158333|375|337|38|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|receiver|14-7|+1.133333|539|267|272|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|receiver|2-1|+0.316667|406|330|76|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|receiver|20-1|-0.229167|397|452|-55|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|receiver|7-14|+0.070833|388|371|17|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|receiver|7-7|+0.387500|379|286|93|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|receiver|8-8|+0.095833|338|315|23|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|day|2021_03_01|+0.245238|735|632|103|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|day|2021_03_08|+0.154762|697|632|65|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|day|2021_03_15|+0.407143|701|530|171|
|UQUALIFIED30__UTEACHER30|practical_mid_urban|day|2021_03_23|+0.297619|689|564|125|
|UQUALIFIED30__UTEACHER30|practical_low_urban|receiver|1-1|+0.179167|456|413|43|
|UQUALIFIED30__UTEACHER30|practical_low_urban|receiver|14-7|+1.191667|617|331|286|
|UQUALIFIED30__UTEACHER30|practical_low_urban|receiver|2-1|+0.600000|511|367|144|
|UQUALIFIED30__UTEACHER30|practical_low_urban|receiver|20-1|+0.375000|533|443|90|
|UQUALIFIED30__UTEACHER30|practical_low_urban|receiver|7-14|+0.320833|493|416|77|
|UQUALIFIED30__UTEACHER30|practical_low_urban|receiver|7-7|+0.379167|480|389|91|
|UQUALIFIED30__UTEACHER30|practical_low_urban|receiver|8-8|+0.450000|470|362|108|
|UQUALIFIED30__UTEACHER30|practical_low_urban|day|2021_03_01|+0.404762|893|723|170|
|UQUALIFIED30__UTEACHER30|practical_low_urban|day|2021_03_08|+0.719048|935|633|302|
|UQUALIFIED30__UTEACHER30|practical_low_urban|day|2021_03_15|+0.423810|855|677|178|
|UQUALIFIED30__UTEACHER30|practical_low_urban|day|2021_03_23|+0.450000|877|688|189|
|UQUALIFIED30__URANDOM30|clean|receiver|1-1|+0.208333|159|109|50|
|UQUALIFIED30__URANDOM30|clean|receiver|14-7|-0.866667|228|436|-208|
|UQUALIFIED30__URANDOM30|clean|receiver|2-1|-0.395833|62|157|-95|
|UQUALIFIED30__URANDOM30|clean|receiver|20-1|-0.595833|47|190|-143|
|UQUALIFIED30__URANDOM30|clean|receiver|7-14|-0.270833|57|122|-65|
|UQUALIFIED30__URANDOM30|clean|receiver|7-7|+0.020833|106|101|5|
|UQUALIFIED30__URANDOM30|clean|receiver|8-8|-2.058333|4|498|-494|
|UQUALIFIED30__URANDOM30|clean|day|2021_03_01|+0.040476|298|281|17|
|UQUALIFIED30__URANDOM30|clean|day|2021_03_08|-1.242857|88|610|-522|
|UQUALIFIED30__URANDOM30|clean|day|2021_03_15|-0.402381|158|327|-169|
|UQUALIFIED30__URANDOM30|clean|day|2021_03_23|-0.657143|119|395|-276|
|UQUALIFIED30__URANDOM30|practical_mid_urban|receiver|1-1|+0.075000|355|337|18|
|UQUALIFIED30__URANDOM30|practical_mid_urban|receiver|14-7|+1.100000|526|262|264|
|UQUALIFIED30__URANDOM30|practical_mid_urban|receiver|2-1|+0.291667|400|330|70|
|UQUALIFIED30__URANDOM30|practical_mid_urban|receiver|20-1|-0.150000|401|437|-36|
|UQUALIFIED30__URANDOM30|practical_mid_urban|receiver|7-14|+0.129167|393|362|31|
|UQUALIFIED30__URANDOM30|practical_mid_urban|receiver|7-7|+0.412500|370|271|99|
|UQUALIFIED30__URANDOM30|practical_mid_urban|receiver|8-8|+0.125000|332|302|30|
|UQUALIFIED30__URANDOM30|practical_mid_urban|day|2021_03_01|+0.130952|704|649|55|
|UQUALIFIED30__URANDOM30|practical_mid_urban|day|2021_03_08|+0.200000|695|611|84|
|UQUALIFIED30__URANDOM30|practical_mid_urban|day|2021_03_15|+0.445238|690|503|187|
|UQUALIFIED30__URANDOM30|practical_mid_urban|day|2021_03_23|+0.357143|688|538|150|
|UQUALIFIED30__URANDOM30|practical_low_urban|receiver|1-1|+0.154167|449|412|37|
|UQUALIFIED30__URANDOM30|practical_low_urban|receiver|14-7|+1.208333|615|325|290|
|UQUALIFIED30__URANDOM30|practical_low_urban|receiver|2-1|+0.579167|498|359|139|
|UQUALIFIED30__URANDOM30|practical_low_urban|receiver|20-1|+0.383333|524|432|92|
|UQUALIFIED30__URANDOM30|practical_low_urban|receiver|7-14|+0.275000|482|416|66|
|UQUALIFIED30__URANDOM30|practical_low_urban|receiver|7-7|+0.387500|470|377|93|
|UQUALIFIED30__URANDOM30|practical_low_urban|receiver|8-8|+0.454167|471|362|109|
|UQUALIFIED30__URANDOM30|practical_low_urban|day|2021_03_01|+0.371429|875|719|156|
|UQUALIFIED30__URANDOM30|practical_low_urban|day|2021_03_08|+0.769048|944|621|323|
|UQUALIFIED30__URANDOM30|practical_low_urban|day|2021_03_15|+0.440476|848|663|185|
|UQUALIFIED30__URANDOM30|practical_low_urban|day|2021_03_23|+0.385714|842|680|162|
|UQUALIFIED30__MATCHED_NOAS30|clean|receiver|1-1|-2.237500|403|940|-537|
|UQUALIFIED30__MATCHED_NOAS30|clean|receiver|14-7|-0.270833|811|876|-65|
|UQUALIFIED30__MATCHED_NOAS30|clean|receiver|2-1|-0.758333|449|631|-182|
|UQUALIFIED30__MATCHED_NOAS30|clean|receiver|20-1|-1.216667|346|638|-292|
|UQUALIFIED30__MATCHED_NOAS30|clean|receiver|7-14|+2.166667|730|210|520|
|UQUALIFIED30__MATCHED_NOAS30|clean|receiver|7-7|-0.054167|454|467|-13|
|UQUALIFIED30__MATCHED_NOAS30|clean|receiver|8-8|-0.954167|375|604|-229|
|UQUALIFIED30__MATCHED_NOAS30|clean|day|2021_03_01|-0.995238|843|1261|-418|
|UQUALIFIED30__MATCHED_NOAS30|clean|day|2021_03_08|-1.097619|957|1418|-461|
|UQUALIFIED30__MATCHED_NOAS30|clean|day|2021_03_15|-0.314286|852|984|-132|
|UQUALIFIED30__MATCHED_NOAS30|clean|day|2021_03_23|+0.507143|916|703|213|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|receiver|1-1|+28.112500|7981|1234|6747|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|receiver|14-7|+28.633333|7734|862|6872|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|receiver|2-1|+26.945833|7473|1006|6467|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|receiver|20-1|+23.816667|7120|1404|5716|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|receiver|7-14|+34.495833|9171|892|8279|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|receiver|7-7|+29.562500|7899|804|7095|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|receiver|8-8|+25.737500|7151|974|6177|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|day|2021_03_01|+27.819048|13945|2261|11684|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|day|2021_03_08|+27.776190|13235|1569|11666|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|day|2021_03_15|+27.052381|13111|1749|11362|
|UQUALIFIED30__MATCHED_NOAS30|practical_mid_urban|day|2021_03_23|+30.097619|14238|1597|12641|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|receiver|1-1|+21.533333|6639|1471|5168|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|receiver|14-7|+22.108333|6501|1195|5306|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|receiver|2-1|+21.166667|6413|1333|5080|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|receiver|20-1|+18.308333|6012|1618|4394|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|receiver|7-14|+26.495833|7607|1248|6359|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|receiver|7-7|+21.920833|6505|1244|5261|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|receiver|8-8|+20.845833|6205|1202|5003|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|day|2021_03_01|+21.259524|11684|2755|8929|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|day|2021_03_08|+21.895238|11287|2091|9196|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|day|2021_03_15|+21.140476|11093|2214|8879|
|UQUALIFIED30__MATCHED_NOAS30|practical_low_urban|day|2021_03_23|+22.778571|11818|2251|9567|

全部六配对、七视图和 RX/day/TX 计数保留在公开 JSON；这里没有重算预测或另作评分。

## 成本与缺项

实际新训练／读出 8703128 NN，共享 smoke 1570 NN，总计 8704698 NN。R/Q 共用的原 As20 父只计一次复用成本，旧 N/C 的 parent20 与累计 E30 native 成本单列；原 UPractical 运行的 770 NN 失败不进入本轮新成本。当前只读比较预检若由主线程另补 0 NN 记录，不改变这些计数。

两个新尾段各 E10／2220 更新，累计 E30；原 L 六前向及 U 一次教师／一次学生前向保持，practical 物理模拟是动态 CPU 成本而非新增神经前向。route host wall 包含 selection／simulation／copy，不能与子项相加，也不能当 CUDA event 或单独 kernel 时间。

汇总状态 ANALYZED；预期 11 文件缺项：0。source prediction wall 未记录：MATCHED_NOAS30, UTEACHER30, URANDOM30, UQUALIFIED30，保持 null。实际日志完整标记=True；独立分析计数为 504 CM／84 分区，矩阵不复制进公开 JSON。

full V27000 只有原生 accuracy/RX 汇总，没有逐包 F1 或错误修复。V1800 的两次信道 draw 不是两个模型种子；源无 high urban。源质量分层是描述性证据，不是额外资格门槛。suburban 差距必须与绝对准确率同时读，clean 降低造成的差距缩小不算保护成功。

四个逻辑条件不是 2×2 因子实验，不报告因子交互。Q−C 为路由总增量、Q−R 为当前资格位置与同本行 k 规则随机位置的比较、Q−N 为阶段 clean 约束；不把后者已有 L practical 优势全部归于 U 路由。n=1 或微小正差不证明稳定双提升。

本次整理 NN／CP／IQ／更新／新增预测／新增评分／远程调用均为 0。报告不生成下一机制、参数或预算决定。


## 执行证据与本轮判断

实际训练与分析代码提交 `fe734e2b571767bae00b86fd0a19fdc784dadcdf`。一次ordinary预检/归档transfer比较/remote compile、唯一独立P0/P1及其bootstrap定点修复、61聚焦负例/39正例、1570NN真实父及L/U无query smoke均已执行；之后实际两GPU训练各2220新增更新，累计E30。初次Windows路径序列化只读预检失败已保留，独立确认无archive/release/run/worker及0NN，6项路径边界修正后才首次传包。该失败不增加训练成本，不向本轮重复计旧Urun的770NN。

两行七视图prediction与独立truth-last均完整，504CM/84分区复计VERIFIED。dispatcher与两worker已自然退出，最终analysis状态ANALYZED，11个聚合文件齐全；源决定保存后才显式启动final分析。未留出训练RX、未更换默认头或正式IQ/信道，未依据正式分数改判或救跑。

本轮唯一假设已得到有限的单seed星地增量支持，但“同时保持clean”的部分被否定，固定资格路由关闭，不E50。Q−C low+0.499405pp且7/7RX、4/4day同向；clean−0.475595pp，5/7RX下降，8-8组净−448/24000、2021_03_08净−471/42000。已看到损伤分布，但隐藏U真值没有读取，不能断言是伪标签错误或某类信道因素导致。R只作控制，微小clean正差不构成备用晋级或重复双提升。

Q相对C的high/mid/lowsuburban准确率分别＋0.005357/＋0.028571/＋0.230952pp，而clean下降0.475595pp，差距缩小不能单独当保护收益。部分high/mid场景宏F1下降，完整表保留，未只汇报准确率正值。最终low51.776190%仍低于60%，suburban与自身clean差3.366071/6.111905/8.714286pp；最终研究目标仍未完成。

R/Q实际native wall1162.845318/1173.114870秒，U物理模拟8713/8668次，适配hostwall16.031957/16.204561秒；统计与资源条件未隔离，不能将微小时间差当加速证据。旧N/C读出复用，同预算NN保持，新增总8704698次样本前向（含1570smoke）。完整训练epoch曲线、资格/伪类/曝光与分层见[公开聚合](results_public.json)，[正式指标](formal_metrics.csv)、[源分层](source_metrics.csv)、[实际成本](costs.json)、[独立终态](run_verification.json)和[原预登记](report.md)。
