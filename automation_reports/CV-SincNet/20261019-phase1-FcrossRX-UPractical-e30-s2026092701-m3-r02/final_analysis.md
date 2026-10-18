# U practical E30：冻结结果整理

源域固定两条件结果为 FAIL；正式 S−匹配 N 的 clean 变化为 -0.150595 个百分点，0.1 个百分点阶段约束未满足。本轮只有 1 个模型种子，尚无可重复双提升证据。

U 视图增量 S−C：clean -0.151190、mid urban -0.040476、low urban +0.083929、urban 三场景均值 -0.032738 个百分点。S−匹配 N 的 mid/low urban 为 +27.869643/+21.352976；S−旧 As 的 clean 为 -0.202381 个百分点。三种配对含义不同，不能互相替代。

源域决定由主线程先保存，随后才单独启动正式汇总。下面的正式分数只描述冻结结果，不改变源决定，不用于调参、扩预算或选择性重跑。


20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m3-r02（immutable m3-r02）；model seed2026092701。N01从原NoAs20，C/S从同一原As20及各自ownEMA追加10轮。旧NoAs30/As30只读复用。

源资格PASS=False，固定配方关闭=True；fullV clean S−matchedN -0.137037pp（≥−.1），pooled两draw且mid/low等权 S−C -0.069444pp（≥+.5）。源先保存并由root读取，final是另一次显式调用；FAIL仍评分，分析不自动扩E50或seed。

正式 S−matchedN clean -0.150595pp、净修复-253，0.1绝对百分点阶段约束=False；mid/low urban +27.869643/+21.352976pp。n=1，不证明可重复双提升。

N/C/S共同去学生strong agreement，原raw ownEMA teacher的RX/day quantile与batch_neighbor资格保留。N/C保留原Gaussian strong(.015)；S先生成整批strong，再覆盖私有均匀B//4位置为完整practical(raw同一U)：256→64，末124→31。每U只有一次teacher和student前向，原L6/1.25/ORD128保留；U CE=.16×teacher-qualified槽位均值（空选为0），entropy=.01×全部U槽均值（概率clamp1e-8），包括未资格与As槽。无额外KL、U真值或新头。S−C是U视图增量；C−旧AS含共同去agreement改变，不是纯星地收益。五条件不是2×2因子。

|条件|视图|ACC %|macro-F1 %|最差RX %|
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
|UPRACTICAL30|clean|85.138690|84.838826|78.362500|
|UPRACTICAL30|practical_high|81.261905|80.935126|71.958333|
|UPRACTICAL30|practical_mid|78.545238|78.232346|70.300000|
|UPRACTICAL30|practical_low_suburban|75.798214|75.492551|67.750000|
|UPRACTICAL30|practical_high_urban|77.657143|77.303670|69.125000|
|UPRACTICAL30|practical_mid_urban|62.771429|62.383879|56.629167|
|UPRACTICAL30|practical_low_urban|51.360714|50.941533|46.641667|

|配对|视图|ΔACC pp|修复|新增|净修复|
|---|---|---:|---:|---:|---:|
|UPRACTICAL30−UTEACHER30|clean|-0.151190|273|527|-254|
|UPRACTICAL30−UTEACHER30|practical_high|-0.180952|637|941|-304|
|UPRACTICAL30−UTEACHER30|practical_mid|-0.128571|804|1020|-216|
|UPRACTICAL30−UTEACHER30|practical_low_suburban|-0.070833|940|1059|-119|
|UPRACTICAL30−UTEACHER30|practical_high_urban|-0.141667|764|1002|-238|
|UPRACTICAL30−UTEACHER30|practical_mid_urban|-0.040476|1197|1265|-68|
|UPRACTICAL30−UTEACHER30|practical_low_urban|+0.083929|1553|1412|141|
|UPRACTICAL30−MATCHED_NOAS30|clean|-0.150595|3709|3962|-253|
|UPRACTICAL30−MATCHED_NOAS30|practical_high|+38.807738|68808|3611|65197|
|UPRACTICAL30−MATCHED_NOAS30|practical_mid|+37.777381|67596|4130|63466|
|UPRACTICAL30−MATCHED_NOAS30|practical_low_suburban|+36.548810|65992|4590|61402|
|UPRACTICAL30−MATCHED_NOAS30|practical_high_urban|+36.478571|65602|4318|61284|
|UPRACTICAL30−MATCHED_NOAS30|practical_mid_urban|+27.869643|53770|6949|46821|
|UPRACTICAL30−MATCHED_NOAS30|practical_low_urban|+21.352976|44904|9031|35873|
|UTEACHER30−MATCHED_NOAS30|clean|+0.000595|3769|3768|1|
|UTEACHER30−MATCHED_NOAS30|practical_high|+38.988690|68900|3399|65501|
|UTEACHER30−MATCHED_NOAS30|practical_mid|+37.905952|67611|3929|63682|
|UTEACHER30−MATCHED_NOAS30|practical_low_suburban|+36.619643|65959|4438|61521|
|UTEACHER30−MATCHED_NOAS30|practical_high_urban|+36.620238|65612|4090|61522|
|UTEACHER30−MATCHED_NOAS30|practical_mid_urban|+27.910119|53637|6748|46889|
|UTEACHER30−MATCHED_NOAS30|practical_low_urban|+21.269048|44585|8853|35732|
|MATCHED_NOAS30−NOAS30|clean|+0.041667|225|155|70|
|MATCHED_NOAS30−NOAS30|practical_high|-0.230952|652|1040|-388|
|MATCHED_NOAS30−NOAS30|practical_mid|-0.227381|646|1028|-382|
|MATCHED_NOAS30−NOAS30|practical_low_suburban|-0.223214|689|1064|-375|
|MATCHED_NOAS30−NOAS30|practical_high_urban|-0.261310|682|1121|-439|
|MATCHED_NOAS30−NOAS30|practical_mid_urban|-0.230952|769|1157|-388|
|MATCHED_NOAS30−NOAS30|practical_low_urban|-0.132143|976|1198|-222|
|UTEACHER30−AS30|clean|-0.051190|105|191|-86|
|UTEACHER30−AS30|practical_high|+0.037500|273|210|63|
|UTEACHER30−AS30|practical_mid|+0.012500|282|261|21|
|UTEACHER30−AS30|practical_low_suburban|+0.007143|278|266|12|
|UTEACHER30−AS30|practical_high_urban|+0.031548|274|221|53|
|UTEACHER30−AS30|practical_mid_urban|+0.005357|325|316|9|
|UTEACHER30−AS30|practical_low_urban|+0.032738|432|377|55|
|UPRACTICAL30−AS30|clean|-0.202381|215|555|-340|
|UPRACTICAL30−AS30|practical_high|-0.143452|703|944|-241|
|UPRACTICAL30−AS30|practical_mid|-0.116071|851|1046|-195|
|UPRACTICAL30−AS30|practical_low_suburban|-0.063690|990|1097|-107|
|UPRACTICAL30−AS30|practical_high_urban|-0.110119|829|1014|-185|
|UPRACTICAL30−AS30|practical_mid_urban|-0.035119|1225|1284|-59|
|UPRACTICAL30−AS30|practical_low_urban|+0.116667|1633|1437|196|
|MATCHED_NOAS30−AS30|clean|-0.051786|3733|3820|-87|
|MATCHED_NOAS30−AS30|practical_high|-38.951190|3381|68819|-65438|
|MATCHED_NOAS30−AS30|practical_mid|-37.893452|3931|67592|-63661|
|MATCHED_NOAS30−AS30|practical_low_suburban|-36.612500|4437|65946|-61509|
|MATCHED_NOAS30−AS30|practical_high_urban|-36.588690|4071|65540|-61469|
|MATCHED_NOAS30−AS30|practical_mid_urban|-27.904762|6750|53630|-46880|
|MATCHED_NOAS30−AS30|practical_low_urban|-21.236310|8839|44516|-35677|
|UTEACHER30−NOAS30|clean|+0.042262|3818|3747|71|
|UTEACHER30−NOAS30|practical_high|+38.757738|68554|3441|65113|
|UTEACHER30−NOAS30|practical_mid|+37.678571|67300|4000|63300|
|UTEACHER30−NOAS30|practical_low_suburban|+36.396429|65632|4486|61146|
|UTEACHER30−NOAS30|practical_high_urban|+36.358929|65216|4133|61083|
|UTEACHER30−NOAS30|practical_mid_urban|+27.679167|53267|6766|46501|
|UTEACHER30−NOAS30|practical_low_urban|+21.136905|44360|8850|35510|
|UPRACTICAL30−NOAS30|clean|-0.108929|3749|3932|-183|
|UPRACTICAL30−NOAS30|practical_high|+38.576786|68465|3656|64809|
|UPRACTICAL30−NOAS30|practical_mid|+37.550000|67289|4205|63084|
|UPRACTICAL30−NOAS30|practical_low_suburban|+36.325595|65666|4639|61027|
|UPRACTICAL30−NOAS30|practical_high_urban|+36.217262|65212|4367|60845|
|UPRACTICAL30−NOAS30|practical_mid_urban|+27.638690|53403|6970|46433|
|UPRACTICAL30−NOAS30|practical_low_urban|+21.220833|44685|9034|35651|
|AS30−NOAS30|clean|+0.093452|3868|3711|157|
|AS30−NOAS30|practical_high|+38.720238|68469|3419|65050|
|AS30−NOAS30|practical_mid|+37.666071|67279|4000|63279|
|AS30−NOAS30|practical_low_suburban|+36.389286|65618|4484|61134|
|AS30−NOAS30|practical_high_urban|+36.327381|65141|4111|61030|
|AS30−NOAS30|practical_mid_urban|+27.673810|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|+21.104167|44297|8842|35455|

|配对|urban三场景均值Δpp|
|---|---:|
|UPRACTICAL30−UTEACHER30|-0.032738|
|UPRACTICAL30−MATCHED_NOAS30|+28.567063|
|UTEACHER30−MATCHED_NOAS30|+28.599802|
|MATCHED_NOAS30−NOAS30|-0.208135|
|UTEACHER30−AS30|+0.023214|
|UPRACTICAL30−AS30|-0.009524|
|MATCHED_NOAS30−AS30|-28.576587|
|UTEACHER30−NOAS30|+28.391667|
|UPRACTICAL30−NOAS30|+28.358929|
|AS30−NOAS30|+28.368452|

|条件|fullV clean %|V1800 clean %|两draw mid/low %|
|---|---:|---:|---:|
|NOAS30|98.359259|98.444444|39.583333|
|AS30|98.255556|98.333333|65.944444|
|MATCHED_NOAS30|98.355556|98.444444|39.208333|
|UTEACHER30|98.233333|98.277778|66.055556|
|UPRACTICAL30|98.218519|98.277778|65.986111|

源CSV保留完整draw/pooled、RX/day/TX、state/lock、实际新增噪声四区间、F1与现有logits margin，不作附加门槛。fullV27000仅原生accuracy/RX，不伪造逐包F1；V1800两draw不是两个模型seed，源无highurban；源panel2.45GHz与训练/正式2.462GHz差异保留，signal/added-noise不是总SNR。

|新条件|qualified U|qualified As|qualified Gaussian|AsU|native秒|
|---|---:|---:|---:|---:|---:|
|MATCHED_NOAS30|8967|0|8967|0|904.277294|
|UTEACHER30|8722|0|8722|0|1158.069700|
|UPRACTICAL30|8701|2165|6536|141750|1388.747024|

完整u_steps重核资格、selected-mean CE/all-U entropy、teacher pseudo六类分布、RX/day/route与As state/lock/noise覆盖；pseudo分布或teacher/student agreement不是U真值精度。保持teacher代码路径不保证模型更新后teacher预测轨迹逐步相等。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source NN|formal NN|
|---|---:|---:|---:|---:|---:|---:|---:|
|NOAS30|6078780|918004|0|963900|7960684|12600|1176002|
|AS30|6078780|918004|0|963900|7960684|12600|1176002|
|MATCHED_NOAS30|2271960|324002|0|567000|3162962|12600|1176002|
|UTEACHER30|2271960|324002|0|567000|3162962|12600|1176002|
|UPRACTICAL30|2271960|324002|0|567000|3162962|12600|1176002|

新增3尾段30modelEpoch/6660更新；训练与读出13054692NN，共享smoke2356NN，新总13057048NN。B00/B01两个真实父去重单列；C/S共享As20，旧E30不重计。各新行2220steps/10epoch、完整L6300/hiddenU56700流全扫，U学生/teacher各567000；S额外141750完整物理模拟不是新增NN，也不是零成本。CPU route包含selection/simulation/copy，不重复相加，不当CUDAevent；detach日志可同步。source prediction wall缺失保持null。

|条件|suburban视图|绝对ACC %|自身clean−suburban pp|
|---|---|---:|---:|
|NOAS30|practical_high|42.685119|42.562500|
|NOAS30|practical_mid|40.995238|44.252381|
|NOAS30|practical_low_suburban|39.472619|45.775000|
|AS30|practical_high|81.405357|3.935714|
|AS30|practical_mid|78.661310|6.679762|
|AS30|practical_low_suburban|75.861905|9.479167|
|MATCHED_NOAS30|practical_high|42.454167|42.835119|
|MATCHED_NOAS30|practical_mid|40.767857|44.521429|
|MATCHED_NOAS30|practical_low_suburban|39.249405|46.039881|
|UTEACHER30|practical_high|81.442857|3.847024|
|UTEACHER30|practical_mid|78.673810|6.616071|
|UTEACHER30|practical_low_suburban|75.869048|9.420833|
|UPRACTICAL30|practical_high|81.261905|3.876786|
|UPRACTICAL30|practical_mid|78.545238|6.593452|
|UPRACTICAL30|practical_low_suburban|75.798214|9.340476|

clean降低也会缩小gap，必须同时看绝对ACC。全部5×7固定预测和独立评分后truth join，重核630CM/105分区。源科学判据与正式阶段分开，默认头联合训练，无RX留出或诊断分类器。完整10配对、F1/最差RX、分层与成本保留；n1/微差不是可重复clean收益。正式不反馈参数/候选/seed或救跑，分析NN/CP/IQ/updates均0。

此前r01首步接口技术失败单列770NN（原NoAs checkpoint前向2＋synthetic L768，U前向/物理传播/更新/worker均0）；旧run/release/log保留。本run有效训练与smoke成本为13057048NN，含失败尝试的候选实际总计13057818NN；失败不是额外科学训练，也不与本次有效smoke重复计数。

## U 资格、路由与覆盖

资格只来自原 raw-U EMA 教师的 RX/day quantile 与 batch-neighbor；三条新尾段共同关闭学生 strong-agreement。N/C 的指定四分之一仍是原 Gaussian strong，不是 raw。S 在整批原 strong 生成后，以同一批 U 的 raw 输入生成完整 practical 并覆盖四分之一。teacher-only 代码相同不保证训练后的教师预测轨迹相同。

|条件|路由|伪类 0|伪类 1|伪类 2|伪类 3|伪类 4|伪类 5|
|---|---|---:|---:|---:|---:|---:|---:|
|MATCHED_NOAS30|Gaussian|94307|95542|95290|92764|94625|94472|
|MATCHED_NOAS30|Gaussian_qualified|1466|1411|1633|1304|1475|1678|
|UTEACHER30|Gaussian|94410|95997|94850|91944|95047|94752|
|UTEACHER30|Gaussian_qualified|1422|1350|1605|1208|1462|1675|
|UPRACTICAL30|As|23497|24179|23900|22910|23828|23436|
|UPRACTICAL30|As_qualified|364|333|403|301|359|405|
|UPRACTICAL30|Gaussian|70902|71801|70942|69018|71257|71330|
|UPRACTICAL30|Gaussian_qualified|1057|1013|1198|899|1100|1269|

这些是教师伪标签计数，不能称为 U 的身份准确率。公开 JSON 保留全部 RX/day/route 的总量及资格量、As 的 scene/state/lock/实际新增噪声区间曝光；不包含 U 真值、逐包向量或物理 ID。

|条件|RX|day|路由|包数|教师资格包数|
|---|---|---|---|---:|---:|
|MATCHED_NOAS30|1|1|Gaussian|37800|598|
|MATCHED_NOAS30|1|2|Gaussian|37800|655|
|MATCHED_NOAS30|1|3|Gaussian|37800|612|
|MATCHED_NOAS30|3|1|Gaussian|37800|512|
|MATCHED_NOAS30|3|2|Gaussian|37800|565|
|MATCHED_NOAS30|3|3|Gaussian|37800|429|
|MATCHED_NOAS30|4|1|Gaussian|37800|674|
|MATCHED_NOAS30|4|2|Gaussian|37800|588|
|MATCHED_NOAS30|4|3|Gaussian|37800|692|
|MATCHED_NOAS30|6|1|Gaussian|37800|640|
|MATCHED_NOAS30|6|2|Gaussian|37800|626|
|MATCHED_NOAS30|6|3|Gaussian|37800|605|
|MATCHED_NOAS30|8|1|Gaussian|37800|530|
|MATCHED_NOAS30|8|2|Gaussian|37800|605|
|MATCHED_NOAS30|8|3|Gaussian|37800|636|
|UTEACHER30|1|1|Gaussian|37800|591|
|UTEACHER30|1|2|Gaussian|37800|650|
|UTEACHER30|1|3|Gaussian|37800|596|
|UTEACHER30|3|1|Gaussian|37800|477|
|UTEACHER30|3|2|Gaussian|37800|532|
|UTEACHER30|3|3|Gaussian|37800|392|
|UTEACHER30|4|1|Gaussian|37800|656|
|UTEACHER30|4|2|Gaussian|37800|576|
|UTEACHER30|4|3|Gaussian|37800|689|
|UTEACHER30|6|1|Gaussian|37800|639|
|UTEACHER30|6|2|Gaussian|37800|618|
|UTEACHER30|6|3|Gaussian|37800|603|
|UTEACHER30|8|1|Gaussian|37800|507|
|UTEACHER30|8|2|Gaussian|37800|583|
|UTEACHER30|8|3|Gaussian|37800|613|
|UPRACTICAL30|1|1|As|9375|144|
|UPRACTICAL30|1|1|Gaussian|28425|446|
|UPRACTICAL30|1|2|As|9330|158|
|UPRACTICAL30|1|2|Gaussian|28470|491|
|UPRACTICAL30|1|3|As|9463|150|
|UPRACTICAL30|1|3|Gaussian|28337|445|
|UPRACTICAL30|3|1|As|9549|121|
|UPRACTICAL30|3|1|Gaussian|28251|354|
|UPRACTICAL30|3|2|As|9637|144|
|UPRACTICAL30|3|2|Gaussian|28163|386|
|UPRACTICAL30|3|3|As|9362|94|
|UPRACTICAL30|3|3|Gaussian|28438|292|
|UPRACTICAL30|4|1|As|9380|155|
|UPRACTICAL30|4|1|Gaussian|28420|501|
|UPRACTICAL30|4|2|As|9469|149|
|UPRACTICAL30|4|2|Gaussian|28331|425|
|UPRACTICAL30|4|3|As|9579|172|
|UPRACTICAL30|4|3|Gaussian|28221|517|
|UPRACTICAL30|6|1|As|9374|163|
|UPRACTICAL30|6|1|Gaussian|28426|476|
|UPRACTICAL30|6|2|As|9450|155|
|UPRACTICAL30|6|2|Gaussian|28350|464|
|UPRACTICAL30|6|3|As|9365|135|
|UPRACTICAL30|6|3|Gaussian|28435|468|
|UPRACTICAL30|8|1|As|9346|124|
|UPRACTICAL30|8|1|Gaussian|28454|380|
|UPRACTICAL30|8|2|As|9573|151|
|UPRACTICAL30|8|2|Gaussian|28227|430|
|UPRACTICAL30|8|3|As|9498|150|
|UPRACTICAL30|8|3|Gaussian|28302|461|

|条件|CPU 记录|host wall 秒|
|---|---|---:|
|MATCHED_NOAS30|original_strong_cpu_wall_seconds|0.178464|
|MATCHED_NOAS30|selection_cpu_wall_seconds|0.612547|
|MATCHED_NOAS30|simulation_adapter_cpu_wall_seconds|0.000000|
|MATCHED_NOAS30|replacement_copy_cpu_wall_seconds|0.000000|
|MATCHED_NOAS30|route_cpu_wall_seconds|1.344282|
|MATCHED_NOAS30|logging_cpu_wall_seconds|1.056274|
|UTEACHER30|original_strong_cpu_wall_seconds|0.190363|
|UTEACHER30|selection_cpu_wall_seconds|0.681202|
|UTEACHER30|simulation_adapter_cpu_wall_seconds|0.000000|
|UTEACHER30|replacement_copy_cpu_wall_seconds|0.000000|
|UTEACHER30|route_cpu_wall_seconds|1.421453|
|UTEACHER30|logging_cpu_wall_seconds|1.131709|
|UPRACTICAL30|original_strong_cpu_wall_seconds|0.180885|
|UPRACTICAL30|selection_cpu_wall_seconds|0.635193|
|UPRACTICAL30|simulation_adapter_cpu_wall_seconds|235.613343|
|UPRACTICAL30|replacement_copy_cpu_wall_seconds|0.175304|
|UPRACTICAL30|route_cpu_wall_seconds|239.608292|
|UPRACTICAL30|logging_cpu_wall_seconds|1.131577|

route wall 已包含 selection、完整物理模拟适配和 copy，不能与子项相加；日志 detach 可能同步。这里不是 CUDA event 或纯 kernel 时间，也不据此声称加速。

## 完整性与边界

汇总文件缺项：0（11 个预期文件齐备）。完成状态：ANALYZED。source prediction wall 未单列：NOAS30, AS30, MATCHED_NOAS30, UTEACHER30, UPRACTICAL30。未记录的时间保持 null，不估算。

新三条尾段各使用已有 E20 父模型追加 E10，分别计入实际 student/teacher、源读出和正式预测；两个不同父模型去重，旧 E30 训练与读出复用。有效 2356 NN smoke 与失败 r01 的 770 NN 分列。E20 的历史 U 训练保留，不能称全程重新从零或全程改变 U。

source full V27000 只有原生准确率及 RX 汇总，没有伪造逐包 F1 或修复；V1800 的两次信道 draw 不是两个模型 seed。源 quality 分层与真实资格/曝光用于描述，不增加任何筛选门槛。正式无同类 physics 标签时不报告正式 state/noise 性能。

五个条件不是 2×2 因子设计，不报告因子交互。S−C 是路由增量；C−旧 As 同时含共同去除学生 agreement 的改变；S−N 是本轮匹配的 clean 阶段约束。最差 RX 使用 receiver 维度原标签，不从标签字面推断 TX/RX 角色。所有 gap 同时报绝对准确率，clean 降低造成的 gap 缩小不能当作保护成功。

本次整理只读取现有聚合 JSON；NN、checkpoint、IQ、更新、新预测、新评分、远程调用均为 0。完整 630 CM 与 105 个分区已由原分析产物记录，这里不复制矩阵或重复评分。


## 执行证据与交付

实际训练与分析代码提交为 `fa9b272aa400ed3403e8f550c50a6957ec0f8588`，固定 FINAL30／默认 cosine 头随骨干联合训练。唯一针对接口的定向 P0/P1 复核通过；协议负例、真实父 no-query smoke、一次发布 archive 比较及远端 compile、启动进程/命令/GPU/日志核实均已执行。三行最终各 2220 次新增更新，累计 6660 次；七视图预测固定后独立 truth-last，完成 630 个混淆矩阵与 105 个分区复计。dispatcher 与三 worker 均已自然退出，终态和产物独立读回 VERIFIED，无缺失评估。实际代码未在运行中修改。

完整矩阵、数据角色、父模型来源、命令和停止规则见 [预登记报告](report.md) 与 [experiment.json](experiment.json)。[正式全分层指标](formal_metrics.csv)、[源域全分层指标](source_metrics.csv)、[公开配对与训练聚合](results_public.json)、[实际成本](costs.json)、[独立终态](run_verification.json) 保留全部结果。两个旧 E30 参照只读复用，不重复训练或按目标成绩选模。

本轮唯一假设是：用四分之一已有 U 学生前向学习完整 practical，能够在保持 clean 的同时获得原 L As 以外的识别增量。固定配方已被源域两条件否定，正式 clean 阶段限制也未满足。结论限于该配方，不等于所有 U 星地训练都无效。没有新增损失、分支、诊断头、RX 留出、目标反馈或 E50 救跑。后续方案须独立预登记并与另一对话去重，不能依据本轮正式分数调比例、资格或种子。
