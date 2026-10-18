# LRaw固定配方 E50：八条件预算分析

固定LRaw配方由E30延长到E50，没有解决clean—星地矛盾：mid/low urban分别提高0.435119/0.873214个绝对百分点，但clean下降1.214286个百分点。相对原As50，low仅＋0.083333个百分点，mid下降0.559524、clean下降2.522619个百分点。本次固定单种子预算验证已完成，尚无可重复双提升证据。

E50新As clean为83.138095%，相对匹配新NoAs +0.564286个绝对百分点，通过下降不超过0.1个百分点的阶段约束。low/mid urban为52.470833%/63.342262%。模型种子n=1，尚不能证明可重复的星地与clean双提升。

LRawAs50−ownLRawAs30的clean/low/mid分别为-1.214286/+0.873214/+0.435119个百分点；匹配NoAs自身clean预算变化-0.144643个百分点。同E50 LRawAs−原As的clean/low/mid为-2.522619/+0.083333/-0.559524个百分点。预算收益与方法增量分别报告，不把匹配NoAs恶化当作clean保护或As纯收益。

20261018-phase1-FcrossRX-LRaw-e50-s2026092701-m2-r01。新2个LRaw50、旧6个冻结条件，合计8×7。两个新行各从其LRaw FINAL30学生/ownEMA追加20轮，累计E50/11100更新；旧原30/50与LRaw30训练审计、源读出、正式预测/评分全部复用。总轨迹20+10+20并重置新增段AdamW/cos20，不是从零连续cos50。

E50扩展在读取E30正式结果前依据保存的E30源双规则PASS固定。E50 source仅描述同配方的预算结果，无新晋级阈值、不自动扩预算/种子、不从正式结果选择性重跑。源结果与正式stage是两个不同结论。

固定配方保持6L/768NN每step/总L尺度1.25、原.5/.5/.25监督、默认cosine头联合训练、As(raw64)/NoAs raw64、ORD128成员和practical物理处理。完整ordinary仍每step一次128包执行，输出弃用，L及既有ORD128实际输入raw128。Uweak/strong、U目标、学生U前向、ownEMA教师和更新全部保留。

## 七视图绝对结果

每view n=168000；urban均值仅三场urban accuracy等权，不等同模型seed平均。

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
|LRAW_NOAS30|clean|82.718452|82.337475|75.050000|
|LRAW_NOAS30|practical_high|41.388690|37.473265|30.070833|
|LRAW_NOAS30|practical_mid|39.658929|36.409801|29.370833|
|LRAW_NOAS30|practical_low_suburban|37.877381|35.118040|28.483333|
|LRAW_NOAS30|practical_high_urban|39.981548|36.630720|29.745833|
|LRAW_NOAS30|practical_mid_urban|33.397024|31.932659|25.962500|
|LRAW_NOAS30|practical_low_urban|28.450595|27.924007|22.520833|
|LRAW_AS30|clean|84.352381|83.962034|77.091667|
|LRAW_AS30|practical_high|81.120238|80.799680|70.254167|
|LRAW_AS30|practical_mid|78.575595|78.282313|69.000000|
|LRAW_AS30|practical_low_suburban|75.966667|75.700869|66.775000|
|LRAW_AS30|practical_high_urban|77.566667|77.260552|67.687500|
|LRAW_AS30|practical_mid_urban|62.907143|62.672125|55.991667|
|LRAW_AS30|practical_low_urban|51.597619|51.275459|46.458333|
|NOAS50|clean|85.295238|85.009461|77.479167|
|NOAS50|practical_high|39.952381|34.167858|33.212500|
|NOAS50|practical_mid|39.140476|34.087783|32.204167|
|NOAS50|practical_low_suburban|38.218452|33.750550|31.241667|
|NOAS50|practical_high_urban|38.994048|33.810253|32.766667|
|NOAS50|practical_mid_urban|34.036905|30.802839|28.666667|
|NOAS50|practical_low_urban|29.878571|27.842578|25.845833|
|AS50|clean|85.660714|85.381047|79.025000|
|AS50|practical_high|82.239286|82.031208|72.279167|
|AS50|practical_mid|79.556548|79.342102|70.287500|
|AS50|practical_low_suburban|76.776190|76.523952|67.625000|
|AS50|practical_high_urban|78.743452|78.521008|69.708333|
|AS50|practical_mid_urban|63.901786|63.536348|57.245833|
|AS50|practical_low_urban|52.387500|51.848684|47.233333|
|LRAW_NOAS50|clean|82.573810|82.168088|74.233333|
|LRAW_NOAS50|practical_high|41.635714|36.507489|27.425000|
|LRAW_NOAS50|practical_mid|39.597024|35.115704|26.587500|
|LRAW_NOAS50|practical_low_suburban|37.867857|33.902972|26.662500|
|LRAW_NOAS50|practical_high_urban|39.955952|35.642461|27.087500|
|LRAW_NOAS50|practical_mid_urban|33.318452|31.334391|24.062500|
|LRAW_NOAS50|practical_low_urban|28.358929|27.373844|21.879167|
|LRAW_AS50|clean|83.138095|82.644314|76.800000|
|LRAW_AS50|practical_high|80.645833|80.357570|67.070833|
|LRAW_AS50|practical_mid|78.381548|78.113765|66.237500|
|LRAW_AS50|practical_low_suburban|76.017857|75.743246|64.737500|
|LRAW_AS50|practical_high_urban|77.311310|77.029236|65.041667|
|LRAW_AS50|practical_mid_urban|63.342262|63.047910|54.870833|
|LRAW_AS50|practical_low_urban|52.470833|52.009788|46.787500|

|条件|urban三场均值 %|
|---|---|
|NOAS30|35.570833|
|AS30|63.939286|
|LRAW_NOAS30|33.943056|
|LRAW_AS30|64.023810|
|NOAS50|34.303175|
|AS50|65.010913|
|LRAW_NOAS50|33.877778|
|LRAW_AS50|64.374802|

## 十三组配对与交互

新As−新NoAs是同E50新配方的As总效应；各自50−30是预算变化；新As50−原As50、新NoAs50−原NoAs50是同预算输入机制增量。不能将这些差值相加预测未来成绩。

|比较|视图|accuracy Δpp|macro-F1 Δpp|最差RX Δpp|修复|新增错误|净修复|
|---|---|---|---|---|---|---|---|
|AS30−NOAS30|clean|+0.093452|+0.064839|-0.087500|3868|3711|157|
|AS30−NOAS30|practical_high|+38.720238|+41.772814|+37.604167|68469|3419|65050|
|AS30−NOAS30|practical_mid|+37.666071|+40.404893|+36.691667|67279|4000|63279|
|AS30−NOAS30|practical_low_suburban|+36.389286|+38.814010|+35.687500|65618|4484|61134|
|AS30−NOAS30|practical_high_urban|+36.327381|+38.942178|+35.108333|65141|4111|61030|
|AS30−NOAS30|practical_mid_urban|+27.673810|+29.003860|+26.962500|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|+21.104167|+21.702446|+20.629167|44297|8842|35455|
|LRAW_AS30−LRAW_NOAS30|clean|+1.633929|+1.624560|+2.041667|6724|3979|2745|
|LRAW_AS30−LRAW_NOAS30|practical_high|+39.731548|+43.326416|+40.183333|71136|4387|66749|
|LRAW_AS30−LRAW_NOAS30|practical_mid|+38.916667|+41.872512|+39.629167|70438|5058|65380|
|LRAW_AS30−LRAW_NOAS30|practical_low_suburban|+38.089286|+40.582828|+38.291667|69702|5712|63990|
|LRAW_AS30−LRAW_NOAS30|practical_high_urban|+37.585119|+40.629832|+37.941667|68269|5126|63143|
|LRAW_AS30−LRAW_NOAS30|practical_mid_urban|+29.510119|+30.739466|+30.029167|57793|8216|49577|
|LRAW_AS30−LRAW_NOAS30|practical_low_urban|+23.147024|+23.351452|+23.937500|49377|10490|38887|
|AS50−NOAS50|clean|+0.365476|+0.371586|+1.545833|4153|3539|614|
|AS50−NOAS50|practical_high|+42.286905|+47.863349|+39.066667|73787|2745|71042|
|AS50−NOAS50|practical_mid|+40.416071|+45.254319|+38.083333|71258|3359|67899|
|AS50−NOAS50|practical_low_suburban|+38.557738|+42.773401|+36.383333|68696|3919|64777|
|AS50−NOAS50|practical_high_urban|+39.749405|+44.710756|+36.941667|70308|3529|66779|
|AS50−NOAS50|practical_mid_urban|+29.864881|+32.733508|+28.579167|56661|6488|50173|
|AS50−NOAS50|practical_low_urban|+22.508929|+24.006107|+21.387500|46642|8827|37815|
|LRAW_AS50−LRAW_NOAS50|clean|+0.564286|+0.476226|+2.566667|6285|5337|948|
|LRAW_AS50−LRAW_NOAS50|practical_high|+39.010119|+43.850081|+39.645833|70051|4514|65537|
|LRAW_AS50−LRAW_NOAS50|practical_mid|+38.784524|+42.998061|+39.650000|70306|5148|65158|
|LRAW_AS50−LRAW_NOAS50|practical_low_suburban|+38.150000|+41.840274|+38.075000|69916|5824|64092|
|LRAW_AS50−LRAW_NOAS50|practical_high_urban|+37.355357|+41.386775|+37.954167|67977|5220|62757|
|LRAW_AS50−LRAW_NOAS50|practical_mid_urban|+30.023810|+31.713519|+30.808333|59037|8597|50440|
|LRAW_AS50−LRAW_NOAS50|practical_low_urban|+24.111905|+24.635943|+24.908333|51590|11082|40508|
|LRAW_AS30−AS30|clean|-0.988690|-1.082440|-1.270833|1719|3380|-1661|
|LRAW_AS30−AS30|practical_high|-0.285119|-0.284550|-1.654167|1964|2443|-479|
|LRAW_AS30−AS30|practical_mid|-0.085714|-0.077393|-1.212500|2670|2814|-144|
|LRAW_AS30−AS30|practical_low_suburban|+0.104762|+0.128128|-0.966667|3200|3024|176|
|LRAW_AS30−AS30|practical_high_urban|-0.200595|-0.162655|-1.420833|2505|2842|-337|
|LRAW_AS30−AS30|practical_mid_urban|+0.100595|+0.221662|-0.466667|4695|4526|169|
|LRAW_AS30−AS30|practical_low_urban|+0.353571|+0.388424|+0.079167|5983|5389|594|
|LRAW_NOAS30−NOAS30|clean|-2.529167|-2.642161|-3.400000|1661|5910|-4249|
|LRAW_NOAS30−NOAS30|practical_high|-1.296429|-1.838152|-4.233333|13213|15391|-2178|
|LRAW_NOAS30−NOAS30|practical_mid|-1.336310|-1.545012|-4.150000|13204|15449|-2245|
|LRAW_NOAS30−NOAS30|practical_low_suburban|-1.595238|-1.640690|-3.570833|12725|15405|-2680|
|LRAW_NOAS30−NOAS30|practical_high_urban|-1.458333|-1.850310|-4.254167|12990|15440|-2450|
|LRAW_NOAS30−NOAS30|practical_mid_urban|-1.735714|-1.513944|-3.533333|11721|14637|-2916|
|LRAW_NOAS30−NOAS30|practical_low_urban|-1.689286|-1.260582|-3.229167|10828|13666|-2838|
|LRAW_AS50−AS50|clean|-2.522619|-2.736733|-2.225000|2331|6569|-4238|
|LRAW_AS50−AS50|practical_high|-1.593452|-1.673638|-5.208333|2665|5342|-2677|
|LRAW_AS50−AS50|practical_mid|-1.175000|-1.228337|-4.050000|3569|5543|-1974|
|LRAW_AS50−AS50|practical_low_suburban|-0.758333|-0.780705|-2.887500|4287|5561|-1274|
|LRAW_AS50−AS50|practical_high_urban|-1.432143|-1.491772|-4.666667|3390|5796|-2406|
|LRAW_AS50−AS50|practical_mid_urban|-0.559524|-0.488438|-2.375000|6597|7537|-940|
|LRAW_AS50−AS50|practical_low_urban|+0.083333|+0.161103|-0.445833|8759|8619|140|
|LRAW_NOAS50−NOAS50|clean|-2.721429|-2.841374|-3.245833|2533|7105|-4572|
|LRAW_NOAS50−NOAS50|practical_high|+1.683333|+2.339631|-5.787500|27900|25072|2828|
|LRAW_NOAS50−NOAS50|practical_mid|+0.456548|+1.027921|-5.616667|25899|25132|767|
|LRAW_NOAS50−NOAS50|practical_low_suburban|-0.350595|+0.152421|-4.579167|24375|24964|-589|
|LRAW_NOAS50−NOAS50|practical_high_urban|+0.961905|+1.832209|-5.679167|26440|24824|1616|
|LRAW_NOAS50−NOAS50|practical_mid_urban|-0.718452|+0.531552|-4.604167|21646|22853|-1207|
|LRAW_NOAS50−NOAS50|practical_low_urban|-1.519643|-0.468733|-3.966667|18261|20814|-2553|
|AS50−AS30|clean|+0.319643|+0.336573|+0.662500|2432|1895|537|
|AS50−AS30|practical_high|+0.833929|+0.946978|+0.370833|3952|2551|1401|
|AS50−AS30|practical_mid|+0.895238|+0.982396|+0.075000|4408|2904|1504|
|AS50−AS30|practical_low_suburban|+0.914286|+0.951211|-0.116667|4764|3228|1536|
|AS50−AS30|practical_high_urban|+0.976190|+1.097801|+0.600000|4421|2781|1640|
|AS50−AS30|practical_mid_urban|+1.095238|+1.085884|+0.787500|5920|4080|1840|
|AS50−AS30|practical_low_urban|+1.143452|+0.961649|+0.854167|6711|4790|1921|
|NOAS50−NOAS30|clean|+0.047619|+0.029826|-0.970833|2104|2024|80|
|NOAS50−NOAS30|practical_high|-2.732738|-5.143558|-1.091667|6685|11276|-4591|
|NOAS50−NOAS30|practical_mid|-1.854762|-3.867030|-1.316667|6992|10108|-3116|
|NOAS50−NOAS30|practical_low_suburban|-1.254167|-3.008180|-0.812500|7169|9276|-2107|
|NOAS50−NOAS30|practical_high_urban|-2.445833|-4.670777|-1.233333|6832|10941|-4109|
|NOAS50−NOAS30|practical_mid_urban|-1.095833|-2.643763|-0.829167|7490|9331|-1841|
|NOAS50−NOAS30|practical_low_urban|-0.261310|-1.342012|+0.095833|7810|8249|-439|
|LRAW_AS50−LRAW_AS30|clean|-1.214286|-1.317720|-0.291667|1725|3765|-2040|
|LRAW_AS50−LRAW_AS30|practical_high|-0.474405|-0.442110|-3.183333|2898|3695|-797|
|LRAW_AS50−LRAW_AS30|practical_mid|-0.194048|-0.168548|-2.762500|3695|4021|-326|
|LRAW_AS50−LRAW_AS30|practical_low_suburban|+0.051190|+0.042378|-2.037500|4361|4275|86|
|LRAW_AS50−LRAW_AS30|practical_high_urban|-0.255357|-0.231316|-2.645833|3469|3898|-429|
|LRAW_AS50−LRAW_AS30|practical_mid_urban|+0.435119|+0.375785|-1.120833|6019|5288|731|
|LRAW_AS50−LRAW_AS30|practical_low_urban|+0.873214|+0.734328|+0.329167|7573|6106|1467|
|LRAW_NOAS50−LRAW_NOAS30|clean|-0.144643|-0.169387|-0.816667|2761|3004|-243|
|LRAW_NOAS50−LRAW_NOAS30|practical_high|+0.247024|-0.965775|-2.645833|10960|10545|415|
|LRAW_NOAS50−LRAW_NOAS30|practical_mid|-0.061905|-1.294097|-2.783333|10926|11030|-104|
|LRAW_NOAS50−LRAW_NOAS30|practical_low_suburban|-0.009524|-1.215069|-1.820833|11137|11153|-16|
|LRAW_NOAS50−LRAW_NOAS30|practical_high_urban|-0.025595|-0.988258|-2.658333|10701|10744|-43|
|LRAW_NOAS50−LRAW_NOAS30|practical_mid_urban|-0.078571|-0.598268|-1.900000|10386|10518|-132|
|LRAW_NOAS50−LRAW_NOAS30|practical_low_urban|-0.091667|-0.550163|-0.641667|9894|10048|-154|
|LRAW_AS50−NOAS50|clean|-2.157143|-2.365147|-0.679167|3804|7428|-3624|
|LRAW_AS50−NOAS50|practical_high|+40.693452|+46.189712|+33.858333|71584|3219|68365|
|LRAW_AS50−NOAS50|practical_mid|+39.241071|+44.025982|+34.033333|69944|4019|65925|
|LRAW_AS50−NOAS50|practical_low_suburban|+37.799405|+41.992696|+33.495833|68190|4687|63503|
|LRAW_AS50−NOAS50|practical_high_urban|+38.317262|+43.218983|+32.275000|68584|4211|64373|
|LRAW_AS50−NOAS50|practical_mid_urban|+29.305357|+32.245070|+26.204167|57284|8051|49233|
|LRAW_AS50−NOAS50|practical_low_urban|+22.592262|+24.167210|+20.941667|48842|10887|37955|

|比较|urban均值 Δpp|
|---|---|
|AS30−NOAS30|+28.368452|
|LRAW_AS30−LRAW_NOAS30|+30.080754|
|AS50−NOAS50|+30.707738|
|LRAW_AS50−LRAW_NOAS50|+30.497024|
|LRAW_AS30−AS30|+0.084524|
|LRAW_NOAS30−NOAS30|-1.627778|
|LRAW_AS50−AS50|-0.636111|
|LRAW_NOAS50−NOAS50|-0.425397|
|AS50−AS30|+1.071627|
|NOAS50−NOAS30|-1.267659|
|LRAW_AS50−LRAW_AS30|+0.350992|
|LRAW_NOAS50−LRAW_NOAS30|-0.065278|
|LRAW_AS50−NOAS50|+30.071627|

|视图|二因子项|交互 Δpp|
|---|---|---|
|clean|LRaw_x_As_at_E30|+1.540476|
|clean|LRaw_x_As_at_E50|+0.198810|
|clean|As_x_budget_original|+0.272024|
|clean|As_x_budget_LRaw|-1.069643|
|clean|LRaw_x_budget_NoAs|-0.192262|
|clean|LRaw_x_budget_As|-1.533929|
|practical_high|LRaw_x_As_at_E30|+1.011310|
|practical_high|LRaw_x_As_at_E50|-3.276786|
|practical_high|As_x_budget_original|+3.566667|
|practical_high|As_x_budget_LRaw|-0.721429|
|practical_high|LRaw_x_budget_NoAs|+2.979762|
|practical_high|LRaw_x_budget_As|-1.308333|
|practical_mid|LRaw_x_As_at_E30|+1.250595|
|practical_mid|LRaw_x_As_at_E50|-1.631548|
|practical_mid|As_x_budget_original|+2.750000|
|practical_mid|As_x_budget_LRaw|-0.132143|
|practical_mid|LRaw_x_budget_NoAs|+1.792857|
|practical_mid|LRaw_x_budget_As|-1.089286|
|practical_low_suburban|LRaw_x_As_at_E30|+1.700000|
|practical_low_suburban|LRaw_x_As_at_E50|-0.407738|
|practical_low_suburban|As_x_budget_original|+2.168452|
|practical_low_suburban|As_x_budget_LRaw|+0.060714|
|practical_low_suburban|LRaw_x_budget_NoAs|+1.244643|
|practical_low_suburban|LRaw_x_budget_As|-0.863095|
|practical_high_urban|LRaw_x_As_at_E30|+1.257738|
|practical_high_urban|LRaw_x_As_at_E50|-2.394048|
|practical_high_urban|As_x_budget_original|+3.422024|
|practical_high_urban|As_x_budget_LRaw|-0.229762|
|practical_high_urban|LRaw_x_budget_NoAs|+2.420238|
|practical_high_urban|LRaw_x_budget_As|-1.231548|
|practical_mid_urban|LRaw_x_As_at_E30|+1.836310|
|practical_mid_urban|LRaw_x_As_at_E50|+0.158929|
|practical_mid_urban|As_x_budget_original|+2.191071|
|practical_mid_urban|As_x_budget_LRaw|+0.513690|
|practical_mid_urban|LRaw_x_budget_NoAs|+1.017262|
|practical_mid_urban|LRaw_x_budget_As|-0.660119|
|practical_low_urban|LRaw_x_As_at_E30|+2.042857|
|practical_low_urban|LRaw_x_As_at_E50|+1.602976|
|practical_low_urban|As_x_budget_original|+1.404762|
|practical_low_urban|As_x_budget_LRaw|+0.964881|
|practical_low_urban|LRaw_x_budget_NoAs|+0.169643|
|practical_low_urban|LRaw_x_budget_As|-0.270238|
|urban_mean|LRaw_x_As_at_E30|+1.712302|
|urban_mean|LRaw_x_As_at_E50|-0.210714|
|urban_mean|As_x_budget_original|+2.339286|
|urban_mean|As_x_budget_LRaw|+0.416270|
|urban_mean|LRaw_x_budget_NoAs|+1.202381|
|urban_mean|LRaw_x_budget_As|-0.720635|

|视图|三因子交互 Δpp|
|---|---|
|clean|-1.341667|
|practical_high|-4.288095|
|practical_mid|-2.882143|
|practical_low_suburban|-2.107738|
|practical_high_urban|-3.651786|
|practical_mid_urban|-1.677381|
|practical_low_urban|-0.439881|
|urban_mean|-1.923016|

三因子=(LRaw×As在E50)−(LRaw×As在E30)，与As×budget和LRaw×budget的等价差核对；NoAs变化没有被隐藏。

## 来源预算读数

固定LRawAs50−ownAs30 fullV clean +0.148148个百分点、两draw mid/low等权+2.083333个百分点。仅为预算读数，无新pass/fail或自动晋级。

|条件|fullV clean %|V1800 clean %|mid/low等权 %|
|---|---|---|---|
|NOAS30|98.359259|98.444444|39.583333|
|AS30|98.255556|98.333333|65.944444|
|LRAW_NOAS30|98.348148|98.555556|39.708333|
|LRAW_AS30|98.318519|98.333333|67.083333|
|NOAS50|98.448148|98.611111|39.916667|
|AS50|98.374074|98.444444|67.750000|
|LRAW_NOAS50|98.407407|98.611111|40.152778|
|LRAW_AS50|98.466667|98.333333|69.166667|

|条件|draw|mid urban %|low urban %|
|---|---|---|---|
|NOAS30|1|42.944444|35.666667|
|NOAS30|2|44.166667|35.555556|
|AS30|1|72.555556|60.111111|
|AS30|2|73.055556|58.055556|
|LRAW_NOAS30|1|43.166667|36.444444|
|LRAW_NOAS30|2|43.888889|35.333333|
|LRAW_AS30|1|73.555556|60.833333|
|LRAW_AS30|2|74.000000|59.944444|
|NOAS50|1|44.000000|36.777778|
|NOAS50|2|43.333333|35.555556|
|AS50|1|74.500000|61.388889|
|AS50|2|75.055556|60.055556|
|LRAW_NOAS50|1|43.611111|37.555556|
|LRAW_NOAS50|2|44.500000|34.944444|
|LRAW_AS50|1|76.611111|63.055556|
|LRAW_AS50|2|76.111111|60.888889|

fullV27000使用冻结原生accuracy/RX，不虚构全V逐包F1/修复；固定V1800两draw×三场独立重算，两draw不是两个模型seed，源没有high urban。源panel2.45GHz与训练/正式2.462GHz差异保留。全5源RX、L6300/U56700标签隐藏/V27000、256点IQ、split392005和权限保持。

## suburban绝对值与自身clean差距

|条件|suburban视图|accuracy %|自身clean−suburban pp|
|---|---|---|---|
|NOAS30|practical_high|42.685119|42.562500|
|NOAS30|practical_mid|40.995238|44.252381|
|NOAS30|practical_low_suburban|39.472619|45.775000|
|AS30|practical_high|81.405357|3.935714|
|AS30|practical_mid|78.661310|6.679762|
|AS30|practical_low_suburban|75.861905|9.479167|
|LRAW_NOAS30|practical_high|41.388690|41.329762|
|LRAW_NOAS30|practical_mid|39.658929|43.059524|
|LRAW_NOAS30|practical_low_suburban|37.877381|44.841071|
|LRAW_AS30|practical_high|81.120238|3.232143|
|LRAW_AS30|practical_mid|78.575595|5.776786|
|LRAW_AS30|practical_low_suburban|75.966667|8.385714|
|NOAS50|practical_high|39.952381|45.342857|
|NOAS50|practical_mid|39.140476|46.154762|
|NOAS50|practical_low_suburban|38.218452|47.076786|
|AS50|practical_high|82.239286|3.421429|
|AS50|practical_mid|79.556548|6.104167|
|AS50|practical_low_suburban|76.776190|8.884524|
|LRAW_NOAS50|practical_high|41.635714|40.938095|
|LRAW_NOAS50|practical_mid|39.597024|42.976786|
|LRAW_NOAS50|practical_low_suburban|37.867857|44.705952|
|LRAW_AS50|practical_high|80.645833|2.492262|
|LRAW_AS50|practical_mid|78.381548|4.756548|
|LRAW_AS50|practical_low_suburban|76.017857|7.120238|

|比较|视图|clean Δpp|suburban Δpp|差距 Δpp|
|---|---|---|---|---|
|LRAW_AS50__LRAW_AS30|practical_high|-1.214286|-0.474405|-0.739881|
|LRAW_AS50__LRAW_AS30|practical_mid|-1.214286|-0.194048|-1.020238|
|LRAW_AS50__LRAW_AS30|practical_low_suburban|-1.214286|+0.051190|-1.265476|
|LRAW_AS50__AS50|practical_high|-2.522619|-1.593452|-0.929167|
|LRAW_AS50__AS50|practical_mid|-2.522619|-1.175000|-1.347619|
|LRAW_AS50__AS50|practical_low_suburban|-2.522619|-0.758333|-1.764286|

差距变化=clean变化−suburban变化。clean下降本身可能缩小差距，不能由此宣称suburban保护。

## 实际成本与完整日志

旧原30 native列是累计E30，LRaw30列是其新增10轮，原50与新50列各是新增20轮；累计成本另列，不重复计父训练。

|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source读出NN|formal预测NN|native秒|
|---|---|---|---|---|---|---|---|---|
|NOAS30|6078780|918004|0|963900|7960684|12600|1176002|2146.190|
|AS30|6078780|918004|0|963900|7960684|12600|1176002|2860.909|
|LRAW_NOAS30|2271960|324002|0|567000|3162962|12600|1176002|815.364|
|LRAW_AS30|2271960|324002|0|567000|3162962|12600|1176002|1044.778|
|NOAS50|4543920|594002|0|1134000|6271922|12600|1176002|1602.126|
|AS50|4543920|594002|0|1134000|6271922|12600|1176002|2069.076|
|LRAW_NOAS50|4543920|594002|0|1134000|6271922|12600|1176002|1596.744|
|LRAW_AS50|4543920|594002|0|1134000|6271922|12600|1176002|2191.173|

|成本组|native NN|训练段和自身读出 NN|
|---|---|---|
|new_LRaw_E50_segments|12543844|14921048|
|old_original_E50_segments|12543844|14921048|
|old_LRaw_E30_segments|6325924|8703128|
|old_original_E30_models|15921368|18298572|

|新行|父30 native NN（复用）|新增20 native NN|累计50 native NN|
|---|---|---|---|
|LRAW_NOAS50|7960684|6271922|14232606|
|LRAW_AS50|7960684|6271922|14232606|

本轮新增40modelEpoch/8880更新，训练及source/formal读出14921048NN，共享smoke1540NN，总新增14922588NN。旧父30、旧6条件及历史读出/smoke按已有审计复用。wall是分段模型秒，不据此推断并行加速。

|新行|解析step/epoch|ordinary调用/包|弃用MSE mean/max|used-raw MSE max|U消费包|EMA local/cum|
|---|---|---|---|---|---|---|
|LRAW_NOAS50|4440/20|4440/568320|0.23615736/2.6144626|0.0|1134000|4440/11100|
|LRAW_AS50|4440/20|4440/568320|0.23615736/2.6144626|0.0|1134000|4440/11100|

|条件|source预测wall秒|formal预测wall秒|
|---|---|---|
|NOAS30|未单列记录|48.08935279189609|
|AS30|未单列记录|49.649732606019825|
|LRAW_NOAS30|未单列记录|48.29190647392534|
|LRAW_AS30|未单列记录|47.518481956096366|
|NOAS50|未单列记录|47.668884380022064|
|AS50|未单列记录|46.84101340197958|
|LRAW_NOAS50|未单列记录|48.834792678942904|
|LRAW_AS50|未单列记录|55.571014045970514|

两新行各4440step/20epoch完整解析，含lraw_steps与ordinary_draw_steps两流及完整L/U序列。旧所有20/30/50日志审计直接复用，不回扫旧stepstream。完整ordinary增强计算仍执行，已包含在native wall，未单独计时不虚构独立节省；source_summary.wall_seconds是离线分析耗时，不能替代source模型预测wall，缺失则明确披露。

## 边界与完整性

前20轮仍有ordinary、U7和ownEMA历史，E21至E50为LRaw，不能称全程NoAug。普通增强cfg始终来自原own20 args冻结E13，非LRaw30 native10 args。增强的同一call RNG消耗保留，不证明完整dropout/教师/伪标签随机轨迹逐步一致。discarded MSE允许0，不按它筛样本或增门槛。

最终状态ANALYZED；8×7固定预测和独立truth-last评分全部齐备后，既有分析独立重核1008CM/168次RX/day/TX分区合并。精简产物不复制CM、逐ID、原预测、CP/IQ或日志；全分层配对在原aggregate保留。

n=1且已暴露基准背景保留。clean阶段PASS、局部urban提高或不足0.1个百分点的clean微增都不能证明可重复双提升；同预算比较与自身预算比较均须看到真实clean代价。正式结果不用于后续设计、选模、参数/种子重排或选择性重跑。

训练及预登记预测评分缺项：0（11个最终汇总文件齐备）；source预测wall未单列记录，属于已披露的成本计时缺失。本整理新增NN/CP/IQ/预测/评分/更新/远程调用均0。


按本轮LRaw匹配NoAs50，clean＋0.564286pp，通过0.1pp阶段界限；对原NoAs50却为−2.157143pp，不能把弱化NoAs对照写成原基线clean保护。suburban自身clean差缩小包含clean下降贡献，三场景距自身clean仍为2.492262/4.756548/7.120238pp。low52.470833%未达到60%。

本研究段E30和E50合计新增60modelEpoch/13320更新/23627256NN（含两次首步smoke共3080NN），不重复计费原20/30/50权重及历史读出。两个E50worker和dispatcher/0NNanalysis均已自然结束，无技术失败、无选择性重跑；本轮授权固定矩阵已收尾，未扩seed或超过E50。
