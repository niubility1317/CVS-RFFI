# As 权重配方 E30→E50：八条件同种子预算分析

20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01；固定seed2026092701。只新增加权NoAs/As各自 ownFINAL30 student/EMA+20；总轨迹20+10+20，每段resetAdamW/cos20（新增段），不是连续cos50。原weight30代码固定3befd47c8041d81f9430eda53562d0821624ff04。

既有NoAs30/As30、加权NoAs30/As30、原NoAs50/As50冻结结果全部复用。E30合法源域两条件通过并保存后才固定E50；新正式结果不参与该决定。新段6L1.25、.125ordinary64/.375selected64、默认头、原U、场景/物理和IQ处理保持；NoAs selected仍是raw64，不是假As或ordinary128。

|视图|NoAs30 %|As30 %|wNoAs30 %|wAs30 %|NoAs50 %|As50 %|wNoAs50 %|wAs50 %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clean|85.247619|85.341071|85.179762|84.852976|85.295238|85.660714|85.291667|85.248810|
|practical_high|42.685119|81.405357|42.148810|81.657143|39.952381|82.239286|39.879762|82.282143|
|practical_mid|40.995238|78.661310|40.477976|79.013690|39.140476|79.556548|38.991667|79.757738|
|practical_low_suburban|39.472619|75.861905|39.070238|76.342262|38.218452|76.776190|38.172619|77.213690|
|practical_high_urban|41.439881|77.767262|40.933929|78.042262|38.994048|78.743452|38.917262|78.798810|
|practical_mid_urban|35.132738|62.806548|34.737500|63.411905|34.036905|63.901786|33.886905|64.483333|
|practical_low_urban|30.139881|51.244048|29.869643|51.961905|29.878571|52.387500|29.845833|53.242857|
|urban_mean|35.570833|63.939286|35.180357|64.472024|34.303175|65.010913|34.216667|65.508333|

|视图|wAs50−wNoAs50 pp|wAs50−wAs30 pp|wNoAs50−wNoAs30 pp|wAs50−As50 pp|wNoAs50−NoAs50 pp|三因子交互 pp|
|---|---:|---:|---:|---:|---:|---:|
|clean|-0.042857|+0.395833|+0.111905|-0.411905|-0.003571|+0.011905|
|practical_high|+42.402381|+0.625000|-2.269048|+0.042857|-0.072619|-0.672619|
|practical_mid|+40.766071|+0.744048|-1.486310|+0.201190|-0.148810|-0.519643|
|practical_low_suburban|+39.041071|+0.871429|-0.897619|+0.437500|-0.045833|-0.399405|
|practical_high_urban|+39.881548|+0.756548|-2.016667|+0.055357|-0.076786|-0.648810|
|practical_mid_urban|+30.596429|+1.071429|-0.850595|+0.581548|-0.150000|-0.269048|
|practical_low_urban|+23.397024|+1.280952|-0.023810|+0.855357|-0.032738|-0.100000|
|urban_mean|+31.291667|+1.036310|-0.963690|+0.497421|-0.086508|-0.339286|

同E50加权As−加权NoAs clean变化-0.042857绝对百分点，阶段≥−.1pp=True。low/mid urban +23.397024/+30.596429pp。相对原NoAs50的clean变化-0.046429pp另列。

每一view完整保存12个单因素配对差、6个二因子交互及三因子交互。三因子=(weight×As在E50)−(weight×As在E30)，并核对等价的As×budget与weight×budget差。NoAs随预算变化与As自己预算收益分开，不能把对照退化全归因As，不能将不同method与budget收益相加预测。

|条件|view|macro-F1 %|最差RX %|
|---|---|---:|---:|
|NOAS30|clean|84.979636|78.450000|
|NOAS30|practical_high|39.311416|34.304167|
|NOAS30|practical_mid|37.954813|33.520833|
|NOAS30|practical_low_suburban|36.758731|32.054167|
|NOAS30|practical_high_urban|38.481030|34.000000|
|NOAS30|practical_mid_urban|33.446603|29.495833|
|NOAS30|practical_low_urban|29.184589|25.750000|
|AS30|clean|85.044474|78.362500|
|AS30|practical_high|81.084230|71.908333|
|AS30|practical_mid|78.359706|70.212500|
|AS30|practical_low_suburban|75.572741|67.741667|
|AS30|practical_high_urban|77.423208|69.108333|
|AS30|practical_mid_urban|62.450463|56.458333|
|AS30|practical_low_urban|50.887035|46.379167|
|WEIGHT_NOAS30|clean|84.905002|78.754167|
|WEIGHT_NOAS30|practical_high|38.816251|34.158333|
|WEIGHT_NOAS30|practical_mid|37.477307|33.379167|
|WEIGHT_NOAS30|practical_low_suburban|36.412564|31.945833|
|WEIGHT_NOAS30|practical_high_urban|38.031781|33.766667|
|WEIGHT_NOAS30|practical_mid_urban|33.156202|29.254167|
|WEIGHT_NOAS30|practical_low_urban|29.060955|25.804167|
|WEIGHT_AS30|clean|84.532314|78.187500|
|WEIGHT_AS30|practical_high|81.329051|72.500000|
|WEIGHT_AS30|practical_mid|78.705621|70.987500|
|WEIGHT_AS30|practical_low_suburban|76.047343|68.612500|
|WEIGHT_AS30|practical_high_urban|77.698154|69.720833|
|WEIGHT_AS30|practical_mid_urban|63.070406|57.445833|
|WEIGHT_AS30|practical_low_urban|51.567754|47.350000|
|NOAS50|clean|85.009461|77.479167|
|NOAS50|practical_high|34.167858|33.212500|
|NOAS50|practical_mid|34.087783|32.204167|
|NOAS50|practical_low_suburban|33.750550|31.241667|
|NOAS50|practical_high_urban|33.810253|32.766667|
|NOAS50|practical_mid_urban|30.802839|28.666667|
|NOAS50|practical_low_urban|27.842578|25.845833|
|AS50|clean|85.381047|79.025000|
|AS50|practical_high|82.031208|72.279167|
|AS50|practical_mid|79.342102|70.287500|
|AS50|practical_low_suburban|76.523952|67.625000|
|AS50|practical_high_urban|78.521008|69.708333|
|AS50|practical_mid_urban|63.536348|57.245833|
|AS50|practical_low_urban|51.848684|47.233333|
|WEIGHT_NOAS50|clean|85.005943|77.687500|
|WEIGHT_NOAS50|practical_high|34.071638|33.308333|
|WEIGHT_NOAS50|practical_mid|33.908242|32.337500|
|WEIGHT_NOAS50|practical_low_suburban|33.718654|31.441667|
|WEIGHT_NOAS50|practical_high_urban|33.737285|32.866667|
|WEIGHT_NOAS50|practical_mid_urban|30.720038|28.791667|
|WEIGHT_NOAS50|practical_low_urban|27.921637|26.012500|
|WEIGHT_AS50|clean|84.961060|78.650000|
|WEIGHT_AS50|practical_high|82.059969|72.370833|
|WEIGHT_AS50|practical_mid|79.535349|70.775000|
|WEIGHT_AS50|practical_low_suburban|76.968695|68.295833|
|WEIGHT_AS50|practical_high_urban|78.569507|69.804167|
|WEIGHT_AS50|practical_mid_urban|64.157115|57.883333|
|WEIGHT_AS50|practical_low_urban|52.727318|48.245833|

|比较|view|修复|新增错误|净修复|
|---|---|---:|---:|---:|
|AS30−NOAS30|clean|3868|3711|157|
|AS30−NOAS30|practical_mid_urban|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|44297|8842|35455|
|WEIGHT_AS30−WEIGHT_NOAS30|clean|3783|4332|-549|
|WEIGHT_AS30−WEIGHT_NOAS30|practical_mid_urban|55234|7061|48173|
|WEIGHT_AS30−WEIGHT_NOAS30|practical_low_urban|46375|9260|37115|
|AS50−NOAS50|clean|4153|3539|614|
|AS50−NOAS50|practical_mid_urban|56661|6488|50173|
|AS50−NOAS50|practical_low_urban|46642|8827|37815|
|WEIGHT_AS50−WEIGHT_NOAS50|clean|4088|4160|-72|
|WEIGHT_AS50−WEIGHT_NOAS50|practical_mid_urban|58028|6626|51402|
|WEIGHT_AS50−WEIGHT_NOAS50|practical_low_urban|48556|9249|39307|
|WEIGHT_AS30−AS30|clean|246|1066|-820|
|WEIGHT_AS30−AS30|practical_mid_urban|2186|1169|1017|
|WEIGHT_AS30−AS30|practical_low_urban|2635|1429|1206|
|WEIGHT_NOAS30−NOAS30|clean|301|415|-114|
|WEIGHT_NOAS30−NOAS30|practical_mid_urban|1262|1926|-664|
|WEIGHT_NOAS30−NOAS30|practical_low_urban|1441|1895|-454|
|WEIGHT_AS50−AS50|clean|425|1117|-692|
|WEIGHT_AS50−AS50|practical_mid_urban|2685|1708|977|
|WEIGHT_AS50−AS50|practical_low_urban|3302|1865|1437|
|WEIGHT_NOAS50−NOAS50|clean|415|421|-6|
|WEIGHT_NOAS50−NOAS50|practical_mid_urban|1393|1645|-252|
|WEIGHT_NOAS50−NOAS50|practical_low_urban|1603|1658|-55|
|AS50−AS30|clean|2432|1895|537|
|AS50−AS30|practical_mid_urban|5920|4080|1840|
|AS50−AS30|practical_low_urban|6711|4790|1921|
|NOAS50−NOAS30|clean|2104|2024|80|
|NOAS50−NOAS30|practical_mid_urban|7490|9331|-1841|
|NOAS50−NOAS30|practical_low_urban|7810|8249|-439|
|WEIGHT_AS50−WEIGHT_AS30|clean|2509|1844|665|
|WEIGHT_AS50−WEIGHT_AS30|practical_mid_urban|5859|4059|1800|
|WEIGHT_AS50−WEIGHT_AS30|practical_low_urban|6866|4714|2152|
|WEIGHT_NOAS50−WEIGHT_NOAS30|clean|2255|2067|188|
|WEIGHT_NOAS50−WEIGHT_NOAS30|practical_mid_urban|7564|8993|-1429|
|WEIGHT_NOAS50−WEIGHT_NOAS30|practical_low_urban|8060|8100|-40|
|WEIGHT_AS50−NOAS50|clean|4042|4120|-78|
|WEIGHT_AS50−NOAS50|practical_mid_urban|57823|6673|51150|
|WEIGHT_AS50−NOAS50|practical_low_urban|48405|9153|39252|

后续固定02/03资格只看两项源条件：wAs50−ownwAs30 fullV clean +0.162963pp（要求≥0）；V1800两draw mid/low等权+2.361111pp（要求≥+.5）。合取PASS=True，固定02/03源资格=True；当前未自动授权多seed。NoAs及原As50源差只报告不加门槛，源科学失败仍完成正式收尾。

|条件|fullV clean %|V1800 clean %|源mid/low等权 %|
|---|---:|---:|---:|
|NOAS30|98.359259|98.444444|39.583333|
|AS30|98.255556|98.333333|65.944444|
|WEIGHT_NOAS30|98.377778|98.444444|39.250000|
|WEIGHT_AS30|98.196296|98.333333|66.722222|
|NOAS50|98.448148|98.611111|39.916667|
|AS50|98.374074|98.444444|67.750000|
|WEIGHT_NOAS50|98.429630|98.666667|39.875000|
|WEIGHT_AS50|98.359259|98.444444|69.083333|

fullV27000为原生冻结clean指标，无全V逐包预测，不能虚构fullV F1/修复；固定V1800两draw×三场景面板独立重算，源没有high urban。源原始opaque数组仅fullindex0由既有producer消费，分析只读保存的logits/metadata；fc2.45GHz与训练/正式2.462GHz既有差异保留。

新两行8880step/40模型epoch全部解析，旧20/30/原50完整audit复用；每条30/50轨迹完整曲线保留。与原E50tail的物理序列、ordinaryMSE及scene/state计数实际比较见training_logs.json，仅声称已记录范围，不声称逐点IQ或逐物理×scene实现相同。

|成本组|本组native NN|训练段+自身源/正式读出 NN|
|---|---:|---:|
|new_weight_E50_segments|12543844|14921048|
|old_original_E50_segments|12543844|14921048|
|old_weight_E30_segments|6325924|8703128|
|old_original_E30_models|15921368|18298572|

本run共享smoke实际1540NN；新增两行+读出+smoke合计14922588NN。ownweighted30亲本及原30/原50/E30测试历史成本分列，不重计新增；组总计不重复加共享prefix20。每row学生train/eval、教师train/eval、actual wall、native累计成本见costs.json。分析0NN/0CP/0IQ/0更新，模型秒不是并行加速证据。

全部8logical×7冻结预测及独立评分齐备后才连接truth；独立核对1008个CM、168次RX/day/TX分区合并。七view全部accuracy/F1/worstRX、分层配对修复/新增错误、suburban相对自身clean差距均保存。

单seed观察不证明可重复；不足0.1pp的clean正变化标为微小观察，阶段基本不降不等于最终双提升。已暴露基准背景保留，不宣称新的盲测；正式IQ、SNR、信道、困难样本不改，正式结果不回流权重、种子、模型选择或选择性重跑。


当前状态VERIFIED / ANALYZED，无技术缺项。dispatcher1276091／两worker1276185、1276263／analysis1277501均已正常退出，failure与stderr为空。训练和分析实际代码`64176e7ff1aa5cd2477a504f8b442bb7966079dc`。11公开汇总单archive传输SHA比较VERIFIED，不取回IQ／checkpoint／原始逐包预测；正式全部8×7固定预测和独立评分齐备后重算1008CM／168分区合并。两新行各20轮／4440更新，累计50／11100，新增总40model-epochs／8880updates／14922588NN含1540smoke。实际native段NoAs1657.592秒／As2178.555秒，学生train4543920、学生eval594002、teachertrain0、teachereval1134000NN各自计费，源12600、正式1176002各行另列。并行wall不以模型秒总和替代。

单seed正式阶段通过，只说明相对匹配NoAs clean基本不降；clean依旧−.042857pp而非正收益，low53.242857%仍未达60%，无重复双提升证据。新权重相对原As50的clean−.411905pp，mid＋.581548pp、low＋.855357pp，clean—星地取舍仍在。相对自身30的clean＋.395833pp／mid＋1.071429pp／low＋1.280952pp是预算收益，不能全算新权重独有；clean／mid／low三因子交互＋.011905／−.269048／−.100000pp，未证明预算与权重有额外正向urban协同。

source资格已先于任何新formal指标读取保存；固定02／03据源fullVclean＋.162963pp及midlow＋2.361111pp推进。此前analysis字段automatic_multiseed_authorized=false表示分析器本身不启动；root现按用户已有源支持总E50授权登记固定四轨迹，见[source决定](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01/source_decision.json>)及[复现预登记](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01/report.md>)。E50正式分数不反向修改这一决定、seed、权重、预算或中间阶段续训。三seed完整后报告全部结果，不挑最好seed。

[机器可读摘要](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01/results.json>)／[全部汇总](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_weight_E50_20261018/collection-final-r01/aggregates>)。本轮八逻辑交付完成，整体研究目标未完成。
