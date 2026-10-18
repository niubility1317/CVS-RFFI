# 完整 practical：stepALT E30 单种子最小对照

本轮已完整完成训练、源判定、clean＋六种 practical 固定预测、独立评分和同一行分析。**新增机制未获得支持，固定配方关闭，不扩 E50。**

相对 NoAs30，clean −0.005952 pp，单种子满足 −0.1 pp 阶段界限；相对原完整 As30，clean −0.099405 pp、mid urban −0.060119 pp、low urban +0.023214 pp、urban 均值 −0.036508 pp。low 的微小涨分伴随最差 RX 下降0.141667pp，不能称为稳定星地增益。仍未取得“星地增强同时改善 clean”的可重复证据。

20261018-phase1-FcrossRX-stepALT-e30-s2026092701-m1-r01。原 As20 student/own EMA 追加10轮；mixed slot每步128As、0As交替，从128As开始。6L前向、1.25L、既有U/头/完整信道和三场景1/3保持。两个原E30冻结对照直接复用。

|视图|NoAs30 %|原As30 %|stepALT30 %|候选−NoAs pp|候选−原As pp|
|---|---:|---:|---:|---:|---:|
|clean|85.247619|85.341071|85.241667|-0.005952|-0.099405|
|practical_high|42.685119|81.405357|81.311905|38.626786|-0.093452|
|practical_mid|40.995238|78.661310|78.574405|37.579167|-0.086905|
|practical_low_suburban|39.472619|75.861905|75.766071|36.293452|-0.095833|
|practical_high_urban|41.439881|77.767262|77.694643|36.254762|-0.072619|
|practical_mid_urban|35.132738|62.806548|62.746429|27.613690|-0.060119|
|practical_low_urban|30.139881|51.244048|51.267262|21.127381|0.023214|
|urban_mean|35.570833|63.939286|63.902778|28.331944|-0.036508|

clean相对同预算NoAs变化-0.005952个百分点；0.1绝对百分点阶段约束：True。low/mid urban收益+21.127381/+27.613690个百分点。n=1不证明可重复；不足0.1个百分点的正变化为微小观察。

|条件|view|macro-F1 %|最差RX %|
|---|---|---:|---:|
|NOAS30_ORIGINAL|clean|84.979636|78.450000|
|NOAS30_ORIGINAL|practical_high|39.311416|34.304167|
|NOAS30_ORIGINAL|practical_mid|37.954813|33.520833|
|NOAS30_ORIGINAL|practical_low_suburban|36.758731|32.054167|
|NOAS30_ORIGINAL|practical_high_urban|38.481030|34.000000|
|NOAS30_ORIGINAL|practical_mid_urban|33.446603|29.495833|
|NOAS30_ORIGINAL|practical_low_urban|29.184589|25.750000|
|AS30_ORIGINAL|clean|85.044474|78.362500|
|AS30_ORIGINAL|practical_high|81.084230|71.908333|
|AS30_ORIGINAL|practical_mid|78.359706|70.212500|
|AS30_ORIGINAL|practical_low_suburban|75.572741|67.741667|
|AS30_ORIGINAL|practical_high_urban|77.423208|69.108333|
|AS30_ORIGINAL|practical_mid_urban|62.450463|56.458333|
|AS30_ORIGINAL|practical_low_urban|50.887035|46.379167|
|AS30_STEPALT|clean|84.946512|78.462500|
|AS30_STEPALT|practical_high|81.000329|71.379167|
|AS30_STEPALT|practical_mid|78.282786|69.733333|
|AS30_STEPALT|practical_low_suburban|75.483621|67.329167|
|AS30_STEPALT|practical_high_urban|77.355724|68.570833|
|AS30_STEPALT|practical_mid_urban|62.389134|56.087500|
|AS30_STEPALT|practical_low_urban|50.904592|46.237500|

|比较|view|修复|新增错误|净修复|
|---|---|---:|---:|---:|
|AS30_STEPALT−NOAS30_ORIGINAL|clean|3744|3754|-10|
|AS30_STEPALT−NOAS30_ORIGINAL|practical_mid_urban|53229|6838|46391|
|AS30_STEPALT−NOAS30_ORIGINAL|practical_low_urban|44414|8920|35494|
|AS30_STEPALT−AS30_ORIGINAL|clean|292|459|-167|
|AS30_STEPALT−AS30_ORIGINAL|practical_mid_urban|1059|1160|-101|
|AS30_STEPALT−AS30_ORIGINAL|practical_low_urban|1396|1357|39|
|AS30_ORIGINAL−NOAS30_ORIGINAL|clean|3868|3711|157|
|AS30_ORIGINAL−NOAS30_ORIGINAL|practical_mid_urban|53261|6769|46492|
|AS30_ORIGINAL−NOAS30_ORIGINAL|practical_low_urban|44297|8842|35455|

源固定判据：fullV clean候选−NoAs -0.148148pp；mid/low等权候选−原As +0.111111pp；PASS=False，固定配方关闭=True。科学负结果仍完成全部正式收尾；不按正式成绩改系数、种子或救跑。

fullV27000 clean来自原生冻结指标，没有全V逐包预测重计数；V1800两draw面板独立重算，源无high urban。面板fc2.45GHz与训练/正式2.462GHz的既有差异保留。分层与CM见source/formal JSON及CSV。

新增2220step/10epoch完整解析，As142080包，每epoch14208；原20和原30的完整audit沿用。物理L/U序列与ordinary MSE逐条比较的实际结果：{"sequence": {"records": 4440, "mismatch": 0, "all_equal": true}, "ordinary_MSE_records_new": 2220, "ordinary_MSE_records_original": 2220, "ordinary_MSE_records_different": 0, "all_equal_not_required": true, "different_actual_arg_keys": ["output_dir", "run_id"]}。stepALT不声称每ID As曝光与原配方相同，实际TX/RX/day/scene及零曝光/次数分布见training_logs.json。

新增训练+源/正式读出4351564NN，共享smoke1538NN，合计4353102NN。ownAs20父成本4797722NN单列，候选累计30轮/6660更新。旧两个E30与旧读出不重计新增成本。学生train/eval和ownEMA教师四counter见costs.json。

全部3logical×7固定预测和独立评分完成后才连接truth；重计数378个CM、63次RX/day/TX分区合并，与原scorer一致。分析0NN/0IQ/0CP/0更新；没有修改正式IQ、SNR、信道或删除困难样本。

## 结果解释与关闭边界

唯一假设是：在固定总量下，把64As/step改为相邻128/0，可能改善 clean—urban 权衡。源 fullV clean 相对 NoAs30 −0.148148pp（少40个正确，超过允许27个），mid/low均值相对原As30仅+0.111111pp，因此预登记源条件失败。源决定在首次读取正式分数前固定保存并提交。正式单种子阶段界限通过，不改变源关闭决定、不启动E50或扩种、不调整周期/相位/比例救跑。此结果只否定这项固定配方的晋级资格，不证明所有步骤分配方式均无效。

原完整 practical 仍作为匹配参照保留；其相对NoAs的强星地增益不能归给这项新增组织方式。本轮没有额外Ag／DG有效性主张，没有基于特征接近、梯度或训练状态宣称识别成功。n=1，无跨种子波动估计；两次source信道draw不是两个模型种子。训练／选模／候选关闭只用合法源域，正式分数不回流。

候选 suburban 相对自身 clean 的差距为 high 3.929762pp、mid 6.667262pp、low suburban 9.475595pp；对应绝对准确率全部略低于原As30。差距略缩小主要伴随clean下降，不能据此宣称suburban保护改善。最初2pp建议未作为本轮新增硬门槛。low urban为51.267262%，未达到最初60%探索愿望。

## 数据、模型与训练成本

保持F／三路SIG_TIME／统一Sinc／joint160／默认cosine6×160头联合训练；NoExplicitCFO与内部逐包补偿保留。原完整L6300／U56700（TX隐藏）／V27000、源RX[1,3,4,6,8]／day[1,2,3]／split392005／IQ256不变，无RX留出或Ldev。测试为原RX[0,2,5,7,9,10,11]／day[0,1,2,3]，每视图168000包；未改received IQ、SNR、信道、类别决策或困难样本集合。

新增10轮2220更新的全部日志已解析：128/0各1110步，6L每步，1.25尺度，每epoch14208 As，共142080；L1704960NN。每个L物理样本均获As，6300个ID零遗漏，每ID11至36次／均值22.552381；不声称与原配方逐ID等量。三场As分别47603／47051／47426；TX／RX／day及270个联合单元实际次数保留在私有training_logs.json，最小464、最大579。原L/U物理序列4440条零差异，ordinary MSE记录2220条零差异；这些是所记录MSE的比较，不冒称额外逐点IQ重计数。最大L损失分解误差1.49384e-7，10轮JSON/CSV一致，全部418行日志已读，无非有限值或错误标记。

|成本项目|实际样本前向 NN|
|---|---:|
|新E10 student train|2271960|
|新E10 student eval|324002|
|新E10 ownEMA teacher train|0|
|新E10 ownEMA teacher eval|567000|
|新源读出|12600|
|新正式预测|1176002|
|本次共享真实CP＋双步smoke|1538|
|**本次实际新增合计**|**4353102**|
|复用As20父原生训练／验证（不重计新增）|4797722|
|候选累计E30原生训练／验证|7960684|
|复用两行旧E30及各自读出（不重计新增）|18298572|

新原生训练及其验证wall1065.561s，10轮epoch时间和1046.539s；新七视图预测wall48.672s，r02只读分析wall16.019s。不同口径不相加冒充完整端到端wall，不宣称速度优势。父20轮／4440更新另列，候选累计30轮／6660更新。

## 验证与交付状态

训练release代码`cc9a2dac5e2667dab9d730dd8b618e8e635fc5ca`，r02分析代码`2e08d8ddabd28ca4bad350485a7a98915c006046`。一次限定P0/P1审查PASS；33协议负例、18py与两段远端AST、真实CP无query2NN＋实际128/0双步1536NN均PASS。实际parent source_roles=EXACT_MATCH、target_contact=false，student／ownEMA严格恢复；实际信道与原As30一致。唯一归档传输比较、远端编译、初始PID/CWD/argv/GPU/日志增长均VERIFIED。

r01只读分析因旧损失字段假设停止，产物／release保留；其0NN／0IQ／0CP／0更新，未读正式truth。r02对真实首末标量与错误raw贡献执行8项定点回归，未放宽tol；只改分析接口，不重训／重预测／重评分。原训练dispatcher1233511／worker1233539、r01分析1235749与r02分析1245938均已退出；最终ANALYZED完成文件、stdout与空stderr相互支持，无技术缺项。

完整3条件×7视图各168000包；独立重算378个CM和63次RX/day/TX分区合并，与既有独立scorer一致。固定全部预测及独立评分完成后才连接truth。source450行和formal378行分层汇总均保留；源fullV仅原生指标，没有全V逐包F1／修复重计数，不能补造。正式state/quality识别准确率未提供，训练state覆盖不能替代它。

已取回11个公开汇总文件，单次归档SHA比较VERIFIED：`13c978a5ca3247fe75e68690dec32b577e789090444e57e353b7d266cc6d38fd`。只取回汇总，不取IQ、checkpoint或原始逐包预测。全局及配对结果见[results.json](results.json)；全部RX/day/TX指标和配对分层见本地[完整汇总目录](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_stepALT_E30_20261018/collection-final-r02/aggregates>)。

研究目标仍未完成。当前无本侧健康训练／预测／分析进程需要延长或修改；下一候选未登记，不从本次正式微差选机制、种子或超参。
