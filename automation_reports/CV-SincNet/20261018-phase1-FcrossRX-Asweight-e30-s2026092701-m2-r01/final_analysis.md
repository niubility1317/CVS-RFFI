# 完整 practical：As 权重转移 E30 四条件最小对照

20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01。两个新行各自 own原FINAL20 student/EMA追加10轮；原NoAs30/As30冻结对照直接复用。固定mixed组权重由.25/.25改为.125/.375，同时作用于CE与对称KL；primary ordinary .5、raw .25、原ORD128 crossRX .1保持。

有效clean监督份额从1降至.875，As从.25升至.375；总尺度1.25、6L前向与次序、原selected64 mask及完整信道保持。NoAs行selected64保持原raw64，真实As数为0。其ordinary/raw份额从.75/.5变为.625/.625；As行ordinary+raw clean从1变为.875、As从.25变为.375。没有新增IQ算子、网络、头或U规则。

|视图|原NoAs30 %|原As30 %|加权NoAs30 %|加权As30 %|新As−新NoAs pp|新As−原As pp|新NoAs−原NoAs pp|交互 pp|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clean|85.247619|85.341071|85.179762|84.852976|-0.326786|-0.488095|-0.067857|-0.420238|
|practical_high|42.685119|81.405357|42.148810|81.657143|39.508333|0.251786|-0.536310|0.788095|
|practical_mid|40.995238|78.661310|40.477976|79.013690|38.535714|0.352381|-0.517262|0.869643|
|practical_low_suburban|39.472619|75.861905|39.070238|76.342262|37.272024|0.480357|-0.402381|0.882738|
|practical_high_urban|41.439881|77.767262|40.933929|78.042262|37.108333|0.275000|-0.505952|0.780952|
|practical_mid_urban|35.132738|62.806548|34.737500|63.411905|28.674405|0.605357|-0.395238|1.000595|
|practical_low_urban|30.139881|51.244048|29.869643|51.961905|22.092262|0.717857|-0.270238|0.988095|
|urban_mean|35.570833|63.939286|35.180357|64.472024|29.291667|0.532738|-0.390476|0.923214|

相对同预算加权NoAs：clean变化-0.326786个百分点，0.1绝对百分点阶段约束=False；low/mid urban变化+22.092262/+28.674405个百分点。相对原NoAs的clean变化另报-0.394643pp。交互=(新As−原As)−(新NoAs−原NoAs)。n=1无种子SD，不证明可重复；不足0.1pp的正变化为微小观察。

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

|比较|view|修复|新增错误|净修复|
|---|---|---:|---:|---:|
|WEIGHT_AS30−WEIGHT_NOAS30|clean|3783|4332|-549|
|WEIGHT_AS30−WEIGHT_NOAS30|practical_mid_urban|55234|7061|48173|
|WEIGHT_AS30−WEIGHT_NOAS30|practical_low_urban|46375|9260|37115|
|WEIGHT_AS30−AS30|clean|246|1066|-820|
|WEIGHT_AS30−AS30|practical_mid_urban|2186|1169|1017|
|WEIGHT_AS30−AS30|practical_low_urban|2635|1429|1206|
|WEIGHT_NOAS30−NOAS30|clean|301|415|-114|
|WEIGHT_NOAS30−NOAS30|practical_mid_urban|1262|1926|-664|
|WEIGHT_NOAS30−NOAS30|practical_low_urban|1441|1895|-454|
|AS30−NOAS30|clean|3868|3711|157|
|AS30−NOAS30|practical_mid_urban|53261|6769|46492|
|AS30−NOAS30|practical_low_urban|44297|8842|35455|
|WEIGHT_AS30−NOAS30|clean|3705|4368|-663|
|WEIGHT_AS30−NOAS30|practical_mid_urban|54549|7040|47509|
|WEIGHT_AS30−NOAS30|practical_low_urban|45946|9285|36661|

源固定E50推进条件：新As−原As fullV clean -0.059259pp（要求≥−.1）；mid/low等权+0.777778pp（要求≥+.5）。两项合取PASS=True，E50 source eligible=True，固定配方关闭=False。加权NoAs源结果及交互完整描述，不新增晋级门槛。源科学失败仍完成全部正式收尾；不按正式成绩调权、选seed或救跑。

|条件|fullV clean %|V1800 clean %|mid/low等权 %|
|---|---:|---:|---:|
|NOAS30|98.359259|98.444444|39.583333|
|AS30|98.255556|98.333333|65.944444|
|WEIGHT_NOAS30|98.377778|98.444444|39.250000|
|WEIGHT_AS30|98.196296|98.333333|66.722222|

fullV27000 clean来自原生冻结指标，未保存逐包fullV预测，不声称独立fullV F1或配对修复；V1800两draw三场景独立重算，全部draw及RX/day/TX分层见source CSV/JSON。源无high urban，不将mid/low称作urban三场均值。面板fc2.45GHz与训练/正式2.462GHz既有差异保留。

历史基准已经暴露；本候选配方与seed在新正式访问前固定，来源与边界保留，结果不声明新的独立盲测。AS50等历史命名中的50%不冒称本候选E50预算或隔离权重因果证据。

新增两行共4440step/20模型epoch完整解析，逐行L1704960，As行实际142080、NoAs行0。原前20/原30完整audit复用。实际物理L/U、ordinary MSE及逐步scene/state计数与对应原tail比较：{"WEIGHT_NOAS30": {"physical_sequence": {"records": 4440, "mismatch": 0, "all_equal": true}, "records": 2220, "ordinary_MSE_records_different": 0, "scene_counts_records_different": 0, "state_counts_records_different": 0, "actual_As_count_records_different": 0, "maximum_ordinary_MSE_difference": 0.0, "supported_physical_mask_and_scene_state_count_scope_only": true, "per_packet_IQ_or_channel_trace_equality_not_claimed": true, "different_actual_arg_keys": ["output_dir", "run_id"]}, "WEIGHT_AS30": {"physical_sequence": {"records": 4440, "mismatch": 0, "all_equal": true}, "records": 2220, "ordinary_MSE_records_different": 0, "scene_counts_records_different": 0, "state_counts_records_different": 0, "actual_As_count_records_different": 0, "maximum_ordinary_MSE_difference": 0.0, "supported_physical_mask_and_scene_state_count_scope_only": true, "per_packet_IQ_or_channel_trace_equality_not_claimed": true, "different_actual_arg_keys": ["output_dir", "run_id"]}}。仅在已记录的mask物理曝光和scene/state计数范围核对，不声称逐点IQ或逐物理ID×scene实现相等；实际分布见training_logs.json。

|条件|student train NN|student eval NN|teacher eval NN|native NN|native wall s|
|---|---:|---:|---:|---:|---:|
|NOAS30|6078780|918004|963900|7960684|2146.190|
|AS30|6078780|918004|963900|7960684|2860.909|
|WEIGHT_NOAS30|2271960|324002|567000|3162962|870.933|
|WEIGHT_AS30|2271960|324002|567000|3162962|1060.936|

新增训练+源/正式读出8703128NN，共享smoke实际1540NN，合计8704668NN。两own原E20父成本与旧两E30读出分别单列，不重计新增成本；每候选累计30轮/6660更新。原own EMA教师保持，teacher train=0；wall为实际模型秒，不据此宣称加速。

全部4logical×7冻结预测和独立评分齐备后才连接truth；独立重算504个CM、84次RX/day/TX分区合并，与原scorer一致。所有view的accuracy/F1/worstRX、配对及分层修复/新增错误、suburban与自身clean差距完整保存。分析0NN/0IQ/0CP/0更新；正式IQ/SNR/信道及困难样本未改。


本轮状态：VERIFIED / ANALYZED，无技术缺项。dispatcher、两训练与0NN分析进程均已退出；两新行各own20＋10，总30轮／6660更新。完整结果取回11个公开汇总文件，单次归档SHA比较VERIFIED；不取回IQ／checkpoint／原始逐包预测。代码与分析均为`3befd47c8041d81f9430eda53562d0821624ff04`。

当前最值得保留的是完整practical监督本身。权重机制在本单seed提高mid／low，但未解决clean代价，不能称为双提升。下一唯一假设为：固定.375监督权重后，总E50是否比自身E30带来源mid／low进一步收益且源clean不降。两条各ownweightedFINAL30＋20，匹配NoAs；不换权重、通道、seed、数据、头或U。推进决定在首次读取本轮正式分数前已保存于[source_decision.json](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01/source_decision.json>)，故不以本轮正式成绩回调。未来扩种仅在As50−ownAs30源fullVclean≥0pp且mid／low均值≥+.5pp时获得支持；失败则放弃固定配方扩种，仍完成全部七视图，不扫权重或epoch救跑。总E50不等于追加50。

完整[机器可读结果](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e30-s2026092701-m2-r01/results.json>)与[全部分层汇总](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_weight_E30_20261018/collection-final-r01/aggregates>)；E50[预登记](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01/report.md>)。研究目标未完成。


冻结结果图：全部七视图四条件[准确率PNG](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_weight_E30_20261018/figures/asweight_E30_seven_view_accuracy.png>)／[PDF](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_weight_E30_20261018/figures/asweight_E30_seven_view_accuracy.pdf>)，候选As增量、NoAs增量与二者[交互PNG](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_weight_E30_20261018/figures/asweight_E30_paired_increments_interaction.png>)／[PDF](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_weight_E30_20261018/figures/asweight_E30_paired_increments_interaction.pdf>)。全精度[坐标表](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_weight_E30_20261018/figures/coordinates.csv>)与冻结汇总核对，最大误差约1.15×10⁻¹⁴pp；n=1无误差棒，图明确标注cleanstage失败，NoAs退化不计作纯As增益。制图没有新增NN／IQ／checkpoint／预测或评分，不影响此前固定E50决定。图与坐标为本地生成产物，不入Git。
