# R20：固定权重的CUDA Graph执行优化

状态：VERIFIED / ANALYZED。

源计时显示R18两版和R19两版的参数减少均未转化为B1加速。本组保持用户原350521父版及R18两个278180版本的全部权重与数学运算，不另训、不调rank，检验固定计算图重放。三行在读取R19成绩前依据源计时固定，不进行目标排序。主对话研究相位/通道结构，本组处理执行开销。

|固定行|完整恢复器参数|继承源最终checkpoint|新增训练|GPU|
|---|---:|---|---:|---:|
|R46_350521_graph|350521|prune_refine E20/4940|0|2|
|R18_factor625_stage_graph|278180|factor625_stage E20/19700|0|2|
|R18_factor625_cond_graph|278180|factor625_cond E20/19700|0|3|

原R46来源commit bbb8daf817de050b958d217b70d4c33f4aefe341，R18来源aca99499d3f0170df71380f6429ac9a01ef412f8。旧权重均按原7/63/30物理源契约、0TX/F/V/target恢复器训练读取核实；图运行不改变其来源与成本。R46各组件累计80轮/串行120，R18各组件100轮/串行140，本组均新增0。F50原seed2026092701/defaulthead只作冻结评测，不进入恢复器训练。既有READY/EXACT_MATCH缓存复用，不重验数据。

每行首步实际2Ls noquery smoke，检查输入替换、旧输出不会被后续重放覆盖、原函数与图函数的IQ/variance/features/logits误差以及状态不变。程序复用原模块参数、反射边界、真实共轭伴随和每stage两步CG，仅把CUDA数值Python检查移到重放后。捕获缓冲区长期保留，每次copy新输入，输出clone保证所有权。原native/其他owner代码保持。

每行固定source1800×7，再完整formal168000×7，B256。所有行无源准确率或速度晋级门槛；每行全部七预测完成后独立CPU truth-last，R06共享score.lock/CPU2/max1ownscorer。对照复用原native七景预测，不重复旧target前向；完整比较IQ/features/logits及argmax。原batch128与本组256可能产生数值差异，逐景报告，不预先承诺目标逐位相同。逐景accuracy/MacroF1、TX/RX/day、净纠错、clean损伤、70%/5pp目标和父0.8pp/原R17历史1pp均单列；已暴露单seed工程基准，不称独立盲测或逐细节TX证明。

同卡64sourceV，对原父和当前候选分别测native/graph完整G及G+F，B1 warm16/64次、B32 warm4/16次，三轮正反顺序。图计时包含新输入copy和输出clone；捕获/预热单列。报告median/p90/p95/逐轮中位/吞吐、完整参数/文件字节/ConvLinearMAC、匹配suite显存及独立单模型G+F五批量graph池内存。输入已在GPU，不含传输/loader/预处理；MAC不含FFT/CG/norm/激活/访存/F，物理3stage/6CG完整执行。共享suite驻留不冒充独立部署内存。

GPU2串行原父与stage，GPU3条件版，本组最多2own/每卡1；实际R19owned句柄退出后、sharedlock下free≥18GB/allowner<2才取得GPU。普通账户、新不可覆盖run/output/release，一次preflight/一次归档两端SHA/一次compile。技术失败保留影响行及缺失测试，无nativefallback/自动重跑/热改或停止其他任务；健康行继续。实际方法commit、命令、环境/CWD、路径、停止规则与产物见experiment.json。

图接口依据[PyTorch2.1 CUDA Graph文档](https://docs.pytorch.org/docs/2.1/notes/cuda.html#cuda-graphs)实现。28项本地聚焦检查通过，其中六项使用真实R46/R18架构零/随机IQ，对原函数逐位比较，并禁止Tensor转Python bool；这些CPU检查不替代真实GPU捕获。唯一独立P0/P1审查发现的主机同步问题已修复，原问题一次定向复核PASS；未重复全审查，真实GPU捕获仍待launcher首步smoke。


## 已核实发布与源计时快照

实际method `c60a675e4bbc2c207e0a16d36c0ceec075238324`。一次归档两端SHA及remote compile通过；三行均以实际PID/CWD/完整命令/commit/GPU/日志增长核实，三行2Ls在B1/8/32/64/256的IQ/variance/features/logits逐位一致，输入替换与输出所有权检查PASS。28项执行检查、18项结果计算检查、唯一独立审查及一次P1定向修复通过。

|已完成profile行|同卡自身native G+F B1 ms|自身graph G+F B1 ms|自身native→graph倍数|同卡父graph→本版graph倍数|
|---|---:|---:|---:|---:|
|R46_350521_graph|64.139907|11.283393|5.6845|1.0000|
|R18_factor625_cond_graph|75.675792|10.965262|6.9014|1.0296|

这是源计时快照，不是正式识别结论。三行继续完成原定七景；原父也获得graph执行优化，必须同时报告优化父对照。图开销包含copy/clone；数值检查位置改变也影响计时，不单独归因于launch。完整graph池reserved/allocated与capture峰值在最终结果单列。


## 完整正式结果

三行均完成source1800×7和formal168000×7，逐行七预测固定后独立CPU truth-last；实际controller/三worker退出，无技术失败。三份原完整权重本地复用并严格加载，静态路径与原函数零IQ逐位一致，无新增权重或训练。

|版本|完整参数|较350521减参|父逐景0.8pp|目标原native决策全一致|70%/5pp|
|---|---:|---:|---|---|---|
|R46_350521_graph|350521|0.000000%|True|True|False/False|
|R18_factor625_stage_graph|278180|20.638136%|True|True|False/False|
|R18_factor625_cond_graph|278180|20.638136%|True|True|False/False|

|版本|场景|准确率%|MacroF1%|较350521pp|较自身原native pp|raw净纠错|
|---|---|---:|---:|---:|---:|---:|
|R46_350521_graph|clean|84.456548|84.156318|+0.000000|+0.000000|-369|
|R46_350521_graph|practical_high|82.452381|82.040850|+0.000000|+0.000000|+70089|
|R46_350521_graph|practical_mid|81.222024|80.857090|+0.000000|+0.000000|+69393|
|R46_350521_graph|practical_low_suburban|79.620238|79.351692|+0.000000|+0.000000|+68192|
|R46_350521_graph|practical_high_urban|81.386905|81.006498|+0.000000|+0.000000|+70037|
|R46_350521_graph|practical_mid_urban|74.447024|74.361224|+0.000000|+0.000000|+67139|
|R46_350521_graph|practical_low_urban|67.510119|67.825990|+0.000000|+0.000000|+62514|
|R18_factor625_stage_graph|clean|84.460714|84.179540|+0.004167|+0.000000|-362|
|R18_factor625_stage_graph|practical_high|82.614881|82.217301|+0.162500|+0.000000|+70362|
|R18_factor625_stage_graph|practical_mid|81.358929|80.980512|+0.136905|+0.000000|+69623|
|R18_factor625_stage_graph|practical_low_suburban|79.835119|79.495611|+0.214881|+0.000000|+68553|
|R18_factor625_stage_graph|practical_high_urban|81.507143|81.135960|+0.120238|+0.000000|+70239|
|R18_factor625_stage_graph|practical_mid_urban|74.617857|74.453097|+0.170833|+0.000000|+67426|
|R18_factor625_stage_graph|practical_low_urban|67.802976|67.832358|+0.292857|+0.000000|+63006|
|R18_factor625_cond_graph|clean|84.497024|84.227256|+0.040476|+0.000000|-301|
|R18_factor625_cond_graph|practical_high|82.644048|82.242109|+0.191667|+0.000000|+70411|
|R18_factor625_cond_graph|practical_mid|81.410119|81.021850|+0.188095|+0.000000|+69709|
|R18_factor625_cond_graph|practical_low_suburban|79.825000|79.480666|+0.204762|+0.000000|+68536|
|R18_factor625_cond_graph|practical_high_urban|81.525000|81.132840|+0.138095|+0.000000|+70269|
|R18_factor625_cond_graph|practical_mid_urban|74.619643|74.371010|+0.172619|+0.000000|+67429|
|R18_factor625_cond_graph|practical_low_urban|67.641667|67.663096|+0.131548|+0.000000|+62735|

0.8pp为逐场景退化容忍上限，不是性能优化目标。原R17历史1pp、逐TX/RX/day、clean损伤、波形/几何、全部原native差异和完整成本见results.json/CSV。单seed已暴露工程基准；不称新盲测、显著性结论或逐细节TX证明。

|版本|口径|父native ms|本版native ms|父graph ms|本版graph ms|本版native→graph倍数|父graph→本版graph倍数|
|---|---|---:|---:|---:|---:|---:|---:|
|R46_350521_graph|G_only_B1|54.680849|54.680849|10.959873|10.959873|4.9892|1.0000|
|R46_350521_graph|G_only_B32|99.104880|99.104880|69.761248|69.761248|1.4206|1.0000|
|R46_350521_graph|G_plus_F_B1|64.139907|64.139907|11.283393|11.283393|5.6845|1.0000|
|R46_350521_graph|G_plus_F_B32|106.394667|106.394667|72.592298|72.592298|1.4656|1.0000|
|R18_factor625_stage_graph|G_only_B1|44.641074|51.791669|8.941385|8.383231|6.1780|1.0666|
|R18_factor625_stage_graph|G_only_B32|87.828908|71.511615|67.417647|47.179899|1.5157|1.4289|
|R18_factor625_stage_graph|G_plus_F_B1|53.872167|60.587931|9.458549|9.391031|6.4517|1.0072|
|R18_factor625_stage_graph|G_plus_F_B32|77.659901|71.069359|53.635680|44.759730|1.5878|1.1983|
|R18_factor625_cond_graph|G_only_B1|54.771997|64.577500|12.240593|12.085964|5.3432|1.0128|
|R18_factor625_cond_graph|G_only_B32|98.557858|89.751519|72.032697|59.226066|1.5154|1.2162|
|R18_factor625_cond_graph|G_plus_F_B1|66.785454|75.675792|11.289904|10.965262|6.9014|1.0296|
|R18_factor625_cond_graph|G_plus_F_B32|109.577657|95.546546|72.272258|59.940270|1.5940|1.2057|

同卡64sourceV、三轮交替顺序；图计时包含输入copy、输出clone，预热/捕获单列，重放后数值检查按本候选调用位置执行。G含完整coarse/action/3prior/6CG，G+F含原默认头；输入预加载GPU，不含传输/loader/预处理。native原函数内部主机检查与图专用路径重放后检查的位置不同，实测改进同时包含该执行安排与图重放；不单独归因于kernel launch。两卡存在其他owner负载，不能以跨卡绝对时间比较结构。父native与父graph两套对照均保留，不能把父版保持native的差距全称为结构收益。

|版本|checkpoint字节|ConvLinearMAC/IQ|单模型G+F五批量graph后allocated MiB|reserved MiB|capture peak MiB|
|---|---:|---:|---:|---:|---:|
|R46_350521_graph|1686812|39081004|53.011|1716.000|310.371|
|R18_factor625_stage_graph|1482748|29727404|52.746|1410.000|250.841|
|R18_factor625_cond_graph|1482748|29727404|52.746|1410.000|250.841|

上述独立部署内存含一个完整G+F及批量1/8/32/64/256的五套graph池，不能当作单批量B1内存，也不能把捕获前仅权重的allocated当作native峰值。匹配计时suite的驻留/peak/incremental、p90/p95、吞吐、各轮中位和每批量setup秒数全部保留于results.json。MAC仅Conv1d+Linear，不包括FFT/CG/norm/激活/访存/F。图不减少权重参数；27.82万减参来自已经固定的R18。

|版本|场景|argmax差异条数|IQ最大差|features最大差|logits最大差|
|---|---|---:|---:|---:|---:|
|R46_350521_graph|clean|0|0|2.86102295e-06|4.76837158e-06|
|R46_350521_graph|practical_high|0|0|2.38418579e-06|3.81469727e-06|
|R46_350521_graph|practical_mid|0|0|2.38418579e-06|3.81469727e-06|
|R46_350521_graph|practical_low_suburban|0|0|2.38418579e-06|3.81469727e-06|
|R46_350521_graph|practical_high_urban|0|0|2.38418579e-06|3.81469727e-06|
|R46_350521_graph|practical_mid_urban|0|0|2.38418579e-06|3.81469727e-06|
|R46_350521_graph|practical_low_urban|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_stage_graph|clean|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_stage_graph|practical_high|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_stage_graph|practical_mid|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_stage_graph|practical_low_suburban|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_stage_graph|practical_high_urban|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_stage_graph|practical_mid_urban|0|0|2.86102295e-06|3.81469727e-06|
|R18_factor625_stage_graph|practical_low_urban|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_cond_graph|clean|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_cond_graph|practical_high|0|0|2.86102295e-06|3.81469727e-06|
|R18_factor625_cond_graph|practical_mid|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_cond_graph|practical_low_suburban|0|0|2.86102295e-06|3.81469727e-06|
|R18_factor625_cond_graph|practical_high_urban|0|0|2.38418579e-06|3.81469727e-06|
|R18_factor625_cond_graph|practical_mid_urban|0|0|2.38418579e-06|4.29153442e-06|
|R18_factor625_cond_graph|practical_low_urban|0|0|2.38418579e-06|3.81469727e-06|

原native预测复用，batch128与新256可能带来浮点差异；不同数值不自动等同于不同分类。逐位/分类一致性分开报告，任何差异都未回流改模型、rank或重跑目标。

部署恢复器可加载原R46/R18完整权重，使用GraphCache(FrozenPipeline(G,graph_safe=True))(IQ)[0]，无需F或teacher；首次各批量捕获及显存开销必须计入具体部署方案。F只在本轮冻结评测的G+F路径使用。

[完整结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-cudagraph-s2026092701-m3-r20/results.json) · [逐TX/RX/day CSV](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-cudagraph-s2026092701-m3-r20/row_metrics.csv)


## 本轮结论与后续边界

三版各自七景分类决策与其原native版全部一致，新增纠错和新增错分均为0。所有正式浮点数组并非逐位相同，因此只能报告分类一致；真实2Ls同批量smoke的逐位一致不扩大到整套不同batch的正式数组。R18相对350521父版的七景小幅正收益保持，本轮没有新增识别训练或新的分类收益。

|版本|自身native G+F B1 ms|自身graph G+F B1 ms|自身native→graph倍数|同卡父graph ms|本版B32 native→graph吞吐条/s|
|---|---:|---:|---:|---:|---|
|R46_350521_graph|64.139907|11.283393|5.6845|11.283393|298.439→439.345|
|R18_factor625_stage_graph|60.587931|9.391031|6.4517|9.458549|445.505→680.970|
|R18_factor625_cond_graph|75.675792|10.965262|6.9014|11.289904|331.882→533.430|

同卡父版获得同样图执行后，两份27.82万版B1中位时间仅约0.7%和2.9%更低，当前三轮共享负载测量不足以证明这种微小差距稳定。大幅收益属于执行安排与图重放，不把6倍左右的自身native加速全归于结构压缩。B32优化父对照下的吞吐比也完整保留，阶段版约1.1485、条件版约1.2131。

27.82万完整恢复器仍较350521减参20.6381%、ConvLinearMAC减少23.9339%、checkpoint文件减少12.0976%；图本身不减少权重，也不删除物理求解。每行跨卡绝对时间不可用于判断两种损失哪个更快。

内存与启动代价：27.82万两版在一个G+F加五批量1/8/32/64/256配置下，实测allocated52.746MiB、reserved1410MiB（约1.377GiB），捕获过程allocated峰值250.841MiB。reserved包含图池与分配器缓存，不能只用52.746MiB宣称全部显存占用。阶段版五批量预热与捕获合计约1.66s，稳态B1延迟不含此一次性开销；单批量B1部署内存未单独测量。

识别边界保持：阶段版lowurban67.802976%、条件版67.641667%，距离70%仍差2.197024pp、2.358333pp；midurban仍较rawclean84.6761904762%低约10.06pp。clean分别较rawclean净损362条、301条。参数、速度与七景保持已验证，完整识别目标仍未达成。

下一步继续从用户原350521基底研究源IQ保真，不从R20目标成绩另选教师或rank。已保存一个尚未启动的源诊断范围：原父及R18两版全部纳入，仅128Ls+128Us均匀不透明位置，clean与既有epoch01/20 received，检验clean恒等、复频谱误差及解析相关残差；不读取F、TX标签、V或target。现有R18/R19成绩与R20完成/容忍标志已暴露，该研究不称新盲测。后续任何新训练版本仍完整七景正式测试。
