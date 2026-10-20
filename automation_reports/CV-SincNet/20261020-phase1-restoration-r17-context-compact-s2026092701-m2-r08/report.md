# 同时压缩粗恢复与信道估计的 R17 精简 R08

状态：ANALYZED。用户要求最大程度精简R17及多GPU批量快速迭代。本轮在R06/R07六行仍训练或预测、未读其新目标准确率时预先固定。动机为纯结构盘点：原coarse1305958/action851589合计2157547参数，占最小R07完整2352389参数约91.7%，仅压prior已接近这条路线的结构下限。CPU半秩分解context长卷积后context为1397619；此计数不证明身份性能。

|固定行|完整参数|比原4665073减少|冻结context|可训练prior|GPU|
|---|---:|---:|---:|---:|---:|
|context_half_prior_quarter|1822977|60.9229%|1397619|425358|4|
|context_half_prior_eighth_pointwise|1592461|65.8642%|1397619|194842|5|

只从原source-selected R17/cd977448/E20/4940初始化与蒸馏，无R04/R05/R06/R07学生。原source-only粗恢复和action估计的84个长Conv1d独立CPU半秩平衡SVD，保留通道数、bias/GN/1x1、谱变换/物理算子/nativevariance范围；分解后context完全冻结。prior为原三步长权重均值后quarter，或eighth并共享原1x1权重；保留各步bias/GN。0初始化IQ/TX/F/V/target/优化更新，CPU分解时间另报。context是原source权重的确定性近似，不再宣称context逐位相同；训练来源与物理数据契约不变。

关键差异：原R17教师必须用自身原coarse/action/nativevariance，学生用自身压缩context。teacher_and_student_context先计算原teacher上下文与原三步目标，再计算student上下文；禁止用student近似物理估计生成teacher目标。源训练、真实smoke均调用同一helper，sentinel负向测试区分两个上下文。仍只优化prior，coarse/action/teacher/F50/defaulthead参数0更新。原3步×每步2CG不变，恢复推理仅当前IQ，0TX/RX/场景/clean/教师/F运行时输入或状态更新。

固定训练同R04/R06/R07：R07原63000=L6300/U56700缓存epochs1..20、15%clean/85%residual，batch256/model2701/data2026102001；AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32，E20/4940。损失10×三步IQNMSE_to原R17 + 冻结原F50/defaulthead T1 KL + .5×unitF160距离平方，task系数1；clean仅detached能量分母floor1e-4/日志。每行L126000/U1134000，原R17/Fteacher/Fstudent各1260000输入；相较此前共享同一context的版本，本轮teacher context和student context各独立前向1260000次，不能忽略额外教师计算。prior继承20+新20=40，coarse20/action20共80继承串行pass，context分解0优化；原F50预训练50/11100、F输入反传及CPU初始化成本分列。使用源监督F知识，不称F-free无监督。

两个候选/秩/预算在新结果前固定，不根据source或target分数筛选/早停/重排/改参数/重跑。每行E20冻结后自动source七视图诊断/profile/独立truth-last，再同168000物理样本×clean与六practical完整target预测，七视图IQ/logit/feature/IDs全部固定后独立CPU才读truth/clean。相对原R17逐场景准确率下降<=1个百分点，以最多损失1680条正确数判断；0pp另诊断。报告accuracy/MacroF1、TX/RX/day/confusion、raw repair/damage/net、full256/prefix232/tail24波形/FFT与配对clean特征差距、完整成本。low urban≥70%及其他环境接近clean的长期目标另报，未自设“接近”阈值；不称新盲测/全局最简/Phase2/在轨保证。

35项合成/协议检查PASS，覆盖两结构完整参数、context冻结、共享alias、原teacher保持、独立context、strictreload、源预算/权限、七视图truth-last及失败隔离。唯一独立P0/P1审查后固定code/config commit。launcher第一步真实原R17/F50/2Ls CPU noquery smoke：teacher阶段输出必须逐位等于原R17自身forward；student prior/F输入梯度有限非零、context/teacher/F/head冻结，checkpoint严格重载逐位一致；0TX/V/target/optimizer更新。PASS立即排队，无批准token。

GPU4/5排队：等待R06根complete/failure终态且其实际ownedGPU句柄退出，仅读PID/命令和终态存在性，不读评分；不修改、停止或重启当前健康六行。之后每卡一个own worker、sourcefree≥18GB/allowner<2，R07健康2/3/6/7继续，保持最多六条own GPU。CPU2线程，所有R17候选共用R06score.lock、最多一个own scorer。一次普通N607预检/新不可覆盖root与release/单archive SHA比较/远端compile/实际PID-CWD-cmd-GPU-loggrowth。契约/nonfinite/OOM/技术错误停止失败行并保留，健康行继续，无自动重试/回退；性能失败不作重启理由。source profile排除GPU checkpointpayload/optimizer，报告resident和增量peak，仍有coowner/顺序限制。命令、环境/CWD、路径与产物见experiment.json。

发布前证据：35项合成/协议检查和唯一独立P0/P1审查PASS，独立复跑35项，未访问真实IQ/权重/远端；检查教师与学生独立context、压缩context冻结/复载、R06实际句柄等待、共享score.lock及自动目标链路。普通N607预检VERIFIED，R06实际GPU句柄仍2条，新root/release不存在；只提交CPU smoke和等待队列，GPU释放后再启动，保留六条健康任务。

实际发布 VERIFIED：方法commit `2e23af45f4a5fc26c3d44c4efa4305ac36ef45db` 已推送且远端OID独立比较一致；单归档SHA比较/远端compile/help通过。实际原R17/F50/2Ls noquery smoke PASS，teacher阶段逐位等于其自身原context前向，使用compressed-context的student各prior梯度和F输入梯度有限非零，context保持冻结，context/teacher/F/head0更新、严格复载逐位输出一致。先等待R06目标测试和评分结束、原实际controller及GPU句柄退出，再在GPU4/5各启动一个新worker；实际PID/CWD/完整命令/commit/GPU/loggrowth与首轮完成均核实，R07健康四卡继续。未读取R06结果改变R08设计/训练/矩阵。

完整汇总工具9项合成检查PASS，包括context_parameters1397619、原teacher context与student context各1260000前向计费、prior-only预算、两固定row/全七视图/分组混淆矩阵、独立assessment与整数1pp重算一致、payload-free推理profile。该工具未改运行中的方法commit；当前无R08完整训练或目标性能结果。


## 完整目标结果

两个固定候选均完成E20/4940、source七视图诊断和同168000物理样本×七目标场景；全七预测固定后独立CPU truth-last，实际controller/worker句柄全部退出。逐场景相对原R17下降<=1个百分点，以整数正确数验收；补报相对已验收3118001参数版本差值。完整分组见results.json/row_metrics.csv。

|版本|场景|目标准确率|原R17准确率|相对原R17（百分点）|相对3118001版（百分点）|1pp验收|
|---|---|---:|---:|---:|---:|---|
|context_half_prior_quarter|clean|69.539881|84.586905|-15.047024|-15.046429|未通过|
|context_half_prior_quarter|practical_high|79.432738|81.319048|-1.886310|-1.522024|未通过|
|context_half_prior_quarter|practical_mid|77.857738|79.577976|-1.720238|-1.461905|未通过|
|context_half_prior_quarter|practical_low_suburban|76.111310|77.996429|-1.885119|-1.525000|未通过|
|context_half_prior_quarter|practical_high_urban|77.217262|80.353571|-3.136310|-2.772024|未通过|
|context_half_prior_quarter|practical_mid_urban|67.571429|73.685119|-6.113690|-6.143452|未通过|
|context_half_prior_quarter|practical_low_urban|58.908333|67.216667|-8.308333|-8.448214|未通过|
|context_half_prior_eighth_pointwise|clean|68.384524|84.586905|-16.202381|-16.201786|未通过|
|context_half_prior_eighth_pointwise|practical_high|79.393452|81.319048|-1.925595|-1.561310|未通过|
|context_half_prior_eighth_pointwise|practical_mid|77.239286|79.577976|-2.338690|-2.080357|未通过|
|context_half_prior_eighth_pointwise|practical_low_suburban|75.055357|77.996429|-2.941071|-2.580952|未通过|
|context_half_prior_eighth_pointwise|practical_high_urban|76.647619|80.353571|-3.705952|-3.341667|未通过|
|context_half_prior_eighth_pointwise|practical_mid_urban|65.224405|73.685119|-8.460714|-8.490476|未通过|
|context_half_prior_eighth_pointwise|practical_low_urban|55.435119|67.216667|-11.781548|-11.921429|未通过|

本轮结论：**两版均未通过七场景最多下降1个百分点验收**。已通过验收的2352389参数R07版本和3118001参数版本保留，本轮全部失败场景和产物保留。这不是全局最简证明。单训练seed/单原F50seed/已暴露基准，不称新盲测或多seed总体保证；所有固定候选都已报告，目标结果未回流训练、超参、父权重、候选重排或选择性重跑。

|版本|完整参数|参数较原R17减少|相对原R17最大下降（百分点）|相对3118001版最大下降（百分点）|零下降诊断|low urban准确率|达到70%|
|---|---:|---:|---:|---:|---|---:|---|
|context_half_prior_quarter|1822977|60.9229%|15.047024|15.046429|False|58.908333|否|
|context_half_prior_eighth_pointwise|1592461|65.8642%|16.202381|16.201786|False|55.435119|否|

压缩1pp验收不等于low urban≥70%及其他环境接近clean的长期目标已完成。“接近”未自设数值阈值。实际原raw clean准确率 84.676190%，各场景绝对差可由完整表直接核对。

### context_half_prior_quarter 的配对clean差距与源诊断

|场景|source准确率|source相对原R17（百分点）|目标全帧NMSE|主体232 NMSE|尾24 NMSE|配对clean特征余弦距离|目标raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|96.333333|-2.111111|0.516754627|0.562498808|0.10801509|0.0650082007|5271/30700/-25429|
|practical_high|97.777778|-0.055556|0.0359214954|0.0224064942|0.167391896|0.0227439888|68665/3649/+65016|
|practical_mid|95.666667|-1.111111|0.047442928|0.0345214568|0.172603682|0.0301267784|68555/4814/+63741|
|practical_low_suburban|94.277778|-1.666667|0.0585502163|0.0459949076|0.179602951|0.037452165|68091/5794/+62297|
|practical_high_urban|95.222222|-1.777778|0.0534599945|0.0405655503|0.1781753|0.0298941098|67743/4711/+63032|
|practical_mid_urban|82.444444|-9.000000|0.110028021|0.0987493992|0.216129631|0.0602603219|64588/9000/+55588|
|practical_low_urban|73.888889|-12.944444|0.151632473|0.141179353|0.247345001|0.0852066427|60453/12390/+48063|

完整20轮日志已分析，采用固定最终E20，未按最小训练loss或V/target择优。各项loss首轮→末轮：loss 6.4868585→1.4549925；iq_nmse 0.60440143→0.17558147；teacher_stage_nmse 0.58303109→0.1241528；task_KL 0.57688642→0.17837107；task_feature_distance 0.079661116→0.035093461。训练耗时2478.883秒，中位每轮123.966秒。

### context_half_prior_eighth_pointwise 的配对clean差距与源诊断

|场景|source准确率|source相对原R17（百分点）|目标全帧NMSE|主体232 NMSE|尾24 NMSE|配对clean特征余弦距离|目标raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|95.000000|-3.444444|0.556023419|0.599415183|0.168099642|0.0665235519|5321/32691/-27370|
|practical_high|97.388889|-0.444444|0.0406562649|0.0248499494|0.193943411|0.0254810378|69820/4870/+64950|
|practical_mid|95.000000|-1.777778|0.0540336967|0.0389492176|0.199298874|0.0331501514|68211/5509/+62702|
|practical_low_suburban|92.611111|-3.333333|0.0669698566|0.0524532162|0.205771536|0.0408157632|66733/6210/+60523|
|practical_high_urban|93.944444|-3.055556|0.0604543053|0.0452564657|0.206990287|0.0350277238|68126/6051/+62075|
|practical_mid_urban|78.277778|-13.166667|0.124005444|0.110202104|0.253115714|0.0724186599|61629/9984/+51645|
|practical_low_urban|68.111111|-18.722222|0.169579014|0.156472668|0.2891635|0.10056778|55746/13518/+42228|

完整20轮日志已分析，采用固定最终E20，未按最小训练loss或V/target择优。各项loss首轮→末轮：loss 6.6276223→1.9576189；iq_nmse 0.62364789→0.19338324；teacher_stage_nmse 0.591566→0.16850258；task_KL 0.62757054→0.22925683；task_feature_distance 0.08439171→0.043336307。训练耗时2460.338秒，中位每轮122.885秒。

成本测量只用固定16条source V、batch1/2warmup，包含完整coarse/action/三步CG；CPU加载后删除checkpointpayload，再放入GPU，无GPU optimizer/F/teacher。两版本测量顺序及其他owner仍影响时间，不视为部署加速证明。

|版本|checkpoint字节|原R17中位ms|精简版中位ms|延迟比|原R17resident字节|精简resident字节|原R17增量peak字节|精简增量peak字节|目标预测秒（含F及写盘）|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|context_half_prior_quarter|7677372|46.793|90.854|1.9416|35870720|24508416|1158656|1116272640|998.366|
|context_half_prior_eighth_pointwise|6752476|45.828|79.859|1.7426|35870720|23585280|1158656|1116272640|972.133|

每行新增20轮/4940更新/L126000/U1134000；原R17 teacher、原Fteacher、Fstudent各1260000输入前向，Fstudent对IQ输入反传，0F/head/context/teacher更新。0校准IQ；CPU均值SVD初始化时间单列。prior继承20+新20=40，coarse20/action20共80继承串行pass；原F50预训练50/11100和教师成本单列，不把E20当作整体训练成本。

两份完整恢复器已取回本地 models/<arm>/final.pt，大小与远端一致、来源/最终预算/共享alias严格加载通过，单条合成零IQ输出有限；0实际IQ/target访问。无optimizer/F/teacher权重，包含完整冻结coarse/action和prior，不提交权重到Git。加载使用experiments.pamf_r17_context_compact.runtime.load_checkpoint，输入原生尺度float32 [B,2,256]、I/Q顺序，输出同形状恢复IQ，下游仍接原F/defaulthead。
