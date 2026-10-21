# R28：冻结 prior 低秩卷积合并

状态：VERIFIED / ANALYZED。

用户要求压缩同时提升性能。本轮固定 BOTH R22 source-final E20/19700（complex625、complex625_identity，方法 e190cf5367b03da8aafced51276a8730ad8274fa），仅作冻结推理的代数变换，无新训练/选模/教师。它们原始训练仍来自用户指定原 R46 prune_refine350521 E20/4940、方法 bbb8daf817de050b958d217b70d4c33f4aefe341，源7/63/30物理角色及完整祖先契约保持。未来训练唯一初始化/教师仍为原350521，本次不把新学生变成训练父版。历史目标已暴露，属于工程执行比较。

|固定版本|源完整参数|合并后完整部署参数|coarse+action / prior|GPU|
|---|---:|---:|---|---:|
|R22_complex625_priorcollapsed|278180|299297|186703 / 112594|2|
|R22_complex625_identity_priorcollapsed|278180|299297|186703 / 112594|3|

只将三阶段 prior 内的 spatial Conv→1x1 Conv 线性对合并，核为 right[:,:,0] @ left.flatten(1)，保留左层stride/padding/dilation与右层bias。28组共享权重合并后仍按真实对象别名共享；阶段bias、GN4、skip、coarse/action、3stage/6CG保持。共84个prior FactorConv替换为单Conv，数据自由完整前向实际减少84次Conv1d调用。参数从278180增到299297，仍较原350521少约14.61%；ConvLinear MAC增加，不能把调用减少写作FLOPs减少或已测加速。完整部署权重/参数和存储均按合并后模型统计，不仅报告源压缩文件。

线性代数相同不等于FP32逐位相同。CPU控制器首步读取原2条L_s的clean/received01，与原F50/defaulthead冻结执行原源模型、合并模型、严格重载模型，检查全部输出有限/形状/正方差、IQ与方差保持原rtol2e-5/atol2e-6接近度、严格重载全部输出逐位一致、原源与F状态不变；完整记录F特征/logits接近度和argmax差异但不据源等价或识别收益筛掉版本。每版先导出独立299297权重，0优化器更新/0TX训练标签/0F训练知识，无sourceV或target用于参数更新。

两版均按冻结矩阵自动完成源1800×七景、同卡完整成本、独立源评分，再正式168000×clean/三郊区/三urban。批量128与原R22预测相同；每行全部新预测固定后，先只读复用旧R22的IQ/features/logits和argmax作差异比较，再独立CPU truth-last。旧native目标不重跑，不假定argmax保持，不从结果调参/选择性重测。逐TX/RX/day、confusion、repair/damage、clean损伤、波形/F几何保留；原父逐景0.8pp/1344条、原R17历史1pp与lowurban70%/其他五景距rawclean5pp/clean保持和强TX真实性分开报告。

同卡64sourceV同时测原R17、原350521父、原R22低秩278180与本次299297，覆盖G与G+原F、B1/B32、三轮正反顺序，记录median/p90/p95/吞吐及每轮值、resident/peak/incremental、完整ConvLinearMAC及权重字节。计时含全部物理解算，排除传输/loader/前处理。四G一F是共享suite驻留，不冒充独立部署占用；MAC不含FFT/CG/norm/访存/F。CPU转换开销和实际GPU成本分别记录，无CUDA Graph或native fallback。

N607普通账户，GPU2/3最多2own/每卡1，严格排在R26同卡所属worker终态及实际退出之后。前驱workers为空时等待；再以共享容量锁free≥18GB/allowner<2分配，不停止/修改任何健康owner。CPU评分2线程、R06score.lock、最多1ownscorer。一个方法commit、新不可覆盖run/release/output、一次preflight、一次归档两端SHA、一次compile，发布后以实际PID/CWD/完整命令/log增长核实。技术失败保留影响行，无自动重启/热改/回退，健康行继续。

25项聚焦检查覆盖卷积合并数学、不同stride/pad/dilation/bias、共享核和阶段偏置、全模型参数与调用计数、保存/严格重载/别名冲突、F/来源/训练权限负例、两版七景链、truth-last与原输出保护；唯一审查由/root/review_r14_prior_compact独立复跑25项PASS，未访问真实IQ/F/TX/远端、未编辑文件。主对话缓存归一化斜率，本组改变prior线性执行，独立固定方案。当前无新增正式结果，完整目标ACTIVE。

R27原run在CPU首步F近零分量等价断言失败，已核实实际退出且0GPU/0target；诊断IQ原阈值均通过，源/F状态不变。新R28只修复F诊断边界，不改变变换/数据/源模型/矩阵/GPU/预算，不覆盖旧失败root；详见R27原报告。


## 实际发布、排队与独立交付

方法commit 7ca5af4f6f5d9f862e07e13b9da15902379044e4已推送并独立匹配远端OID。一次N607预检核实原350521父及两份R22 source-final20/19700、原数据READY/EXACT_MATCH与七景测试句柄；一次归档两端SHA及compile通过。实际release/CWD为r28，未复用旧R27失败root。

真实2Ls clean/received01检查PASS：两版合并后完整299297参数，28组共享核、84个prior factor模块已转换；IQ/方差原阈值通过，全部输出有限、正方差、严格重载四输出逐位一致，源模型与F状态不变。完整保留F features/logits数值差异，当前两条源样本argmax均一致；这不是正式分类保持结论。实际CPU控制器PID/CWD/完整命令/commit/日志增长VERIFIED，两版都等待R26同卡终态及实际退出，0新GPUworker，不改动其他健康任务。

结果工具新增22项检查PASS，覆盖两版全七景、原预测一致性与混淆矩阵的相互核对、源/F权限边界、真实进程退出、完整299297部署参数、四模型成本及缺失/错误证据拒绝、公开逐TX/RX/day CSV和报告输出。仅以聚合artifact独立复算，不读新的IQ/truth或调参。

单一本地完成owner只读等待实际controller/worker退出，之后收集完整source7/formal7/原预测差异/独立truth-last，取回两份合并后完整权重并CPU严格加载，检查组件参数/MAC/有限输出，再报告、提交推送并独立核对远端OID。没有重试、热改、目标重测或复制完成owner；原R23/R24/R26完成owner保持。完整70%/5pp/clean/TX真实性目标仍ACTIVE。


## 完整正式结果

两版均完成源1800×7和正式168000×7，所有新预测固定后比较旧R22原预测，再独立truth-last评分。实际controller/两worker退出，独立299297参数部署权重已取回并CPU严格加载。新增训练0，所有负结果保留。

|版本|部署参数|较350521减参|父0.8pp|原R17历史1pp|原native分类保持|70%/5pp/clean保持|
|---|---:|---:|---|---|---|---|
|R22_complex625_priorcollapsed|299297|14.613675%|True|True|False|False/False/False|
|R22_complex625_identity_priorcollapsed|299297|14.613675%|True|True|False|False/False/False|

|版本|场景|准确率%|MacroF1%|较350521pp|较自身R22 native pp|修复/损坏/净纠错|
|---|---|---:|---:|---:|---:|---|
|R22_complex625_priorcollapsed|clean|84.542857|84.260260|+0.086310|+0.000000|248/472/-224|
|R22_complex625_priorcollapsed|practical_high|82.053571|81.561357|-0.398810|+0.000000|73295/3876/+69419|
|R22_complex625_priorcollapsed|practical_mid|80.845833|80.380230|-0.376190|+0.000000|73143/4382/+68761|
|R22_complex625_priorcollapsed|practical_low_suburban|79.344643|78.917260|-0.275595|+0.000000|72801/5072/+67729|
|R22_complex625_priorcollapsed|practical_high_urban|81.020833|80.566723|-0.366071|+0.000595|73793/4371/+69422|
|R22_complex625_priorcollapsed|practical_mid_urban|74.251786|73.946401|-0.195238|+0.000000|74221/7410/+66811|
|R22_complex625_priorcollapsed|practical_low_urban|67.465476|67.312300|-0.044643|+0.000000|72510/10071/+62439|
|R22_complex625_identity_priorcollapsed|clean|84.605952|84.331257|+0.149405|+0.000000|211/329/-118|
|R22_complex625_identity_priorcollapsed|practical_high|81.854167|81.337937|-0.598214|+0.000000|72735/3651/+69084|
|R22_complex625_identity_priorcollapsed|practical_mid|80.692262|80.189302|-0.529762|+0.000000|72637/4134/+68503|
|R22_complex625_identity_priorcollapsed|practical_low_suburban|79.269048|78.807012|-0.351190|+0.000000|72442/4840/+67602|
|R22_complex625_identity_priorcollapsed|practical_high_urban|80.839881|80.338701|-0.547024|-0.000595|73192/4074/+69118|
|R22_complex625_identity_priorcollapsed|practical_mid_urban|74.536310|74.180915|+0.089286|+0.000000|74215/6926/+67289|
|R22_complex625_identity_priorcollapsed|practical_low_urban|68.055357|67.846422|+0.545238|-0.000595|72990/9560/+63430|

容忍上限不代表性能提升；低urban70%、其他五景距rawclean≤5pp、clean保持和更强TX细节真实性分别判断。每版原R22源权重固定，目标结果未回流转换、选模或重跑。

|版本|场景|argmax差异条数|IQ最大绝对差|F特征最大绝对差|logits最大绝对差|
|---|---|---:|---:|---:|---:|
|R22_complex625_priorcollapsed|clean|0|4.15742397e-06|0.000144481659|8.32080841e-05|
|R22_complex625_priorcollapsed|practical_high|0|9.50694084e-06|0.000541120768|0.000227808952|
|R22_complex625_priorcollapsed|practical_mid|0|8.80658627e-06|0.000190258026|0.000328540802|
|R22_complex625_priorcollapsed|practical_low_suburban|0|1.22487545e-05|0.00126302242|0.000564932823|
|R22_complex625_priorcollapsed|practical_high_urban|1|1.09672546e-05|0.000358736143|0.00022983551|
|R22_complex625_priorcollapsed|practical_mid_urban|0|1.18017197e-05|0.000855326653|0.000871658325|
|R22_complex625_priorcollapsed|practical_low_urban|1|1.52587891e-05|0.000518083572|0.000463724136|
|R22_complex625_identity_priorcollapsed|clean|0|2.2649765e-06|0.000441789627|0.00013923645|
|R22_complex625_identity_priorcollapsed|practical_high|0|9.7155571e-06|0.000366151333|0.00029706955|
|R22_complex625_identity_priorcollapsed|practical_mid|0|1.20401382e-05|0.496560276|2.01859951|
|R22_complex625_identity_priorcollapsed|practical_low_suburban|0|1.59740448e-05|0.114418805|0.311322331|
|R22_complex625_identity_priorcollapsed|practical_high_urban|1|1.01625919e-05|0.000386357307|0.00059530139|
|R22_complex625_identity_priorcollapsed|practical_mid_urban|0|1.69873238e-05|0.000711604953|0.000676691532|
|R22_complex625_identity_priorcollapsed|practical_low_urban|1|1.67489052e-05|0.000773429871|0.0013435483|

|版本|口径|原350521/原低秩/合并版中位ms|合并版p95ms|原低秩/合并版吞吐条/s|较原低秩延迟变化|
|---|---|---|---:|---|---:|
|R22_complex625_priorcollapsed|G_only_B1|40.681103/48.904638/44.895561|46.812998|20.421/22.201|-8.1977%|
|R22_complex625_priorcollapsed|G_only_B32|56.776680/54.346703/56.551154|59.933159|586.297/561.966|+4.0563%|
|R22_complex625_priorcollapsed|G_plus_F_B1|52.117525/60.398090/56.260586|58.195052|16.560/17.760|-6.8504%|
|R22_complex625_priorcollapsed|G_plus_F_B32|66.224041/63.565267/65.761764|69.313803|501.705/484.415|+3.4555%|
|R22_complex625_identity_priorcollapsed|G_only_B1|49.874842/59.078299/54.715077|57.135419|16.806/18.204|-7.3855%|
|R22_complex625_identity_priorcollapsed|G_only_B32|61.696974/61.400331/63.483314|66.440632|518.621/501.930|+3.3925%|
|R22_complex625_identity_priorcollapsed|G_plus_F_B1|64.268089/74.071619/69.306518|72.043302|13.515/14.392|-6.4331%|
|R22_complex625_identity_priorcollapsed|G_plus_F_B32|74.383430/74.886134/76.378339|79.784889|427.143/416.434|+1.9926%|

计时含完整coarse/action/3prior/6CG，G+F含原默认头；同卡64sourceV、B1/B32三轮交替，预加载GPU输入，不含传输/预处理/loader。四G一F驻留是共享suite，不称单模型部署显存；完整每轮median/p90/p95/吞吐/resident/peak/incremental保存在results.json。

|版本|coarse/action/prior参数|完整权重字节|原低秩/合并版ConvLinearMAC|CPU转换秒|
|---|---|---:|---|---:|
|R22_complex625_priorcollapsed|106658/80045/112594|1546228|29727404/34949420|0.217154|
|R22_complex625_identity_priorcollapsed|106658/80045/112594|1546228|29727404/34949420|0.309270|

84次卷积调用减少、28组共享核保留，部署参数从278180增加至299297；MAC增加，不能用源压缩文件替代部署权重大小。MAC不含FFT/CG/norm/激活/访存/F。新训练轮次/更新/训练IQ均0；原R22继承E20/19700及其祖先成本不消失。

|版本|场景|IQ NMSE|prefix232 NMSE|tail24 NMSE|FFTlogpower L1|
|---|---|---:|---:|---:|---:|
|R22_complex625_priorcollapsed|clean|0.000120223798|0.000124569531|7.31806504e-05|0.0139766876|
|R22_complex625_priorcollapsed|practical_high|0.0156991854|0.00686451048|0.100724317|0.251369983|
|R22_complex625_priorcollapsed|practical_mid|0.020635169|0.0118868575|0.104696311|0.28261748|
|R22_complex625_priorcollapsed|practical_low_suburban|0.0261056982|0.0174254682|0.109384172|0.311600268|
|R22_complex625_priorcollapsed|practical_high_urban|0.0209404714|0.0122333169|0.10457243|0.268439174|
|R22_complex625_priorcollapsed|practical_mid_urban|0.0454998501|0.0373549089|0.122789368|0.355371237|
|R22_complex625_priorcollapsed|practical_low_urban|0.0685784146|0.060784325|0.141185805|0.422803193|
|R22_complex625_identity_priorcollapsed|clean|4.09193963e-05|4.21238256e-05|2.98536343e-05|0.0142119722|
|R22_complex625_identity_priorcollapsed|practical_high|0.0158022363|0.00685482519|0.101531096|0.252665609|
|R22_complex625_identity_priorcollapsed|practical_mid|0.0207823999|0.0119372401|0.105463065|0.283867598|
|R22_complex625_identity_priorcollapsed|practical_low_suburban|0.0262748823|0.0175072867|0.110142551|0.31276679|
|R22_complex625_identity_priorcollapsed|practical_high_urban|0.0210536215|0.0122395502|0.105404936|0.269757003|
|R22_complex625_identity_priorcollapsed|practical_mid_urban|0.0457096435|0.0374773219|0.123765402|0.356823921|
|R22_complex625_identity_priorcollapsed|practical_low_urban|0.0688387603|0.0609728321|0.142138958|0.424488008|

[完整结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-prior-collapse-s2026092701-m2-r28/results.json)与[全部逐TX/RX/day CSV](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-prior-collapse-s2026092701-m2-r28/row_metrics.csv)保留混淆矩阵、净纠错、波形和F几何。浮点差异与分类差异分别报告；已暴露单seed工程基准，不称新盲测或TX逐细节真实性证明。

独立完整权重：local_artifacts/pamf_restoration/r46_prior_collapse_20261021_r28/models/<arm>/final.pt；使用experiments.pamf_r46_prior_collapse.runtime.load_checkpoint加载，FP32[B,2,256]输入/同形输出，部署无需F/teacher/clean；权重、IQ和原始日志不入Git。
