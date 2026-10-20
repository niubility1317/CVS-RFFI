# R49：六点压缩、训练吞吐与正式推理成本

性能优先结论：六行均未同时实现逐场景识别提升与原70%/其他五景gap5pp目标。29.31万prune_balanced的lowurban为67.961310%，较指定父提高0.451190pp，但四个受损场景下降；其他五行的六个受损场景全部低于指定父。原每景1pp只作退化上限，不能把通过容差、减参或训练提速表述为性能优化成功。指定R46 prune_refine350521继续保留为唯一直接基准，不根据本轮正式结果另选父。

状态：VERIFIED / ANALYZED。六固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

用户在R46结果揭示后指定350521父，exact retain_refine为同预算训练优化控制，其余五行为GN4重要性全链/context/prior不同分配的剪枝点；同原接收链/3prior/6CG，无其他恢复网络。六行均pairedclean IQ+.05三尺度相对功率log1p误差+1→.1冻结父三stageIQ教师，所有coarse/action/prior联合。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。无TX标签模拟配对源域自监督，不称无clean盲学习。

方法commit `e96397bc3c7036f085317f9cc0e59074fb6ee24f`。完整登记、数据与祖先、命令和停止规则见[登记](../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49/experiment.json)；全部20轮与TX/RX/day见[结果](../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49/row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|retain_refine|350521|0.000000%|True|True|66.822024|
|prune_near|329885|5.887236%|True|False|66.757143|
|prune_balanced|293069|16.390459%|True|True|67.961310|
|prune_small|215049|38.648754%|False|False|65.825595|
|context_slim|268257|23.469065%|True|False|66.395238|
|prior_slim|310905|11.302033%|True|False|66.997024|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|retain_refine|clean|84.441071|84.143598|-0.015476|-0.064881|-0.145833|-395|
|retain_refine|practical_high|81.881548|81.366509|-0.570833|-0.536905|+0.562500|+69130|
|retain_refine|practical_mid|80.426190|79.952098|-0.795833|-0.583929|+0.848214|+68056|
|retain_refine|practical_low_suburban|78.936905|78.550153|-0.683333|-0.439286|+0.940476|+67044|
|retain_refine|practical_high_urban|80.761905|80.297723|-0.625000|-0.555357|+0.408333|+68987|
|retain_refine|practical_mid_urban|73.730952|73.611351|-0.716071|-0.495833|+0.045833|+65936|
|retain_refine|practical_low_urban|66.822024|67.133893|-0.688095|+0.049405|-0.394643|+61358|
|prune_near|clean|84.468452|84.175142|+0.011905|-0.037500|-0.118452|-349|
|prune_near|practical_high|81.656548|81.155792|-0.795833|-0.761905|+0.337500|+68752|
|prune_near|practical_mid|80.196429|79.721455|-1.025595|-0.813690|+0.618452|+67670|
|prune_near|practical_low_suburban|78.563095|78.145808|-1.057143|-0.813095|+0.566667|+66416|
|prune_near|practical_high_urban|80.547619|80.086420|-0.839286|-0.769643|+0.194048|+68627|
|prune_near|practical_mid_urban|73.610714|73.424180|-0.836310|-0.616071|-0.074405|+65734|
|prune_near|practical_low_urban|66.757143|66.912942|-0.752976|-0.015476|-0.459524|+61249|
|prune_balanced|clean|84.561310|84.268333|+0.104762|+0.055357|-0.025595|-193|
|prune_balanced|practical_high|81.931548|81.462168|-0.520833|-0.486905|+0.612500|+69214|
|prune_balanced|practical_mid|80.698810|80.244544|-0.523214|-0.311310|+1.120833|+68514|
|prune_balanced|practical_low_suburban|79.299405|78.912698|-0.320833|-0.076786|+1.302976|+67653|
|prune_balanced|practical_high_urban|80.905357|80.444455|-0.481548|-0.411905|+0.551786|+69228|
|prune_balanced|practical_mid_urban|74.575595|74.274679|+0.128571|+0.348810|+0.890476|+67355|
|prune_balanced|practical_low_urban|67.961310|67.902141|+0.451190|+1.188690|+0.744643|+63272|
|prune_small|clean|84.501190|84.218347|+0.044643|-0.004762|-0.085714|-294|
|prune_small|practical_high|80.453571|79.876800|-1.998810|-1.964881|-0.865476|+66731|
|prune_small|practical_mid|79.133929|78.569910|-2.088095|-1.876190|-0.444048|+65885|
|prune_small|practical_low_suburban|77.605952|77.102797|-2.014286|-1.770238|-0.390476|+64808|
|prune_small|practical_high_urban|79.333929|78.784482|-2.052976|-1.983333|-1.019643|+66588|
|prune_small|practical_mid_urban|72.551190|72.287390|-1.895833|-1.675595|-1.133929|+63954|
|prune_small|practical_low_urban|65.825595|65.971650|-1.684524|-0.947024|-1.391071|+59684|
|context_slim|clean|84.604762|84.327034|+0.148214|+0.098810|+0.017857|-120|
|context_slim|practical_high|81.380952|80.880316|-1.071429|-1.037500|+0.061905|+68289|
|context_slim|practical_mid|80.302381|79.854475|-0.919643|-0.707738|+0.724405|+67848|
|context_slim|practical_low_suburban|78.818452|78.472356|-0.801786|-0.557738|+0.822024|+66845|
|context_slim|practical_high_urban|80.234524|79.762961|-1.152381|-1.082738|-0.119048|+68101|
|context_slim|practical_mid_urban|73.271429|73.097960|-1.175595|-0.955357|-0.413690|+65164|
|context_slim|practical_low_urban|66.395238|66.611990|-1.114881|-0.377381|-0.821429|+60641|
|prior_slim|clean|84.482738|84.188302|+0.026190|-0.023214|-0.104167|-325|
|prior_slim|practical_high|81.797619|81.267986|-0.654762|-0.620833|+0.478571|+68989|
|prior_slim|practical_mid|80.287500|79.769180|-0.934524|-0.722619|+0.709524|+67823|
|prior_slim|practical_low_suburban|78.601190|78.134056|-1.019048|-0.775000|+0.604762|+66480|
|prior_slim|practical_high_urban|80.746429|80.229938|-0.640476|-0.570833|+0.392857|+68961|
|prior_slim|practical_mid_urban|73.859524|73.480361|-0.587500|-0.367262|+0.174405|+66152|
|prior_slim|practical_low_urban|66.997024|66.812475|-0.513095|+0.224405|-0.219643|+61652|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|retain_refine|136890/101037/112594|1686364|39081004|
|prune_near|127954/95237/106694|1605788|37989928|
|prune_balanced|112194/85533/95342|1452636|34540072|
|prune_small|85670/61085/68294|1146780|26390560|
|context_slim|89866/65797/112594|1358748|33935396|
|prior_slim|136890/101037/72978|1529372|32291884|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|retain_refine|G_only_B1|44.6075/44.5250|48.3775|22.2114|67194880/67194880|
|retain_refine|G_only_B32|80.9515/81.5294|89.1899|391.5788|276216832/276216832|
|retain_refine|G_plus_F_B1|54.8240/54.8702|59.8024|18.0848|67194880/67194880|
|retain_refine|G_plus_F_B32|90.6399/90.3532|97.6854|352.8159|276216832/276216832|
|prune_near|G_only_B1|45.9973/44.1574|61.3752|21.1898|67194880/1088000|
|prune_near|G_only_B32|83.4319/82.5391|90.1656|385.3717|276216832/276167680|
|prune_near|G_plus_F_B1|54.7709/53.8840|58.1520|18.4307|67194880/1201152|
|prune_near|G_plus_F_B32|91.6516/90.6443|102.4955|351.7908|276216832/276167680|
|prune_balanced|G_only_B1|42.3437/41.2085|42.4010|24.2367|67194880/1086976|
|prune_balanced|G_only_B32|56.5763/51.8985|64.8323|592.9033|276216832/227757056|
|prune_balanced|G_plus_F_B1|52.3321/51.3496|52.6641|19.4631|67194880/1201152|
|prune_balanced|G_plus_F_B32|65.5635/60.2804|61.8530|529.1422|276216832/227757056|
|prune_small|G_only_B1|42.4602/43.3987|45.0318|22.9727|67194880/79976960|
|prune_small|G_only_B32|56.7747/46.2169|47.3842|691.6884|276216832/181954560|
|prune_small|G_plus_F_B1|52.2932/53.3125|54.8236|18.7495|67194880/79976960|
|prune_small|G_plus_F_B32|66.1022/55.2849|56.8737|577.7237|276216832/181954560|
|context_slim|G_only_B1|42.8052/43.6513|44.8550|22.9062|67194880/79950336|
|context_slim|G_only_B32|57.5396/53.8231|64.6295|581.5011|276216832/276216832|
|context_slim|G_plus_F_B1|53.1800/53.8204|55.6401|18.5396|67194880/79950336|
|context_slim|G_plus_F_B32|66.9698/62.7960|65.0991|507.2773|276216832/276216832|
|prior_slim|G_only_B1|43.4693/43.8420|45.7176|22.7404|67194880/79978496|
|prior_slim|G_only_B32|57.3654/51.3828|52.5180|621.2118|276216832/275410944|
|prior_slim|G_plus_F_B1|53.4682/53.8428|55.2896|18.5603|67194880/79978496|
|prior_slim|G_plus_F_B32|66.5822/60.5604|61.9497|527.2047|276216832/275410944|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。

retain_refine首轮→末轮：loss 0.025517663→0.021753739；iq_nmse 0.024150934→0.021364156；spectral_error 0.0063670567→0.0056479812；teacher_stage_nmse 0.0010483766→0.0010718354；中位每轮76.518s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_near首轮→末轮：loss 0.031994163→0.021878085；iq_nmse 0.027539723→0.021470603；spectral_error 0.0070460502→0.0056680475；teacher_stage_nmse 0.0041021381→0.0012407895；中位每轮75.882s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_balanced首轮→末轮：loss 0.058816204→0.022414212；iq_nmse 0.041508431→0.021953657；spectral_error 0.010454877→0.005783176；teacher_stage_nmse 0.016785028→0.0017139645；中位每轮51.976s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prune_small首轮→末轮：loss 0.22493016→0.025032627；iq_nmse 0.12475796→0.024309154；spectral_error 0.023923676→0.0063684188；teacher_stage_nmse 0.098976011→0.0040505156；中位每轮49.592s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
context_slim首轮→末轮：loss 0.1375679→0.022461186；iq_nmse 0.078665779→0.021984676；spectral_error 0.016254056→0.00575757；teacher_stage_nmse 0.058089423→0.0018863109；中位每轮52.471s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
prior_slim首轮→末轮：loss 0.069341158→0.023597765；iq_nmse 0.048404996→0.022992251；spectral_error 0.013857961→0.006024355；teacher_stage_nmse 0.020243264→0.0030429678；中位每轮52.018s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。六行同预算、同损失配方、同全局batch256/micro256，采用direct float32 buffer，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_multipoint_fast_r49_20261020/models/<arm>/final.pt；使用experiments.pamf_r46_multipoint_fast.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。


## 固定方案和历史启动记录

以下为结果揭示前的方法与启动记录，当前训练和正式测试已完成。


状态：**VERIFIED / RUNNING**。

用户最新要求“发布多点压缩优化版本，提高显卡利用率，提升效率”。本轮直接从用户指定的 R46 `prune_refine` 350521 源最终权重出发，固定一个同宽吞吐优化控制和五个不同压缩点。原 R48 两行保持健康运行，不热改或重启。普通账户只读核实当前八张3090中GPU0/1为R48，GPU2至7无计算进程后，新增六行各占一张空闲卡，每卡一条本任务worker；新矩阵最多6条，与R48合计最多8条。本扩展来自用户最新多点并行要求，不影响其他owner。

|行|GPU|完整参数|context参数|prior唯一参数|主要差异|
|---|---:|---:|---:|---:|---|
|retain_refine|2|350521|237927|112594|精确复制指定父，训练吞吐优化对照|
|prune_near|3|329885|223191|106694|较保守的全链通道裁剪|
|prune_balanced|4|293069|197727|95342|中等全链裁剪|
|prune_small|5|215049|146755|68294|更小的全链裁剪|
|context_slim|6|268257|155663|112594|优先裁剪context，保留父prior宽度|
|prior_slim|7|310905|237927|72978|保留父context，只缩prior|

仍为原接收链/context/三步共享长权重prior/六次CG架构；按每个GN4组内权重重要性裁剪，正确映射encoder、decode、skip、GN及action全局头。五个裁剪点的宽度在新target访问前固定，和R48原273765点形成更多固定观察点；本轮全部行均正式测试，不依据源或target成绩重排、取消或选择性重跑。每个组件实际宽度由design.configuration唯一维护，不另拟合身份头。

源数据、教师及目标函数与R48相同：原L6300/U56700、20轮received缓存；配对clean IQ NMSE＋0.05三尺度相对功率log1p频谱误差＋三阶段冻结父IQ教师误差线性1→0.1。全部coarse/action/prior联合优化，恢复器训练0TX标签、F权重/输出、V或target。两角色都利用，90%曝光来自U；正确称为TX无标签的模拟配对源自监督，不称无clean盲学习。

固定每行20epochs/4940更新、全局batch256、L126000/U1134000曝光；重置AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/关闭TF32，末批24用真实分母。微批从R48的64改为256，每次全局更新只执行一次微批；以float32原始内存buffer传给Torch，绕过旧tolist和Torch2.1/NumPy2不可用的C桥，不安装或改环境。真实远端合成buffer逐值一致检查通过；梯度累积对照使用同样真实分母核对。数学目标和数据曝光不变，FP32归约顺序可能引入小数值差异，报告同宽控制完整结果及实际每轮时间，不预先承诺加速倍数。

直接父与冻结IQ教师唯一来源：`20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46/prune_refine/native_map_prior/final.pt`，方法commit `bbb8daf817de050b958d217b70d4c33f4aefe341`，固定源E20/4940、参数350521。继承全部上游原7/63/30物理源划分，load前核对实际launch/方案/源训练/严格checkpoint契约，复用READY/EXACT_MATCH数据。每组件继承80轮/本次20/累计100，组件轮数和300、串行训练总140；F50原训练另计。用户在旧七场景揭示后人工指定父，本轮属于暴露工程基准，不称新独立盲测或干净泛化证明。

每行最终固定E20，不源早停/跨行筛选；完成即自动进入源1800×7、同卡推理成本、独立源评分、正式168000×7及独立truth-last评分。正式场景为clean、郊区high/mid/low、urban high/mid/low；全七视图固定后独立CPU scorer才连接truth。原F50/defaulthead只冻结评测，FFT/PA/time/stats由原流程计算，不换头或另训头。原R17和350521父逐景1pp分别报告，lowurban70%、其他五景距rawclean84.6761904762%≤5pp及更强TX真实性目标保持，不用参数量替代性能目标。

同卡分别测G和G＋F的batch1、batch32。batch1每轮预热16、计时64、重复3轮；batch32预热4、计时16、重复3轮；候选顺序按轮交替。报告median/p90/p95、各轮中位数、吞吐量、驻留/峰值/增量显存、完整参数/文件及Conv＋Linear MAC。64条sourceV重复计时为相关观测，预加载GPU时间不含传输/预处理/loader；三G一F共享suite驻留不称独立部署显存，MAC排除FFT/复数CG/norm/activation/memory/F。新增六卡并发及其他负载可能影响计时，报告局限，不把训练吞吐增加混为单样本推理加速。

每GPU至多一个本run worker，按free≥18GB/每卡总计算进程<2获取，GPU2至7新root独占。CPU2线程、共用R41 score.lock，R48/R49同一任务最多一个scorer；不会启动额外GPU tester。一次归档SHA/远端compile、launcher首步真实父/F50/两条Ls no-query smoke PASS后立即执行。技术失败保留受影响产物、不自动重试，健康行和其他owner不动。单run一个launch owner，原R48自动测试不重复。

最小聚焦检查覆盖来源/预算/权限拒绝、别名冲突、全行自动正式测试、truth晚于全预测、全部组件梯度/精确父控制、新buffer逐值/只读/非连续数组和不同微批梯度等价；一次独立P0/P1审查。精确矩阵、commit、命令、环境/CWD、路径、停止规则和预期产物写入同run登记，不增加其他gate。


VERIFIED/RUNNING：实际350521父/F50/两Ls配对IQ noquery smoke PASS，六行全部组件有限非零梯度、teacher/F不更新、checkpoint严格重载一致。归档单SHA/compile及controller/六worker实际PID/CWD/argv/空controllerCUDA/各预登记物理GPU2至7和GPU-PID匹配、首轮曝光、micro256/directbuffer和日志增长已核实。GPU2/3发布前已有各一条其他owner轻负载，按free≥18GB/总进程<2共存；GPU4至7预检空闲；无旧进程改动。无本轮正式性能结果；每行E20后自动七正式测试和独立truth-last。


VERIFIED启动效率观察：六行真实E1/247/L6300+U56700、micro256/directbuffer、0F训练输入；retain_refine E1 78.52s/802.33sample/s；prune_near E1 80.04s/787.13sample/s；prune_balanced E1 53.93s/1168.26sample/s；prune_small E1 50.76s/1241.11sample/s；context_slim E1 53.97s/1167.36sample/s；prior_slim E1 54.58s/1154.17sample/s。GPU4至7五次5s间隔采样平均利用率58.6%至67.4%；GPU2/3为94.8%/89.4%且含其他owner负载。同宽控制E1相对R48的耗时比为2.0600，属于不同物理卡/负载/首轮条件的观察，不是隔离方法加速倍数，不能替代正式识别率或完整推理成本。

## 已完成源阶段的推理成本观察

以下四行已完整E20/4940、源1800×7独立评分及同卡G/G+F/B1/B32三轮测量，正式测试正在运行；这是源训练和成本证据，尚无正式识别结论。全部20轮损失/曝光/LR和profile范围由独立汇总工具核对。

|行|参数|每轮中位s|父/本版G B1 median ms|父/本版G+F B1 median ms|父/本版G+F B32 sample/s|
|---|---:|---:|---|---|---|
|prune_balanced|293069|51.976|42.3437/41.2085|52.3321/51.3496|486.907/529.142|
|prune_small|215049|49.592|42.4602/43.3987|52.2932/53.3125|484.142/577.724|
|context_slim|268257|52.471|42.8052/43.6513|53.1800/53.8204|475.661/507.277|
|prior_slim|310905|52.018|43.4693/43.8420|53.4682/53.8428|480.865/527.205|

完整p90/p95、各轮中位数、显存和Conv/Linear MAC见[source阶段与成本汇总](../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49/source_cost_progress.json)。减参未稳定带来batch1加速，下降和变慢全部保留；训练吞吐提高不能混为推理加速。模型仍执行三prior/六CG，不能仅据这些时间断定具体瓶颈。

成本在64条sourceV、预加载GPU的FP32 eval上同步测量，不含传输/预处理/loader；三G一F共享suite驻留不是独立部署显存，增量峰值对应当前前向。GPU负载/温度与相关重复观测限制保留，尚无独立性能置信区间。原70%/5pp/输入TX细节要求继续ACTIVE。

## 性能优先：首批四行完整正式测试

**用户最新要求性能优先。** 识别性能与输入TX保真是主要目标，参数、时延和显存为次要比较；原逐景1pp只是退化上限，不代表优化成功。固定的R48/R49八行保持原方案全部测试，不热改、不按已揭示分数重排或选择性重跑；后续直接父仍为用户指定R46 350521。

以下为按完成时间自然得到的R49四行，均同168000物理记录×全部七视图、原F50/defaulthead；每行全预测固定后独立truth-last评分。R49另两行和R48两行仍在固定流程中，这不是完整八行结论。

|场景|指定父350521|prune_balanced（293069）|prune_small（215049）|context_slim（268257）|prior_slim（310905）|
|---|---:|---:|---:|---:|---:|
|clean|84.456548|84.561310|84.501190|84.604762|84.482738|
|郊区high|82.452381|81.931548|80.453571|81.380952|81.797619|
|郊区mid|81.222024|80.698810|79.133929|80.302381|80.287500|
|郊区low|79.620238|79.299405|77.605952|78.818452|78.601190|
|urban high|81.386905|80.905357|79.333929|80.234524|80.746429|
|urban mid|74.447024|74.575595|72.551190|73.271429|73.859524|
|urban low|67.510119|67.961310|65.825595|66.395238|66.997024|

上述准确率单位为%。相对指定父的完整逐景差值、Macro-F1、TX/RX/day和repair/damage见[首批正式结果](../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49/partial_formal_results.json)。

|行|原R17逐景1pp上限|对指定父逐景1pp参考|lowurban≥70%|其他五景gap≤5pp|
|---|---|---|---|---|
|prune_balanced|True|True|False|False|
|prune_small|False|False|False|False|
|context_slim|True|False|False|False|
|prior_slim|True|False|False|False|

215049版未通过原R17逐景1pp上限，不能作为性能合格压缩版；268257/310905虽通过原R17上限，但对指定父有多景退化，性能优先时不能仅凭减参列为优选。293069的lowurban为67.961310%，较父+0.451190pp，但四个其他受损景下降0.320833至0.523214pp，属于取舍，不是全面提升。四行均未达到原70%/其他五景gap5pp目标。

单模型seed及已暴露工程基准限制保留；当前观察不回流训练或参数。每个恢复细节的TX真实性仍未被证明，固定无F事后关联诊断R50会在全部八行正式评分之后执行，且不阻塞其交付。所有负结果保留。
