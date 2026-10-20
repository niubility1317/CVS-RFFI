# R54：同一压缩拓扑的源通道筛选正式结果

VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。activation_energy 273765参数，七景较父提高1/下降6，lowurban 66.435714%（较父-1.074405pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版41.6582/40.9480ms，G+F B1父/本版53.0625/52.8909ms；projection_sensitivity 273765参数，七景较父提高1/下降6，lowurban 66.919048%（较父-0.591071pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.7906/44.9914ms，G+F B1父/本版58.4112/57.2981ms。activation_energy相对同宽权重筛选R53/prune_decay七景提高1/下降6，lowurban差-2.091667pp；projection_sensitivity相对同宽权重筛选R53/prune_decay七景提高1/下降6，lowurban差-1.608333pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。

状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

唯一直接初始化与冻结IQ教师仍为用户指定R46 prune_refine350521。两行均沿用完整273765拓扑，区别是source激活能量与随机输出投影gain敏感度选择通道；原3prior/6CG、GN4组配额与共同prior map、全部coarse/action/prior联合、pairedclean IQ+.05谱误差+1→.1IQ教师、E20/global256micro256均保持。固定同宽weight-only控制复用R53 prune_decay，不重复训练。校准共用一次128源样本/256冻结G输入、0optimizer更新，另计计算与曝光；不称data-free初始化。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。属于有clean模拟配对的TX无标签源域自监督。

方法commit `554c52baf75ab8da2618a5480851b74271018103`。完整登记、数据与祖先、命令和停止规则见[登记](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-source-prune-s2026092701-m2-r54/experiment.json)；全部20轮与TX/RX/day见[结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-source-prune-s2026092701-m2-r54/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-source-prune-s2026092701-m2-r54/row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|activation_energy|273765|21.897689%|True|False|66.435714|
|projection_sensitivity|273765|21.897689%|True|True|66.919048|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|activation_energy|clean|84.455357|84.153089|-0.001190|-0.050595|-0.131548|-371|
|activation_energy|practical_high|82.469048|82.061644|+0.016667|+0.050595|+1.150000|+70117|
|activation_energy|practical_mid|80.848810|80.435107|-0.373214|-0.161310|+1.270833|+68766|
|activation_energy|practical_low_suburban|78.913690|78.541988|-0.706548|-0.462500|+0.917262|+67005|
|activation_energy|practical_high_urban|81.212500|80.826914|-0.174405|-0.104762|+0.858929|+69744|
|activation_energy|practical_mid_urban|73.793452|73.635062|-0.653571|-0.433333|+0.108333|+66041|
|activation_energy|practical_low_urban|66.435714|66.645823|-1.074405|-0.336905|-0.780952|+60709|
|projection_sensitivity|clean|84.617262|84.331613|+0.160714|+0.111310|+0.030357|-99|
|projection_sensitivity|practical_high|81.789286|81.336497|-0.663095|-0.629167|+0.470238|+68975|
|projection_sensitivity|practical_mid|80.459524|80.061037|-0.762500|-0.550595|+0.881548|+68112|
|projection_sensitivity|practical_low_suburban|78.921429|78.659155|-0.698810|-0.454762|+0.925000|+67018|
|projection_sensitivity|practical_high_urban|80.740476|80.327364|-0.646429|-0.576786|+0.386905|+68951|
|projection_sensitivity|practical_mid_urban|73.769643|73.786442|-0.677381|-0.457143|+0.084524|+66001|
|projection_sensitivity|practical_low_urban|66.919048|67.437611|-0.591071|+0.146429|-0.297619|+61521|

同宽度通道筛选对照：以下只比较预登记R53 prune_decay权重筛选控制，保留全部七景差值；额外共用源校准256次冻结G输入另计，不以正式结果重新选择父、宽度或筛选方式。

|源通道筛选行|权重筛选控制|场景|准确率变化（pp）|MacroF1变化（pp）|
|---|---|---|---:|---:|
|activation_energy|R53/prune_decay|clean|-0.073214|-0.090822|
|activation_energy|R53/prune_decay|practical_high|+0.060119|+0.090149|
|activation_energy|R53/prune_decay|practical_mid|-0.500000|-0.493910|
|activation_energy|R53/prune_decay|practical_low_suburban|-0.944643|-0.952016|
|activation_energy|R53/prune_decay|practical_high_urban|-0.199405|-0.164657|
|activation_energy|R53/prune_decay|practical_mid_urban|-1.267857|-1.196345|
|activation_energy|R53/prune_decay|practical_low_urban|-2.091667|-1.926580|
|projection_sensitivity|R53/prune_decay|clean|+0.088690|+0.087702|
|projection_sensitivity|R53/prune_decay|practical_high|-0.619643|-0.634999|
|projection_sensitivity|R53/prune_decay|practical_mid|-0.889286|-0.867981|
|projection_sensitivity|R53/prune_decay|practical_low_suburban|-0.936905|-0.834849|
|projection_sensitivity|R53/prune_decay|practical_high_urban|-0.671429|-0.664207|
|projection_sensitivity|R53/prune_decay|practical_mid_urban|-1.291667|-1.044965|
|projection_sensitivity|R53/prune_decay|practical_low_urban|-1.608333|-1.134793|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|activation_energy|106138/79477/88150|1388124|30761512|
|projection_sensitivity|106138/79477/88150|1388124|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|activation_energy|G_only_B1|41.6582/40.9480|42.6066|24.1912|67194880/1084928|
|activation_energy|G_only_B32|56.7120/50.9963|52.4088|625.4821|276216832/227691520|
|activation_energy|G_plus_F_B1|53.0625/52.8909|60.4331|18.5866|67194880/1201152|
|activation_energy|G_plus_F_B32|66.2411/60.4560|64.2429|520.8585|276216832/227691520|
|projection_sensitivity|G_only_B1|45.7906/44.9914|49.7962|22.0389|67194880/1084928|
|projection_sensitivity|G_only_B32|81.1216/71.6087|80.6873|442.4472|276216832/227691520|
|projection_sensitivity|G_plus_F_B1|58.4112/57.2981|60.8298|17.3375|67194880/1201152|
|projection_sensitivity|G_plus_F_B32|91.1957/81.1561|86.0988|395.5609|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。


共用源校准：128条物理样本（64L_s/64U_s）、256次冻结G输入、耗时9.5506s；source received01/20、两次Rademacher投影；0TX/F/V/target/optimizer更新、父状态不变。具体分数/map/行位置保持私有，不是运行时输入。

activation_energy首轮→末轮：loss 0.14409604→0.023421186；iq_nmse 0.084582491→0.022831636；spectral_error 0.020308574→0.0059982311；teacher_stage_nmse 0.058498123→0.0028963785；中位每轮49.705s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
projection_sensitivity首轮→末轮：loss 0.1344389→0.023329311；iq_nmse 0.079862335→0.022750676；spectral_error 0.018631943→0.0059851801；teacher_stage_nmse 0.053644969→0.0027937648；中位每轮52.969s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。两行同E20训练预算、同batch256/micro256/directbuffer、同1→.1教师；额外共用256次冻结G源校准、0optimizer更新，不重复计算为两份或隐藏为0数据初始化，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_source_prune_r54_20261021/models/<arm>/final.pt；使用experiments.pamf_r46_source_prune.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。


## 固定方案与历史发布记录

# R54：同一压缩拓扑的源响应与输出敏感度通道筛选

状态：**VERIFIED / RUNNING**，activation_energy 已完成正式评分，projection_sensitivity 在 GPU2 继续训练。性能优先；没有新正式结果，不把通道评分、源IQ误差、容差或减参当作识别优化完成。

固定两行 activation_energy、projection_sensitivity，均完整273765参数、context185615；沿用R48最初完整两宽度矩阵的273765拓扑，现有350521宽度保留全通道时筛选恒等，原R49同宽控制保留，不为恒等初始化重复训练。仅用源响应替换原权重能量通道排序，不更换网络族、宽度分配、3prior/6CG或推理分支。唯一直接初始化/冻结IQ教师为用户指定R46 prune_refine350521 E20/4940，方法bbb8daf817de050b958d217b70d4c33f4aefe341；不加载R48/R49/R53学生，不根据其正式分数重排或另选父。固定同宽权重剪枝控制复用R53 prune_decay（方法df36c3d3da4e026b747776b65b58d4921dc1a71e），不重复训练。方案参数在读取R53正式分数之前固定；R53健康训练/预测保持原计划。

源校准在CPU2共同执行一次：seed2026102154从原6300L_s/56700U_s各抽64条，使用原R07 cachedreceived01/20，共128物理样本、256次冻结G输入；0TX标签/F/V/target/optimizer更新。clean仅用于每例输出能量归一化，均方下限1e-4。activation_energy按encoder输出均方评分；projection_sensitivity使用独立Rademacher输出投影（seed2026102155、draw2、除sqrt512和sqrt(clean能量)），求每通道整体gain的方向导数，再平均其平方。通道gain导数为sum_t activation乘output-projection对activation的梯度，不把平方的逐时间均值误写成整体通道gain敏感度。该局部近似不保证识别或GroupNorm裁剪后的全局误差单调。全部模型参数冻结、无参数梯度或状态更新，父逐张量核对不变。

每个GN4组按固定配额、stable排序选通道，保留组内原索引顺序。prior0/1/2先汇总平均评分，再使用完全相同的通道map，避免共享长卷积索引错位；各stage独立bias/GroupNorm保留。coarse spectral、temporal与action分别评分，global head同步切输入列，decoder skip使用原上游宽度偏移。校准分数/map和行位置留私有run，推理只加载独立最终state，不再读取源样本/校准文件或F。

两行继续原pairedclean IQNMSE＋0.05三尺度相对功率STFT＋线性1→0.1冻结父三stage IQNMSE，所有coarse/action/prior联合。原model2026092701、split392005、data2026102001、augmentation2026101912、formal2027，原L/U完整source、R07 epoch01..20、global256micro256/directfloat32buffer、ResetAdamW、LR2e-4暖247/cos至2e-5、WD1e-4/clip1、FP32/TF32关闭。每行新增E20/4940、L126000/U1134000、1260000学生和IQ教师输入；继承每组件80轮后累计100、组件和300、串行140pass，F50成本另计。另单列共用校准256输入（L128/U128次视图曝光），不伪装为0数据初始化、不计成梯度训练epoch或重复计入两行训练。模型/校准均0TX/F/V/target训练。

原source READY/EXACT_MATCH、original7/63/30物理ID及R46/R44/R38/R17/R07/R04来源复用，parent/source契约在实际初始化核对。属于有clean模拟配对的TX无标签源域自监督，不能称无clean盲学习。原F50联合训练默认头仅在训练后评测；无另训头。历史工程基准已暴露，不声称新的独立盲测或逐细节TX真实性。

资源仅物理GPU2/3，R53＋R54本任务合计最多两条GPU worker、每卡一条。activation_energy在GPU3等待R53 retain_anchor完整正式完成且实际所属PID退出；projection_sensitivity在GPU2等待排队中的R53 prune_anchor完整完成且该卡所有R53所属PID退出，不能只等第一条prune_decay。等待进程使用CPU、不预占GPU；不读取前驱性能分数、不停止或热改任何健康进程。普通账户、CPU2、同一共享score.lock最多一个scorer；新不可覆盖release/run/output，唯一owner codex/root/R46-source-prune。

37项聚焦检查与CLIhelp通过；一次限P0/P1审查以实际review.json为准。首步真实checkpoint源校准及R46/F50/2Ls noquery smoke，PASS直接按卡排队训练；一次N607 preflight、单归档两端SHA、compile和实际PID/CWD/argv/GPU/日志核实。技术失败保留受影响输出，无自动重试/热改/回退，低准确率不触发停止。额外设计审批/签名/成员哈希等不属于最小流程，REJECTED_EXTRA_GATE，不等待。

每行固定最终E20/4940，自动source1800×7、同卡G与G+F/B1/B32三轮成本、正式168000×7预测；每行七景预测固定后独立truth-last评分并逐行报告，源域无提升也不取消测试。主要指标为逐场景accuracy/Macro-F1、相对同宽weight-only控制和R17/R46、逐TX/RX/day及repair/damage；时延/吞吐/显存/参数次级。成本为64源V、同步预加载FP32、不含传输/前处理/loader，3G＋1F共享驻留不是单模型显存；ConvLinearMAC不含FFT/CG/norm/激活/F。原R17每景1pp只是退化容差；lowurban≥70%、其他五景距rawclean84.6761904762%≤5pp与更强输入TX细节真实性目标继续ACTIVE。精确commit、命令、环境/CWD、路径和矩阵见experiment.json。


VERIFIED/RUNNING：真实源校准128物理样本/256冻结G输入完成（9.5506s），48组map独立复算通过，0TX/F/V/target/optimizer更新、父状态不变；两行实际R46/F50/2Ls smoke PASS，所有模块有限非零梯度、严格重载一致。单归档SHA/compile、controller/两worker实际PID/CWD/argv/线程/设备均核实。activation_energy实际GPU3首轮与日志增长核实；projection_sensitivity是CPU等待，尚未分配GPU，正确等待R53第三行GPU2完整结束和实际PID退出。跨R53/R54 GPU2/3各仅一条本任务GPU worker。无R54正式性能结论，全部新行仍固定E20后自动七正式测试；R53已评分两行结果不回流本方法。

## 已完成行的正式测试（1/2）

以下均完整168000条×七景、原冻结F50默认头、预测固定后独立truth-last；其余行继续固定训练/预测，不能把部分行冒充完整矩阵。性能优先，所有负结果保留，不回流选模、参数或父。

|场景|指定父350521|activation_energy（273765）|
|---|---:|---:|
|clean|84.456548|84.455357|
|郊区高|82.452381|82.469048|
|郊区中|81.222024|80.848810|
|郊区低|79.620238|78.913690|
|urban高|81.386905|81.212500|
|urban中|74.447024|73.793452|
|urban低|67.510119|66.435714|

准确率单位为%。逐TX/RX/day、Macro-F1与repair/damage见[逐行正式结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-source-prune-s2026092701-m2-r54/partial_formal_results.json)。

|行|R17逐景1pp容差|指定父逐景1pp参考|lowurban≥70%|其他五景raw-clean差≤5pp|
|---|---|---|---|---|
|activation_energy|True|False|False|False|

activation_energy相对固定R53同宽权重筛选控制的七景差（百分点，顺序同上）：-0.073214、+0.060119、-0.500000、-0.944643、-0.199405、-1.267857、-2.091667。

activation_energy 的 low urban 为 66.435714%，较指定父低 1.074405 个百分点，未通过父版逐景 1pp 参考；相对固定同宽权重筛选控制，七景仅郊区高提高，其余六景下降。不能将这一路线报告为性能优化成功。

首行权重已交付：项目根 `local_artifacts/pamf_restoration/r46_source_prune_r54_20261021/models/activation_energy/final.pt`。CPU严格加载、原生float32[B,2,256]输出有限、参数/ConvLinearMAC核对通过，仅合成零IQ、0实际IQ/F/标签访问。使用 `experiments.pamf_r46_source_prune.runtime.load_checkpoint`；独立G无需源数据/教师/校准文件，接原冻结F50默认头时从同一恢复IQ计算FFT/PA/时域/统计分支。完整元数据见[已交付权重](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-source-prune-s2026092701-m2-r54/partial_model_delivery.json)。

首行完整20轮训练与同卡成本见[源阶段与成本](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-source-prune-s2026092701-m2-r54/source_cost_progress.json)。额外源校准共用128物理样本/256冻结G输入、9.5506s、0optimizer更新，独立列账、不重复计入两行训练。成本为64源V、同步预加载FP32、三轮重复，不含传输/预处理/loader；三G一F共享驻留不是独立模型显存。

|行|参数|每轮中位s|父/本版G B1 median ms|父/本版G+F B1 median ms|
|---|---:|---:|---|---|
|activation_energy|273765|49.705|41.6582/40.9480|53.0625/52.8909|

原350521直接父不变；原完整识别与更强输入TX细节真实性目标ACTIVE。单seed、已暴露工程基准；识别率和波形误差不单独证明逐细节身份。原70%/5pp目标未达成。

VERIFIED/RUNNING：activation_energy完整20轮/4940、168000x7正式预测固定后独立truth-last并已退出；27.38万参数lowurban66.435714%，较指定父-1.074405pp，同宽固定控制1景提高/6景下降，保留负结果和独立权重。projection_sensitivity已按原守护条件接续GPU2，实际PID/GPU/CWD/argv与12轮日志已核实，继续原E20与自动七正式测试；GPU3本任务worker已退出。R53全部完成/所属进程退出，不重复发布。原父和全部方法参数不变，目标ACTIVE。
