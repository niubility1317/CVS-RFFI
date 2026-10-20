# R55：同一压缩拓扑的源通道重建合并正式结果

VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。fold_full 273765参数，七景较父提高0/下降7，lowurban 66.324405%（较父-1.185714pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版43.0100/42.1524ms，G+F B1父/本版54.3029/53.6499ms；fold_half 273765参数，七景较父提高0/下降7，lowurban 67.173810%（较父-0.336310pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版43.5420/42.7307ms，G+F B1父/本版53.4489/52.6415ms。fold_full相对同宽权重切片R53/prune_decay七景提高0/下降7，lowurban差-2.202976pp；fold_half相对同宽权重切片R53/prune_decay七景提高0/下降7，lowurban差-1.353571pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。

状态：VERIFIED / ANALYZED。两固定行完整E20/4940、source1800×7及target168000×7；逐行全预测固定后独立truth-last，所属controller/worker已退出。全部结果保留，不回流调参或选择性重跑。

唯一直接初始化与冻结IQ教师仍为用户指定R46 prune_refine350521。两行均沿用完整273765拓扑，区别是同一weight-only通道map上分别全量/半量合并源岭回归重建的被删输入通道贡献；原3prior/6CG、GN4组配额与共同prior map、全部coarse/action/prior联合、pairedclean IQ+.05谱误差+1→.1IQ教师、E20/global256micro256均保持。固定同宽weight-only控制复用R53 prune_decay，不重复训练。校准共用一次128源样本/256冻结G输入、0optimizer更新，另计计算与曝光；不称data-free初始化。0TX/F/V/target优化，L126000/U1134000；原F50/defaulthead只冻结评测。属于有clean模拟配对的TX无标签源域自监督。

方法commit `6bea50605a1c2cfb49645b5c2f7ba8306b39791d`。完整登记、数据与祖先、命令和停止规则见[登记](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-fold-compensate-s2026092701-m2-r55/experiment.json)；全部20轮与TX/RX/day见[结果](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-fold-compensate-s2026092701-m2-r55/results.json)及[逐组CSV](../automation_reports/CV-SincNet/20261021-phase1-restoration-R46-fold-compensate-s2026092701-m2-r55/row_metrics.csv)。

|行|完整参数|较350521减参|R17逐景1pp|父350521逐景1pp|lowurban（%）|
|---|---:|---:|---|---|---:|
|fold_full|273765|21.897689%|True|False|66.324405|
|fold_half|273765|21.897689%|True|True|67.173810|

逐景1pp容差、性能提升与原lowurban70%/其他五景距rawclean84.6761904762%≤5pp分列；不能用均值掩盖单景退化。R44只作完整描述对照。单seed、相关视图、已暴露且人选父的工程基准，不称新独立盲测或干净泛化证明。

|行|场景|准确率（%）|MacroF1（%）|较350521父（pp）|较R44（pp）|较原R17（pp）|raw净修复|
|---|---|---:|---:|---:|---:|---:|---:|
|fold_full|clean|84.388690|84.080715|-0.067857|-0.117262|-0.198214|-483|
|fold_full|practical_high|81.004167|80.403161|-1.448214|-1.414286|-0.314881|+67656|
|fold_full|practical_mid|79.790476|79.241718|-1.431548|-1.219643|+0.212500|+66988|
|fold_full|practical_low_suburban|78.213690|77.779458|-1.406548|-1.162500|+0.217262|+65829|
|fold_full|practical_high_urban|79.845833|79.274028|-1.541071|-1.471429|-0.507738|+67448|
|fold_full|practical_mid_urban|73.128571|72.865146|-1.318452|-1.098214|-0.556548|+64924|
|fold_full|practical_low_urban|66.324405|66.438763|-1.185714|-0.448214|-0.892262|+60522|
|fold_half|clean|84.413095|84.111633|-0.043452|-0.092857|-0.173810|-442|
|fold_half|practical_high|81.849405|81.311756|-0.602976|-0.569048|+0.530357|+69076|
|fold_half|practical_mid|80.610714|80.121050|-0.611310|-0.399405|+1.032738|+68366|
|fold_half|practical_low_suburban|79.048810|78.643136|-0.571429|-0.327381|+1.052381|+67232|
|fold_half|practical_high_urban|80.815476|80.326042|-0.571429|-0.501786|+0.461905|+69077|
|fold_half|practical_mid_urban|74.107143|73.851024|-0.339881|-0.119643|+0.422024|+66568|
|fold_half|practical_low_urban|67.173810|67.095201|-0.336310|+0.401190|-0.042857|+61949|

同宽度通道重建合并对照：以下只比较预登记R53 prune_decay权重筛选控制，保留全部七景差值；额外共用源校准256次冻结G输入另计，不以正式结果重新选择父、宽度或筛选方式。

|源通道重建合并行|权重筛选控制|场景|准确率变化（pp）|MacroF1变化（pp）|
|---|---|---|---:|---:|
|fold_full|R53/prune_decay|clean|-0.139881|-0.163196|
|fold_full|R53/prune_decay|practical_high|-1.404762|-1.568335|
|fold_full|R53/prune_decay|practical_mid|-1.558333|-1.687299|
|fold_full|R53/prune_decay|practical_low_suburban|-1.644643|-1.714545|
|fold_full|R53/prune_decay|practical_high_urban|-1.566071|-1.717543|
|fold_full|R53/prune_decay|practical_mid_urban|-1.932738|-1.966261|
|fold_full|R53/prune_decay|practical_low_urban|-2.202976|-2.133641|
|fold_half|R53/prune_decay|clean|-0.115476|-0.132278|
|fold_half|R53/prune_decay|practical_high|-0.559524|-0.659739|
|fold_half|R53/prune_decay|practical_mid|-0.738095|-0.807968|
|fold_half|R53/prune_decay|practical_low_suburban|-0.809524|-0.850867|
|fold_half|R53/prune_decay|practical_high_urban|-0.596429|-0.665528|
|fold_half|R53/prune_decay|practical_mid_urban|-0.954167|-0.980383|
|fold_half|R53/prune_decay|practical_low_urban|-1.353571|-1.477202|

|行|coarse/action/prior参数|文件字节|Conv+Linear MAC/IQ|
|---|---|---:|---:|
|fold_full|106138/79477/88150|1388700|30761512|
|fold_half|106138/79477/88150|1388700|30761512|

|行|推理口径|父/本版median ms|本版p95 ms|本版sample/s|父/本版增量峰值字节|
|---|---|---|---:|---:|---|
|fold_full|G_only_B1|43.0100/42.1524|44.5081|23.6860|67194880/1084928|
|fold_full|G_only_B32|81.5550/71.3495|87.8283|431.2906|276216832/227691520|
|fold_full|G_plus_F_B1|54.3029/53.6499|60.4891|18.3251|67194880/1201152|
|fold_full|G_plus_F_B32|92.3456/79.4919|91.3166|394.7644|276216832/227691520|
|fold_half|G_only_B1|43.5420/42.7307|45.6492|23.2664|67194880/1084928|
|fold_half|G_only_B32|81.0240/70.3010|77.6970|452.9230|276216832/227691520|
|fold_half|G_plus_F_B1|53.4489/52.6415|56.6078|18.8434|67194880/1201152|
|fold_half|G_plus_F_B32|90.2153/82.1262|86.7000|392.1499|276216832/227691520|

同卡64条sourceV，G与G+原冻结F分别测B1/B32，多轮预热和3轮交替顺序；全部轮次、p90/p95和驻留/峰值见JSON。计时为同步预加载GPU输入，排除传输/预处理/loader。驻留为三G+一F共享suite，增量峰值对应当前前向，不称独立部署显存；MAC只含G的Conv/Linear，FFT/CG/norm/activation/memory及F不计。参数减少不推定时延或显存同比减少。


共用源校准：128条物理样本（64L_s/64U_s）、256次冻结G输入、耗时3.9369s；source received01/20、Conv输入16固定空间位置、动作头实际池化输入；固定ridge1e-3、无截距，先合并共享prior矩再求单一投影；校准0clean/TX/F/V/target/optimizer更新、父状态不变。具体分数/map/行位置保持私有，不是运行时输入。

fold_full首轮→末轮：loss 0.086491323→0.023466999；iq_nmse 0.055717787→0.022870751；spectral_error 0.012206017→0.0059995518；teacher_stage_nmse 0.030163234→0.0029627027；中位每轮70.002s。lowurban70=False；其他五景gap5pp=False。日志类别：[]
fold_half首轮→末轮：loss 0.085844885→0.023227391；iq_nmse 0.055147978→0.022656197；spectral_error 0.012661649→0.0059458558；teacher_stage_nmse 0.030063824→0.0027390127；中位每轮67.980s。lowurban70=False；其他五景gap5pp=False。日志类别：[]

全部训练记录、全帧/主体232/尾24 NMSE、FFT误差、F配对clean特征几何、逐TX/RX/day和repair/damage均保留。低重建误差、识别率或F几何都不单独证明每个恢复细节来自输入TX。恢复器0TX/F训练与独立评分使用真值分别说明。

继承120串行pass+本次20=140；每组件累计100、组件轮次和300，原F50训练另计。两行同E20训练预算、同batch256/micro256/directbuffer、同1→.1教师；额外共用256次冻结G源校准、0optimizer更新，不重复计算为两份或隐藏为0数据初始化，均重置优化器并以指定350521父作IQ教师；与旧父相比增加了训练预算及可能的通道裁剪，不把追加训练收益全部归因于减参。

独立恢复器已取回，CPU严格加载、synthetic零IQ有限输出通过。路径local_artifacts/pamf_restoration/r46_fold_compensate_r55_20261021/models/<arm>/final.pt；使用experiments.pamf_r46_fold_compensate.runtime.load_checkpoint，float32[B,2,256]输入/同形输出；无需teacher/F即可运行恢复器，接原F50/defaulthead完成识别。权重/数据/原始日志不入Git。总目标继续active，是否性能达标按实际逐景数据判断。


## 固定方案与历史发布记录

# R55：源通道重建合并到原有剪枝权重

状态：**VERIFIED / RUNNING**，GPU3/2 两行实际训练。性能优先；两固定行fold_full、fold_half，均273765完整参数，不引入推理分支、矩阵或校准状态。唯一直接初始化和冻结IQ教师仍为人选R46 prune_refine350521 E20/4940、方法bbb8daf817de050b958d217b70d4c33f4aefe341；不继承R48/R49/R53/R54学生。固定weight-only GN4通道map与原273765拓扑不变，只补偿删除输入通道的相关贡献。理论依据是原切片把被删通道置零，源输入通道间相关性可用于线性重建；不依据正式分数重排父、宽度或超参数。R53 prune_decay作为固定同宽/同micro/同E20控制复用，不重复训练。

共同CPU2校准一次：seed2026102156从原6300L_s/56700U_s分别抽64条，使用原R07 received01/20，128物理样本/256冻结父G输入。校准不读取clean、TX标签、F、V或target，无optimizer更新。每个需要减输入通道的Conv采16个固定、互异的空间位置，action global_head采实际池化后输入；在float64累计未去均值的通道二阶矩。共享prior长卷积的各stage先合并二阶矩，再为同一共享权重求一个投影，保留step独立bias/GN。原权重map的GN4配额及decoder旧上游宽度偏移保持。

重建矩阵P的保留行严格为单位矩阵，被删行作岭回归，固定ridge=1e-3×保留通道平均二阶矩对角线（尺度下限1e-12）。不拟合截距，因此线性通道合并与卷积零填充边界一致。原完整卷积先保留输出行，再沿输入通道合并W×P；fold_full使用完整合并，fold_half使用原切片与完整合并各一半。线性动作头同样合并，bias和GN切片保持。投影、二阶矩和源行位置留私有run，部署仅加载既有结构的最终权重。局部通道重建误差可以核对，不代表剪枝后GN、完整G输出或TX识别必然改善；半量行用于固定比较这项局部近似的幅度，不作源/目标网格选择。

两行沿用pairedclean IQNMSE＋0.05三尺度相对功率STFT＋线性1→0.1冻结父三stage IQ教师，全部coarse/action/prior联合。固定E20/4940、global256micro256/directfloat32buffer、ResetAdamW、LR2e-4暖247/cos至2e-5、WD1e-4/clip1、FP32/TF32关闭；model2026092701、split392005、data2026102001、augmentation2026101912、formal2027。每行L126000/U1134000、1260000学生/教师IQ输入；每组件累计100、组件和300、串行140pass，F50另计。额外共用校准256次冻结G输入、L128/U128视图曝光独立列账，不重复计入两行、不冒充0数据初始化或梯度epoch。

source READY/EXACT_MATCH、原7/63/30物理划分及全部R46/R44/R38/R17/R07/R04来源复用，实际加载前仍核对原R46训练契约。原F50/defaulthead只作冻结评测，无另训头。恢复器训练用模拟受损/clean配对，属于TX无标签源域自监督，使用U_s而不读取其TX标签；不能称无clean盲学习。历史工程基准及人选父已暴露，不声称新独立盲测、干净泛化晋级或输入TX逐细节真实性。

资源仅物理GPU3(fold_full)/GPU2(fold_half)，每卡一条本任务GPU worker、跨R54/R55最多两条；守护先检查固定R54同卡行完成并实际所属PID退出，不读取前驱性能。普通账户、CPU2、共享capacity lock、同一score.lock最多一条scorer，旧健康任务不停止/重启/热改。新run/release/output不可覆盖；唯一owner codex/root/R46-fold-compensate。

31项聚焦检查与CLIhelp通过，一次独立P0/P1审查PASS；第一步实际校准和R46/F50/2Ls noquery smoke，PASS直接继续。一次preflight、一个归档两端SHA及远端compile，再核实实际PID/CWD/argv/GPU/日志增长；技术失败保留输出、无自动重试或回退，低性能不触发停止。额外签名/设计审批/成员哈希/重复审查均REJECTED_EXTRA_GATE，不等待。

所有新行固定最终E20/4940，不依据源分数取消正式测试，自动source1800×7、同卡G及G+F/B1/B32三轮成本、正式168000×7预测；每行七景预测固定后独立CPU truth-last。主要报告每景accuracy/MacroF1、相对固定同宽控制和R17/R46、逐TX/RX/day与repair/damage；时延/吞吐/显存/参数次级。原R17每景1pp为退化上限，不当作优化成功；lowurban≥70%、其他五景距rawclean84.6761904762%≤5pp及更强TX细节目标继续ACTIVE。

成本使用64源V同步预加载FP32，不含传输/前处理/loader；三G一F共享suite驻留不是独立部署显存。ConvLinearMAC排除FFT/CG/norm/激活及F；参数、MAC和实际时延分别报告。精确commit、命令、环境/CWD、路径和矩阵见experiment.json。


VERIFIED/RUNNING：真实源校准128物理样本/256冻结G输入完成（3.9369s），101组通道投影、141处模块使用的岭回归独立复算通过；校准0clean/TX/F/V/target/optimizer更新、父状态不变。两行实际R46/F50/2Ls smoke PASS，所有组件有限非零梯度、严格重载一致。单归档SHA/compile、controller与两worker实际PID/CWD/argv/线程/GPU及首轮日志增长已核实，GPU3fold_full与GPU2fold_half各一条本任务worker；R54固定退出守护已通过。仅说明实现与运行正常，尚无正式识别收益。两行继续固定E20/4940后自动七正式预测/独立truth-last；原指定父和原完整目标ACTIVE。
