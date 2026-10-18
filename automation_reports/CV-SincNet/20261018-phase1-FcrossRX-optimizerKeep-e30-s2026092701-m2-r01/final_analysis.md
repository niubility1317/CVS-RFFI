# P69 最终分析：保留 AdamW 状态没有带来实质 urban 增量

状态：ANALYZED，四个逻辑条件×七环境全部完成，missing=[]。仅一个模型种子2026092701。固定源规则两项均FAIL，方案关闭，不扩E50。整体目标仍ACTIVE；low urban≥60%、可重复clean正收益与suburban接近clean尚未达成。准确率/F1/最差RX使用%，配对变化使用绝对百分点（pp）。

## 结果与归因

相对同父的原As30，AS_KEEP的默认头clean只提高0.007143pp（净修复12/168000），mid urban提高0.020833pp（净修复35），low urban下降0.022024pp（净新增37个错误），urban三场景均值仅提高0.006151pp。urban修复1545／新增1514，净31/504000；urban各场景最差RX的均值下降0.081944pp。不能将这一结果视为新增机制的实质识别收益。

相对同预算、同种子、同数据角色、同KEEP规则的NOAS_KEEP，AS_KEEP clean提高0.075595pp，mid提高27.859524pp、low提高21.164881pp、urban均值提高28.532143pp。正式单种子满足clean下降不超过0.1pp的阶段目标，且观察到小幅clean与大幅星地共同改善；clean上涨不足0.1pp且只有一个seed，只能记为初步观察，不能声明可重复双提升。既有完整As的适应收益不能全部归给AdamW继承。

NOAS_KEEP相对原NoAs30的urban均值下降0.157540pp，因此urban交互为+0.163690pp，主要对应无As侧退化，并非有As侧明显提升。交互定义为(AS_KEEP−NOAS_KEEP)−(AS30−NOAS30)，不是Ag×As交互；optimizer状态继承不是IQ泛化增强。

新行low urban为51.222024%，距60%尚差8.777976pp。suburban的high/mid/low相对自身clean差距为3.888095/6.635119/9.445833pp，均未接近建议的2pp探索目标；没有降低clean来缩小差距。没有跨模型种子复现，也没有证明新增机制稳定缓解某一种urban物理退化。

## 唯一实现差异与真实执行基线

实际训练代码commit d2ddf0d48a48b7c9f8416e8c4219185155faf298；启动记录commit fd757aa1ab191a660207628bc52a64a8ad9b007a；均已推送且独立GitHub compare identical。起点0ba6945f52f0709f7246f6d0e39a26364752ef53。源码为experiments/cvs_optimizer_continuity69/的18个py及analysis/cvs_optimizer_continuity69_analysis_20261018.py／对应checks，CPU复用原P67/P66解析器。

旧continuous30改变全程LR，viewMoment两侧均重置状态，不能代替纯moment继承对照；旧纯CE一步诊断没有证明实际practical收益。RXpopulation、校准刷新、时机、场景/比例等已有实训，本轮不换名重做。另一对话common_channel未被本轮重复。已合格的原As/Asweight/LRaw/SAT18路线均已做E50，没有待补的合格扩预算项。

新NOAS_KEEP/AS_KEEP分别严格加载P43 B00/B01自己的scratch20 student＋ownEMA＋AdamW optimizer；旧NOAS30/AS30就是对应相同父的RESET10，直接复用既有预测评分。四条件全量L6300／hidden U56700／V27000，RX1/3/4/6/8、day1/2/3、六TX、split392005，无heldRX／Ldev，无U真值、目标训练或目标选模。父来源与角色精确匹配、epoch20/4440、final_only，未借用AS_CE20、其他seed或不明上游。

固定F＋CROSS_RX，SIG_TIME_X3／统一Sinc／CENTER_UNITS／joint160／默认cosine30，900722参数、206状态键，原头与骨干正常联合训练。6L×128＝768个L前向样本/步，总L尺度1.25，原clean监督份额、同IQ RDrop、ordinary128 SupCon、As64及其CE、原batch_neighbor旧U/ownEMA规则均保持；NoAs对应位置保持其原控制。没有额外头/分支/损失/教师/IQ扰动，没有新增Ag或AsU，有标签As不经过学生置信度门。

唯一变化发生在原native AdamW构造完成、首更新之前：保留父的exp_avg/exp_avg_sq/step，克隆state避免别名修改父payload，检查位置IDs、同factory的参数组顺序、当前named layout与父model键顺序/shape/dtype/finite，恢复新建groups全部超参与当前参数指针。历史optimizer不含参数名，不能声称CP自带name映射；这里以相同模型/factory和layout验证位置映射。未使用参数允许没有state，不要求全部190参数都有moment。

两个实际父均有74项state、每项step4440；116项参数无state。真实8NN smoke核实m/v逐项相等，body LR从父1e-6恢复新段2e-4，Sinc group从父.001恢复新段.2，Sinc/clip/hook计数0。最终两行74项step均6660，新段Sinc/clip计数2220。保留moment但重开原cos10属于带状态微调，不冒称完整原轨迹恢复。原teacher/EMA本地段更新计数仍2220，累计预算另计。默认头确实联合更新，不另训或换头。

实际nativeargs相对各自RESET控制仅output_dir/run_id变化，角色IDs EXACT_MATCH；真正leo_practical来自PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/leo_practical/{channel,residual}.py，不是顶层code。DataAugmentation和SSDG.train_ssdg也由同pinned路径导入。实际__file__、参数和末态见runtime_configuration.json。

fs25MHz、IQ2×256、practical前逐包RMS/center=false、网络内CENTER_UNITS保持。训练fc2.462GHz、full residual/post_sync、EQ=false即G=1；质量适配、相位跟踪、IQ补偿、AGC开启，reference_snr_db30/quality_reference20、reflect、AGC±30dB。相位跟踪残差1deg，残余CFO标准差100Hz、粗补偿误差500Hz，未把CFO当身份特征。15个RX/day/eq硬件session固定、不按TX定义，传播/噪声/状态/补偿依赖不变。训练仅原lowSub/midUrban/lowUrban各1/3，全部六config构造不等于全部六场景训练。正式固定IQ、fc2.45GHz、SNR、六practical和困难样本不变。新增噪声比不是总SNR，也没有反演原录音未知TX硬件。

## 源域决定与正式七环境

预登记恰两项源晋级条件：AS_KEEP完整V27000 clean−NOAS_KEEP≥−0.1pp；V1800×2固定draw的pooled mid/low AS_KEEP−原AS30≥＋0.5pp。实测前者−0.129630pp，后者＋0.097222pp，两项均FAIL。完整V clean分别NOAS_KEEP98.370370%、AS_KEEP98.240741%；它们不同于V1800面板clean98.444444/98.277778%，不得混用。完整V未保存macroF1，不合成该指标。

源mid两draw相对原As为＋0.055556/＋0.111111pp，low为＋0.166667/＋0.055556pp；两draw不是两个模型seed。源fullV对原NoAs也下降0.118519pp。源决定在新As prediction/scoring完成前写出，owner保存source_closure后才读取本轮正式指标。SOURCE_FAIL不因正式clean微升改判，不扩E50、不扫LR/betas/moment子集/seed救跑。所有已训练行仍实际完成七环境。

以下每个逻辑行、每个环境168000个固定物理样本，504个混淆矩阵与84个RX/day分区已独立重数，四行七视图prediction固定后才truth-last。旧RESET产物无重训/重推理/重评分。

| 环境 | NOAS30 RESET | AS30 RESET | NOAS_KEEP | AS_KEEP |
|---|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.272619 | 85.348214 |
| practical_high | 42.685119 | 81.405357 | 42.459524 | 81.460119 |
| practical_mid | 40.995238 | 78.661310 | 40.773810 | 78.713095 |
| practical_low_suburban | 39.472619 | 75.861905 | 39.271429 | 75.902381 |
| practical_high_urban | 41.439881 | 77.767262 | 41.214881 | 77.786905 |
| practical_mid_urban | 35.132738 | 62.806548 | 34.967857 | 62.827381 |
| practical_low_urban | 30.139881 | 51.244048 | 30.057143 | 51.222024 |
| urban 三场景均值 | 35.570833 | 63.939286 | 35.413294 | 63.945437 |

全部宏F1与最差RX如下。最差RX ID：两As行clean为1-1，其余所有格为14-7；完整逐RX/day指标见CSV。各场景最差RX的平均值不等于同一物理RX汇总。

| 环境 | NOAS30 F1／最差RX | AS30 F1／最差RX | NOAS_KEEP F1／最差RX | AS_KEEP F1／最差RX |
|---|---:|---:|---:|---:|
| clean | 84.979636／78.450000 | 85.044474／78.362500 | 85.002193／78.429167 | 85.051722／78.625000 |
| practical_high | 39.311416／34.304167 | 81.084230／71.908333 | 39.183535／33.416667 | 81.149060／71.937500 |
| practical_mid | 37.954813／33.520833 | 78.359706／70.212500 | 37.861196／32.825000 | 78.420506／70.254167 |
| practical_low_suburban | 36.758731／32.054167 | 75.572741／67.741667 | 36.674211／31.366667 | 75.624881／67.700000 |
| practical_high_urban | 38.481030／34.000000 | 77.423208／69.108333 | 38.361102／33.191667 | 77.452826／68.979167 |
| practical_mid_urban | 33.446603／29.495833 | 62.450463／56.458333 | 33.409998／28.850000 | 62.496130／56.370833 |
| practical_low_urban | 29.184589／25.750000 | 50.887035／46.379167 | 29.215888／25.500000 | 50.912662／46.350000 |

新增机制与交互的准确率变化：

| 环境 | KEEP−RESET，无As | KEEP−RESET，有As | 优化器×As交互 |
|---|---:|---:|---:|
| clean | +0.025000 | +0.007143 | -0.017857 |
| practical_high | -0.225595 | +0.054762 | +0.280357 |
| practical_mid | -0.221429 | +0.051786 | +0.273214 |
| practical_low_suburban | -0.201190 | +0.040476 | +0.241667 |
| practical_high_urban | -0.225000 | +0.019643 | +0.244643 |
| practical_mid_urban | -0.164881 | +0.020833 | +0.185714 |
| practical_low_urban | -0.082738 | -0.022024 | +0.060714 |

AS_KEEP−AS30的clean/mid/low正向RX分别4/7、5/7、3/7（low另1RX持平）；日期均2/4正向。该分层是同一个模型的观察，不是跨seed复现。全部修复/新增错误、固定NoAs30 clean正确集合保持率与margin/30在formal_paired_effects.csv和result_numbers.json；不挑最好RX或把F1小涨替换为accuracy改进。

## 机制分层、父集合与覆盖

源mid BLOCKED仅净修复2个观测、low BLOCKED净修复6个；low LOS/PARTIAL各净新增1个错误。部分小增量集中在BLOCKED/unlocked，但不足以认定该物理退化被稳定缓解。正式预测没有state/lock标签，不能把源分层当正式分层，也没有固定接收状态的新增单损伤因果干预。完整As相对NoAs的复合信道适应仍存在；新增optimizer机制没有可靠的物理类别收益证据。

精确P43 B01 As20学生的源V1800 clean正确集合为1765，不是P68 AS_CE20的1767。父缓存既有12602NN直接复用，本轮此分析0NN。相对原As30，AS_KEEP在此集合clean少保留1个，pooled mid/low多保留1/4个重复观测；clean/mid/low margin/30分别下降约0.000843/0.000522/0.000343。保持率/间隔与微小正式变化不能互相替代。NoAs20自身父集合及正式As20父集合没有缓存，未伪造或新增推理；NoAs行在As20集合上只是比较项，另有固定NoAs30集合读出。

AS_KEEP实际生成142080个完整信道观测，覆盖90/90 TX×RX×scene单元；物理分层计数与原As控制一致。两新行各1704960个LNN，全部原L/U物理序列匹配对应RESET；支持全部源RX/TX和LOS/PARTIAL/BLOCKED、locked/degraded/unlocked。原U各观察567000包，接受NOAS_KEEP8960/AS_KEEP8716，接受数不是伪标签精度，U真值始终隐藏。20个实际optimizer epoch记录与完整可用累计E1–30训练曲线均保留。曲线高度接近/梯度非零/状态连续不等于身份收益；预算耗尽不等于收敛。

## 实际成本与运行核实

| 项 | 实测成本 |
|---|---:|
| 本次新增两尾段 | 各E10/2220更新，合计20model-epochs/4440更新 |
| 各学生累计预算 | E30/6660更新，自身E20/4440复用 |
| 两父既有native NN | 各4797722，合计9595444；不计成本为零 |
| 新native学生train/eval/教师eval | 4543920/648004/1134000 NN |
| 新native合计 | 6325924 NN |
| 两行source/正式七视图 | 25200/2352004 NN |
| 共享真实父student/EMA smoke | 8 NN/0更新/0真实IQ |
| 本轮新增总NN | 8703136 |
| 本地真实CVSNN／分析NN | 0／0 |
| 新NoAs/As native墙钟 | 829.918/1086.908 s |
| 新两行native墙钟之和 | 1916.826 s；并行，不能当端到端墙钟 |
| 新NoAs/As source | 1.851/1.817 s |
| 新NoAs/As formal | 49.682/47.593 s |
| CPU完整分析 | 30.855 s/0NN/IQ/CP |
| 提交到分析完成端到端 | 1185.216 s，约19.754 min |

同两个父各20只计一次，四条件独有训练80model-epochs/17760更新，包含复用RESET尾段，不是四次从零E30全部新支出。两父native墙钟1329.394/1810.249s、原RESET尾段816.796/1050.660s都是历史成本。本轮没有额外学生/教师训练前向，也未再跑父cohort。NN指模型处理样本量，不是API调用次数；optimizer状态验证/统计仍有计算及存储开销。检回128项产物＋manifest共约34.99MB，checkpoint/IQ/原始formal logit数组保留远端，不入Git。

本地23项协议/配置检查＋40项纯张量optimizer检查PASS，18核心py compile PASS；分析8项纯toy/compile PASS。一次限定P0/P1审查PASS，无额外review或门槛。发布代码/配置一个commit固定，单归档SHA e432bfef1c3ca3042fa9b759f8bdb8231c4551af543afc247edb40026fc6f2cb、本地/远端一次比较VERIFIED，83059bytes/22py远端compile PASS。首步真实CP smoke PASS8NN/0更新；GPU1/2所属PID/CWD/argv/GPU计算及日志增长已核实，未修改健康其他任务。Windows SCP首次客户端未产生远端文件，确认目录空后改既有验证客户端成功；不是训练失败或技术重跑。

终态独立读回dispatcher1558466、NoAs1558557、As1559046均消失，failure不存在，两个训练/源读出/七环境评分均齐，后继analysis/complete.json为ANALYZED。主runner的complete.json保留SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS这一先发生状态，不把它当仍在训练；后继分析终态已证明闭环。源决定mtime早于As prediction/scoring完成，未用正式结果晋级。

## 交付和剩余目标

实际命令：在PROJECT/releases/cvs_FcrossRX_optimizer_keep_E30_20261018_r01执行CVS-RFFI/python -u experiments/cvs_optimizer_continuity69/runner.py；末CPU分析由dispatcher执行同release的analysis/cvs_optimizer_continuity69_analysis_20261018.py --project PROJECT --out PROJECT/runs/20261018-phase1-FcrossRX-optimizerKeep-e30-s2026092701-m2-r01/analysis_r01 --local-nn 0 --smoke-nn 8 --code-commit d2ddf0d48a48b7c9f8416e8c4219185155faf298。唯一输出根PROJECT/runs/20261018-phase1-FcrossRX-optimizerKeep-e30-s2026092701-m2-r01。配置、来源、owner、GPU/停止规则和逐行矩阵见experiment.json；实际状态见verification.json、launch_verification.json、runtime_configuration.json和delivery.json。

四PNG已逐张目视查看，均非空且坐标/图例可读，对应PDF存在。大As效应会压缩很小的optimizer增量，精确数值表优先。

- [完整E1–30训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/full_training.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/full_training.pdf)
- [源状态/质量/锁定/噪声分层](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/source_state_quality_lock_noise.pdf)
- [正式配对和交互](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_interaction.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_interaction.pdf)
- [真实优化器状态轨迹](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/optimizer_state_readout.png)，[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/optimizer_state_readout.pdf)
- [全部正式指标](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/formal_metrics.csv)、[配对修复/新增错误](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv)、[交互](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/formal_interactions.csv)
- [全部源分层](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/source_strata.csv)、[源配对](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/source_paired_effects.csv)、[父集合](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/P43_As20_clean_cohort.csv)
- [完整训练曲线数据](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/training_curves.csv)、[optimizer记录](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/optimizer_epochs.csv)、[TX×RX物理覆盖](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/AS_KEEP_TX_RX_scene_state_lock.csv)
- [全部聚合数值及逐seed边界](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/result_numbers.json)、[成本](E:/type10-7/local_artifacts/cvs_FcrossRX_optimizer_keep_E30_20261018_r01/retrieved_r01/analysis_r01/costs.json)

本轮正式七环境无缺项。仍未完成跨seed复现、正式low urban超过60%、可重复clean正收益、suburban接近clean，以及新增机制的物理退化归因；没有源合格E50资格。原完整As真实标签监督及clean份额继续保留，这个optimizer继承配方关闭，不由正式微增翻案。
