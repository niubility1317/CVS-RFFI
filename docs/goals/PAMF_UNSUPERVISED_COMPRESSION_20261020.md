# 无 TX 标签恢复器的压缩与性能联合优化

2026-10-21 当前R62：VERIFIED / ANALYZED：两行完整E20/4940、源1800×7、正式168000×7、逐行七预测固定后独立truth-last、全20轮/同卡成本/实际所属进程退出和独立权重CPU严格重载已核实。group_preserve 248828参数，较指定M35七景提高0/下降7，low urban 66.230952%（较M35-1.279167pp）；较R49同增量预算提高0/下降7，low差-0.591071pp；70%=False、其他五景gap≤5pp=False；同卡G B1父/本版48.9053/48.2824ms，G+F B1父/本版59.1257/57.9017ms；regroup_control 248828参数，较指定M35七景提高0/下降7，low urban 66.207738%（较M35-1.302381pp）；较R49同增量预算提高0/下降7，low差-0.614286pp；70%=False、其他五景gap≤5pp=False；同卡G B1父/本版46.9371/47.3193ms，G+F B1父/本版57.3480/57.7004ms。同初始权重/通道/预算下GN3相对GN1七景提高6/下降1，low urban差+0.023214pp；归约和实际成本不同，不称完美计算匹配。直接指定350521初始化/冻结IQ教师不变，全prior保留并更新，原pairedcleanIQ/谱/1→.1IQ教师，全组件联合；0TX/F/V/target优化/新校准/新参数/额外G前向。全部正负结果保留，不把减参或1pp容差当作性能提升或逐细节TX真实性。完整目标ACTIVE。 [全部七景与成本](../PAMF_R46_WHOLE_GROUP_R62_20261021.md)。

R63已实际VERIFIED/WAITING_PARENT_FORMAL：CPU2/0GPU、真实2Ls解析no-query smoke、单archiveSHA/compile/实际PID/CWD/argv/空CUDA及日志增长核实。未消费R62源验证或正式准确率；同父M35、同增量预算R49retain与GN3/GN1两个固定读出。复用56、新算84，总140固定后独立truth-last2016组/336条件对照/140配对差。无新恢复器/F/checkpoint/神经预测/训练/校准/头拟合，不改父，不作逐细节真实性证明。[R63固定评价](../PAMF_GROUP_FORMAL_IDENTITY_R63_20261021.md)。

2026-10-21 当前 R62：VERIFIED / TRAINING：固定方法 7a8fa7bd836946c3493635cbe0966591ecf14e1e 已发布，普通账户、单归档 SHA 比对和远端 compile 通过；真实指定父/F50/两条 L_s no-query smoke PASS。两固定行物理 GPU3/2 的实际 PID/CWD/argv/CUDA/CPU2、GPU PID、日志增长和至少一轮有限损失已核实。每卡一条本任务训练，总数2；其他 owner 不变。整组 GN3/GN1 同 248828 参数及初值，全 prior、原波形目标、E20/4940 后两行均自动七景正式测试/独立 truth-last/同卡成本，无源分数筛选，无新识别收益结论，完整目标 ACTIVE。 [固定矩阵](../PAMF_R46_WHOLE_GROUP_R62_20261021.md)。

2026-10-21 R62 LOCAL_VERIFIED：整组裁剪既有上下文 U-Net，完整 prior 保留；GN3/GN1 两行同 248828 参数、同初始权重、同源预算。直接指定 M35 父与冻结 IQ 教师，权重选组无需 IQ，原 paired IQ＋谱误差＋1→0.1 IQ 教师，全 E20/4940/micro256，0 TX/F/V/target 优化。56 checks/CLI/真实父合成 CPU 梯度/ONE P0P1 PASS；首步真实 2 L_s smoke 后两行自动七景正式测试、独立 truth-last 与同卡成本。只读复用 R49 retain_refine 同增量预算对照，不继承任何子版；性能优先，70/5pp/逐细节真实性目标 ACTIVE。[R62 固定方案](../PAMF_R46_WHOLE_GROUP_R62_20261021.md)。

二十一个固定新版的正式性能、郊区/urban 全场景和配对推理成本均已完整汇总，按预登记顺序保留全部结果；性能优先，不用通过退化容差、减参或提速替代逐场景性能改善。二十一行中相对指定父七景全部提高的版本为：R48 prune_refine（273765参数，low urban 68.257738%）、R60 context_hint（273765参数，low urban 68.232143%）。完整数据按固定矩阵顺序报告，不据此另选父、修改超参数或选择性重跑。只反映单 seed 工程基准，70%/5pp以逐行验收表为准；输入TX逐细节真实性尚未证明，原完整目标继续ACTIVE。 见[二十一版性能总表](../PAMF_R46_COMPRESSION_PERFORMANCE_20261020.md)。指定 350521 父版不变，原完整目标继续 ACTIVE。

2026-10-21 当前 R60：VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。context_hint 273765参数，七景较父提高7/下降0，lowurban 68.232143%（较父+0.722024pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版47.6136/47.1164ms，G+F B1父/本版59.5221/58.5740ms；all_hint 273765参数，七景较父提高1/下降6，lowurban 67.719643%（较父+0.209524pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.9480/45.2744ms，G+F B1父/本版57.7602/56.0171ms。context_hint相对预登记同图同预算无hint控制R53prune_decay七景提高3/下降4，lowurban差-0.295238pp；all_hint相对预登记同图同预算无hint控制R53prune_decay七景提高0/下降7，lowurban差-0.807738pp。相同.02context权重，all_hint比context_hint额外.02prior：七景提高0/下降7，lowurban差-0.512500pp。原参数80+20=100轮，context/prior各84模块的训练期特征约束；0额外G前向/校准IQ/新增参数，全部hooks保存前移除，额外训练开销已列账。原指定350521直接父保持，正负结果保留；不把容差、减参或中间特征拟合当作整体提升或逐细节TX真实性。原完整目标ACTIVE。 [全部七景与同图控制](../PAMF_R46_FEATURE_HINT_R60_20261021.md)。

R61已完成VERIFIED/ANALYZED：140解析单元固定后truth；2016分组/336条件阴性对照/140配对差独立复算，R46父及R53同宽控制1008记录与R56一致；0F/checkpoint/额外神经预测/训练/headfit/新校准/GPU，所属CPU进程EXITED。有限同TX关联不是识别率或逐细节真实性证明，不改变父实验，原目标ACTIVE。 [R61全部四候选正式关联](../PAMF_HINT_FORMAL_IDENTITY_R61_20261021.md)。

2026-10-21 当前 R58：VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。calibrated 276453参数，七景较父提高0/下降7，lowurban 67.383333%（较父-0.126786pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.4561/121.1858ms，G+F B1父/本版56.2264/132.8959ms；identity_control 276453参数，七景较父提高0/下降7，lowurban 66.771429%（较父-0.738690pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.1120/119.8319ms，G+F B1父/本版55.7322/130.8974ms。同图calibrated相对identity_control七景提高5/下降2，lowurban差+0.611905pp；R55仅旧归一化参考、非完整参数/计算匹配控制。新2688统计参数仅20轮/4940次；原参数80+20=100轮，所有参数联合优化，成本分别报告。原指定350521直接父保持，正负结果保留；不把容差、减参、源拟合或解析关联当作整体提升或逐细节TX真实性。原完整目标ACTIVE。 [全部七景与同图控制](../PAMF_R46_STAT_TRANSPORT_R58_20261021.md)。

2026-10-21 R58执行缓存LOCAL_PROTOTYPE：同一276453图只缓存冻结权重派生斜率，21项CPU合成检查通过、输出/输入梯度/portable state逐位一致；每完整G前向index/clamp/exp各少168次dispatcher调用，不是GPU kernel或提速证据。额外非持久缓存5376B，默认clone另持完整模型参数1105812B及其状态，显存须实测。未读取真实IQ/checkpoint/F/source或target分数；无新训练候选、当前发布代码/矩阵/权重/进程不变，GPU与正式预测一致性尚待验证。主线仍为R58全部七正式、R59固定身份诊断和R60现图训练优化/全部正式，性能优先/原完整目标ACTIVE。[缓存原型及边界](../PAMF_R58_COEFFICIENT_CACHE_20261021.md)。

2026-10-21 当前R60：既有273765普通GN/U-Net/3prior6CG的训练期中间特征约束；直接人选350521父，context_hint/all_hint均+.02context，后一行另+.02prior，各84GN按保留通道逐IQ能量归一化。47checks/CLI/ONEP0P1PASS，真实source2Ls noquerysmoke在launcher首步执行；无新增推理参数/buffer/分支、额外G前向或校准IQ。全E20/4940/globalmicro256/所有组件联合源pairedclean自监督，0TX/F/V/target优化；两行自动source与168000×7正式/独立truthlast/同卡成本。核心0549b81原型在R58验证/正式分数读取前固定，源校准与新训练仍未执行；GPU3/2须等待R58同卡实际退出。VERIFIED/WAITING_R58_ACTUAL_EXIT：真实2Ls smoke/严格重载/hooks移除和两条CPU等待进程/日志增长已核实，未分配GPU或训练，无新性能结论，性能优先/原完整目标ACTIVE。[R60固定方案](../PAMF_R46_FEATURE_HINT_R60_20261021.md)。

R59已完成VERIFIED/ANALYZED：196解析单元固定后truth；3024分组/504条件阴性对照/280配对差独立复算，R46父及R55旧归一化参考1008记录与R56一致；0F/checkpoint/额外神经预测/训练/headfit/新校准/GPU，所属CPU进程EXITED。有限同TX关联不是识别率或逐细节真实性证明，不改变父实验，原目标ACTIVE。 [R59全部六候选正式关联](../PAMF_NORM_FORMAL_IDENTITY_R59_20261021.md)。

2026-10-21 当前 R57：VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。gn_full 273765参数，七景较父提高4/下降3，lowurban 67.330357%（较父-0.179762pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.8344/45.3467ms，G+F B1父/本版57.1810/57.1344ms；gn_half 273765参数，七景较父提高4/下降3，lowurban 67.588095%（较父+0.077976pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版47.8450/47.4498ms，G+F B1父/本版58.4878/58.0378ms。gn_full相对同宽full-fold控制R55/fold_full七景提高6/下降1，lowurban差+1.005952pp；gn_half相对同宽full-fold控制R55/fold_full七景提高7/下降0，lowurban差+1.263690pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。 [全部七景与同宽控制](../PAMF_R46_GN_RESPONSE_R57_20261021.md)。

2026-10-21 当前R58：现有压缩图168GN组内可学习统计校正，完整276453/context186959、新参数2688；同图calibrated/identity_control两行，直接指定350521父/source-onlyR55fullfold统计、无训练学生继承。31checks/CLI/一次P0P1PASS，CPU2共128source received/额外256父+256恒等子G输入，0clean/TX/F/V/target/optimizer校准；所有新统计参数随原参数联合E20/4940，原参数100轮/新参数仅20轮分别记账；两行自动source及168000×7正式测试。核心方案/prototype在R57分数读取前固定，GPU3/2等待R57实际退出。VERIFIED/RUNNING：两行实际GPU3/2首轮训练、真实组统计校准/smoke/PID与日志增长已核实，正式性能待完成；性能优先，原目标ACTIVE。[R58固定方案](../PAMF_R46_STAT_TRANSPORT_R58_20261021.md)。

2026-10-21 当前 R55：VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。fold_full 273765参数，七景较父提高0/下降7，lowurban 66.324405%（较父-1.185714pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版43.0100/42.1524ms，G+F B1父/本版54.3029/53.6499ms；fold_half 273765参数，七景较父提高0/下降7，lowurban 67.173810%（较父-0.336310pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版43.5420/42.7307ms，G+F B1父/本版53.4489/52.6415ms。fold_full相对同宽权重切片R53/prune_decay七景提高0/下降7，lowurban差-2.202976pp；fold_half相对同宽权重切片R53/prune_decay七景提高0/下降7，lowurban差-1.353571pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。 [全部七景与同宽控制](../PAMF_R46_FOLD_COMPENSATE_R55_20261021.md)。

2026-10-21 当前R57：在读取R55正式分数前固定的既有273765拓扑GN响应校正，沿用R55 source-only full-fold投影但不加载R55训练权重；直接人选350521父，CPU2在同128源received上额外256父+256初始子G输入，一次拟合168原GN，full/half两行；0clean/TX/F/V/target校准，0TX/F/V/target训练，完整E20/4940及自动全部七景。固定同宽控制R55fold_full复用、GPU3/2各一条并等待R55实际退出。VERIFIED/RUNNING：两行实际GPU3/2首轮训练、真实源GN校准/168模块独立复算/smoke/PID与日志增长已核实，正式性能待完成；性能优先，原完整目标ACTIVE。[R57固定方案](../PAMF_R46_GN_RESPONSE_R57_20261021.md)。

R56已完成VERIFIED/ANALYZED：280解析单元固定后truth；4536分组/756条件阴性对照/392配对差独立复算，R46父及R49同宽控制1008记录与R50一致；0F/checkpoint/额外神经预测/训练/headfit/新校准/GPU，所属CPU进程EXITED。有限同TX关联不是识别率或逐细节真实性证明，不改变父实验，原目标ACTIVE。 [R56全部九候选正式关联](../PAMF_COMPRESSION_FORMAL_IDENTITY_R56_20261021.md)。

2026-10-21 当前R55：原273765拓扑与weight-only GN4 map，源通道岭回归合并到既有Conv/Linear权重，full/half两固定幅度；直接人选350521父，CPU2共同128物理样本/256冻结G输入、校准无clean/TX/F/V/target/更新；两行全部联合E20/4940后自动七正式测试及成本。唯一同宽控制R53prune_decay复用，GPU3/2各一条、固定R54实际退出保护。VERIFIED/RUNNING：两行实际GPU3/2首轮训练、真实校准/101组投影复算/smoke/PID与日志增长已核实，正式性能待完成；原性能/TX细节目标ACTIVE。[R55方案](../PAMF_R46_FOLD_COMPENSATE_R55_20261021.md)。

2026-10-21 当前 R54：VERIFIED/ANALYZED：两行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及两套独立权重CPU严格重载均已核实。activation_energy 273765参数，七景较父提高1/下降6，lowurban 66.435714%（较父-1.074405pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版41.6582/40.9480ms，G+F B1父/本版53.0625/52.8909ms；projection_sensitivity 273765参数，七景较父提高1/下降6，lowurban 66.919048%（较父-0.591071pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版45.7906/44.9914ms，G+F B1父/本版58.4112/57.2981ms。activation_energy相对同宽权重筛选R53/prune_decay七景提高1/下降6，lowurban差-2.091667pp；projection_sensitivity相对同宽权重筛选R53/prune_decay七景提高1/下降6，lowurban差-1.608333pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。 [全部七景与同宽控制](../PAMF_R46_SOURCE_PRUNE_R54_20261021.md)。

2026-10-21 当前R54首行：VERIFIED/RUNNING：activation_energy完整20轮/4940、168000x7正式预测固定后独立truth-last并已退出；27.38万参数lowurban66.435714%，较指定父-1.074405pp，同宽固定控制1景提高/6景下降，保留负结果和独立权重。projection_sensitivity已按原守护条件接续GPU2，实际PID/GPU/CWD/argv与12轮日志已核实，继续原E20与自动七正式测试；GPU3本任务worker已退出。R53全部完成/所属进程退出，不重复发布。原父和全部方法参数不变，目标ACTIVE。 [正式结果](../PAMF_R46_SOURCE_PRUNE_R54_20261021.md)。

2026-10-21 当前 R53：VERIFIED/ANALYZED：三行完整E20/4940、源1800×7和正式168000×7、每行七预测固定后独立truth-last、完整曲线/同卡成本、所属controller/worker退出及三套独立权重CPU严格重载均已核实。prune_decay 273765参数，七景较父提高6/下降1，lowurban 68.527381%（较父+1.017262pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版44.1931/43.3225ms，G+F B1父/本版56.2200/54.1512ms；retain_anchor 350521参数，七景较父提高2/下降5，lowurban 67.636310%（较父+0.126190pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版44.2953/44.1067ms，G+F B1父/本版55.4548/54.3477ms；prune_anchor 273765参数，七景较父提高4/下降3，lowurban 68.347619%（较父+0.837500pp），70%=False、其他五景gap5pp=False；同卡G B1父/本版43.4629/42.8090ms，G+F B1父/本版54.6030/54.1270ms。retain_anchor相对同宽R49/retain_refine七景提高7/下降0，lowurban差+0.814286pp；prune_anchor相对同宽R53/prune_decay七景提高0/下降7，lowurban差-0.179762pp。原指定350521直接父保持，全部正负结果保留；不把容差通过、减参、源域或解析关联当作正式提升或逐细节TX真实性。原完整目标ACTIVE。 [全部七景与同宽控制](../PAMF_R46_ANCHOR_REFINE_R53_20261021.md)。

2026-10-21 当前：R54固定两种源数据通道选择，完整273765拓扑、直接人选350521父，CPU2共用128源物理样本/256冻结G输入，0TX/F/V/target；共享prior采用相同map，全部联合E20/4940后逐行自动七正式测试与成本。固定控制复用R53prune_decay，不依其未知正式分数调参；GPU3等R53retain_anchor完成且PID退出，GPU2等R53queuedprune_anchor完成且同卡所属PID退出，跨R53/R54每卡最多一GPU worker、合计两条。VERIFIED/RUNNING：GPU3实际首轮训练、GPU2仅CPU等待，源校准与真实smoke/48组map复算已核实；无R54正式结论。原性能/TX细节目标ACTIVE。 [R54方案](../PAMF_R46_SOURCE_PRUNE_R54_20261021.md)。

2026-10-21 当前：R52源相位参考对照已完成并独立核实108单元五指标和R51旧108数组，0F/TX/V/target/truth/训练，父逐张量未变；大角度C4下固定clean和坐标传递clean误差都明显，不能据此施加等变/不变训练或声称身份缺陷。回到性能优先的现有压缩结构：R53固定273765衰减控制和350521/273765持续IQ教师约束三行，直接人选350521父、全组件联合E20/4940、GPU2/3各1、全部自动七正式测试及成本；复用R49同宽衰减控制，不重复训练。当前VERIFIED/RUNNING，GPU2/3两行首轮与日志增长核实、第三行排队，尚无新性能结论。原70%/五景gap5pp/更强TX细节目标ACTIVE。 [R52参考对照](../PAMF_R46_PHASE_REFERENCE_R52_20261021.md)；[R53方案](../PAMF_R46_ANCHOR_REFINE_R53_20261021.md)。

2026-10-21 当前：R50完整无F正式关联已独立核实并交付（4536组/756条件阴性对照/420配对差，父504条不变；全部所属CPU进程退出），27.38万lowurban combined margin0.069414，高于raw0.002280、低于clean0.124015；属于有限同TX关联，不是逐细节证明。TX2的clean解析margin也为负，保留坐标局限，不把该负数单独当恢复器错误。R51源域128条C4检验也已完成并独立复算108单元，0F/TX/V/target/truth/训练，原350521父逐张量不变；received非零C4的final等变NMSE0.82160833。接收恢复本来要补偿相位，大等变误差不直接证明缺陷，下一步先补充“只旋转received、clean固定”与“联合坐标旋转”的源对照，不直接据此加入全局相位等变训练。全部历史权重/正式预测保留，指定350521直接父不变，性能优先、原70%/其他五景gap5pp/更强TX细节目标继续ACTIVE。 [R50关联解释](../PAMF_MULTIPOINT_FORMAL_IDENTITY_R50_20261020.md)；[R51完整源检验](../PAMF_R46_SOURCE_PHASE_R51_20261021.md)。

八个固定新版的正式性能、郊区/urban 全场景和配对推理成本均已完整汇总，按预登记顺序保留全部结果；性能优先，不用通过退化容差、减参或提速替代逐场景性能改善。本轮固定矩阵的实际正收益是 R48 prune_refine：完整 273765 参数，较指定父减少 21.8977%，七场景准确率全部提高。Low urban 为 68.257738%（+0.747619pp）。这是单 seed、已暴露工程基准的描述结果，保留其权重与完整负结果；不因此私自更换后续直接父或用正式分数调参。原完整目标尚未达成。 见[八版性能总表](../PAMF_R46_COMPRESSION_PERFORMANCE_20261020.md)。指定 350521 父版不变，原完整目标继续 ACTIVE。

2026-10-20 当前正式结果：**R48两行全部完成七场景、重复时延/吞吐测量并交付独立权重。** 性能优先：retain_refine七景提高1景、下降6景；prune_refine七景提高7景、下降0景。逐景1pp仅是退化上限，不把通过容差、减参或训练提速自动表述为性能提升；逐场景数据优先于均值。指定R46 prune_refine350521继续为唯一直接基准，已揭示正式结果不用于另选父或选择性重跑。 VERIFIED/ANALYZED：两行E20/4940、源1800×7、正式168000×7全部预测固定后独立truth-last、所属控制器和GPU进程自然退出、完整权重取回与CPU严格加载均核实。retain_refine350521参数，lowurban66.852976%（较350521父-0.657143pp），原R17逐景1pp=True，父逐景1pp=True，70%=False，其他五景gap5pp=False；G_only_B1父/本版median 44.5498/44.5913ms，本版P95 45.8523ms，G_plus_F_B1父/本版median 55.1817/55.6139ms，本版P95 57.8439ms，G_only_B32父/本版吞吐551.4749/549.9111sample/s，G_plus_F_B32父/本版吞吐466.5235/468.6024sample/s；prune_refine273765参数，lowurban68.257738%（较350521父+0.747619pp），原R17逐景1pp=True，父逐景1pp=True，70%=False，其他五景gap5pp=False；G_only_B1父/本版median 41.4050/40.9928ms，本版P95 42.5749ms，G_plus_F_B1父/本版median 52.2722/51.4586ms，本版P95 53.1965ms，G_only_B32父/本版吞吐561.7997/619.6688sample/s，G_plus_F_B32父/本版吞吐485.1169/530.7903sample/s。同卡预加载IQ计时，不含传输/预处理/loader；显存为三G一F共享suite及当前前向增量。参数、时延、吞吐和性能分别报告，全部负结果保留，不回流调参或选择性重跑。原完整目标继续ACTIVE，识别率/损失/关联均不单独证明逐细节TX真实性。 [R48全部正式结果与推理成本](../../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-compact-refine-s2026092701-m2-r48/report.md)。

用户指定的R46 prune_refine350521仍是后续唯一直接基准，所有父权重与历史产物保留；未依据本轮正式分数另选父或选择性重跑。R47无F正式关联诊断已独立完成，属于有标签事后评价，不是逐细节真实性证明。原lowurban70%、其他五景距rawclean≤5pp、原R17逐景最多1pp及更强输入TX细节真实性要求保持。下文为历史记录，以本段为当前R48状态。

2026-10-20 当前正式结果：**R49六行全部完成七场景、重复时延/吞吐测量并交付独立权重。** 性能优先结论：六行均未同时实现逐场景识别提升与原70%/其他五景gap5pp目标。29.31万prune_balanced的lowurban为67.961310%，较指定父提高0.451190pp，但四个受损场景下降；其他五行的六个受损场景全部低于指定父。原每景1pp只作退化上限，不能把通过容差、减参或训练提速表述为性能优化成功。指定R46 prune_refine350521继续保留为唯一直接基准，不根据本轮正式结果另选父。 VERIFIED/ANALYZED：六行E20/4940、源1800×7、正式168000×7全部预测固定后独立truth-last、所属控制器和GPU进程自然退出、完整权重取回与CPU严格加载均核实。retain_refine350521参数，lowurban66.822024%（较350521父-0.688095pp），原R17逐景1pp=True，父逐景1pp=True，70%=False，其他五景gap5pp=False；G_only_B1父/本版median 44.6075/44.5250ms，本版P95 48.3775ms，G_plus_F_B1父/本版median 54.8240/54.8702ms，本版P95 59.8024ms，G_only_B32父/本版吞吐393.8300/391.5788sample/s，G_plus_F_B32父/本版吞吐353.0949/352.8159sample/s；prune_near329885参数，lowurban66.757143%（较350521父-0.752976pp），原R17逐景1pp=True，父逐景1pp=False，70%=False，其他五景gap5pp=False；G_only_B1父/本版median 45.9973/44.1574ms，本版P95 61.3752ms，G_plus_F_B1父/本版median 54.7709/53.8840ms，本版P95 58.1520ms，G_only_B32父/本版吞吐382.2783/385.3717sample/s，G_plus_F_B32父/本版吞吐347.5274/351.7908sample/s；prune_balanced293069参数，lowurban67.961310%（较350521父+0.451190pp），原R17逐景1pp=True，父逐景1pp=True，70%=False，其他五景gap5pp=False；G_only_B1父/本版median 42.3437/41.2085ms，本版P95 42.4010ms，G_plus_F_B1父/本版median 52.3321/51.3496ms，本版P95 52.6641ms，G_only_B32父/本版吞吐556.5820/592.9033sample/s，G_plus_F_B32父/本版吞吐486.9068/529.1422sample/s；prune_small215049参数，lowurban65.825595%（较350521父-1.684524pp），原R17逐景1pp=False，父逐景1pp=False，70%=False，其他五景gap5pp=False；G_only_B1父/本版median 42.4602/43.3987ms，本版P95 45.0318ms，G_plus_F_B1父/本版median 52.2932/53.3125ms，本版P95 54.8236ms，G_only_B32父/本版吞吐563.4631/691.6884sample/s，G_plus_F_B32父/本版吞吐484.1417/577.7237sample/s；context_slim268257参数，lowurban66.395238%（较350521父-1.114881pp），原R17逐景1pp=True，父逐景1pp=False，70%=False，其他五景gap5pp=False；G_only_B1父/本版median 42.8052/43.6513ms，本版P95 44.8550ms，G_plus_F_B1父/本版median 53.1800/53.8204ms，本版P95 55.6401ms，G_only_B32父/本版吞吐550.4561/581.5011sample/s，G_plus_F_B32父/本版吞吐475.6606/507.2773sample/s；prior_slim310905参数，lowurban66.997024%（较350521父-0.513095pp），原R17逐景1pp=True，父逐景1pp=False，70%=False，其他五景gap5pp=False；G_only_B1父/本版median 43.4693/43.8420ms，本版P95 45.7176ms，G_plus_F_B1父/本版median 53.4682/53.8428ms，本版P95 55.2896ms，G_only_B32父/本版吞吐556.5201/621.2118sample/s，G_plus_F_B32父/本版吞吐480.8649/527.2047sample/s。同卡预加载IQ计时，不含传输/预处理/loader；显存为三G一F共享suite及当前前向增量。参数、时延、吞吐和性能分别报告，全部负结果保留，不回流调参或选择性重跑。原完整目标继续ACTIVE，识别率/损失/关联均不单独证明逐细节TX真实性。 [R49全部正式结果与推理成本](../../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49/report.md)。

原R48两行另按已固定计划收尾，用户指定的R46 prune_refine350521仍是后续唯一直接基准，所有父权重与历史产物保留；未依据本轮正式分数另选父或选择性重跑。R47无F正式关联诊断已独立完成，属于有标签事后评价，不是逐细节真实性证明。原lowurban70%、其他五景距rawclean≤5pp、原R17逐景最多1pp及更强输入TX细节真实性要求保持。下文为历史记录，以本段为当前R49状态。

2026-10-20 用户最新明确：**性能优先**。后续在指定R46 prune_refine350521上继续优化/压缩，识别性能与输入TX保真优先，参数/时延/显存次要；原每景1pp仅为退化上限，不以减参或加速掩盖识别下降。当前R48/R49八行保持固定方案与自动七正式测试；已揭示正式分数不回流调参、重排或选择性重跑，不私自更换直接父。首批四行完整正式结果已保留：[R49阶段报告](../../automation_reports/CV-SincNet/20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49/report.md)。原70%/其他五景gap5pp/更强TX细节目标继续ACTIVE。

R50已完成VERIFIED/ANALYZED：280解析单元固定后truth；4536分组/756条件阴性对照/420配对差独立复算，R46父504记录与R47一致；0F/checkpoint/额外神经预测/训练/headfit/新校准/GPU，所属CPU进程EXITED。有限同TX关联不是识别率或逐细节真实性证明，不改变父实验，原目标ACTIVE。 [R50全部九候选正式关联](../PAMF_MULTIPOINT_FORMAL_IDENTITY_R50_20261020.md)。

2026-10-20 用户最新要求多点压缩并提高显卡利用率：已固定[R49六点并行与micro256吞吐优化](../PAMF_R46_MULTIPOINT_FAST_R49_20261020.md)，同宽350521及329885/310905/293069/268257/215049参数，直接指定R46父，新增GPU2至7各一行；R48GPU0/1两行保持。每行完整E20/4940、原全局batch256和全部L/U曝光，micro256/directbuffer；0TX/F/V/target训练，全部自动七正式测试和G/G+F成本。当前VERIFIED/RUNNING，六行实际GPU/PID/首轮日志增长核实，尚无新性能结论。参数与训练/推理时延、吞吐、显存分别报告，原70%/5pp/更强TX细节真实性目标保持ACTIVE。

2026-10-20 用户最新指定：**以R46 prune_refine350521继续优化、压缩，同时关注时延、吞吐量与显存。** 已固定[R48同预算350521优化对照与273765进一步剪枝](../PAMF_R46_COMPACT_REFINE_R48_20261020.md)，两行均直接从指定父分叉，新增E20/4940，0TX/F/V/target训练，逐行自动七正式测试；同卡测G与G+F的B1/B32多轮时延/吞吐和内存，减参不预设加速。当前VERIFIED/RUNNING，实际两卡首轮和日志增长核实，尚无新性能结论。保留所有父权重/结果，人工指定已暴露工程基准的限制及原70%/5pp/真实性目标保持ACTIVE。

2026-10-20 当前正式结果：**R46两行全部完成并交付独立权重。** VERIFIED/ANALYZED：两行E20/4940、源1800×7、正式168000×7全部预测固定后独立truth-last、所属控制器和GPU进程自然退出、完整权重取回与CPU严格加载均核实。retain_refine442805参数，lowurban67.964881%（较442805父+1.192262pp），原R17逐景1pp=True，父逐景1pp=True，70%=False，其他五景gap5pp=False；prune_refine350521参数，lowurban67.510119%（较442805父+0.737500pp），原R17逐景1pp=True，父逐景1pp=True，70%=False，其他五景gap5pp=False。全部负结果和成本保留，不回流调参或选择性重跑。原完整目标继续ACTIVE，识别率/损失/关联均不单独证明逐细节TX真实性。 [R46全部七场景、完整曲线与成本](../../automation_reports/CV-SincNet/20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46/report.md)。

用户指定的原R44 group_pruned442805权重与所有历史产物继续保留。R47独立无F同TX评价按固定流程继续，实际进度见[R47登记](../../automation_reports/CV-SincNet/20261020-diagnostic-restoration-R46-formal-analytic-TX-s2026092701-m2-r47/report.md)，不作为R46正式交付门槛。原lowurban70%、其他五景距rawclean≤5pp、原R17逐景最多1pp及更强输入TX细节真实性要求保持，不因产物齐全或参数下降缩小目标。下文为历史方案与启动记录，以本段为R46当前状态。

R47已完成VERIFIED/ANALYZED：112正式特征单元先于truth，56复用/56新算；1512分组/252条件对照/84配对差已独立复算，R44父504条记录与R45完全一致，全部所属CPU进程退出。lowurban combined margin为r44_442805 0.055390、retain_refine 0.067473、prune_refine 0.066143。属于有标签F独立事后关联，不是识别率或逐细节真实性证明，不改变R46训练和预测。[R47完整评价](../PAMF_R46_FORMAL_IDENTITY_R47_20261020.md)。

2026-10-20 用户再次指定：**保留R44 group_pruned442805并继续优化、进一步减参。** 已固定[R46同预算优化控制与350521进一步剪枝](../PAMF_R44_PRUNED_REFINE_R46_20261020.md)，两行均直接从该父分叉，完整E20，0TX/F/V/target训练，逐行自动七正式测试。用户在已揭示R44结果后指定父，报告保留工程基准暴露背景，不称独立盲测。当前VERIFIED/FORMAL_PREDICTING，两版完整E20/4940及source七视图已完成，正式预测已实际启动、尚无正式评分结果；旧44万权重/结果保持，下文R38基底说明为先前阶段。

2026-10-20 当前结果：**R44 两种不同压缩均已完成正式测试并交付独立权重；继续保留用户指定 R38 794529 作为工程基准。** 重要性剪枝442805减参44.2682%，七景中郊区high/mid与urban high较R38提高，但lowurban66.7726%相对68.2696%下降1.4970pp；它通过用户原始R17逐景1pp容差，没有达到更保守的R38逐景1pp。低秩530161减参33.2735%，lowurban65.7714%，相对R38下降2.4982pp，也未通过原R17容差。两版实测batch1延迟增加，不能以更小宣称性能/成本联合优化达成。全部结果、20轮完整记录、逐TX/RX/day、成本与负结果均保留；所属GPU进程和控制器已退出，权重取回并CPU严格验证。[R44正式结果](../../automation_reports/CV-SincNet/20261020-phase1-restoration-R38-compression-methods-s2026092701-m2-r44/report.md)。

另已启动固定CPU2/0GPU的[R45正式无F同TX关联评价](../PAMF_FORMAL_ANALYTIC_IDENTITY_R45_20261020.md)：双方全部正式评分完成后才读取保存IQ，112特征单元固定后独立打开truth；0F/checkpoint/新模型预测/训练/分类头拟合或新校准，不改变R44决策。该诊断已完成且全部所属CPU进程退出，1512分组/252条件对照/84配对差已独立复算；正式lowurban combined margin为r38_794529 0.051890、low_rank 0.056720、group_pruned 0.055390。这些是F独立有标签事后关联统计，不是准确率或逐细节TX真实性证明，不改变训练或模型选择。原70%/5pp与逐细节真实性目标保持ACTIVE。下文为历史方法与进度记录，以本段为当前状态。

用户随后进一步要求“探索另外的不同压缩，优化提升”：当前实际下一轮改为 [R44 低秩分解与重要性剪枝](../PAMF_R38_COMPRESSION_METHODS_R44_20261020.md)，完整530161/442805参数，两方法均直接从指定79.45万R38初始化，0TX/F/V/target优化并自动七正式测试。低秩保留通道宽度，剪枝按每个GN4组内的权重重要性选择通道；不追加其他恢复网络。R43仅登记/预检，未发布、未运行，暂存不占GPU；R44已实际RUNNING（按卡接续）：方法 `2f5247dbd5fe50f28defa38a0252d70e692934af`，真实R38/F50/两Ls配对IQ smoke通过，单archiveSHA/compile/controller/CWD/argv/日志增长已核实；按卡等待R41后接续，尚无R44性能结果。其下文计划保留为历史记录，以本段为当前方向。

2026-10-20 用户最新收窄方向：**继续在 794529 参数的 R38 dense_half 基础上减参和优化，不再追加其他恢复网络路线。** 此要求覆盖下文旧的“下一步从原 R17 初始化”及 R41 物理锚点后续研发方向。现有健康 R41/R42 保留按已冻结计划收尾，不扩展；新的 GPU 工作只推进低于 794529 参数的既有压缩结构。父基底是用户在已暴露测试基准上明确指定的工程起点，不称新的独立盲测。

已读回 R13 两档无 F 训练压缩模型的完整七场景正式完成状态：442805/204561 参数，low urban64.3321%/56.4655%，均超过此前每景 1pp 容差，不能交付为性能合格压缩版。固定两档继续做训练优化，仍直接从 R38 dense_half 初始化和蒸馏，不继承 R13/R41 学生；完整 E20、配对 IQ 与 10→1 衰减教师约束、0 TX/F/V/target 优化，每版自动七正式预测及独立 truth-last。[R43 最小方案](../PAMF_R38_COMPRESSION_REFINE_R43_20261020.md)。原 R17 容差仍保留，并逐场景报告相对 79.45 万版的变化；尚未产生本轮训练结果。

状态：ACTIVE。2026-10-20 用户将后续研发目标改为前面最优版本的压缩精简，随后确认“压缩的同时也要优化”。用户已确认正式测试每场景相对原 R17 最多下降 1 个百分点。此为允许的最坏退化，不是主动追求的结果；研究目标是在完整网络更小、推理成本更低的同时获得更好的测试识别表现和输入 TX 保真证据。

## 固定原版和测试口径

原版固定为 `20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17/unroll_unshared`，方法 commit `cd97744884b5e460926e48c8b7efb7d78c6da0f4`，原 source-only 选定最终 E20/4940。使用原物理划分 L6300/U56700/V、原 F50 seed2026092701 和联合训练默认头；不以新目标分数选择父模型、调参或选择性重跑。原 `unroll_shared` 只作为已有共享参数对照；不把它按目标排名重新选成训练来源。

每个新版本均正式测试原 R05 固定 168000 个物理样本的 clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。源域用于合法冻结、诊断及事前设计依据，源域无提升不取消新版测试。每版全部七视图预测固定后，独立 CPU scorer 才连接 truth；报告所有结果、下降和技术缺项。该历史基准已暴露，不声明新的独立盲测。

|场景|原 R17 正式准确率（%）|最多下降 1pp 的下限（%）|
|---|---:|---:|
|clean|84.5869047619|83.5869047619|
|高郊区|81.3190476190|80.3190476190|
|中郊区|79.5779761905|78.5779761905|
|低郊区|77.9964285714|76.9964285714|
|高 urban|80.3535714286|79.3535714286|
|中 urban|73.6851190476|72.6851190476|
|低 urban|67.2166666667|66.2166666667|

实际验收按整数正确数，每场景最多减少 1680 条，含等号；不用舍入后的百分数或七视图均值覆盖超限。报告 accuracy/Macro-F1、逐 TX/RX/day、原版正确丢失/错误修复、原 raw 的 repair/damage、clean 损伤与配对波形/FFT 差距。性能增益须按同一固定 F/seed/场景对照，并注明单 seed 点估计和相关视图的局限。

原 low urban≥70% 与其他五个受损场景距离 raw clean84.67619047619047% 不超过5个百分点，继续单列实际差距。通过压缩退化容差不等于这些原识别目标已经完成，也不证明每个恢复细节真实属于输入 TX。

## 已独立核实的完整参数构成

本次严格加载本地原最终 checkpoint 的完整状态并计数；0 IQ、F、TX 标签或 target 读取，0 forward/训练更新。公开计数见 [原版参数汇总](PAMF_UNSUPERVISED_COMPRESSION_PARAMETERS_20261020.json)。

|组成|原 unshared 参数|已有 shared 参数|
|---|---:|---:|
|coarse|1305958|1305958|
|action estimator|851589|851589|
|prior 唯一参数|2507526|835842|
|完整恢复器|4665073|2993389|

coarse/action 合计2157547参数，占原版46.25%，占共享版72.08%。因此只压缩prior有明显结构下限，需要同时研究上下文估计网络。共享版本依然执行三次prior及六次CG，不推定其MAC或延迟随参数等比例降低。参数字节仅表示唯一参数张量，完整部署还另报 buffers/序列化开销；不把删除训练optimizer或重复teacher文件称为网络压缩。下游原F成本单列。

## 后续研究和无监督边界

优先利用既有结构实现做完整链压缩，覆盖粗恢复、信道估计和三步先验；候选从原 R17 合规源教师派生，独立新 root，不继承其他 owner 已暴露目标的学生权重。第一轮工程方向是将完整恢复器压到约200万参数以下，并在同一数据/训练预算下优化恢复质量；这是待验证方向，不是已达到的交付或准确率承诺。

同时研究可训练的紧凑物理估计和轻量先验，使用原合法模拟配对/当前 IQ 的物理一致性、clean 重建及原 R17 IQ 函数约束。蒸馏只提供源 IQ 参照与稳定初始化，不能把“更像教师”当成性能优化成功；最终收益必须由正式测试证明。是否联合优化压缩的coarse/action、具体损失、结构/秩、预算和矩阵在新 run 发布前固定，不因 R37 或其他目标分数调整。

保持 TX 无标签模拟配对自监督：L_s/U_s 都参与，恢复器优化不读 TX 真值、训练好的 F 权重/输出、V 或 target。原 R17 IQ 教师冻结；原 F/defaulthead 仅用于冻结评测及合法无 query smoke。合法 clean 配对参与自监督须如实披露，不称不使用 clean 的盲去噪。推理只用当前 `[B,2,256]` IQ，输出一份恢复 IQ，由原 PAMF 统一计算 FFT、PA、时域及统计输入。

已有另一路 R17 精简工作覆盖参数共享/低秩/上下文压缩，部分使用源监督 F 蒸馏且已独立登记。其结果可作为明确注明训练权限差异的参考，不静默继承 F 蒸馏学生，不重复启动对方健康任务，也不据其目标分数挑选本路线结构。相关证据入口：[原 R17](../PAMF_UNROLLED_RESTORE_R17_20261020.md)、[既有精简路线](R17_COMPACT_NO_REGRESSION_20261020.md)。

## 当前运行与交付

R37 已完成此前固定 R32 两版七场景正式测试、独立评分、成本及报告，所有所属进程自然退出；实际 low urban 为控制67.2208%/响应约束66.9917%，源方向响应改善没有转化为该正式场景收益。其结果保留，不作为压缩父模型或结构依据。[完整结果](../../automation_reports/CV-SincNet/20261020-evaluation-restoration-R32-recovery-all-versions-F50-s2026092701-m2-r37/report.md)。

本路线已实际发布 [R38 两档完整链压缩与性能联合优化](../PAMF_IQ_JOINT_COMPACT_R38_20261020.md)：1709713/794529参数，全部coarse/action/prior联合优化；paired-clean IQ NMSE加衰减原R17 IQ教师约束，0TX/F/V/target训练。真实权重无query smoke和所属GPU0/1进程读回通过，固定E20后各自自动七场景正式测试，不以源成绩取消。R38宽度与目标在读取R37完整正式分数前固定；实际正式七场景已完成：79万版七场景均提高，六受损平均+1.1111pp；两版均满足每场景最多下降1pp。batch1延迟基本不变或小幅下降，79万版forward峰值显存反而更高，不宣称同比例加速/省显存。

另已发布 [R39固定F独立解析源诊断](../PAMF_COMPACT_ANALYTIC_IDENTITY_R39_20261020.md)，CPU2线程、0GPU，复用R34坐标/原LU校准，已在R38两版源预测冻结后完成跨RX/day的同TX关联；1260组/252条件对照/84比较均独立复算，lowurban combined源margin两版均低于原R17，0神经前向/F/target/新校准。该诊断不筛模型，也不替代R38正式测试或逐细节真实性要求。

已启动实验不热改。每GPU最多一条本任务训练/预测worker、总计最多两条，并遵守共享容量；不修改其他owner。R39与R38共用score.lock控制本任务CPU评分并发。两版完整结果和CPU独立加载已验证，所有所属GPU/CPU进程均退出，最终权重已取回。原70%/5pp和更强TX细节真实性尚未完成；79万版逐场景全部提高但六视图联合全对样本较原版少1165条，作为补充事实保留，不新增验收门槛。

最终交付须同时包含：可加载的独立恢复器、完整参数/模型大小、同口径实测延迟和显存、七场景正式性能与相对原版差值、全部失败与身份保真证据边界。成本和性能至少形成一个明确的有效取舍，不能仅用“更小”宣布联合优化达成。本轮已交付两版完整模型和全部正式数据；79万版达到本次单seed的压缩及逐场景识别改善，但不能据此完成原性能与TX细节真实性总目标。后续继续依据合法源域证据设计，不用这些正式分数挑选新父权重、超参数或选择性重跑，目标继续active。


R40 已完成固定原 R17 与 R38 两档的源域接收链诊断：CPU2/0GPU，37800 次 context-only 估计，所有21单元固定后独立 NumPy oracle 评分；189个RX/day/场景汇总与2079项统计独立核实，0TX/F/target/拟合/新网络，所属进程均退出。低仰角 urban coarse NMSE 原版/171万/79万为0.074163/0.151927/0.279164，四探针算子误差为0.047689/0.106442/0.182144。中间物理量偏离不能直接推断最终性能，但提供源域物理约束的可检验依据。下一步仍从原R17独立初始化，不从已暴露目标成绩挑学生；固定损失与矩阵后再训练，并按每新版七正式测试报告。原70%/5pp及更强输入TX细节真实性目标继续ACTIVE。[R40全部源结果](../../automation_reports/CV-SincNet/20261020-diagnostic-restoration-compact-physics-s2026092701-m3-r40/report.md)。


R41 已完整收尾：两行E20/4940、同168000物理记录×七正式场景、独立truth-last和所属进程自然退出均VERIFIED。原始预测独立复算210组源分类/252组正式分类，40轮完整训练和126源物理记录/1386统计均核实；两权重已取回并从原方法快照CPU严格独立加载。相对原R17，两版都通过每景最多下降1pp；171万版七景均提高，79万版4升3降。但79万版相对用户指定同宽R38七景均下降，lowurban67.8476%相对68.2696%为−0.4220pp。原70%/5pp和逐细节TX真实性仍未达成；物理路线不再扩展，R44低秩/重要性剪枝继续从指定R38派生，不继承R41学生。[R41完整正式结果](../../automation_reports/CV-SincNet/20261020-phase1-restoration-IQ-physics-compact-s2026092701-m2-r41/report.md)。


R42 已完成固定无F解析同TX关联源诊断，VERIFIED/ANALYZED，最终结果commit `46aaaf5cca30a23201396158c81513240920bae2` 已推送并独立核实远端OID。复用原R34/R39固定LU8192校准，112特征单元在读V标签前全固定，1260组/252条件对照/84配对比较均独立复算，原R17参考420组保持一致；0神经前向/F/target/新拟合/新GPU，三个所属CPU进程均自然退出。lowurban combined margin 原R17/171万R41/79万R41 为0.029034/0.025525/0.022017，属于有限源关联证据，不是识别准确率或逐细节TX真实性证明。R41已完成两行E20及七正式测试并退出；不扩展其同参数物理路线，下一轮保持当前R44低秩/重要性剪枝压缩。R42不筛选或更改任何训练/测试。[R42完整源诊断](../../automation_reports/CV-SincNet/20261020-diagnostic-restoration-physics-compact-analytic-TX-s2026092701-m2-r42/report.md)。

R44实际首轮记录：GPU实际训练VERIFIED：group_pruned (GPU1,E1)；实际PID/CWD/argv/commit/GPU和首轮曝光/日志增长核实，0TX/F训练。其余行按既定队列接续，无正式识别结论。

R44实际首轮记录：GPU实际训练VERIFIED：low_rank (GPU0,E1)；实际PID/CWD/argv/commit/GPU和首轮曝光/日志增长核实，0TX/F训练。其余行按既定队列接续，无正式识别结论。
