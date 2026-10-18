# 固定128个对比位置的practical训练：E30最小对照

状态：ANALYZED。运行 20261018-phase1-FcrossRX-mixedContrast-e30-s2026092701-m2-r01，唯一owner codex/root/practical-mixedContrast30。代码、配置、分析提交 6849fbf648895c055398e6db159844dba0de3187；两条新E30轨迹及四逻辑条件×七视图分析全部完成，无技术缺项。固定源规则失败，配方关闭，不扩E50。正式clean下降.665476pp，未满足.1pp阶段要求。详见[完整结果](final_analysis.md)及[机器可读汇总](results.json)。

唯一假设：在既有CROSS_RX SupCon中，用已有mixed128 maps替换ordinary128 maps，保持原L batch的128个位置各一次，不人工追加selected64 positions，能否改善完整practical的low／mid urban，而不增加clean代价。原sampler若已有重复物理ID保持不变，不新增去重／重采样要求。P41 ORD128／DUP192／SAT192已有合法source负结果：SAT−ORD两draw×midlow为−.389／−.389／+.278／+.333pp，旧预定晋级失败。它使用192 anchors／4L，本次关键差异是固定128 positions、当前P43六L及既有raw监督；不能把广义As进入SupCon说成未试或预期必然有效。加性As、cleanretain37与HQ的历史负结果及契约差异继续保留，不借本次重启这些路线。

矩阵：仅新训B00/B01固定modelseed2026092701，各自严格加载原P43 ownFINAL20 student及ownEMA，新10轮／2220更新，总E30／6660，resetAdamW lr2e−4、WD1e−4、cos10、min1e−6。共20新增model-epochs／4440更新，非重训20或连续cos30。原P43的NoAs30／As30两个完整冻结结果和审计复用，形成4逻辑条件NOAS30／AS30／MIXED_NOAS30／MIXED_AS30，不取其它种子或新加权父模型。具体row矩阵、六类seed与路径仅维护在[experiment.json](experiment.json)。

仅SupCon视图成员改变，公式／time_down与t2.pool层位／mean+std描述子／detached batch标准化／L2／tau.2／lambda.1／warm5-ramp10保持。两次RDrop分别用对应的mixed maps，再平均两pass和两层；每个原L位置的真实TX／物理RX标签保持，同TX不同RX正对、异TX负、同TX同RX排除。只使用合法L标签，不把U加入有标签关系，不加teacher到该项，不强制全部feature或joint160等于clean。

原损失为.5RDordinary128＋.5RDmixed128＋.25RDraw128，等价于.5primary＋.25mixedordinary64＋.25selected64＋.25raw128，CE/KL总尺度1.25；不使用已关闭的.125/.375。NoAs selected实际raw.clone64，As selected为完整practical(raw64)。6L／每step768个L样本前向，SupCon复用maps不增加NN。As每step64包，三场low_suburban／midurban／lowurban均匀1/3，全部合法L As监督，不要求学生高置信；所有原ordinary／mask／dropout／U流及ownEMA规则保持，RDrop两次使用同一已生成IQ而不重抽增强。普通增强仍固定自身原20参数epoch13，SupCon／channel累计offset20／4440。

F＋CROSS_RX／SIG_TIME×3／统一Sinc／真实joint160／默认cosine6×160scale30随骨干联合训练，保留NoExplicitCFO与内部逐包补偿，不新增部署分支、头、诊断分类器或冻结骨干换头。原fullL6300／U56700隐藏TX／V27000、5个源RX[1,3,4,6,8]、days[1,2,3]、split392005、IQ256／25MHz／dataset eq1保持，不留训练RX或Ldev。practical接收EQ=false、residual/post_sync、reference_snr30与原source／正式收到IQ保持；dataset eq1不等于practical接收均衡开关。源／训练／正式载频与两个source draw等既有差异分别记录，不改通道或SNR，不删除困难包。

来源：原P43 run20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01 B00/B01-s2026092701/stage20/final_ssdg.pth，各自原20为scratch、4440更新、source roles与本轮完全一致、ownEMA、source-only固定FINAL20。复用已核实来源；本次一次preflight核对当前相同句柄与角色，严格加载前由原parent_payload检查，不用目标结果选择父权重，不回退其他checkpoint。固定P43基础release为cvs_FcrossRX_impulse_factorial_E30_20261018_r01，commit19aedc3604fed9c193237dd92366164178af0983。旧config中的legacyE200引用仅架构／配方来源，不能冒称加载E200训练权重；本轮直接父为own scratch20。

源条件提前固定：newAs−originalAs的fullV clean≥−.1个绝对百分点，V1800两个固定draw的mid／low等权均值≥＋.5pp。无每cell门槛，不把source模型健康或遥测通过当性能收益。任一失败关闭此固定候选，不扩E50／扫描mask或参数／选seed救跑；单seed通过只支持后续研究，不证明正式或重复双提升。本次不自动启动E50。科学正负均自动完成所有新row的七视图正式闭合。

每新row按固定FINAL30冻结，先source读出，再clean＋六practical预测全部完成后独立truth-last评分。正式RX[0,2,5,7,9,10,11]／day[0,1,2,3]／168000每view与既有receivedIQ不变。统一0NN分析在4逻辑行×7固定pred／独立score齐备后才连接truth重算504CM／84分区，报告newAs−matchednewNoAs、newAs−originalAs、newNoAs−originalNoAs及交互，各viewaccuracy／macroF1／最差RX、urban3mean、RX/day/TX、修复／新增错误、suburban绝对值及自身clean差。正式阶段newAs−matchednewNoAs clean≥−.1pp；最终仍需重复clean正收益＋星地提升，单seed或不足.1pp小涨不算完成。目标结果不回流任何候选、系数、seed、epoch或重跑。

发布环境：普通N607既有CVS-RFFI/PyTorch fullFP32、CPU2、无admin／安装。不可覆盖release `PROJECT/releases/cvs_FcrossRX_mixedContrast_E30_20261018_r01`，输出`PROJECT/runs/20261018-phase1-FcrossRX-mixedContrast-e30-s2026092701-m2-r01`，CWD为该release。精确dispatcher命令`CVS-RFFI/python -u experiments/cvs_practical_mixed_contrast30/runner.py`；row命令使用`--mode worker --row {row_id}`，随后同PID独立exec predict／score。共享capacity队列按每GPU所有owner总compute进程≤3且free≥12000MB分配，不动健康任务。一次preflight、单release archive一次本远端SHA比较、remotecompile；launcher首步两个真实own20无query4NN＋两6L1536NN共1540NN，PASS立即训练，不另建smoke许可。发布后核实PID/CWD/argv/GPU/log增长。

停止规则：仅本run的数据／父来源／默认头／数值／实际成本／不可变输出或无法合法预测等系统技术错误；依准确PID/CWD/cmdline/runroot只停止所属故障任务、保留所有产物，不自动重试。不以低准确率、source科学失败或正式不达标停止健康行。无新审批／签名／memberSHA／重复数据验证／报告审查；唯一P0/P1限会让下一次真实运行错误或越权的缺陷。

实际新增更新、完整epoch／step、L/U曝光与mask、学生train/eval、teachertrain/eval、source／formal／smokeNN、wall与继承父成本分列。预期无额外SupCon前向，是否匹配由真实计数报告；旧20及旧E30不重复收费。单seed、已暴露基准与地面practical代理的边界保留，当前不承诺lowurban>60%或双提升。

预登记时已读取引用对话《开发 CVS Phase1 星地增强训练》，其当时研究satellite CE温度T2及等比例减权HALF控制；本项只改existing SupCon的输入maps，CE温度／监督系数不改，不重复该实验。没有向另一任务发送消息或干预其健康进程。

发布前记录：本地实现已冻结，19py／73聚焦负例与两发布脚本AST PASS；analysis4文件／9focused tests PASS；唯一合并P0/P1审查为0／0。未远端／NN／CP／IQ／更新；真实1540NN首步烟测和发布后PID／GPU核实待执行，不能冒称训练已经开始。

实际代码／配置已固定提交`6849fbf648895c055398e6db159844dba0de3187`，远端分支OID已独立核对一致。本次唯一preflight／release archive比较／remotecompile／真实首步烟测按此版本执行。


## 已实际发布并训练

一次普通N607 preflight核实own20与冻结30控制metadata／source roles EXACT_MATCH；release archive单次比较与compile VERIFIED。1540NN真实首步smoke PASS后直接训练。dispatcher1353621（CPU）；B00-s2026092701 worker1353649／GPU0，B01-s2026092701 worker1353729／GPU1。PID／CWD／argv／CUDA与nvidia实际GPU分配／log增长独立VERIFIED，全GPU总进程cap3符合，健康旧任务未动。

实际训练输出source_roles均匹配、target_contact=false、strict own20 EMA，optimizer为空resetAdamW／cos10／lr.0002／WD.0001。首步actual6L、128原位置／128anchors／4mixed map terms、rawpairmaps0、extraSupConNN0、trueAs0或64，未改原sampler。contrast_steps在段末统一保存，当前未落盘不代表缺少应有最终产物；当前没有新性能结论。

0NN分析1355032已核实CWD／argv／emptyCUDA，WAIT_SOURCE／truth未开，将按既定4×7冻结预测和独立评分闭合。分析单archive比较／compile PASS，未重复本地synthetic tests。


## 完成状态与结论

两条新轨迹各完成10个新增epoch／2220更新，总E30／6660更新；全部source及七视图独立truth-last评分完成。4逻辑条件×7视图独立重计数504CM、84分区均核对通过，所有本run进程正常退出。新增20model-epochs／4440updates／8704668NN（含1540smoke），原父20和旧E30控制另列。没有训练RX留出、数据/默认头/测试IQ变更或新增模型前向。

源结果先取回并保存[source关闭决定](source_decision.json)：newAs相对originalAs fullVclean＋.033333pp、midlow均值＋.069444pp，后者未达预登记＋.5pp，故在读取新正式数值前已关闭固定配方，不扩E50、不扫描或换seed救跑。正式科学负结果仍完整交付。

正式n=1，MIXED_AS30相对匹配MIXED_NOAS30的clean−.665476pp、mid／lowurban＋27.692262／＋21.134524pp，clean阶段FAIL；相对原完整AS30，clean−.847619pp，六practical均略低，未获得新增机制收益。原完整As的大幅星地收益不能归给mixed128。本次唯一假设已否定；无重复clean正收益证据，整体研究目标未完成。

完整accuracy／macroF1／最差RX／修复与新增错误／RX-day-TX／交互／suburban差距／成本／限制在[最终分析](final_analysis.md)。科研图：[绝对表现](figures/mixedcontrast_single_seed_absolute.png)、[配对变化与交互](figures/mixedcontrast_single_seed_paired.png)、[clean护栏及F1/最差RX](figures/mixedcontrast_single_seed_clean_and_quality.png)。三图130导出点与原汇总逐值核对，最大误差1.42e−14pp，n=1无SD/CI。

本轮收尾时另一对话转向U教师来源；本路线不重复该实验或干预其任务。当前本侧无已登记新候选，无在训或评分进程；E50授权只用于合法源证据支持的配方。已关闭机制不因正式微差重新启动。
