# F＋跨 RX：注入接收机 session 覆盖的冻结源域反证

run_id：20261017-diagnostic-FcrossRX-RXsession-fitL-s2026092701-m1-r01。ANALYZED／执行VERIFIED／固定理由科学FAIL；全部源预测、独立全量复算和图表已完成。唯一逐行配置见 experiment.json；本次仅一个冻结 ALL 模型，不是新 E20 训练或正式目标测试。

代码审计确认，原 PairedChannel 每包按 epoch/step/physical ID 刷新传播与底层噪声，而 receiver_for_session 始终由 receiver_seed20261016102 与 rx/day/eq 定义。四个 fit RX×三日只有12个注入硬件参数实现；同 session 跨 TX 稳定。既有波形自身未知 TX/RX 效应始终保留，不反演未知 TX。这只是覆盖事实，尚不是低 urban 错误的原因。

唯一假设：已经按这些12个 session 训练的 ALL 默认头，在同样传播与底层噪声下，对原 session 有可观察的优势；如果优势不存在，就没有依据将“注入 session 覆盖不足”作为下一训练方向。它属于 As 的覆盖诊断，不是已验证 IQ 泛化增强 Ag。

## 固定对照与判据

仅用 P15 ALL 的 FINAL20（实际运行 f788309f5f9e27689ec1e98ae99a6bd83237e1f9）：F＋跨 RX 同 TX SupCon、三 SIG_TIME、统一 Sinc、NoExplicitCFO、joint160/defaultcos30。原 scratch E20／4440 联合更新／fit L5040 U45360 V21600、hidden U／ownEMA、FINAL-only、无祖先/目标参与的来源，在加载前与实际 config/init/completion/source_fold/freeze/result 核对。不是 E50/E200 权重，不替换或另训头。

冻结评估只消费父训练已见 fit L5040，RX3/4/6/8、day1/2/3、全部6TX、72格各70。原缓存反序列化 L6300 及未用 V/历史特征，RX1的1260包在选用IQ前排除；U只有opaque角色ID，没有U IQ/真值，V/held/target不参与。实际 source_cache ID 是 opaque 字符串，不能从中解码TX或物理字段；RX/day来自已验证L元数据，eq1沿用现有载体契约。只保存合法可见L的ID→TX/RX/day供独立读者核对，不新建权限链或解封隐藏真值。

|视图|源IQ|传播/底层噪声|注入RX参数|额外训练|
|---|---|---|---|---:|
|clean|原缓存RMS IQ|无新增传播|保留原录音效应|0|
|original|同一raw经原normalize_input|每包每场景一个固定命名实现|原rx/day/eq session|0|
|bank1/2/3|同上|与original完全相同H/底层噪声seed|同分布三个固定virtual bank后缀，跨TX稳定|0|
|original replay控制|同上|原seed重放|原session|无额外NN|

只比较 mid/low urban 两场景。实际入口沿用P15不可变后端的experiments/adv3b02_xuc/code/leo_practical，运行保存__file__/实际全六配置；fs25MHz、fc2.462GHz、IQ256、residual/post_sync、EQoff G=1、reference30、reflect、接收质量/相位跟踪/IQ补偿/AGC保持。四种session共用H/数值边界/质量/锁定状态；RX广义线性处理会相应改变处理后噪声和AGC，不能把共享底层噪声说成输出噪声逐点相同。新增噪声SNR不等于总SNR。不得提高SNR、改测试参数或删难包。

在真实IQ访问前固定，两个场景均必须满足：original准确率比三bank均值至少高1个百分点；original到三bank的平均hard判决变化至少3%；任何单bank比original高不得超过0.5个百分点。全部45360判决／10080物理记录／50400原生process／48稳定session、原重放0MSE、状态/预算/模型冻结检查也必须满足。源判断独立全量复算；不根据某个bank最差分数选新的训练session。

FAIL关闭这个固定session覆盖理由，不换bank、seed、场景、RX或阈值救跑。PASS也只支持另行预登记匹配训练，不能证明Ag、泛化收益或因果分离；没有自动训练。父已见fitL结果不冒充heldRX、V或target表现，三个物理bank不是三个模型seed。

## 执行与产物

0新model-epoch／优化器／教师／EMA／诊断头拟合；1个已有CP读取，launcher第一步strict load后2个合成包no-query smoke，PASS立即继续，合计45362 NN。5040 clean＋5040×两场景×四bank＝45360源判决；每包每场景4次原生process加1次replay，共50400调用。历史ALL训练成本另列E20／4440更新／5823318 NN，不作为本轮训练或继承收益。

普通N607既有CVS-RFFI环境，CPU2线程，CUDA_VISIBLE_DEVICES为空，单进程，不使用GPU训练容量、不装包或干预其他任务。CWD：PROJECT/releases/cvs_FcrossRX_RXsession_fitL_20261017_r01；命令：CVS-RFFI/bin/python -u PROJECT/releases/cvs_FcrossRX_RXsession_fitL_20261017_r01/experiments/cvs_rxsession_probe17/runner.py。精确可执行路径、CWD、run/log根在experiment.json；不可变run根PROJECT/runs/20261017-diagnostic-FcrossRX-RXsession-fitL-s2026092701-m1-r01，日志在release/dispatcher.log，本地证据E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017。唯一owner codex/root/FcrossRX-RXsession-source-20261017。

只因本run角色/来源、非有限、冻结状态、预算、输出碰撞或不可启动等系统故障停止并保留产物，不自动重试；低分不是技术故障。按项目最小流程完成一个Git代码/config提交、一次限定P0/P1、一次普通N607 preflight、一个release归档传输比较与remote compile、真实CP smoke、启动后PID/CWD/argv/CUDA/log增长或自然终态独立核验。

本地最终51个协议反例、6个科学反例、30次原生合成process PASS（三次局部检查累计90次原生合成process），0真实IQ/CP/NN/更新。一次审查发现并修复两个直接错误：四RX source fold实际标记与原carrier不同；fit source_cache采用opaque ID而非held readout的五字段元组。新增合法四RX来源通过、旧标记/污染/角色混用拒绝，以及opaque ID越界/重复缺失回归；未启动真实run补救。本候选只进行原问题定向复核，不重复全审。

产物：实际配置/模块/默认头、合法fitL标签与opaqueID元数据、四bank全部logits/argmax/margin30、完整物理元数据/48session及replay检查、全部40批stdout、独立准确率/F1/CE/配对修复损坏/固定clean正确集保持和TX/RX/day/状态/质量分层、输入变化/成本/图/同row报告登记。独立读者只读取冻结预测与保存元数据，不重新打开IQ或运行网络。IQ、权重、逐包ID/原预测日志、私有机器信息及生成图不提交Git。

REJECTED_EXTRA_GATE：通用skill的额外设计审批、固定spec文件、hash/signature或重复review不在项目八项白名单；当前已有授权内继续最小源反证。模型、分类头、L/U/V和正式clean＋六practical保持。当前无新的识别收益，Ag独立价值、As稳定性、B00/B10/B01/B11交互、正式low>60%、suburban接近clean与跨seed联合目标均未完成，保持ACTIVE。


## 完整源结果与固定判定

实际运行代码31e63c58cfea58cb94cf7259f9ab87941cf59348。原PID199424自然终止，无failure、重启或参数热改。一次普通N607 preflight、一个release归档传输比较、remotecompile与真实匹配CP2合成NN no-query smoke均PASS；独立/proc CWD/argv/CUDA隐藏与日志276→297字节核对通过，完整40批stdout和终态保留。

独立 reader 读取全部45360个六类logits／hard判决与10080个完整物理控制记录，重算argmax、CE、宏F1、margin/30、配对修复与损坏、固定clean正确集、全部192个分层及原判据，与运行输出一致。opaque ID只作键，标签与RX/day来自合法可见L元数据，成员恰等于父source_contract的fitL；没有从opaque ID推导TX，没有读U真值或新IQ。

|场景|RX session|accuracy %|macro F1 %|CE|margin/30|与original判决不同 %|
|---|---|---:|---:|---:|---:|---:|
|practical_mid_urban|original|73.253968|73.114064|0.684837|0.095507|0.000000|
|practical_mid_urban|bank1|73.273810|73.142587|0.685244|0.095463|0.396825|
|practical_mid_urban|bank2|73.273810|73.137854|0.685096|0.095485|0.515873|
|practical_mid_urban|bank3|73.313492|73.175015|0.685166|0.095503|0.436508|
|practical_low_urban|original|57.738095|57.491104|1.058247|0.052842|0.000000|
|practical_low_urban|bank1|57.698413|57.446248|1.058840|0.052803|0.853175|
|practical_low_urban|bank2|57.777778|57.536376|1.057988|0.052849|1.011905|
|practical_low_urban|bank3|57.817460|57.561752|1.058637|0.052804|1.071429|

同一父模型的源fit clean：accuracy 99.702381%、宏F1 99.702310%、CE 0.018757，固定clean正确集5025包。此分数来自训练已见L，不是held RX1或target。模型没有再训练，因此本轮没有新增训练收益或clean代价估计。

|固定量|mid urban|low urban|原条件|
|---|---:|---:|---|
|original准确率−三bank均值，pp|-0.033069|-0.026455|每场景≥+1，均FAIL|
|original到三bank平均判决变化，%|0.449735|0.978836|每场景≥3%，均FAIL|
|任一bank相对original最高增量，pp|0.059524|0.079365|≤+0.5，均PASS|

**结论：FAIL_FIXED_SESSION_COVERAGE_RATIONALE_CLOSED。** original没有显示源session优势，三bank均值反而仅高约0.03pp；输入与判决有实际变化，但远小于预登记的有意义变化条件。不以bank3的微小正差选择新训练session，不调bank、种子、场景、RX或门槛救跑，不启动session-bank E20。该结论只关闭当前有限覆盖理由，不证明所有RX硬件因素无关或普遍不需要覆盖。

![全部冻结源结果](E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/final/figures/frozen_session_accuracy.png)

## 配对错误、输入变化和固定正确集

|场景|bank|修复|新增错误|净正确数|相对IQ RMS变化中位数 %|相对IQ RMS最大值 %|固定clean正确集保持率 %|
|---|---|---:|---:|---:|---:|---:|---:|
|practical_mid_urban|original|0|0|0|0.000000|0.000000|73.452736|
|practical_mid_urban|bank1|6|5|1|0.373150|2.226737|73.472637|
|practical_mid_urban|bank2|8|7|1|0.488919|2.129466|73.472637|
|practical_mid_urban|bank3|7|4|3|0.494199|2.152361|73.512438|
|practical_low_urban|original|0|0|0|0.000000|0.000000|57.870647|
|practical_low_urban|bank1|6|8|-2|0.437531|2.287389|57.830846|
|practical_low_urban|bank2|11|9|2|0.589452|2.158917|57.910448|
|practical_low_urban|bank3|12|8|4|0.551025|2.158873|57.950249|

mid每bank只净增加1至3个正确包，low为−2／+2／+4包；这些是同一冻结模型对输入变化的判决差，不是新模型学到的增量。相对IQ RMS变化中位数mid约0.373至0.494%、low约0.438至0.589%，全包最大2.287389%；不能把扰动确实生效等同泛化有效。

![实际输入与配对判决变化](E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/final/figures/session_perturbation_and_decisions.png)

## low urban分层与可支持机制

|维度|层|n|original %|bank1 %|bank2 %|bank3 %|
|---|---|---:|---:|---:|---:|---:|
|state|BLOCKED|2833|30.356513|30.427109|30.497706|30.568302|
|state|LOS|1480|97.635135|97.500000|97.500000|97.500000|
|state|PARTIAL|727|83.218707|82.943604|83.218707|83.218707|
|lock|unlocked|2225|27.280899|27.415730|27.505618|27.460674|
|lock|locked|1902|96.214511|96.109359|96.109359|96.109359|
|lock|degraded|913|51.807229|51.478642|51.697700|52.026287|
|quality|[0,5)|1495|30.167224|30.234114|30.301003|30.301003|
|quality|>=20|1539|97.530864|97.400910|97.400910|97.400910|
|quality|<0|730|21.369863|21.643836|21.780822|21.643836|
|quality|[10,20)|654|82.415902|82.110092|82.415902|82.415902|
|quality|[5,10)|622|42.282958|42.122186|42.122186|42.604502|

BLOCKED为30.356513%→30.427109/30.497706/30.568302%，unlocked为27.280899%→27.415730/27.505618/27.460674%；LOS约97.5%、locked约96.1%，困难状态与较好状态之间的大差距依旧存在。四session只让BLOCKED净增2／4／6个正确包，不能称为困难urban得到实际训练缓解。这是父已见fit源上的条件观察，不把状态间差异解释为噪声/多径/IQ各自的独立误差份额。

原生处理及独立元数据检查确认，同一包四bank的传播核/能量、几何、数值延迟、边界、质量proxy、锁定状态和残余scale相同，原session replay逐点0MSE。EQ关闭时G=1；注入RX的广义线性主项/镜像项仍处理信号和噪声，补偿参数按新session真实改变，AGC依原过程重算。共享底层噪声不能宣称处理后噪声完全相同；这些是控制解释的边界。未新增内部gate记录或特征对齐，因而不据此归因于门控。

![low urban全部状态与质量](E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/final/figures/low_urban_session_strata.png)

## 各源RX结果与费用

下表RX均在父模型训练中出现；“最差fit RX”不等于最差未见RX。每RX每bank每场景1260包。

|场景|fit RX|original %|bank1 %|bank2 %|bank3 %|
|---|---:|---:|---:|---:|---:|
|practical_mid_urban|3|70.793651|70.793651|70.714286|70.873016|
|practical_mid_urban|4|72.063492|71.904762|71.984127|71.984127|
|practical_mid_urban|6|74.365079|74.603175|74.523810|74.523810|
|practical_mid_urban|8|75.793651|75.793651|75.873016|75.873016|
|practical_low_urban|3|56.587302|56.507937|56.587302|56.825397|
|practical_low_urban|4|55.634921|55.476190|55.396825|55.555556|
|practical_low_urban|6|59.841270|60.000000|60.000000|59.841270|
|practical_low_urban|8|58.888889|58.809524|59.126984|59.047619|

完整TX/day/RX/状态/质量192个分层均保存，全部合法5040包参与，无低质量筛选或困难样本删除。只有一个model seed和三个新硬件bank，不能做跨模型seed统计；也没有正式clean＋六practical的新测试结果。

实际1个CP读取、45362 NN＝5040clean＋5040×2场景×4bank＋2合成smoke；10080个配对物理记录、50400次原生process、12原session＋36新session＝48。0新增epoch/优化器/教师/EMA/头拟合/目标访问，模型参数和buffer进入/结束逐项相同，.grad均为空。父ALL历史成本单独E20／4440更新／5823318 NN；没有续训或扩大预算。

运行主体耗时100.299359秒（包含模型加载/smoke、源物理生成和前向，不含传输、独立分析、绘图与本地准备）。本地三次局部检查累计90个原生合成process、0真实IQ/CP/NN；最终固定版本51协议反例与6科学反例PASS。新离线harvest/plot代码提交edc08179a042d889bc62eface5a9fa630b39f6df，不修改已执行release。实际取回16个固定文件，无IQ或权重，独立分析和制图0NN；3PNG＋3PDF已实际渲染并逐张查看，无裁切。没有新训练，所以没有本轮训练曲线，不用旧曲线充数。

## 文件、命令与未完成项

实现包experiments/cvs_rxsession_probe17：design/physics/source/runner/run/publish为实际31e63c58c运行；analyze独立全记录评分；checks为源/协议/科学反例；harvest/plot为完成后的只读工具。实际运行命令/环境/CWD/输出/GPU/stop rule在experiment.json及上文保留；实际模块__file__见私有actual_modules.json。所有实验代码、配置和登记已纳入Git，不提交IQ/权重/逐包ID或原始预测/生成图。

完整复算命令：F:/App/miniconda3/python.exe -m experiments.cvs_rxsession_probe17.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/final，CWD=E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014。绘图用相同Python执行 E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/plot_local.py，仅发送已完成聚合JSON到已有MPL环境。

证据：[完整独立结果](<E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/final/analysis_summary.json>)、[全部分层](<E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/final/source_strata.json>)、[完整运行日志](<E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/final/dispatcher.log>)、[自然终态](<E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/latest_readback.json>)、[图索引](<E:/type10-7/local_artifacts/cvs_FcrossRX_RXsession_20261017/final/figures/manifest.json>)。完整默认头logits/物理记录/合法L元数据和runtime只在私有本地产物及原不可变run保留，不把来源CPU诊断当成新发布模型。

本轮证实“只有12个原注入session”这一覆盖事实，但否定了其对当前已训默认头产生预定幅度偏差的固定理由。没有证明新增Ag、默认头训练提高、clean代价、跨seed重现或正式低urban超过60%；三个bank不等于三个seed，fit low57.738095%不能冒充未见RX1或target成绩。总体Ag＋As联合目标保持ACTIVE。

下一项已有独立预登记的假设是：复用原浅/中层跨RX同TX对比项，让其覆盖ordinary/As能否保护urban身份而保持clean。它属于As训练组织，不是新Ag；本轮不重复派发该矩阵，也不把尚未核实的执行状态或结果写成完成。新的独立Ag及匹配四格/交互、正式六practical、最差未见RX和多seed仍缺证据。
