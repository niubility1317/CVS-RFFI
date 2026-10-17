# 完整源普通星地拼接：E20 固定参照

## 完整矩阵结论

状态：ANALYZED（3/3模型训练、固定预测、独立评分和同row分析完成）。完整联合研究仍ACTIVE；没有新增Ag，没有跨seed证明，也未达到联合目标。沿用已经暴露的既有测试基准，不称新的独立盲测。

普通拼接在本seed提供了真实practical适应：相对匹配4前向REPEAT，六个practical准确率全部提高，midurban＋25.5982pp、lowurban＋20.3762pp；但clean−2.3815pp。原NATIVE对照下clean−2.8042pp。不能把源统计、RDrop或星地增量相加预测联合成绩。

|视图|NATIVE准确率%|REPEAT准确率%|CONCAT准确率%|CONCAT−REPEAT pp|CONCAT宏F1%|CONCAT最差RX准确率%|
|---|---:|---:|---:|---:|---:|---:|
|clean|84.9929|84.5702|82.1887|-2.3815|81.6662|71.4167|
|practical_high|44.6673|46.9738|80.7595|33.7857|80.2817|72.3750|
|practical_mid|42.8851|44.9905|78.3054|33.3149|77.8454|70.7333|
|practical_low_suburban|41.1238|42.7732|75.8929|33.1196|75.4704|68.9417|
|practical_high_urban|43.2857|45.5464|77.3018|31.7554|76.8176|69.9208|
|practical_mid_urban|36.3369|37.7667|63.3649|25.5982|62.9185|57.5208|
|practical_low_urban|31.1214|32.0125|52.3887|20.3762|51.8882|47.7875|

CONCAT的三个suburban与自身clean差依次为1.4292、3.8833、6.2958pp，只有high满足预登记2pp探索目标；lowurban52.3887%未超过60%，clean也没有净正收益。普通拼接作为参照的As价值在这一固定seed得到支持，当前配方的联合达标假设被结果否定。一次seed不建立稳定性结论，未自动扩种子／续训／改系数。

### 逐包身份收益与代价

|比较|视图|修复错误|新增错误|准确率净变化pp|clean且该视图同时正确净变化pp|宏F1净变化pp|
|---|---|---:|---:|---:|---:|---:|
|CONCAT-REPEAT|clean|3782|7783|-2.3815|-2.3815|-2.5932|
|CONCAT-REPEAT|practical_high|62128|5368|33.7857|32.5244|37.6861|
|CONCAT-REPEAT|practical_mid|61727|5758|33.3149|31.8083|36.8087|
|CONCAT-REPEAT|practical_low_suburban|61830|6189|33.1196|31.3530|36.2605|
|CONCAT-REPEAT|practical_high_urban|59474|6125|31.7554|30.4815|35.3216|
|CONCAT-REPEAT|practical_mid_urban|51614|8609|25.5982|23.9667|27.9032|
|CONCAT-REPEAT|practical_low_urban|44818|10586|20.3762|18.7875|21.6928|
|CONCAT-NATIVE|clean|3697|8408|-2.8042|-2.8042|-3.0855|
|CONCAT-NATIVE|practical_high|64989|4354|36.0923|34.1417|39.0440|
|CONCAT-NATIVE|practical_mid|64279|4773|35.4202|33.1964|38.1231|
|CONCAT-NATIVE|practical_low_suburban|63576|5164|34.7690|32.3810|37.3087|
|CONCAT-NATIVE|practical_high_urban|62303|5156|34.0161|32.0560|36.6861|
|CONCAT-NATIVE|practical_mid_urban|53433|8026|27.0280|24.8827|28.8893|
|CONCAT-NATIVE|practical_low_urban|46078|10349|21.2673|19.2667|22.3464|
|REPEAT-NATIVE|clean|1458|2168|-0.4226|-0.4226|-0.4923|
|REPEAT-NATIVE|practical_high|8968|5093|2.3065|1.6173|1.3579|
|REPEAT-NATIVE|practical_mid|8679|5142|2.1054|1.3881|1.3145|
|REPEAT-NATIVE|practical_low_suburban|8052|5281|1.6494|1.0280|1.0482|
|REPEAT-NATIVE|practical_high_urban|8821|5023|2.2607|1.5744|1.3645|
|REPEAT-NATIVE|practical_mid_urban|7793|5391|1.4298|0.9161|0.9861|
|REPEAT-NATIVE|practical_low_urban|7153|5656|0.8911|0.4792|0.6535|

CONCAT相对REPEAT在lowurban修复44818个错误、新增10586个错误；clean修复3782个错误却新增7783个错误，净增4001错。clean与lowurban同时正确比例仍净增18.7875pp；两模型共同clean正确的134295包中，lowurban修复39449包、新增8004错。这说明有实际星地识别改善，不能据此否认clean代价或冒称联合成功。每项差值均来自同一168000物理ID、固定预测的逐包比较。

### 完整源数据、曲线和实际成本

所有行都覆盖原5RX×3day、全部6300 L与56700 U唯一物理ID；没有留出训练域或以5580/720替代。三行完整5994条L/U批次序列逐条一致，各20轮、4440条step/reference/Sinc-unit记录连续完整；66项实际共有增强／U／预算字段无差异，REPEAT与CONCAT共有ordinary的逐轮输入MSE均值也一致。各自L物理暴露568320、U396900，源V27000只作健康；没有非有限更新、U真值可用、traceback、OOM、Killed或自动重试标记。sourceV最后分别98.0778%、98.0593%、97.7296%，不作为目标调参或候选排序依据。

|实际成本|NATIVE|REPEAT|CONCAT|
|---|---:|---:|---:|
|从零model-epoch|20|20|20|
|实际optimizer更新|4440|4440|4440|
|L学生前向包|1136640|2273280|2273280|
|学生训练总前向包（含原U）|1533540|2670180|2670180|
|学生源评估／smoke前向包|594002|594002|594002|
|教师eval前向包|396900|396900|396900|
|教师train前向包|0|0|0|
|As实际生成包|0|0|568320|
|固定正式预测决策|1176000|1176000|1176000|
|正式预测前向（含2包smoke）|1176002|1176002|1176002|
|训练墙钟秒|1021.32|1200.99|2171.68|
|预测墙钟秒|41.66|43.77|42.80|
|独立CPU评分秒|3.53|3.38|3.35|

合计60个新model-epoch、13320次真实更新、3528000个正式决策；0父模型初始化或续训。教师计数已扣除deepcopy继承偏移，教师0训练模式前向；原U规则与模型参数量保持。匹配E50 CP只作兼容性smoke（1读／2合成NN），未进入训练／teacher初始化。r02前置合成检查1292 NN／158物理合成包／3合成optimizer步骤单列；r01仅测试桩失败，其同量合成成本为按已到达路径推算，真实IQ／训练／target均0，不冒充测量或科学结果。全部原始私有产物保留。

CONCAT完整As仅施加一次原practical(raw)，无ordinary前置损伤、无置信度筛除／丢包、无新U视图；15个RX/day/eq会话，90个TX×RX×训练scene组全部有监督。实际scene包量：practical_low_suburban=189631，practical_low_urban=189260，practical_mid_urban=189429；各组最少6130包。模型固定默认cosine头联合训练，未另训诊断分类器，NoExplicitCFO与内部逐包补偿保留，正式received IQ与六场景不变。

### 图表、分层结果和声明边界

- [三组匹配比较图](../../../local_artifacts/F_full_source_concat_20261017/figures/MATRIX_v2/matched_matrix.png)／[PDF](../../../local_artifacts/F_full_source_concat_20261017/figures/MATRIX_v2/matched_matrix.pdf)
- [NATIVE全20轮与七视图](../../../local_artifacts/F_full_source_concat_20261017/figures/NATIVE/curves_and_seven_views.png)
- [REPEAT全20轮与七视图](../../../local_artifacts/F_full_source_concat_20261017/figures/REPEAT/curves_and_seven_views.png)
- [CONCAT全20轮与七视图](../../../local_artifacts/F_full_source_concat_20261017/figures/CONCAT/curves_and_seven_views.png)
- [全部RX／day／TX分层结果（378条）](stratified_results.md)
- [共有普通增强实际作用、概率、归一化与边界](ordinary_augmentation_actual.md)

图表从全部20轮与固定评分生成，PNG/PDF独立读回并检查像素；MATRIX v1图例留白问题仅在v2修正，旧图保留，未改变任何数值。离线分析和绘图0新增IQ／checkpoint／原始truth读取、0网络前向和0训练更新。

本轮只完成普通As的完整源参照和额外前向控制。B10/B11尚未建立，Ag独立clean价值与四因子交互项均不可计算；当前模型未达到原始联合目标。历史加法As保护raw路线已在原减少覆盖契约下两seed确认失败，记录保留，不换名当作新路线；源统计／轻IQ旧结果也不当作星地成功。后续机制须依据合法源证据、保留完整训练域并与本普通拼接作直接匹配比较，不能由这些目标分数选择增强幅度、系数、候选或重跑seed。


---


以下保留预登记、启动与两行先完成时的历史过程；当前完成状态以本页“完整矩阵结论”为准。训练不会留出一个源RX。原候选唯一P0/P1审查及路径P1定向复核PASS，r02只修评分测试桩，不重复全审；本地语法与33个配置／协议反例PASS，真实F合成检查安排在发布后、任何真实训练之前执行。

用户于2026-10-17指出近期探索偏离星地识别、部分新方法不如普通拼接，并要求不再减少源训练域。当前唯一问题是：固定F＋跨RX同TX和完整源覆盖下，普通源／practical配对监督的真实clean与星地收益如何。先建立该参照，后续新机制必须直接比较，不把源统计风险下降算作性能。

底座：三路SIG_TIME、完整统一Sinc／CENTER_25K+CENTER_UNITS、真实joint160、默认cosine30头联合训练、NoExplicitCFO及必要内部逐包补偿、原ordinary、RDrop1／LS.01／浅中层CROSS_RX（time_down/t2.pool、T=.2、最大.1、warm5/ramp10）。不新增部署参数、头、对齐损失、课程、门控或U增强。

数据：原完整source RX1/3/4/6/8、day1/2/3、全部6TX，L6300/U56700隐藏TX/V27000，split392005，原physical IDs。无源训练RX留出、无5580/720替代；V仅原source健康诊断，无反传／EMA／状态更新。全部行从零，不加载历史训练父权重。完整source_contract元数据已核对L_s6300/U_s56700/V27000。既有VALIDATED_ONCE数据复用，无新数据验证或额外hash。

|row|有标签目标|每物理L前向|固定L前向包量|完整practical训练包量|
|---|---|---:|---:|---:|
|NATIVE|原RD(ordinary)＋原SupCon|2|1136640|0|
|REPEAT|.5RD(同ordinary对1)＋.5RD(同ordinary对2)＋原SupCon(对1)|4|2273280|0|
|CONCAT|.5RD(ordinary)＋.5RD(practical(raw))＋原SupCon(ordinary)|4|2273280|568320|

所有模型seed2026092701，各scratchE20=13L＋7U、222步/轮、4440更新，固定FINAL20学生默认头。合计3新模型／60model-epoch／13320更新，0续训。REPEAT和CONCAT共有ordinary对权重.5，另一个槽显式替换；不隐瞒native ordinary总权重由1到.5的差异，也不声称保留了native完整clean损失。每对是同一已生成IQ两次dropout，REPEAT不重抽ordinary。原SupCon只作用共有第一对，不扩展到As或U。三行保留原L/U采样，新增视图不改训练集合。

As沿用旧最朴素C1：均匀low_suburban/mid_urban/low_urban，完整residual/post_sync、EQ off/G=1、25MHz、2.462GHz、referenceSNR30、原质量/噪声/传播/补偿/AGC/reflect耦合；硬件按RX/day/eq稳定且不按TX定义。直接处理原规范化IQ，不把ordinary PA/AWGN/multipath预先叠在practical上。每个合法有标签satellite进入监督，没有高置信筛除、删难包或重复完整信道。实际导入、完整Config和普通增强resolved args会保存。正式测试保留原固定2.45GHz观测等配置，不提高SNR或改received IQ。

一份匹配完整源数据契约的固定final50 CROSS_RX checkpoint只用于兼容性no-query smoke（2合成包），不初始化训练或teacher。已核实它从零、无祖先／父CP、sourceV27000、无heldRX、target_contact=false、固定final50；发布首步再执行一次实际strict-load/no-query前向。每新模型native自身scratchCP另有2包smoke，计入实际计数。历史E50训练成本不算新训练，也不参与成绩对照。

比较固定CONCAT−REPEAT（完整practical视图增量）、REPEAT−NATIVE（额外前向效应）、CONCAT−NATIVE（整配方差）。普通拼接没有净星地收益、或只以clean下跌换差距缩小，均不能称本假设成立；负结果完整保留，不扫描配比／seed／RX救援。这里建立参照，不是新Ag或联合成功声明，不根据目标结果晋级或调参。

每行训练完按预定final20冻结，即自动逐行预测原clean＋六种practical：high、mid、low_suburban、high_urban、mid_urban、low_urban；全部168000×7预测与logp保存后，以exec启动独立CUDA为空CPU scorer连接truth，默认头准确率／宏F1／全部CM以第二实现复算。原正式观测manifest已核实VALIDATED_ONCE／168000／六场景齐备；不生成新正式IQ。全部3行都完成，不按前一行分数停止其他训练。报告每RX/day/TX、最差RX、urban均值、成对修复/新错、各suburban绝对值和与自身clean差。suburban探索差≤2pp和lowurban>60%预先量化，均非承诺；单seed不能证明稳定联合收益。

运行ID：20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02。配置/代码身份为实际Git提交（release_commit及resolved_config记录HEAD）；源码experiments/cvs_fullsource_concat20。CWD PROJECT/releases/cvs_F_fullsource_concat_e20_20261017_r02。命令：`CVS-RFFI/bin/python -u experiments/cvs_fullsource_concat20/runner.py`。N607普通账户、CPU2 dispatcher／worker、共享每GPU最多3个总进程（计入其他owner）。唯一owner codex/root/F-fullsource-concat-20261017。一个不可覆盖root PROJECT/runs/20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02，逐行source/prediction/scores和logs。

仅协议越界、角色／输出冲突、非有限更新、实现无法执行或不能生成合法prediction触发系统技术停止；保存失败／已有产物，停止pending，健康旧任务和已启动健康行继续，无自动重试或按分数早停。预期产物见experiment.json，含完整20轮／逐步日志、实际L/U序列、Sinc更新、As覆盖、默认头checkpoint、固定7视图预测／独立scores、实际成本、同row分析图与登记。IQ／权重／原始日志和预测不提交Git。额外真实训练、合成检查3个optimizer步骤、教师、U、target前向及墙钟分别实测，不能把计划数当已执行。完整Ag／As／四因子及交互／跨seed联合目标ACTIVE。


r02 技术恢复：r01在合成评分反例中误拒仓库reference_recipe.json，真实run未创建、所属进程0、真实训练/CP/target0。原release及日志保留。已在本地复现该异常，修正测试桩允许读取唯一仓库配置并用Path.name处理两端路径，六项不完整预测拒绝回归PASS。模型、数据、As、损失、seed、E20均不变；r02新不可变root，不复用r01。


## 实际启动读回

执行commit `b94076b191648103daeb8a9cb1c17bdcc9c0e231`；r02单archive比较、远端编译、真实F合成检查PASS（33配置＋10角色/评分反例、1292合成NN包、158物理合成包、3个合成optimizer步骤），匹配完整源CP的no-query smoke PASS（1CP、2合成NN、0查询），均在真实训练之前完成。

三个worker PID346632／346705／346862（GPU2／3／4）与CPU dispatcher346544均经独立/proc证据核对CWD、argv、start_ticks、GPU及日志增长；全部source_contract角色6300/56700/27000、source_fold_held_rx与training_receivers为空、实际E20/13L7U/seed2026092701、baseline和teacher初始化路径为空。当前观察到NATIVE E3/666、REPEAT E3/666、CONCAT E1/222，均无failure；这是已完成轮端点快照，不是最终训练成本或收益。CONCAT首轮已监督28416个完整practical L包。自动逐行clean＋六practical和独立CPUtruth-last路径已随worker发布，尚未发生正式评分。

r01失败的3个合成optimizer步骤与1292NN／158physics按固定到达路径推算单列；它没有真实训练或target访问。r02真实运行继续原预算，无健康旧任务变动、无延长或改参。


## 运行中全日志解析

新增独立只读汇总工具 `experiments/cvs_fullsource_concat20/analyze_logs.py`，不改变已冻结训练 release。源日志模式不打开 IQ、checkpoint、truth 或目标评分；正式收尾模式等待原矩阵 ARTIFACTS_COMPLETE 后，收集全部既有评分，严格要求每行完整 20 轮和三类 4440 条更新记录。解析全部 stdout、逐步数值、嵌套层位记录、完整 L/U 物理采样序列及实际成本，不用尾部抽样代替全程分析。

首次实时全量解析分别获得 NATIVE 3581、REPEAT 3160、CONCAT 1691 条完整 step 记录；当时 epoch 文件为 16／14／7 轮。已解析记录的更新序号连续，全部非有限更新及 pseudo_truth_available 标记为 0，原源角色数均为 L6300/U56700/V27000；三组共同采样前缀无差异。压缩流仍在写入，因此 partial/EOF 标记如实保留，不能据此认定技术失败或冒充完整训练。该读取不访问目标评分，不形成识别收益结论。

15 项本地合成解析检查 PASS，包含完整结束正例、截断 gzip、缺日志拒绝、嵌套数值、全 stdout 扫描与逐物理 ID 序列比较；0 实际 IQ／CP／NN／更新／target。闭合后的完整曲线、七视图结果与成本分析仍待真实实验完成。


## 已完成的两行正式结果（矩阵未完成）

同一模型 seed2026092701、完整source RX1/3/4/6/8、L6300/U56700/V27000、split392005；每行固定final20学生默认cosine头，168000个物理样本×7个不变视图。以下是固定比较的单seed结果，不用于其他健康行调参、停止、续训或选择性重跑。CONCAT尚未完成，不能提前判断As收益或联合目标。

|视图|NATIVE准确率%|NATIVE宏F1%|REPEAT准确率%|REPEAT宏F1%|REPEAT−NATIVE准确率pp|
|---|---:|---:|---:|---:|---:|
|clean|84.9929|84.7517|84.5702|84.2593|-0.4226|
|practical_high|44.6673|41.2377|46.9738|42.5956|2.3065|
|practical_mid|42.8851|39.7222|44.9905|41.0367|2.1054|
|practical_low_suburban|41.1238|38.1617|42.7732|39.2099|1.6494|
|practical_high_urban|43.2857|40.1315|45.5464|41.4960|2.2607|
|practical_mid_urban|36.3369|34.0291|37.7667|35.0153|1.4298|
|practical_low_urban|31.1214|29.5418|32.0125|30.1954|0.8911|

NATIVE的三个suburban与自身clean差为40.3256／42.1077／43.8690 pp；REPEAT对应为37.5964／39.5798／41.7970 pp。两行均未达到lowurban>60%或suburban差≤2pp的探索目标。REPEAT仍然没有新增IQ视图或As；其lowurban增量不能算作星地增强证据。

逐包固定预测比较：

|视图|REPEAT修复NATIVE错误|REPEAT新增错误|clean且该视图同时正确的净变化pp|
|---|---:|---:|---:|
|clean|1458|2168|-0.4226|
|practical_high|8968|5093|1.6173|
|practical_mid|8679|5142|1.3881|
|practical_low_suburban|8052|5281|1.0280|
|practical_high_urban|8821|5023|1.5744|
|practical_mid_urban|7793|5391|0.9161|
|practical_low_urban|7153|5656|0.4792|

完整日志已逐条解析：两行各20个epoch、4440条step／reference／Sinc-unit记录，5994条L/U批序列；各自覆盖全部6300个L和56700个U唯一物理样本，累计L物理暴露568320、U396900，所有5RX×3day保留。无非有限更新或U真值可用标记。NATIVE末轮source V为98.0778%，REPEAT为98.0593%，仅作seen-source健康记录，未据此选模。

|成本|NATIVE|REPEAT|
|---|---:|---:|
|实际更新|4440|4440|
|L学生前向包|1136640|2273280|
|学生训练总前向包|1533540|2670180|
|学生源评估／smoke前向包|594002|594002|
|教师评估前向包|396900|396900|
|教师训练模式前向包|0|0|
|正式预测＋2包smoke前向|1176002|1176002|
|训练墙钟秒|1021.32|1200.99|
|预测墙钟秒|41.66|43.77|
|独立CPU评分秒|3.53|3.38|

原始20轮曲线、全部126条分RX/day/TX/overall指标、score派生同包correctness和完整日志保留在各自原run及本地私有收集目录。NATIVE在clean正确的142788包中，lowurban仍正确48839包（34.2039%）；因此仅有高clean并不等于practical适应。上述后处理0新增IQ／checkpoint／原始truth读取、0模型前向、0优化更新。

实际ordinary审计补充：读取了实际DataAugmentation、cvsrffi.schedule.build_aug_base_cfg、configure_augmentor_for_epoch和L调用。标签签名开关false、DAC概率0；PA normal开关true且调用no_pa=false，属于共有旧配方，不能未经独立验证称为标签保持Ag。E1实际scale=.1，E20实际scale=.2506003339；E20 time_shift概率.0877101、最大8点，CFO概率.0877101、最大.0001002401 cycles/sample，AWGN概率.1002401且条件SNR20至36dB，multipath概率.0451081，PA概率.0350840。循环移位按模L索引把切片首尾相接，具有接缝与固定参考窗口改变风险。顺序为DC／taper→shift→gain→phase→CFO→phase-noise→multipath→AWGN→可选PA；class signature关闭不等于PA身份保持已证明。augmentor末端无RMS归一化；后续分支对增益的实际保留仍按对应特征归一化路径解释，不以输入MSE宣称身份收益。RDrop每对复用同一已生成IQ；As从ordinary前原IQ单次生成，不叠加本ordinary噪声／多径／PA。


## 完整曲线与普通增强补充

NATIVE、REPEAT已分别生成并读回PNG和PDF（2340×1475 PNG），完成像素检查，无裁切／标签重叠。图包含全部20轮L目标／普通CE、seen-source健康曲线、七视图准确率／宏F1及三个suburban相对自身clean差；没有增加模型、IQ或truth读取。

- [NATIVE图](../../../local_artifacts/F_full_source_concat_20261017/figures/NATIVE/curves_and_seven_views.png)
- [REPEAT图](../../../local_artifacts/F_full_source_concat_20261017/figures/REPEAT/curves_and_seven_views.png)
- [共有普通增强的实际作用](ordinary_augmentation_actual.md)

增益路径已核实：RMS在ordinary之前，后面没有统一输入RMS把增益全部消掉；参考支路平坦复增益商的局部性质不能外推为整个模型。实际aug日程在E13后保持.2506003339，因为训练明确传min(epoch,label_epochs)。PA虽低概率，Rapp饱和与部分非线性强度并未全部随scale收缩；不能称已验证的轻度身份保持增强。该代码核对不改变正在运行的配方。
