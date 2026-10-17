# 完整源采样率／practical四因子：E20完整结果

状态：ANALYZED。3个新模型全部scratchE20、13320真实更新、3528000次正式决策与独立truth-last评分完成；复用2个固定E20参照，五模型均为同数据、4次L前向、固定final20与默认分类头。所属训练／评分进程已自然退出。

**本轮未取得联合收益。** 仅Ag的clean增加0.119048个百分点、宏F1增加0.123800个百分点，但与As联合后，相对仅As的clean下降0.159524个百分点、lowurban下降0.003571个百分点。不能将这个单种子小幅独立增益称为稳健跨RX泛化成立。B11相对B00的clean净差−1.025000个百分点，lowurban50.596429%，不达联合目标。

普通完整拼接仍是必要直接参照：B11比FULL_AS_REF的clean高1.356548个百分点，但lowurban低1.792262个百分点；B01/B11的As观测与监督剂量均只有完整拼接的一半，这体现了本配方的clean／practical取舍，不能归因于Ag成功。

E30是用户在本轮目标成绩读取前明确授权并固定的预算修订；五条父source-final20与同样追加10轮均已预先登记，不由这些分数选父／筛行。E30实际五行继续运行，不热改、不停止，不据E20结果扫描ppm／损失／seed／场景或延长预算；E30未完成，完整研究目标仍ACTIVE。

## 七场景正式结果

单位：准确率%。每格168000个同一物理样本，单model seed2026092701；不是多种子均值。B00不是完全无增强，保留共有旧ordinary／RDrop／CROSS_RX。

|视图|B00|B10：仅Ag|B01：半As|B11：Ag＋半As|完整As参照|
|---|---:|---:|---:|---:|---:|
|clean|84.570238|84.689286|83.704762|83.545238|82.188690|
|practical_high|46.973810|47.346429|80.073214|80.029167|80.759524|
|practical_mid|44.990476|45.267857|77.255357|77.191667|78.305357|
|practical_low_suburban|42.773214|43.058929|74.574405|74.547024|75.892857|
|practical_high_urban|45.546429|45.788690|76.506548|76.467857|77.301786|
|practical_mid_urban|37.766667|37.973810|61.805357|61.764286|63.364881|
|practical_low_urban|32.012500|32.133929|50.600000|50.596429|52.388690|

## 预登记对比与交互

单位：百分点，正值为左组更高。所有六组对比的完整5994条L/U物理序列逐条相同。

|比较|clean|midurban|lowurban|
|---|---:|---:|---:|
|B10-B00|+0.119048|+0.207143|+0.121429|
|B01-B00|-0.865476|+24.038690|+18.587500|
|B11-B00|-1.025000|+23.997619|+18.583929|
|B11-B10|-1.144048|+23.790476|+18.462500|
|B11-B01|-0.159524|-0.041071|-0.003571|
|B11-FULL_AS_REF|+1.356548|-1.600595|-1.792262|

|视图|A11−A10−A01＋A00|
|---|---:|
|clean|-0.278571|
|practical_high|-0.416667|
|practical_mid|-0.341071|
|practical_low_suburban|-0.313095|
|practical_high_urban|-0.280952|
|practical_mid_urban|-0.248214|
|practical_low_urban|-0.125000|

七场景交互项全部为负。B11相对B01的七场景准确率也全部为负；lowurban只多6个错误，不能解释为有实际星地增量。B10独立clean修复485个、新增285个错误，净200个；B11相对B01的clean修复312个、新增580个，净增加268个错误。

## Suburban绝对值与自身clean差

括号为自身clean减该场景的百分点；预登记探索目标为每场景≤2pp。

|模型|clean|sub high（差）|sub mid（差）|sub low（差）|
|---|---:|---:|---:|---:|
|B00|84.570238|46.973810（37.596429）|44.990476（39.579762）|42.773214（41.797024）|
|B10|84.689286|47.346429（37.342857）|45.267857（39.421429）|43.058929（41.630357）|
|B01|83.704762|80.073214（3.631548）|77.255357（6.449405）|74.574405（9.130357）|
|B11|83.545238|80.029167（3.516071）|77.191667（6.353571）|74.547024（8.998214）|
|FULL_AS_REF|82.188690|80.759524（1.429167）|78.305357（3.883333）|75.892857（6.295833）|

B11三个suburban差均超过2pp。相对B01，B11的suburban差略缩小，但三场景绝对准确率与clean都下降；差距缩小不能作为保护suburban成功。完整拼接只有sub high达到本轮2pp探索目标，mid／low仍失败。

## 同一物理包的修复与新增错误

|比较／视图|修复|新增错误|同包clean与该视图同时正确净差pp|共同clean正确子集内修复／新增|
|---|---:|---:|---:|---:|
|B10-B00／clean|485|285|+0.119048|0／0|
|B10-B00／practical_low_urban|1453|1249|+0.176786|1354／1103|
|B01-B00／clean|3338|4792|-0.865476|0／0|
|B01-B00／practical_low_urban|40350|9123|+17.425595|35889／7174|
|B11-B00／clean|3190|4912|-1.025000|0／0|
|B11-B00／practical_low_urban|40297|9076|+17.411905|35873／7128|
|B11-B01／clean|312|580|-0.159524|0／0|
|B11-B01／practical_low_urban|1128|1134|-0.013690|943／908|
|B11-FULL_AS_REF／clean|4096|1817|+1.356548|0／0|
|B11-FULL_AS_REF／practical_low_urban|6760|9771|-1.375595|5010／7988|

B11相对B01的lowurban虽在共同clean正确子集内净修复35个，但总体lowurban净新增6错，同包clean＋lowurban同时正确也净减少23个（−0.013690pp）。这不支持把错误重新分配称为联合收益。

## 完整训练与实际执行证据

全5个源RX1/3/4/6/8、day1/2/3、L6300/U56700隐藏TX/V27000、split392005；没有留出训练RX或切5580／720。三新模型均无初始化父权重，13Lonly＋7LwithU、4440成功更新，无nonfinite跳步／U真值；默认头与骨干联合训练，正式统一final20，日志中的best_epoch=13不用于正式checkpoint。

已读取新三行所有20轮、每行4440条step／reference／Sinc-unit记录及5994条物理序列，全部闭合无截断。B00／FULL_AS_REF复用已完成的完整日志分析；与每个新行逐条核对序列完全一致。68项实际augmentation／预算／U关键args相同，20轮ordinary输入MSE均逐轮完全一致。每行L物理观测568320，U396900，唯一L6300／U56700；共有SupCon非零3330步／13320层位项。

实际Ag每启用模型284160包，最大79.9999445ppm，最大网格偏移0.010199993点，平均IQ MSE 8.733571e−6；B10／B11完全相同。这证明算子实际执行，不证明保身份或泛化有效。As每个半剂量模型284160包、完整参照568320包，全部合法有标签视图监督，0筛除；原传播／噪声／接收质量／补偿依赖、post_sync／noEQ与正式receivedIQ不变。

共有ordinary对固定权重.5，mixed对.5且按128归一化；G64／S64缺失因子回填同一ordinary。同一pair复用同一生成IQ、四行均4次L前向、无As(Ag)组合。额外槽中的变换使有效未新增强ordinary权重分别为B00=1、B10=.75、B01=.75、B11=.5；完整As参照ordinary=.5。共有原对不变，但不能把这些不同总ordinary剂量掩盖成完全相同的clean暴露。

[旧ordinary逐项实际审计](../20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02/ordinary_augmentation_actual.md)保持；本矩阵共有旧PA／CFO／AWGN／multipath开关没有借Ag名义改动。As走raw-before-ordinary，不再叠一次ordinary噪声或PA。

|模型|源V准确率%（仅健康）|训练秒|学生训练NN|学生源评估含smoke|教师eval NN|L NN|Ag包／As包|
|---|---:|---:|---:|---:|---:|---:|---:|
|B00|98.059259|1200.985|2670180|594002|396900|2273280|0／0|
|B10|98.051852|1204.172|2670180|594002|396900|2273280|284160／0|
|B01|97.885185|1674.995|2670180|594002|396900|2273280|0／284160|
|B11|97.888889|1749.879|2670180|594002|396900|2273280|284160／284160|
|FULL_AS_REF|97.729630|2171.681|2670180|594002|396900|2273280|0／568320|

源V在97.73%至98.06%，只作健康信息；B10源V相对B00还低0.007407pp，不以源V小数或辅助损失支持DG成功。所有教师train-mode前向为0，eval计数已扣复制时继承偏移。每行正式预测NN1176002（含2包自身smoke），正式决策1176000；本轮新增3行60modelE／13320更新／3528000正式决策，复用两参照40modelE与2352000原决策另计。新三行学生训练NN8010540、源评估NN1782006、教师eval NN1190700、正式预测NN3528006；另有本run发布合成2576NN／4次合成更新、真实CP兼容性2包，分析／绘图0新NN、0新训练。

直接从既有混淆矩阵重计全部630条总体／RX／TX／day指标，准确率与宏F1一致；六组配对差与修复−新增计数一致。没有重新连接raw truth或运行模型。[全部630条分层指标](stratified_results.md)。

## 图与当前决定

[五模型／交互／错误矩阵图](../../../local_artifacts/F_clock_factorial_20261017/figures/matrix/matched_factorial.png)；同目录保留PNG／PDF及五行20轮完整曲线。图已检查布局；不把曲线形状或源健康作为正式身份收益。

E20结果支持As有明显practical适应，但有clean代价；只支持Ag的单seed小幅独立clean差，否定其在本E20组合中的星地增量。联合目标未达成。按用户已固定的新预算完成五行E20＋10后，再按同一预登记阈值判定E30；不增加损失／分支／诊断头，不扩U机制，不组合As(Ag)，不扫描幅度或选择性重跑。

## 原预登记与发布证据

唯一假设：轻度、包内非恒定的采样速率观测变化，能在不新增部署结构／分类头／损失的情况下改善跨RX身份提取；与真实practical联合后保留其收益。它不是已失败的±.125点固定分数延时，也不是CFO或常相位旋转；没有依据声称它已经保身份或已经改善星地。

物理依据：仓库实际DataAugmentation的sample_dg_env_params包含5/25/80/150ppm采样速率范围；本候选一次性固定±80ppm，不拟合源RX或目标。GNU Radio原始信道模型也将采样速率差表示为接近1的步进插值器，而将CFO单列：[原始数据生成论文](https://pubs.gnuradio.org/index.php/grcon/article/download/11/10/23)、[官方动态信道说明](https://wiki.gnuradio.org/index.php/Dynamic_Channel_Model)。这些依据支持算子类型，不证明80ppm是本数据的实测接收机误差，更不证明RFFI收益。

Ag：对已生成共有ordinary的G槽执行u[n]=n+epsilon*(n−127.5)，epsilon均匀于±80ppm，每个物理包／epoch／step独立命名流，不使用TX标签。Lanczos半宽16，反射延拓，无循环接缝、无输出RMS、无新噪声／PA／CFO。最大几何偏移.0102点；不是改变包起点的恒定延时。数字变换不等同替换真实RX。As仍直接处理ordinary之前原规范化IQ，沿用旧完整residual/post_sync／noEQ／Fs25MHz／fc2.462GHz／SNRref30／reflect及质量、噪声、补偿依赖。只生成一次，不串接Ag与As；正式IQ和六场景保持原2.45GHz等既定配置。

底座F＋CROSS_RX，三路SIG_TIME、统一Sinc/CENTER_25K+CENTER_UNITS、joint160、默认cos30头正常联合训练、NoExplicitCFO与内部补偿。原ordinary、RDrop1、LS.01、浅中层SupCon仅共有第一对，原oldgate/ownEMA/cos及13Lonly+7LwithU规则保持。全部5源RX1/3/4/6/8、day1/2/3、L6300/U56700隐藏TX/V27000，split392005，不留训练RX，不切5580/720，V不训练或更新EMA。新模型全部scratch20，不以任何旧权重初始化。

每批L128先生成一次ordinary并做RDrop对。第二个mixed128视图按(batch_position+global_step)%2固定为G64/S64；不存在的因子回填同一ordinary。两个对各权重.5、各自按128物理包平均；共有ordinary监督.5稳定，总L尺度1。每个pair两次前向复用同一IQ，所有模型4次L前向，无新U／教师视图。标签、质量、置信度不控制分配，所有生成As都监督。

|行|G槽64|S槽64|实际新预算|
|---|---|---|---|
|B00|ordinary|ordinary|复用固定REPEAT；先以实际F合成loss/gradient/BN状态等价检查确认|
|B10|Ag(ordinary)|ordinary|scratchE20/4440|
|B01|ordinary|As(raw)|scratchE20/4440|
|B11|Ag(ordinary)|As(raw)|scratchE20/4440|
|FULL_AS_REF|旧ordinary128与practical128完整配对|复用固定CONCAT|0新训练，独立保留全剂量参照|

B01/B11每模型As284160包，为全As参照568320的一半；B10/B11每模型Ag284160包。不能把B11−B01的正差当作超越全As。B00和FULL_AS_REF来源为20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02，固定final20且未据目标选模；仅复用已冻结结果，不加载为训练父权重。匹配真实CP只作兼容性no-query smoke（2合成包），随后丢弃，3新模型从零。

固定model seed2026092701。3新模型×20E=60modelE、13320更新，每模型2273280 L前向；2个既有20E参照的历史40modelE单列，不冒充零累计成本。U、教师真实前向、Ag/As生成、评估和smoke另计。每row固定final20学生默认头后立即自动预测原clean＋六practical，每视图168000；同row全部7视图和logp固定后exec独立CUDA为空CPU scorer连接truth，复用已有CM/宏F1第二实现重计。固定所有row，不根据前一row分数停其他行。

比较B10−B00、B01−B00、B11−B00、B11−B10、B11−B01、B11−FULL_AS_REF；逐视图报告交互A11−A10−A01+A00、准确率／宏F1／最差RX、RX/day/TX分层、成对修复/新增错误、各suburban绝对值与自身clean差。单seed不能完成稳定联合目标。

预登记弃用条件：B10不能提供clean正增益或宏F1下降，或B11不能提供相对B00的clean正增益、lowurban>60%、三个suburban各差≤2pp、lowurban同时高于B01和FULL_AS_REF，则结束此固定时钟／分槽配方。记录所有负结果，不扫描ppm、系数、RX、seed、场景概率或追加E救援。源V只作原健康与机制证据，不据目标选候选或追逐源V微小涨分。指标不触发中途停止健康训练；系统技术停止只限协议／角色越界、非有限更新、输出冲突、无法执行或不能合法预测。保留产物、健康行和旧任务，无自动重试。

运行20261017-phase1-F-crossRX-clock-factorial-e20-s2026092701-m3-r01；代码experiments/cvs_clock_factorial20，实际Git commit发布时写入release_commit/resolved_config；本报告不额外要求设计或文件SHA。命令CVS-RFFI/bin/python -u experiments/cvs_clock_factorial20/runner.py，CWD PROJECT/releases/cvs_F_clock_factorial_e20_20261017_r01。N607普通账号、每worker CPU2、共享每GPU最多3总进程计入其他owner。唯一owner codex/root/F-clock-factorial-20261017；不可覆盖root PROJECT/runs/20261017-phase1-F-crossRX-clock-factorial-e20-s2026092701-m3-r01。实际code/config/source日志、最终CP、全部预测／scores、clock与As覆盖、全程曲线和交互分析保存在同run。期望产物与逐行字段以experiment.json为准。

本地与审查证据：48项配置越权负例通过；四行均为5个完整源RX、L6300/U56700，4次L前向；本地未读取IQ/CP或执行网络。唯一独立P0/P1审查为PASS，无阻断缺陷，不追加重复审查。预检查首次因PowerShell远程文本CRLF而FAILED；同一只读脚本改为LF的私有副本后VERIFIED，账号／项目／时间／8块3090可见、当时显存各1MiB。未修改公共脚本或旧任务。

实际发布：单归档传输比较／远端编译VERIFIED；实际F合成57项负例PASS、2576合成NN、128合成物理包、132采样时钟包、4次合成更新、0真实IQ/CP/target；B00与既有REPEAT的loss／全部gradient／BN状态等价。一次匹配真实CP no-query smoke PASS（2合成包、0查询／更新，丢弃CP），随后3行scratch。独立相隔62.85s核实dispatcher与3worker的PID/CWD/argv/GPU和各自step日志增长，GPU0/1/2各1个本run任务，无失败。启动证据见execution_evidence.json；不把检查通过称为Ag／As收益。
