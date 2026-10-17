# ordinary／As 跨视图 dropout 相关性：E30 正式结果

**ANALYZED／VERIFIED。** 两行均完成同父E20＋10、累计E30／6660更新，以及七视图默认头预测和独立truth-last评分。执行commit a56e02cada3f11db002115afedafb4c024151348。

固定识别收益条件失败，关闭本配方；不改掩码key／p／父／seed／预算救跑，不扩种子。完整联合目标仍未证明。

## 固定比较

两行 INDEPENDENT／ANTITHETIC，model seed2026092701。都从原全源 ordinaryAs CONCAT-s2026092701 的 scratch fixed-final20 学生及自身 EMA 出发，独立重置同 AdamW lr0.0002／wd0.0001、cos10；追加10轮，每行2220新更新、累计E30／6660更新。不是 scratch30 或精确 optimizer resume，父成本 E20／4440 单列。父来源与全部源物理角色沿用既有合规核对，不继承失败变体或按目标排名换父。

F 三路 SIG_TIME、统一 Sinc CENTER_25K／CENTER_UNITS、joint160、默认 cosine W6×160 与骨干正常联合训练；不新增网络、部署分支、头拟合或损失。NoExplicitCFO 与必要内部补偿保留。全部5源 RX1/3/4/6/8、day1/2/3，L6300／U56700 隐藏 TX／V27000，split392005；没有训练 RX 留出，也不改为5580/720。用户确认无可靠长上下文映射，仅使用现有256点IQ。

原损失仍为 .5 ordinaryRD＋.5 AsRD＋.1共有ordinary浅中层跨RX同TX SupCon；每个L128物理样本共4次身份前向，原生CE correction不变。每对复用同一个已生成IQ；As从raw-before-ordinary走原 complete residual/post_sync/NoEQ，全部合法有标签 As 监督。传播、噪声、接收质量、补偿、采样先验和 formal received IQ 不改。U、ownEMA、旧门控和半监督规则保持，V仅健康监测，fixed-final10冻结。

按 O1、O2、S1、S2 顺序，仅操作四次共有的身份 encoder 中既有 nn.Dropout。INDEPENDENT 四掩码独立；ANTITHETIC 令S1/S2分别使用O1/O2离散随机数的反射值，O1/O2及S1/S2对内仍独立。p不调，有限均匀支持 M=2^24，keep threshold=round((1−p)M)、原1/(1−p)缩放；相对原生理想p的离散误差≤1/(2M)，两行相同。私有seed20261017561只依赖step/module/occurrence，不读IQ、TX、RX或day。两行都抽4份同大小随机张量并计算反射，匹配额外抽样与反射工作；As前向不重抽IQ。原生首次前向中身份网络之外的dropout不改；U／teacher／eval范围外使用原生 dropout。新增可训练参数、NN前向、教师、loss和U规则均0；掩码缓存／抽样成本按实际记录。


## 七视图正式身份识别

|场景|独立掩码准确率 %|反射掩码准确率 %|差 pp|反射掩码宏F1 %|反射掩码最差RX %|
|---|---:|---:|---:|---:|---:|
|clean|82.998214|83.020238|+0.022024|82.609283|74.070833|
|suburban high|81.770238|81.722024|-0.048214|81.293004|73.504167|
|suburban mid|79.483929|79.447024|-0.036905|79.027910|71.966667|
|suburban low|77.074405|77.048214|-0.026190|76.652253|70.050000|
|urban high|78.316667|78.236310|-0.080357|77.785544|70.783333|
|urban mid|64.460714|64.392857|-0.067857|63.954087|58.412500|
|urban low|53.470833|53.414286|-0.056548|52.889392|49.004167|

固定判定：{"low_urban_gain_at_least_0_5pp": false, "clean_not_lower": true, "mid_urban_not_lower": false, "each_suburban_no_more_than_0_25pp_lower": true}。全部14份view／2352000正式决策固定，逐row七view完成后独立CPU评分。接收IQ、测试SNR／信道和样本不改，直接checkpoint默认头。

三个suburban距自身clean的差（pp）：INDEPENDENT 1.227976／3.514286／5.923810；ANTITHETIC 1.298214／3.573214／5.972024。不能把clean下降造成的差距缩小算作联合收益。

同预算B00 clean仅作原已冻结背景：85.268452%。两行clean净收益（pp）分别为-2.270238／-2.248214；完整目标仍要求clean正收益、low>60%、suburban接近clean及新的合法Ag和因子证据，本项训练组织比较不能替代。

## 配对错误与独立复算

|场景|修复|新增错误|差 pp|clean与场景共同正确差 pp|
|---|---:|---:|---:|---:|
|clean|326|289|+0.022024|+0.022024|
|suburban high|379|460|-0.048214|-0.020238|
|suburban mid|458|520|-0.036905|-0.006548|
|suburban low|511|555|-0.026190|-0.009524|
|urban high|430|565|-0.080357|-0.044643|
|urban mid|685|799|-0.067857|-0.057143|
|urban low|855|950|-0.056548|-0.083333|

low与clean共同正确：反射82905／独立83045；共同clean正确139148包中，low修复658／新增803。

252个混淆矩阵／42个分区相加／7个paired差值独立重计一致，最大指标误差2.22e-16。保留[全部分层](stratified_results.md)、[独立复算](result_summary.json)和[完整源日志分析](source_analysis.json)。

## 实际匹配与成本

两行857项实际原生参数仅['candidate_id', 'output_dir']不同；4440条L/U序列逐条一致，原6300／56700物理角色完整，源训练RX无留出。每行2220条参考目标、单位更新和掩码记录完整，重算目标最大误差1.19e-07；每步4L／128As、同IQ不重抽、全As监督、原U隐藏规则与默认头联合训练符合。

|行|L NN（已含于student train）|student train|student eval|teacher train/eval|预测NN|源墙钟 s|预测墙钟 s|评分CPU墙钟 s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|INDEPENDENT|1136640|1703640|324002|0/567000|1176002|1216.244|41.458|3.267|
|ANTITHETIC|1136640|1703640|324002|0/567000|1176002|1234.027|43.524|3.218|

新增包级身份NN前向合计7543210，已计两行训练/teacher/eval/预测、共用真实CP smoke2以及实现期1920；父E20／4440成本复用另列。父训练包级NN为3661082，student train/eval及teacher eval为2670180/594002/396900；同一父只实际复用一次，不把每行累计预算相加当新增训练。实现期384合成As处理和3次失败探针单列，0真实IQ／CP／更新，不是额外训练。每行As284160次、私有mask随机张量115440个／随机元素30116413440个，两行同抽样/反射工作；额外NN=0不等于额外算子开销=0，缓存与随机生成有GPU成本，不用共享资源墙钟差宣称提速。

[完整曲线与固定测试图](figures/dropout_coupling_matched_E30.png)包含父20轮和新增10轮全部源目标，不截取有利epoch。仅显示冻结结果，不回流调参、排序、父／seed或选择性重跑。分析只读既有日志/分数/正确性，0新IQ／CP／rawtruth／NN／更新。

20条旧可选诊断N/A保持null并列出路径；两行全部loss/梯度更新及正式评分有限，未把可选诊断缺项当作非有限训练失败。只验证本固定单seed配方未达条件，不能据此证明所有dropout相关性都无用。旧RDrop、AsPairCE和噪声antithetic结论不改判，本项未新增Ag或组合视图。

这是已暴露固定基准上的持续研究，不声明新的独立盲测；目标不回流幅度、mask key、p、父、选模、晋级、seed或选择性重跑。执行状态见[终态证据](execution_evidence.json)，默认头/源角色与857参数核对见[实际配置核对](actual_config_verification.json)。归档/编译/唯一P0P1检查和Git交付保留。
