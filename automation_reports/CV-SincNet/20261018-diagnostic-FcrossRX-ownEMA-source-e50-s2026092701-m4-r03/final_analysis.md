# 完整 practical E50：固定 own EMA 源域比较负结果

**已完成四条件源域比较；固定 EMA 读出未提供新增收益，关闭本候选。** 仅 seed2026092701，既有 NoAs50／As50 的 student／own EMA，原默认头与全部模型状态联合来源保持。0新训练／EMA更新／头拟合，未访问formal。

fullV27000的AsEMA相对As学生修复5个、增加7个错误，净−2（−0.007407 pp）；mid／low两场合并修复14个、增加25个错误，净−11，等权均值−0.152778 pp。AsEMA−NoAsEMA的fullV clean为−0.066667 pp，在源域0.1pp范围内，但这个匹配保持并不等于EMA改善了原As。三项预登记判定中，匹配clean保持通过、As自身clean保护与urban增量均未通过。

这只是当前固定种子／固定状态的负结果，变化小，不声称所有EMA都无效。不改变EMA衰减、epoch、种子或预算救跑；不扩种子、不进入正式测试。旧E20 clean-only EMA负例保留，本次没有以CE下降代替身份收益。

## 完整源 V clean

|条件|准确率 %|宏F1 %|最差源RX %|正确／27000|
|---|---:|---:|---:|---:|
|NOAS_STUDENT|98.448148|98.447431|95.814815|26581|
|NOAS_EMA|98.433333|98.432582|95.814815|26577|
|AS_STUDENT|98.374074|98.373404|95.592593|26561|
|AS_EMA|98.366667|98.365908|95.574074|26559|

学生NoAs／As命中数与原生fullV记录完全一致，均未出现一包差异。原V角色全部27000个物理ID，源RX1/3/4/6/8、day1/2/3；没有留训练RX或另拆Ldev。独立从120个保存的fullV CM重算命中／样本数／宏F1，16次RX／day／TX／RXday分区合并全部一致；这是保存CM层面的重算，不冒充再次独立逐包模型预测。

## 固定 V1800 面板

clean分母1800；每practical合并两draw分母3600，仍只来自1800个物理样本，不是两个模型种子。完整physical ID、TX／RX／day逐项一致，使用原缓存11个条件的index0 full，其余反事实条件没有进入本次分类读出。

|条件|场景|准确率 %|宏F1 %|最差源RX %|
|---|---|---:|---:|---:|
|NOAS_STUDENT|clean|98.611111|98.609077|96.111111|
|NOAS_STUDENT|practical_low_suburban|49.361111|45.400918|42.500000|
|NOAS_STUDENT|practical_mid_urban|43.666667|40.802522|36.805556|
|NOAS_STUDENT|practical_low_urban|36.166667|34.507057|30.972222|
|NOAS_EMA|clean|98.611111|98.608626|96.111111|
|NOAS_EMA|practical_low_suburban|48.944444|45.029979|42.083333|
|NOAS_EMA|practical_mid_urban|43.583333|40.711229|37.638889|
|NOAS_EMA|practical_low_urban|36.222222|34.582188|30.972222|
|AS_STUDENT|clean|98.444444|98.441556|95.833333|
|AS_STUDENT|practical_low_suburban|90.111111|90.023766|84.305556|
|AS_STUDENT|practical_mid_urban|74.777778|74.660448|70.972222|
|AS_STUDENT|practical_low_urban|60.722222|60.378648|57.222222|
|AS_EMA|clean|98.388889|98.385052|95.833333|
|AS_EMA|practical_low_suburban|90.055556|89.966444|84.305556|
|AS_EMA|practical_mid_urban|74.666667|74.541269|70.833333|
|AS_EMA|practical_low_urban|60.527778|60.162640|57.361111|

|AsEMA−AsStudent|draw|修复|新增错误|净修复|准确率变化 pp|
|---|---:|---:|---:|---:|---:|
|clean|合并|0|1|-1|-0.055556|
|practical_low_suburban|0|2|1|1|0.055556|
|practical_low_suburban|1|0|3|-3|-0.166667|
|practical_low_suburban|合并|2|4|-2|-0.055556|
|practical_mid_urban|0|2|5|-3|-0.166667|
|practical_mid_urban|1|4|5|-1|-0.055556|
|practical_mid_urban|合并|6|10|-4|-0.111111|
|practical_low_urban|0|5|9|-4|-0.222222|
|practical_low_urban|1|3|6|-3|-0.166667|
|practical_low_urban|合并|8|15|-7|-0.194444|

low urban最差源RX从57.222222%升至57.361111%（+0.138889 pp），但整体准确率和宏F1均下降，不能用这一项替代身份识别收益。面板clean−0.055556pp与fullV clean−0.007407pp是不同样本口径，分别保留。源面板没有high urban，不报告源urban三场均值；面板fc2.45GHz／原训练formal2.462GHz的既有差异保留。这里low urban的60.527778%是源面板准确率，不能写成正式测试超过60%。

## 匹配差分与归因

|比较|fullV clean pp|panel clean pp|low suburban pp|mid urban pp|low urban pp|
|---|---:|---:|---:|---:|---:|
|AS_EMA_minus_NOAS_EMA|-0.066667|-0.222222|41.111111|31.083333|24.305556|
|AS_STUDENT_minus_NOAS_STUDENT|-0.074074|-0.166667|40.750000|31.111111|24.555556|
|AS_EMA_minus_AS_STUDENT|-0.007407|-0.055556|-0.055556|-0.111111|-0.194444|
|NOAS_EMA_minus_NOAS_STUDENT|-0.014815|0.000000|-0.416667|-0.083333|0.055556|

EMA交互=(AsEMA−AsStudent)−(NoAsEMA−NoAsStudent)：fullV clean 0.007407pp；面板clean／low suburban／mid／low分别-0.055556／0.361111／-0.027778／-0.250000pp。clean交互的微小正值不是实际clean增益；AsEMA自身clean净损伤2包，urban也未提高，不能声称修复了As训练机制或取得稳定双提升。

## 实际成本、技术修复与边界

成功r03：133204次样本前向，其中fullV学生54000、fullV已有EMA54000、EMA面板25200、真实双CP双状态smoke4；6次CP读取（含原严格父链），explicit V raw27000包、panel clean1800包、固定观测10800包。新增模型epoch／优化／EMA／头更新均0；U迭代／U真值／target IQ／target truth读取均0。模型参数和buffer前后变化0。原两模型累计100 model-epochs成本引用父报告，不重新计入。成功worker实测15.559s，不宣称加速比。

两次技术失败全部保留：r01把opaqueID误与native辅助坐标比较，r02误将11条件轴要求为1；各108004NN／6CP／0训练更新／0target。r03用实际header与completion确认结构[2,3,11,1800,2,256]／float32／C-order，canonicalID连接并核TX／RX／day，输入检查前移到任何模型前向之前。原科学参数、模型、数据角色、IQ和信道始终不变。两失败216008NN，加成功133204NN，**本路线实际新增总349212NN、18次CP读取、0训练更新**。技术错误不写成科学负结果，未覆盖／重启失败root，也不因分数选择性重跑。

实际执行提交 `649487724adda1b69243f9efd441f2c05160f781`；原teacher/student父release固定`0f399c910324e21a8b1a60346d879be5c2964a9c`。普通N607账户的单run preflight／archive单次SHA比较／remote compile与checks通过。原PID1218189启动CWD／argv／CUDA环境均独立核实，随后日志增长887字节、终态SOURCE_ANALYZED且PID已退出，无failure。短任务的nvidia-smi快照未捕获活跃GPU PID；实际进程环境固定GPU0，runner使用cuda:0完成上述真实前向，不将未捕获写成已观测。

本次公开汇总11文件经单archive传输SHA比较VERIFIED：`b2a6ad8603621daa14482300ff2f831651a0eaa9cd1dffad8fb7285f6c6384e0`。源原始logits、IQ、CP未取回或入Git。完整1320行metrics／配对错误／CM保留在私有local_artifacts/F_ownEMA_source_E50_20261018_r03/collection-final-r01；公开交付只含本报告与摘要，不提交原始数据。

本轮没有新增formal预测或评分，六种practical的正式结论沿用[E50三种子完整报告](../20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01/final_analysis.md)：As50相对同预算NoAs50的clean均值−0.150198pp，阶段目标仍未满足，可重复clean＋星地双提升仍未获得。固定EMA源假设至此关闭；后续继续研究增强训练组织，不把本次负结果外推为EMA普遍无效，也不从正式分数挑参数。
