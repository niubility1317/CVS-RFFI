# R17 精简版源 clean 直接纠正 R10

状态：ANALYZED。用户要求压缩同时优化，本轮固定两种既定R08/R09结构，检验去掉模仿原R17恢复波形后，直接源clean波形/身份约束能否提高识别。原R17仅作CPU权重初始化；原F50/defaulthead冻结。目标逐场景原R17−1pp为下降上限，不能把容差通过等同提分。low urban≥70%、其他环境距clean和配对clean恢复差距独立报告，不自设“接近”的阈值。

|固定行|完整/可训练参数|比原4665073减少|GPU|波形教师权重|F参照|
|---|---:|---:|---:|---:|---|
|clean_only_quarter|1822977|60.9229%|4|0|source paired clean|
|clean_only_eighth_pointwise|1592461|65.8642%|5|0|source paired clean|

只原source-selected R17/cd977448/E20/4940初始化，无已测学生继承。84个context长卷积CPU半秩、prior quarter共享或eighth并共享原1×1，保留各步bias/GN；原结构/初始权重构造同R08/R09，三步×2CG不变。全部coarse/action/prior可训练；直接调用coarse与action estimator保留梯度，不走旧coarse no_grad路径。原R17初始化后保留CPU，训练开始即删除，optimizer只含compact参数。

固定目标：最终恢复与对应source-clean IQ的NMSE（clean mean energy detached、下限1e-4）+ KL_T1(Fclean||Frestored) + 0.5×单位160维特征距离平方。IQ、KL系数各1，feature系数0.5，无原R17输出/中间阶段模仿项。与对应R09 clean arm相比只删除10倍R17三阶段波形模仿，原架构/初始化/数据/顺序/seed/优化器/预算保持；同时移除无用teacher前向，因此不称相同训练算量。不预先断言去掉teacher一定提分。

只原source训练缓存63000=L6300/U56700、20个既定received视图epoch、15%clean/85%residual；配对clean来自合法source L/U，0新TX标签、0V/target训练、0头拟合。使用配对波形及监督F来源知识，不称严格无监督或F-free。推理只当前received IQ，0clean/Fteacher/TX/RX/场景/cache/测试状态更新；识别仍接原F50及其联合训练默认头。

E20/4940，B256/model2026092701/data2026102001，AdamW2e-4/warm247/cos2e-5/WD1e-4/clip1/FP32/noTF32。每行L126000/U1134000/source1260000，原R17 teacher/context训练前向0，studentcontext1260000、Fclean1260000、Fstudent1260000并对student IQ反传。继承coarse20/action20/R17prior20加一次joint20共80串行训练pass，各组件累计40、component-epoch sum120；F50预训练50/11100及smoke/profile开销另计。不能把20轮当整体成本，不能把120当串行轮数。

固定最终E20，无早停、V选模或源性能资格门槛；每行自动source七视图诊断/profile/独立truth-last，再正式168000×clean和六practical全部预测，所有七视图IQ/logit/feature/physical IDs固定后独立CPU才读truth/clean。报告原始正确数1pp整数验收、真实涨分、MacroF1、TX/RX/day、raw修复/损伤/净修复、full256/prefix232/tail24/FFT/clean特征距离、完整参数/文件/延迟/显存与20轮日志。全部固定行和负结果保留。

38项合成/协议检查PASS，唯一独立P0/P1审查PASS，独立复跑38项。真实原R17初始化/F50/2Ls noquery smoke为launcher首步，检查coarse/action/各prior与F输入有限非零梯度、原初始化父/F/head不更新、严格复载逐位输出，0TX/V/target/optimizer更新，PASS即继续。

GPU4/5先等R08根终态且实际ownedGPU进程全部退出，不读评分；随后各一own worker，R09健康2/3/6/7继续，共六路。所有R17 scorer共享R06/score.lock，CPU2线程/最多一个own scorer。新不可覆盖run/output与release；一次普通N607预检、一个archive SHA比较/远端compile/help与实际PID/CWD/完整命令/commit/GPU/loggrowth。技术错误停止失败行、保留产物，健康行继续，无自动重试/回退；性能负结果不重启。profile排除GPU payload/optimizer，source-only16/batch1/2warmup，不把共存负载/测量顺序差称部署加速。

新方案由用户要求与合法源训练目标的匹配消融决定，在任何R08/R09目标分数前固定，已暴露基准不称新盲测。不以目标结果调秩、系数、预算、父权重、重排或选择性重跑。具体环境/CWD/命令/路径/停止规则和预期产物见experiment.json。

发布前证据：38项聚焦合成/协议检查和唯一独立P0/P1审查PASS，审查0真实IQ/checkpoint/远端访问；已核对正式训练0R17前向、完整联合梯度、F冻结、严格复载、实际R08句柄等待和自动全目标truth-last。普通N607预检VERIFIED，R08仍2个实际GPU进程，新root/release不存在；source/R17/F50句柄复用，只提交CPU smoke与等待，不抢占健康任务。

实际发布 VERIFIED：方法commit `d113672a6e51e75eb3ea0f7ecce0500e67f92966` 已推送且远端OID独立比较一致；单归档SHA比较/远端compile/help通过。实际原R17初始化/F50/2Ls noquery smoke PASS，两行student coarse/action/各prior及F输入梯度有限非零，原CPU初始化父/F/head0更新，严格复载输出逐位一致。R08完整测试/评分及原实际GPU句柄退出后，在GPU4/5各启动一个新worker。实际PID/CWD/完整命令/commit/GPU/loggrowth与E1均核实，R09健康2/3/6/7继续；原R17正式训练前向删除，未用新目标结果改变设计/矩阵。

完整汇总工具26项聚焦检查PASS，检查全联合组件预算、0R17 teacher/context计费/source曝光、直接clean损失、同全七视图/分组/原始整数1pp、实际涨分/70%与容差分开、payload-free profile。汇总工具未改运行中方法commit；当前尚无R10目标准确率。

元数据说明：修正继承的 teacher_IQ_examples_train 为 0，并将父权重角色文字改为 CPU 初始化、coarse/action 联合训练，与实际训练日志和实现一致。未修改运行中代码、配方、预算、权重或预测。


## 完整目标结果

两个固定候选均完成E20/4940、source七视图诊断和同168000物理样本×七目标场景；全七预测固定后独立CPU truth-last，实际controller/worker句柄全部退出。逐场景相对原R17下降<=1个百分点，以整数正确数验收；补报相对已验收3118001参数版本差值。完整分组见results.json/row_metrics.csv。

|版本|场景|目标准确率|原R17准确率|相对原R17（百分点）|相对3118001版（百分点）|1pp验收|
|---|---|---:|---:|---:|---:|---|
|clean_only_quarter|clean|85.236310|84.586905|+0.649405|+0.650000|通过|
|clean_only_quarter|practical_high|79.373214|81.319048|-1.945833|-1.581548|未通过|
|clean_only_quarter|practical_mid|77.928571|79.577976|-1.649405|-1.391071|未通过|
|clean_only_quarter|practical_low_suburban|76.441071|77.996429|-1.555357|-1.195238|未通过|
|clean_only_quarter|practical_high_urban|78.348810|80.353571|-2.004762|-1.640476|未通过|
|clean_only_quarter|practical_mid_urban|72.808929|73.685119|-0.876190|-0.905952|通过|
|clean_only_quarter|practical_low_urban|66.647619|67.216667|-0.569048|-0.708929|通过|
|clean_only_eighth_pointwise|clean|84.752976|84.586905|+0.166071|+0.166667|通过|
|clean_only_eighth_pointwise|practical_high|80.552381|81.319048|-0.766667|-0.402381|通过|
|clean_only_eighth_pointwise|practical_mid|79.192857|79.577976|-0.385119|-0.126786|通过|
|clean_only_eighth_pointwise|practical_low_suburban|77.615476|77.996429|-0.380952|-0.020833|通过|
|clean_only_eighth_pointwise|practical_high_urban|79.555952|80.353571|-0.797619|-0.433333|通过|
|clean_only_eighth_pointwise|practical_mid_urban|73.713690|73.685119|+0.028571|-0.001190|通过|
|clean_only_eighth_pointwise|practical_low_urban|67.600595|67.216667|+0.383929|+0.244048|通过|

本轮结论：**符合七场景1个百分点容差：clean_only_eighth_pointwise**。本轮固定已测试候选中，达标最少完整参数为 1592461（clean_only_eighth_pointwise），比原R17减少65.8642%，比已验收3118001参数版再减少48.9269%。这不是全局最简证明。单训练seed/单原F50seed/已暴露基准，不称新盲测或多seed总体保证；所有固定候选都已报告，目标结果未回流训练、超参、父权重、候选重排或选择性重跑。

|版本|完整参数|参数较原R17减少|相对原R17最大下降（百分点）|相对3118001版最大下降（百分点）|零下降诊断|low urban准确率|达到70%|
|---|---:|---:|---:|---:|---|---:|---|
|clean_only_quarter|1822977|60.9229%|2.004762|1.640476|False|66.647619|否|
|clean_only_eighth_pointwise|1592461|65.8642%|0.797619|0.433333|False|67.600595|否|

压缩容差通过不等于优化成功；下面逐行报告实际涨分和退化，不用均值替代逐场景验收。单训练seed/已暴露基准不能证明稳定总体提升，lowurban是否达到70%依实际结果；其他环境接近clean未自设阈值。

|版本|上涨场景|不变场景|下降场景|七场景均不下降|lowurban达到70%|
|---|---|---|---|---|---|
|clean_only_quarter|clean||practical_high, practical_mid, practical_low_suburban, practical_high_urban, practical_mid_urban, practical_low_urban|False|False|
|clean_only_eighth_pointwise|clean, practical_mid_urban, practical_low_urban||practical_high, practical_mid, practical_low_suburban, practical_high_urban|False|False|

两种架构均使用最终source-clean IQ与冻结F50 clean-reference纠正目标，无R17输出模仿。与对应R09 clean arm的匹配比较须复用其完整冻结汇总；目标结果不回流当前训练、预算、权重或重跑。

### 匹配波形模仿消融与同参数已验收版

复用完整冻结R09汇总比较，不新增模型前向或truth访问。同结构、原始父初始化、source数据/顺序/seed/优化器/E20预算及clean-F参照保持；R10删除10倍三阶段R17波形模仿和对应teacher前向，训练算量有变化。以下为R10减R09对应clean arm的百分点差；单seed汇总不能证明样本配对显著性或稳定总体提升。

|R10版本|R09匹配行|clean|high|mid|low suburban|high urban|mid urban|low urban|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|clean_only_quarter|joint_clean_quarter|-0.031548|-0.691667|-0.785119|-0.822619|-0.766667|-0.488690|-0.434524|
|clean_only_eighth_pointwise|joint_clean_eighth_pointwise|-0.080357|-0.897619|-0.679762|-0.679167|-0.723810|-0.076190|+0.493452|

相对已验收R09同参数1592461版的七场景差值保存于results.json的relative_to_accepted_1592461，主验收仍是原R17逐场景1pp。两版各有场景优势，保留全部权重与结果，不以low urban单项覆盖其他场景退化。

相对已验收2352389参数版的补充对照如下；不改变固定原R17逐场景1pp主验收，也不以补充结果回流训练或重排。每场景原始正确数/波形/分组差值保存在results.json的relative_to_accepted_2352389。最大下降为有符号差，若全部提高则可为负。

|版本|完整参数|相对2352389再减参数|相对2352389最大下降（百分点）|lowurban准确率|lowurban相对2352389（百分点）|
|---|---:|---:|---:|---:|---:|
|clean_only_quarter|1822977|22.505291%|1.823214|66.647619|-0.714286|
|clean_only_eighth_pointwise|1592461|32.304521%|0.626786|67.600595|+0.238690|

### clean_only_quarter 的配对clean差距与源诊断

|场景|source准确率|source相对原R17（百分点）|目标全帧NMSE|主体232 NMSE|尾24 NMSE|配对clean特征余弦距离|目标raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.500000|+0.055556|0.00432469603|0.00434549246|0.00409660162|0.00162988913|2082/1141/+941|
|practical_high|98.166667|+0.333333|0.0222785734|0.00905234739|0.152798429|0.0148829361|69190/4274/+64916|
|practical_mid|97.722222|+0.944444|0.0303860158|0.0176354367|0.156237647|0.0195491817|68663/4803/+63860|
|practical_low_suburban|96.944444|+1.000000|0.0389583111|0.0266309697|0.160635456|0.0245799273|68339/5488/+62851|
|practical_high_urban|97.888889|+0.888889|0.0309494417|0.0180424079|0.158192456|0.018548185|69667/4734/+64933|
|practical_mid_urban|92.944444|+1.500000|0.0696574822|0.0580855422|0.183212683|0.0366186686|71206/6819/+64387|
|practical_low_urban|87.555556|+0.722222|0.105216689|0.094376415|0.210298777|0.0537066199|69849/8784/+61065|

完整20轮日志已分析，采用固定最终E20，未按最小训练loss或V/target择优。各项loss首轮→末轮：loss 0.7706084→0.14825332；iq_nmse 0.29070715→0.048869829；task_KL 0.42139673→0.083351353；task_feature_distance 0.058504524→0.016032137。训练耗时2172.901秒，中位每轮108.627秒。

### clean_only_eighth_pointwise 的配对clean差距与源诊断

|场景|source准确率|source相对原R17（百分点）|目标全帧NMSE|主体232 NMSE|尾24 NMSE|配对clean特征余弦距离|目标raw修复/损伤/净修复|
|---|---:|---:|---:|---:|---:|---:|---|
|clean|98.388889|-0.055556|0.00410178816|0.00414363714|0.00376506592|0.00172198378|1440/1311/+129|
|practical_high|98.111111|+0.277778|0.0225436613|0.00911527686|0.155208707|0.0147951748|70970/4073/+66897|
|practical_mid|97.722222|+0.944444|0.030617075|0.0176216867|0.159074351|0.0190983303|70512/4528/+65984|
|practical_low_suburban|96.833333|+0.888889|0.0391399041|0.0265309382|0.163880676|0.0238385834|69909/5085/+64824|
|practical_high_urban|97.944444|+0.944444|0.0314873233|0.0183795784|0.160853058|0.0183584616|71477/4516/+66961|
|practical_mid_urban|92.722222|+1.277778|0.0712516382|0.0594788305|0.187096953|0.0359665714|72474/6567/+65907|
|practical_low_urban|86.444444|-0.388889|0.10787379|0.0968602002|0.215257585|0.0529849455|71265/8599/+62666|

完整20轮日志已分析，采用固定最终E20，未按最小训练loss或V/target择优。各项loss首轮→末轮：loss 0.75819617→0.15256042；iq_nmse 0.27752584→0.049884931；task_KL 0.42276158→0.086374119；task_feature_distance 0.05790875→0.01630137。训练耗时2156.744秒，中位每轮107.853秒。

成本测量只用固定16条source V、batch1/2warmup，包含完整coarse/action/三步CG；CPU加载后删除checkpointpayload，再放入GPU，无GPU optimizer/F/teacher。各版本测量顺序及其他owner仍影响时间，不视为部署加速证明。

|版本|checkpoint字节|原R17中位ms|精简版中位ms|延迟比|原R17resident字节|精简resident字节|原R17增量peak字节|精简增量peak字节|目标预测秒（含F及写盘）|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clean_only_quarter|7678332|48.719|93.279|1.9146|35870720|24508416|1158656|1116272640|1005.862|
|clean_only_eighth_pointwise|6753436|48.159|81.637|1.6951|35870720|23585280|1158656|1116272640|987.734|

每行新增20轮/4940更新/L126000/U1134000；原R17只CPU初始化，训练teacher/context前向均0；source1260000、student context/Fclean/Fstudent各1260000输入前向，student的完整context/prior及F输入反传，coarse/action/prior均4940更新，0F/head/原R17更新。F参照固定source paired clean，clean IQ权重1、R17波形模仿权重0。0校准IQ；CPU均值SVD初始化时间单列。继承coarse20+action20+R17prior20与本轮joint20共80串行pass，每组件累计40，component-epoch总和120，不把120称串行轮；原F50预训练50/11100和教师成本单列。

两份完整恢复器已取回本地 models/<arm>/final.pt，大小与远端一致、来源/最终预算/共享alias严格加载通过，单条合成零IQ输出有限；0实际IQ/target访问。无optimizer/F/teacher权重，包含完整联合优化后的coarse/action/prior，不提交权重到Git。加载使用experiments.pamf_r17_clean_corrective.runtime.load_checkpoint，输入原生尺度float32 [B,2,256]、I/Q顺序，输出同形状恢复IQ，下游仍接原F/defaulthead。
