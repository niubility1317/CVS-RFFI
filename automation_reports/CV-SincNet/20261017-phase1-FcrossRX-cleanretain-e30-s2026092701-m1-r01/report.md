# 全源 F：保留 clean 监督的 E30 配对结果

状态：ANALYZED／VERIFIED。新 R25 与两个复用控制完成原七环境预测及独立 truth-last；整体联合目标仍 ACTIVE。

**本轮固定源保护条件未通过。** R25 相对同预算 C0 的 clean 为 -0.639286 个百分点；相对 AS50，clean +1.460119、mid urban -1.780357、low urban -2.167262、urban 均值 -1.709921 个百分点。

这是一个 model seed2026092701、全部五源 RX、scratch30/20L10U 的结果。75%普通视图／25%完整 practical 是用户明确要求参考的训练组织，本轮没有新增物理 Ag，也不能把源保护通过当作域泛化创新或联合目标完成。

## 默认头七环境准确率

单位为%。所有模型固定既有默认6×160 cosine头/scale30，骨干与头联合训练；没有换头、另训诊断头或目标拟合。

|环境|C0 无As|AS50|R25|
|---|---:|---:|---:|
|clean|85.156548|83.057143|84.517262|
|practical_high|46.513095|81.772619|80.688095|
|practical_mid|44.677976|79.417857|78.013095|
|practical_low_suburban|42.714286|77.039286|75.378571|
|practical_high_urban|45.233333|78.350000|77.167857|
|practical_mid_urban|37.582738|64.337500|62.557143|
|practical_low_urban|31.985714|53.394643|51.227381|

## 收益、代价与配对错误

R25 urban 均值 63.650794%；low urban 51.227381%；clean 84.517262%。low>60%、clean正净收益、每个suburban距自身clean≤2pp仍按原探索目标报告，不能靠clean下降缩小差距。

|模型|clean宏F1 %|low宏F1 %|low最差RX %|
|---|---:|---:|---:|
|C0|84.891668|30.218017|27.700000|
|AS50|82.666197|52.824516|48.737500|
|R25|84.154010|50.668058|45.725000|

|R25差值参照／视图|准确率差pp|修复|新增错误|clean与该视图同时正确净差pp|
|---|---:|---:|---:|---:|
|C0/clean|-0.639286|3952|5026|-0.639286|
|C0/practical_mid_urban|+24.974405|48684|6727|+23.369643|
|C0/practical_low_urban|+19.241667|40934|8608|+17.754167|
|AS50/clean|+1.460119|4522|2069|+1.460119|
|AS50/practical_mid_urban|-1.780357|5539|8530|-1.180357|
|AS50/practical_low_urban|-2.167262|6716|10357|-1.636310|

|模型|high suburban距clean pp|mid suburban距clean pp|low suburban距clean pp|
|---|---:|---:|---:|
|C0|38.643452|40.478571|42.442262|
|AS50|1.284524|3.639286|6.017857|
|R25|3.829167|6.504167|9.138690|

固定C0 clean正确集合共143063包；R25 low保持率55.940390%，AS50为58.281317%。固定集合只是诊断参照，两个控制均非新训练父模型。

## 源判据、分层和因果边界

正式结果读取前已落盘源判据。fullV clean R25−C0=-0.100000pp；两个独立draw×mid/low相对AS50的差=[[-2.888889, -2.0], [-3.0, -3.555556]]pp。固定要求clean≥−.25pp、四个urban差均≥−2pp。任何失败即本轮保护假设失败，不换份额、扰动、seed、阈值或预算救跑；不自动扩种子。

全部物理实现比较：426240个选中As包、6660步，完整记录一致426240，差异0。首次批次吻合不能代替该全量统计；若有差异，应同时报告对应字段，不将效果单独归因为监督份额。

源V1800使用同一真实IQ和两组固定完整practical观测，按LOS/PARTIAL/BLOCKED、locked/degraded/unlocked、RX/day/TX及新增噪声比分层。新增噪声比不是录音已有噪声与新增噪声合计的总SNR。正式IQ没有逐包状态/质量旁表，所以不能把源分层冒称目标urban的状态因果效果。

|源low urban新增噪声比|样本数|C0 %|AS50 %|R25 %|
|---|---:|---:|---:|---:|
|<3 dB|1127|20.408163|30.079858|27.329193|
|3-8 dB|794|27.833753|44.332494|40.428212|
|8-15 dB|303|40.924092|78.217822|73.597360|
|>=15 dB|1376|57.921512|95.130814|93.386628|

## 完整训练、数据与预算

全部RX1/3/4/6/8、day1/2/3、六TX、L6300/U56700隐藏TX/V27000、7/63/30物理ID保持；不留训练RX，不合并V，不解封U TX。F为SIG_TIME_X3、Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、真实joint160；900722参数/206状态；保留原必要包内补偿、旧U门控、自身EMA、SupCon和RDrop。

每个L128，原ordinary128作为第一半；第二半交替64个ordinary和64个原practical(raw)，全部生成As监督。每对RDrop共享同一生成IQ，无增强重抽。CE/KL两半归一化再.5/.5，实际ordinary.75/satellite.25；SupCon只用原第一半。4L NN/物理样本，不增加部署模块或损失。

实际导入为原不可变release中experiments/adv3b02_xuc/code/leo_practical，不是顶层code。25MHz/IQ256、dataset原RMS，As前原unitRMS、As后无额外RMS；完整residual/post_sync/EQ关闭即G1/referenceSNR30/fc2.45GHz/reflect，质量适配、相位跟踪、RXIQ补偿、AGC及相互依赖保持；62tap核和24点数值延迟不变。没有改变正式IQ、SNR、边界或困难样本。

本轮与C0、AS50的全部训练序列比较：{"AS50-C0": {"records_compared": 8880, "mismatched_records": 0, "identical": true, "no_hash_gate": true}, "R25-C0": {"records_compared": 8880, "mismatched_records": 0, "identical": true, "no_hash_gate": true}, "R25-AS50": {"records_compared": 8880, "mismatched_records": 0, "identical": true, "no_hash_gate": true}}。完整30轮/6660step、原SupCon5550有效步、Sinc6660更新、全量physics及活动损失均由独立分析读取。RDrop是原随机子网络正则，不记作新Ag。

监督分量的独立尺度复算最大误差4.76837158e-07。共有ordinary完整日程与波形统计比较：{"C0": {"probability_epoch_mismatches": 0, "waveform_observation_epoch_mismatches": 0}, "AS50": {"probability_epoch_mismatches": 0, "waveform_observation_epoch_mismatches": 0}, "R25": {"probability_epoch_mismatches": 0, "waveform_observation_epoch_mismatches": 0}}。默认头/Sinc参数运动只证明联合更新，不证明分支或门控导致收益；没有保存逐包门控/分支梯度，不补造该机制证据。

|模型|训练源NN|训练秒|正式NN|预测秒|新训练/复用|
|---|---:|---:|---:|---:|---|
|C0|5407922|1540.012254|1176002|47.865980|复用历史30E/6660更新|
|AS50|5407922|2956.008296|1176002|47.819214|复用历史30E/6660更新|
|R25|5407922|2283.089897|1176002|48.236964|新30E/6660更新|

新R25父训练0、30轮/6660更新、3409920L网络前向、426240真实As训练视图；U/学生/教师/评估分别由实际counter计入，teacher训练0。兼容smoke2、随机模型合成512、新C0源读出12602、新R25源读出12602、formal1176002分别保存；本轮新增总NN=6609642，历史两个控制NN=13180450单列，不能记作新增或父继承。调度到终态墙钟2364.776987秒；独立分析CPU墙钟92.326699秒，不包含等待训练。预算结束不等于收敛，不延长至E50/E200。

## 代码、验证和可检查产物

训练代码c577b3daded412337fb4181be05078ce051a64f1；独立结果分析1656d1a30669153718496672a50d4eb96ef7c7ce；物理实现比较56cfd3a9f5aed9ba8e176db6fa6e49aeda6c6274。训练实现experiments/cvs_cleanretain37；原匹配控制P25 B00 ordinary-repeat和P32 B01 rawpractical；E20＋10的旧B01只作路线背景，不混作scratch30控制，也没有加载旧目标选中权重。

本地8项训练聚焦检查、4项独立分析检查、3项物理配对反例通过。一次独立P0/P1审查发现合成验证输出冲突，已隔离且定向复核；首步真实CP2包、原生64As/512NN随机模型验证通过，无query、0optimizer；848个native args仅运行标识/输出位置不同。一次发布传输比较/编译，实际PID/CWD/argv/GPU/log增长已核实。所有其他健康任务保持。

训练命令见[experiment.json](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01/experiment.json)；环境/CWD/GPU/停止规则及source权限在原预登记中保持。所有正式预测固定后独立CPU scorer，378个混淆矩阵独立复算；模型默认头无额外拟合。

[逐row/seed/RX/day/TX](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/per_row_seed.csv)；[完整训练曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/training_curves.csv)；[源状态质量分层](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/source_strata.csv)；[独立完整统计与成本](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/analysis_summary.json)。

![formal_accuracy](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/formal_accuracy.png)

![training_curves](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/training_curves.png)

![source_added_noise_accuracy](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/source_added_noise_accuracy.png)

![source_state_lock_accuracy](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/source_state_lock_accuracy.png)

![R25_supervision_components](E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017/final_r01/analysis_r01/R25_supervision_components.png)

## 未完成项

当前只有一个模型seed，不能回答跨seed稳定复现。源两draw不是两个模型seed。没有独立有效Ag，不能虚构Ag×As四格交互。low>60%、clean净收益、所有suburban接近clean和跨seed联合验证仍按实际结果分别审查，整体goal不标complete。

固定CVS和固定practical条件下，本轮可核实的是完整组合practical的训练组织效应：默认头相对AS50的low变化-2.167262pp、mid-1.780357pp；clean相对C0变化-0.639286pp、相对AS50变化+1.460119pp。单个多径/频偏/噪声因素的训练因果增益及跨seed复现尚不能由这一行证明。

## 有效覆盖、状态与原参考边界

全部6300个L物理样本均实际进入过satellite视图，未进入样本数0；全6660步共426240As观测，15个RX/day单元全部有覆盖。TX×RX×场景和TX×RX×状态均90个有效单元，未筛掉困难样本。完整序列比对三组各8880条、差异0；活动loss均有限、无跳步/错误，stdout中300处NaN文本属于未启用统计，不作为活动梯度失败。

源low urban的LOS/locked为C0 59.905660%、AS50 96.037736%、R25 94.811321%；PARTIAL/degraded为41.897233%／85.375494%／81.422925%；BLOCKED/unlocked为22.300319%／33.610224%／30.287540%。可恢复受损观测明显适应，噪声主导失锁仍弱，R25在全部新增噪声比区间均低于AS50。该描述只来自同一源面板，不能把它解释为正式目标的特异多径或Doppler因果改善。

最差RX（accuracy比例）为{"C0":{"clean":{"RX":"14-7","accuracy":0.7443333333333333},"practical_low_urban":{"RX":"14-7","accuracy":0.277}},"AS50":{"clean":{"RX":"14-7","accuracy":0.7408333333333333},"practical_low_urban":{"RX":"20-1","accuracy":0.487375}},"R25":{"clean":{"RX":"14-7","accuracy":0.7551666666666667},"practical_low_urban":{"RX":"14-7","accuracy":0.45725}}}。R25比AS50在low修复6716包却新增10357错；恢复一部分clean时损失更多urban，不能称双方突破。

本轮与原参考的训练阶段、LR/optimizer重置、ordinary强度、4×128/2×256前向组织、场景抽样及训练载频不同；仅复制监督份额不等于复现原方案。完整已核实差异见[reference_reconciliation.md](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01/reference_reconciliation.md)。后续沿原完整参照继续，当前不启动调份额救跑。

全部10张PNG已实际逐张查看，20个PNG/PDF已取回；标签、单位、源/目标边界和tiny strata说明可读。四个本run进程自然退出，原CP和全部7个logits文件保留。

修改实现：[固定配置](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_cleanretain37/design.py)、[真实视图路由](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_cleanretain37/route.py)、[原默认头训练](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_cleanretain37/model.py)、[观察与native接入](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_cleanretain37/observer.py)、[自动串联](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_cleanretain37/dispatch.py)；结果分析与报告工具位于同工作树analysis/cvs_cleanretain_*。本轮不增加部署参数、分类头、数据权限或测试难度变化。
