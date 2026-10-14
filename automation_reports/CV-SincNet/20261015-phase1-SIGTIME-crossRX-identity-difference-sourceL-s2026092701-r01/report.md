# 跨 RX 类间特征差方向：真实 L 源否证

当前状态 **ANALYZED / VERIFIED，固定source gate0/8**。以下保留预登记方案。最新固定可信U补偿普查仅有U3/V4活跃，关闭该入口；本轮独立检验真实L的相对身份监督。核心任务是同一对TX在不同RX里的类间特征差方向是否一致，有标签身份直接给出含义，不依赖新伪标签判定。

固定BASE_CE第一seed2026092701自身INITIAL/FINAL20，各四个L180（ordinal0/1、17/18、35/36、52/53）。各side90为6TX×5RX×3day，临时类均值每RX/类/side3包。独立heldL180为68/69；全部900包物理不交，原L池6300、原7/63/30角色保持。heldL仍是父训练见过的源数据，不称盲测/未见RX。U/V/target IQ不读，U真值保持隐藏。

actualjoint=`((a+b)+c)+s`，每项从同次前向直接取得。辅助图数值逐位相同，但PA/ref两端点、三个gain detached；仅a/sRaw接收辅助梯度，原主CE/defaultW联合梯度保持。先在rawjoint坐标里作同RX类均值差，再eps1e-4单位化；不同RX有序对20×无序类对15=300。固定 `.1/300 sum relu(.9-cos)`，达到方向停止值不继续逼到1。无W、新投影/分类头/参数或持久bank。两端均attached。SELF比较同RX的A/B，75项重复4次也计算300；ZERO返回仅a/sRaw零图；每mode均计算全部SELF/CROSS600方向项，0extraIQ。

同RX相减若近似含公共RX偏移可减少其影响，这是表示近似，不是严格硬件/因果解耦；类依赖失真仍存在。方向一致也可能塌缩或损害类内结构，因此不能只看辅助loss/非零gradient。相关理论基础见[SupCon](https://arxiv.org/abs/2004.11362)、[MatchDG](https://arxiv.org/abs/2006.07500)、[NDCL](https://arxiv.org/abs/2601.21999)，具体新机制的完整原创性和泛化收益未证明。设计见[机制说明](../../../analysis/phase1_crossrx_identity_difference_20261015.md)。

INITIAL是原AdamW空矩/lr2e-4的明确假想一步；FINAL20用实际保存矩/lr1e-6。每cell独立simulate主CE＋SELF、主CE＋CROSS，分别clip5；两个纯CEbase_after须完全一致，ZERO直接用该纯CE提议（不因explicitzero改变unused参数AdamW语义）。静态部署feature范围68tensor/194561coord，排除W和3gain。后评价临时模型固定更新前W/全部gain，actualjoint160 rawgeometry的身份间隔、mean/perRXCE/defaultWmargin、实际位移/RMS/Sinc核和浅中层变化均保留。auxdetach不等于全模块冻结，shared/main/globalclip可能间接改变其他参数。

科学gate提前固定：至少2个FINAL20单元同时满足CROSS活跃≥30/300、两组所有类差norm≥1e-4、CROSS/SELF辅助梯度相对差>.01、CROSS/ZERO实际F位移相对差>.01、heldrawjointRMS>1e-7、至少1个浅中层tensor改变、held身份间隔相对ZERO及SELF各增加>1e-5、heldmeanCE相对两对照各增加≤1e-6。INITIAL结果只解释初始状态，不替代FINAL20资格。条件只判源信号，不证明DG；失败不扫描方向停止值/系数/归一化或按target挑模型。

真实预算900uniqueL/6300metadata/6124packet主网络前向/32临时CPUAdamW；0实际训练/头拟合/EMA/U/V/target。技术来源/角色/ID/状态不一致、非有限或未处理异常均保存产物并只停止本run，不重试或改其他健康任务。入口两份真实CP无querysmokePASS后直接继续。正式训练只有本源诊断成立才另行固定同父E20+20配纯CE/原SSL/SELF/CROSS，累计40，正常原头逐rowclean truth-last，不从E50串接。

run `20261015-phase1-SIGTIME-crossRX-identity-difference-sourceL-s2026092701-r01`，owner `codex/root/identity-difference-source-20261015`；实际GitHEAD发布时写release_commit.txt，父E20执行版本6693e4480ef990bc23c5021ff1440b547fe5fcaf。普通N607账户/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01/experiments/cvs_identity_difference20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-crossRX-identity-difference-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01/source_probe.stdout.log`。完整矩阵/路径/环境/stop/artifacts见[experiment.json](experiment.json)。

历史SIG_TIME、CENTER25K、COMPACT_LINEAR、LONG_SILU正收益独立保留，不能叠加或据target高分挑父。84%默认头clean三seed目标active、尚未完成；尚无本轮真实source结果。


## 完整实际源结果

执行 `1c629a7f54fd69a0b1dc9751368f38165b21505a`，代码提交/推送和remote OID一致；归档SHA比较、远端编译、own PID/CWD/cmdline/CPU/GPU/log增长以及两份真实CP无querysmoke核实。独立读回全部8单元、summary和自然终止，无failure，源进程约13.25 s。固定gate INITIAL0/4、FINAL20 0/4。

| 状态 | ordinal | CROSS活跃/300 | 位移差/原CE | held RMS差 | 相对CE身份间隔增益 | 相对SELF身份间隔增益 | held CE相对SELF增量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| INITIAL | 0 | 300 | 0.116458 | 6.968e-04 | +7.345e-06 | +5.895e-06 | -3.971e-04 |
| INITIAL | 17 | 300 | 0.109152 | 3.784e-04 | +5.132e-06 | +4.585e-06 | +7.473e-05 |
| INITIAL | 35 | 300 | 0.116332 | 1.297e-03 | +8.470e-06 | +4.572e-06 | -4.177e-05 |
| INITIAL | 52 | 300 | 0.113717 | 4.244e-04 | +6.633e-06 | +3.339e-06 | -7.087e-05 |
| FINAL20 | 0 | 127 | 0.020405 | 9.421e-07 | +3.335e-07 | +5.160e-07 | +6.922e-09 |
| FINAL20 | 17 | 129 | 0.024517 | 8.785e-07 | +3.527e-07 | +4.195e-07 | +8.179e-09 |
| FINAL20 | 35 | 103 | 0.018631 | 8.229e-07 | +2.850e-07 | +3.514e-07 | +7.808e-09 |
| FINAL20 | 52 | 111 | 0.018955 | 7.692e-07 | +1.904e-07 | +2.318e-07 | +6.336e-09 |

本轮与上一可信U补偿的稀疏机会不同。FINAL20每单元有103至129/300跨RX方向未达到停止值，SELF同RX只有4至6/75基础项活跃。所有类差norm最小2.31至3.02，未处于eps近零区；CROSS/SELF辅助梯度相对差2.21至4.22。任务本身有足够监督，也产生了不同方向，不能把本轮失败解释成原门控没接受样本或loss未接线。

FINAL20相对原CE，真实F位移差占原位移1.8631%至2.4517%，held实际joint RMS差7.69e-7至9.42e-7，40至42feature张量不同，其中14至16个浅中层/SIG张量；Sinc额外改变8/6/4/5个核。原W/全部gain在有限步评价中固定，不能归为仅头变化。这些变化仍不是训练收益。

固定gate失败的直接原因是独立held-L的raw特征关系身份间隔增益幅度：FINAL20相对原CE为1.90e-7至3.53e-7，相对SELF为2.32e-7至5.16e-7，四单元符号都为正，但远低于预登记1e-5。此身份间隔明确是同TX跨RX同day平均余弦减不同TX同RX同day平均余弦，不是分类概率或另训头分数。相对原CE的mean CE增加3.29e-9至8.41e-9，相对SELF增加6.34e-9至8.18e-9，均在固定1e-6容差内；不能把这些极小增量写成明显性能下降，也不能把弱正几何迹象写成机制有效。

INITIAL四单元300/300活跃，位移相对差10.92%至11.65%，held RMS3.78e-4至1.30e-3。相对原CE的mean CE均减少6.67e-5至1.99e-4，相对SELF三减一增（ordinal17增加7.47e-5）。raw几何身份间隔增益相对原CE5.13e-6至8.47e-6、相对SELF3.34e-6至5.89e-6，也未达到1e-5；INITIAL结果按预登记不得替代FINAL20资格。这些初始弱正信号保留，不因总体gate失败删去。

实际900uniqueL、6300metadata、6124packet前向、32临时CPUAdamW、120条perRX记录。0实际训练/头拟合/EMA/U/V/target，未重测clean。原model/optimizer/grad/RNG保持；PA/ref端点的aux None不表示模块整体冻结，INITIAL的PA参数仍有极小clip耦合差，记录原值。独立heldL是原L的一部分、已被父训练见过，不构成新的盲测或未见RX证据。

按固定规则关闭这个identity-difference源变体，不扩原配置E20+20，不扫描coefficient/.9停止值/normalization，也不借INITIAL结果晋级。保留全部实现与弱正方向，但当前证据不足以支持后续正式训练。

本诊断的FINAL20是原保存AdamW矩和末态LR1e-6，INITIAL则为空矩LR2e-4；它们不等于原设计提到的E20+20 optimizer-reset实际阶段。不能据此断言重置后也无学习余量。后续最多优先做一项独立基线诊断：只用原纯CE核对拟执行续训阶段的重置状态及合法源学习余量，匹配恢复状态对照；不直接复活这个CROSS候选、不用target结果挑LR/父权重。确认阶段条件后，再研究不同监督内容，避免让无关阶段的微小一步替代真实训练机制判断。

原SIG_TIME、固定CENTER25K、COMPACT_LINEAR、LONG_SILU正收益独立保留，不能相加不同成本/配方。84%默认头clean三seed目标仍active、未达成。完整[8单元](source_cells.csv)、[120条RX指标](source_RX_metrics.csv)、[标量证据](result_evidence.json)、[运行终态](launch_evidence.json)可检查，私有原输出/日志/权重保留。
