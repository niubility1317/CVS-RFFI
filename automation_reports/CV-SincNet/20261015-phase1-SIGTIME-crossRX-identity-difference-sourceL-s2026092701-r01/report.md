# 跨 RX 类间特征差方向：真实 L 源否证

状态 PLANNED。最新固定可信U补偿普查仅有U3/V4活跃，关闭该入口；本轮独立检验真实L的相对身份监督。核心任务是同一对TX在不同RX里的类间特征差方向是否一致，有标签身份直接给出含义，不依赖新伪标签判定。

固定BASE_CE第一seed2026092701自身INITIAL/FINAL20，各四个L180（ordinal0/1、17/18、35/36、52/53）。各side90为6TX×5RX×3day，临时类均值每RX/类/side3包。独立heldL180为68/69；全部900包物理不交，原L池6300、原7/63/30角色保持。heldL仍是父训练见过的源数据，不称盲测/未见RX。U/V/target IQ不读，U真值保持隐藏。

actualjoint=`((a+b)+c)+s`，每项从同次前向直接取得。辅助图数值逐位相同，但PA/ref两端点、三个gain detached；仅a/sRaw接收辅助梯度，原主CE/defaultW联合梯度保持。先在rawjoint坐标里作同RX类均值差，再eps1e-4单位化；不同RX有序对20×无序类对15=300。固定 `.1/300 sum relu(.9-cos)`，达到方向停止值不继续逼到1。无W、新投影/分类头/参数或持久bank。两端均attached。SELF比较同RX的A/B，75项重复4次也计算300；ZERO返回仅a/sRaw零图；每mode均计算全部SELF/CROSS600方向项，0extraIQ。

同RX相减若近似含公共RX偏移可减少其影响，这是表示近似，不是严格硬件/因果解耦；类依赖失真仍存在。方向一致也可能塌缩或损害类内结构，因此不能只看辅助loss/非零gradient。相关理论基础见[SupCon](https://arxiv.org/abs/2004.11362)、[MatchDG](https://arxiv.org/abs/2006.07500)、[NDCL](https://arxiv.org/abs/2601.21999)，具体新机制的完整原创性和泛化收益未证明。设计见[机制说明](../../../analysis/phase1_crossrx_identity_difference_20261015.md)。

INITIAL是原AdamW空矩/lr2e-4的明确假想一步；FINAL20用实际保存矩/lr1e-6。每cell独立simulate主CE＋SELF、主CE＋CROSS，分别clip5；两个纯CEbase_after须完全一致，ZERO直接用该纯CE提议（不因explicitzero改变unused参数AdamW语义）。静态部署feature范围68tensor/194561coord，排除W和3gain。后评价临时模型固定更新前W/全部gain，actualjoint160 rawgeometry的身份间隔、mean/perRXCE/defaultWmargin、实际位移/RMS/Sinc核和浅中层变化均保留。auxdetach不等于全模块冻结，shared/main/globalclip可能间接改变其他参数。

科学gate提前固定：至少2个FINAL20单元同时满足CROSS活跃≥30/300、两组所有类差norm≥1e-4、CROSS/SELF辅助梯度相对差>.01、CROSS/ZERO实际F位移相对差>.01、heldrawjointRMS>1e-7、至少1个浅中层tensor改变、held身份间隔相对ZERO及SELF各增加>1e-5、heldmeanCE相对两对照各增加≤1e-6。INITIAL结果只解释初始状态，不替代FINAL20资格。条件只判源信号，不证明DG；失败不扫描方向停止值/系数/归一化或按target挑模型。

真实预算900uniqueL/6300metadata/6124packet主网络前向/32临时CPUAdamW；0实际训练/头拟合/EMA/U/V/target。技术来源/角色/ID/状态不一致、非有限或未处理异常均保存产物并只停止本run，不重试或改其他健康任务。入口两份真实CP无querysmokePASS后直接继续。正式训练只有本源诊断成立才另行固定同父E20+20配纯CE/原SSL/SELF/CROSS，累计40，正常原头逐rowclean truth-last，不从E50串接。

run `20261015-phase1-SIGTIME-crossRX-identity-difference-sourceL-s2026092701-r01`，owner `codex/root/identity-difference-source-20261015`；实际GitHEAD发布时写release_commit.txt，父E20执行版本6693e4480ef990bc23c5021ff1440b547fe5fcaf。普通N607账户/CVS-RFFI Python/CPU2线程/GPU0，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01/experiments/cvs_identity_difference20/runner.py /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01/resolved_request.json`，唯一输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-crossRX-identity-difference-sourceL-s2026092701-r01/source_diagnostic`，日志 `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_crossRX_identity_difference_sourceL_20261015_r01/source_probe.stdout.log`。完整矩阵/路径/环境/stop/artifacts见[experiment.json](experiment.json)。

历史SIG_TIME、CENTER25K、COMPACT_LINEAR、LONG_SILU正收益独立保留，不能叠加或据target高分挑父。84%默认头clean三seed目标active、尚未完成；尚无本轮真实source结果。
