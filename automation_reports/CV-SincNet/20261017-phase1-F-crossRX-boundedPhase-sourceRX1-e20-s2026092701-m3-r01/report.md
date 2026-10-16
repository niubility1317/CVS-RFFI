# F＋跨 RX 同 TX：有界非恒定相位的三组匹配源 E20

VERIFIED RUNNING，三组真实训练均已启动，尚无最终源识别结果。唯一逐 row 矩阵见 [experiment.json](experiment.json)。三个新 scratch 模型分别为 B00 额外 raw、C_ordinary 额外同一份 ordinary、B10 额外 bounded phase(raw)。固定 seed2026092701，每 row E20／13L＋7U／4440更新，合计60 model-epoch／13320更新，无父权重、续训或另训头。

保持 F＋跨 RX 同 TX、三 SIG_TIME、统一 Sinc、joint160、NoExplicitCFO、默认 cosine30 头正常联合训练，保留必要内部补偿。公共 raw.75＋原 ordinary.25＋原浅层/中层 CROSS_RX SupCon 完整保留，额外 .25R(view)，R=原 smoothedCE.01均值＋同IQ RDrop1KL。三组都8L前向／总pair权重1.25，不除以视图数；原样本绝对监督权重稳定。C_ordinary复用已生成的实际ordinary tensor，RDrop两次前向共享同一份IQ，无重抽增强、交叉视图一致性损失或新部署分支。

唯一待检验假设：通过源身份保护的这个固定相位视图，能否比同成本额外raw和ordinary都提高跨RX身份提取。此前双默认头G0 PASS_SOURCE_IDENTITY_ONLY、50402NN／0更新，只是源身份保持，不是训练或泛化收益；其权重不作本轮初始化。其他已关闭路线和另一对话的practical噪声配对、时间栅格候选不复刻。

B10唯一新IQ算子与G0相同：CPU Float64 iidGaussian增量sigma.006，cumsum后按固定256点索引正交去常数/线性项，整路径统一缩放使峰值≤.1rad，再全包乘单位模复数并回到原Float32/设备。所有路径保留，不逐点clip/删尾/重抽；没有新增AWGN/MP/PA/DAC/CFO/常相位/时移/量化/后RMS。CPU私有seed202610173401＋optimizer_step并恢复全局CPU/Python/CUDA随机状态；记录CPU构造、IQ往返传输和统计的walltime。新增相位不读TX/RX/day/置信度。原ordinary本来含低概率Wiener，这只是独立全覆盖有界去仿射视图，不是新增强原理；有界过程也不是原始Wiener分布或真实RX替换，不声称覆盖全部源RX差异。

数据为精确fit RX3/4/6/8、day1/2/3、6TX，实际L5040/U45360/V21600；源RX1的原L1260仅final20读出。复用VALIDATED_ONCE原source handles并核对实际角色；不混同fullL6300或L_train5580/L_dev720。U的TX真值始终隐藏，原13L+7U/旧gate/自有EMA/原增强不变，不新增U相位或教师前向。没有donorbank、目标IQ、目标统计或正式truth评分。

每row固定final20冻结后立即自动原默认头预测held-source RX1 clean＋六practical，各1260／每row8820／总26460；场景为practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。所有物理记录、channel/session/evaluation随机流和接收处理沿原框架，测试IQ／信道／SNR／补偿不改。每row结束直接读出，不等整批训练。此为明确source-only证伪，不冒充完整契约或独立target结果。

固定条件：B10相对本轮两个控制都需clean accuracy≥+1pp、clean macro-F1不降、每六practical≥−.5pp、seenV≥−.5pp；完整4440更新、4546560LNN、3330次SupCon、final base L objective≤.5。全部三row和七源视图完成后判定。任一FAIL关闭sigma.006／peak.1rad／全覆盖相位训练，不改sigma/峰值/系数/归一化、筛RX/TX/day/包/抽样、换seed或延预算救跑；不事后晋级控制。PASS只支持独立seed确认，不等于DG／As／联合成功。

每row4546560 L NN，其中1136640额外；原ordinary568320每row，新相位仅B10 568320，控制0，训练As0。原U/EMA/V、held8820、每rowown2NN scratch smoke和墙钟实际分别记录；dispatcher首项真实匹配CP合成2包no-query smoke，绝不用于训练初始化，PASS立即继续。全部13320 native/view/Sincunitsteps、L/U序列、原默认头更新和新相位RMS/峰值/仿射矩/缩放/输入输出功率/IQ/参考窗变化和72格覆盖私有保存。

本地28负例、4114合成NN／3合成联合更新、640合成phase／128ordinary PASS，0realIQ/CP/target。三组初始权重与原U参数相同、CPU/Python RNG终态相同；B00损失和默认头梯度精确匹配既有additive raw公式、sameIQ八前向、G0相位输出与160导出一致。后来只增加非阻断相位生成/传输walltime记录，不改目标或随机流。一轮独立训练代码P0/P1审查PASS，见review.json；未重复审查G0或新增gate。旧Torch NumPy桥未调用、不安装环境。

唯一owner codex/root/F-bounded-nonaffine-phase-E20-20261017。普通N607已有CVS-RFFI Python，dispatcherCPU2、workers FP32/loader0/CPU2，原共享容量锁计入全部owner、每GPU最多3进程且free≥12000MiB，不干预健康旧任务。CWD PROJECT/releases/cvs_F_bounded_nonaffine_phase_sourceRX1_e20_20261017_r01；入口CVS-RFFI/bin/python -u experiments/cvs_bounded_phase20/runner.py，逐row命令见矩阵。不可变输出PROJECT/runs/20261017-phase1-F-crossRX-boundedPhase-sourceRX1-e20-s2026092701-m3-r01；row日志logs/，release/diagnostic.stdout.log。

发布入口 E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_bounded_phase20.publish，CWD当前managed工作树。Git commit为代码身份，写release_commit.txt/resolved；一个普通账户preflight、一份archive／一次传输SHA比较、remotecompile、一次launch，独立PID/CWD/argv/GPU/loggrowth核实。系统性protocol/角色/初始化/预算/路径/非有限/启动或非法prediction错误保留failure/partial、取消pending、健康worker继续，无自动重试。

期望完整训练/实际args/默认头checkpoint/固定冻结、每row七源视图、全部曲线/配对修复新错/分层/suburban绝对值与自身clean差距/实际NN教师更新和walltime、独立复算及同row报告/登记/Git。IQ、权重、私有ID、逐包预测、完整日志、私有运行路径不入Git。完整契约Ag/As四格交互、跨seed、正式clean＋六practical、lowurban>60%和suburban距clean≤2pp探索目标仍ACTIVE。


实际发布：VERIFIED RUNNING，执行commit 51fc25d6862b7773e8d67fcc95a852c2e52ecc86。一个archive传输比较／remotecompile／真实匹配CP两合成包smoke PASS；三row真实进程CWD/argv/GPU及存活期loggrowth已独立比对，GPU0/1/2，未修改旧任务。详细状态见[launch_summary.json](launch_summary.json)。当前已开始训练，尚无最终heldRX读出／科学判定／正式评分，不能声称有界相位Ag有效或联合成功。
