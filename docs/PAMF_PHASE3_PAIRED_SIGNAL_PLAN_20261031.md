# Phase3 同记录接收／恢复信号的开放表征学习

Run `20261031-phase3-paired-signal-s2026092701-r01`。沿当前 PCD 研究基线，检验同一观测的星地 received P0 与 DURA 输出 P1 能否提供互补的开放身份信息。既有 paired evidence 只在冻结 F0 描述符后训练 MLP，源 RU43.7143%未达标；本轮学习位置改为信号分支及其端到端融合，不能继承旧方法成绩，也不预先断言 DURA 删除了身份。

## 固定模型与三行矩阵

原 F0/default6、DURA v1R69、原 pure-source r03 和历史 PCD 均冻结。每个新学生独立从 source-only r03 E_SIGNAL2220（code dc01a402582f93e452dbaa3ef59a9fbb44151818）初始化共享 SignalOpen334721，冻结其 Sinc48 与 buffers，更新其余334673参数。原F0身份始终只用P1，不更改默认头或闭集预测。

共享编码器对两输入各前向一次，得到单位z0/z1∈R160，第二输入始终P1。新残差为 `delta=.1/sqrt(160)*tanh(Linear320to160(concat[z0-z1,z0*z1]))`，最终z=normalize(z1+delta)。新Linear含bias，全零初始化，51360参数，delta L2≤.1。数学初值是原P1表示，FP32再次归一化不宣称逐bit相同。单学生386081，训练386033，单主F0＋学生＋G共886025参数；临时目标16×160原型不部署。

| 行 | 第一／第二输入槽 | 源训练 | 目标无标签项 |
|---|---|---|---|
| SOURCE_P1 | P1/P1 | L192/OE64，444步 | 关闭，U0 |
| SOURCE_PAIR | P0/P1 | 同上、同索引、同初始化 | 关闭，U0 |
| TARGET_PAIR，唯一主 | P0/P1 | 同上、同索引、同初始化 | 当前PCD global/local，U576 |

SOURCE_P1在模型内部屏蔽received，第一槽改用P1；P0不能经其特征/归一化/统计拟合泄漏。两次共享编码前向及名义结构相同，但有效输入信息不同，此项不伪称信息容量相同。SOURCE_PAIR−SOURCE_P1只诊断源学习阶段的联合输入作用；TARGET_PAIR−SOURCE_PAIR用于同结构适应比较。旧SOURCE444和旧PCD444读出完整精确复用；新旧模型参数量不同，不能将与旧PCD的全部差异归因于单个损失。

源目标保持crossset metric＋OE topology KL＋10×原r03源P1 z锚定。源L/OE采样2026103017/18，全部三臂逐步核对已封存原SOURCE444的同一索引。所有学生独立从同一原r03及零残差初始化，不串接源学生或target-adapted权重。SOURCE_P1和SOURCE_PAIR先完整封存，再打开U。

TARGET_PAIR原样保持当前PCD：原r03弱P1 parent_z确定16容量spherical kmeans++（seed2026103113、25Lloyd）及固定非self近邻；16不是真实未知K。每步当前学生对全部弱U无梯度前向，eps.05/20轮log Sinkhorn停止梯度分配；两强U64各以温度.1预测自身及固定近邻分配，global＋local权重.1；U_rng2026103112。它仍含固定邻接假设，不把本轮输入变化写成已修复该假设。

U P0取已有public固定received，P1三视图复用fresh r01缓存。对P0/P1同步重放seed2026103114的phase/gain（[-pi,pi]/[.95,1.05]）变换；重放P1须与既有强视图一致以确定配对正确，不能把不同抽样或不同物理记录拼成正对。不生成新星地实现、不RMS、不读clean参考。各臂AdamW lr1e-4/wd1e-4/eps1e-8、normclip5、444步，target原型每步单位化；没有EMA/目标TX标签/真实K/配额/查询早停。

## 数据和冻结三条件

源合同392005不变：L6300、OE6134用于重放与冻结后密度，原Us56700只有祖先，本轮新读0；V27000只校准阈值，NV3150只冻结预测及独立truth-last。源五RX为1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23；旧六TX为14-10/14-7/20-15/20-19/6-15/8-20。NV为19-15/19-16/7-1各1050，OE仅outside-six成员及公共RX/day，没有TX/K。source V已参与校准，不能称独立已知测试。

目标开发RX3-2/index34：早日03-01星地U576；未来03-15/23 Q5525/4713，共10238条物理记录，旧4508／四未知5730。三早到新设备5-5/10-7/2-19与后来专有7-11的角色、真实K和配额只builder/scorer可见；后来7-11整类不在U。Q旧六类1491/1182/150/406/1186/93；未知1454/1439/1427/1410。03-23缺旧20-19/8-20，完整AK/CK/H保持NA，不缩小名单认证。源/U/Q物理互斥，clean/star Q同canonical同顺序。

所有Phase1后学习输入仍是同一既定G10 received和冻结DURA输出，不增加独立观测。原G10为600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect，一次inputRMS/输出无RMS，receiver2027，源scene/channel2026103001/4001、未来2026103133/34。source/跨RX clean/跨RX星地全部接入固定DURA v1R69（E20/4940、266951、reuse=True/graph=True/native=False、FP32/TF32off）；新增P0只用于开放证据，原F0闭集保护始终同P1。clean只冻结测试，没有clean U或clean训练。

三学生冻结后，各由源L/OE拟合known90/OE32及L共享残差白化/固定10%收缩、背景容量32/25Lloyd/seed2026103019。然后源V仅按原known12规则校准两分数各6τ，42组完整诊断；3套密度／6组新6τ=36新值，加旧4组24值，共60 FP64=480B。全部阈值先于NV和Q冻结，三条件共享自己的源阈值。

十读出依次为SOURCE/OLD_PCD/SOURCE_P1/SOURCE_PAIR/TARGET_PAIR各pooled_ratio/predicted_ratio，唯一主为TARGET_PAIR_predicted_ratio。全部10×3预测固定后，独立CPU才连接source NV、cleanQ、starQ三truth各一次；Utruth0。主与SOURCE_PAIR/OLD_PCD的星地AK须提高且RU不降，并满足clean RU≥95%、星地RU≥90%、CK≥90%、原同P1 F0闭集−AK≤2pp及主景覆盖才记适应通过；不能挑pooled或源控制换主。

报告逐真实TX宏AK/RU/H，伴CK/原F0闭集损、AUROC/OSCR、RX/day/TX/六scene、同ID净纠错与三行差分。无发现/匿名注册，AN/HOW=NA；单开发RX/seed、源V校准、NV/Q已曝光、global historyUNKNOWN保持。正式RX24-18不访问，原PCD研究基线和完整Phase3目标ACTIVE不因本行低分改写。

## 执行、成本与停止

默认N607，一个训练进程内三臂顺序执行，另预测/独立CPU评分顺序串联；每卡新增最多2个本任务GPU worker，实际free≥4096MiB，其他owner不停止/修改。先真实F0/r03/DURA无query合成smoke，PASS立即继续；合成检查零初值、delta界、共享编码、P1控制独立性、配对增强、源/U隔离及严格三臂回载。一个候选一次P0/P1审查，实际代码commit、最小登记、不可覆盖root、单包一次SHA比较/remotecompile、独立PID/CWD/argv/GPU/log-growth及预测后评分，白名单外审批不阻断。

新优化1332。每source臂复合信号113664次、共享base前向227328行；target复合426240、base852480；合计base1307136，另源parent_z12434一次。训练F0/G真实0，源及U P1复用既有DURA缓存；U P0新增消费576但不是新物理观测。评估3学生复合source127752＋Q61428=189180，base378360行；F0/G各20476，三学生共用同一query恢复结果。合成计数另列，祖先F0/r03/旧PCD/G/cache成本保留，不以参数量推断B1时延。

非法祖先/角色、非有限值、冻结状态变化、输出冲突或无法产出合法prediction记技术失败并保留；科学低分完整完成三臂与全部测试，不热改、补轮或按Q改融合界/容量/阈值。具体实际commit/命令/环境/CWD/路径/GPU/owner和产物见run experiment.json。
