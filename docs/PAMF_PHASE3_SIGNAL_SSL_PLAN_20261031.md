# PAMF 源重放与目标双视图的开放表征适应

Run `20261031-phase3-signal-ssl-s2026092701-r01`。在固定DURA v1.0＋原F0默认身份头之上，复制已完成P1 signal r03为两个独立学生，比较同预算source-only续训和源重放＋无标签U同记录对比。唯一主 `SSL_ratio`；不按Q选择损失、迭代或主列。

## 研究依据和固定矩阵

此前公共/类别相关统计偏移提高部分known接收但损害unknown拒识，没有满足联合保护。现有reciprocal仅训练最终特征10432参数MLP；已有源端IQ/VICReg训练没有将目标U的两种IQ增强用于深层encoder适应。此轮实际更新独立signal的Sinc、时频/PA/参考/时序主干和metric投影，原身份网络不更新。

对比表征学习可用于已知与新类混合的无标签数据；本轮借鉴[GCD原研究](https://www.robots.ox.ac.uk/~vgg/research/gcd/)的表征学习思路，只实现下面明确的目标实例对比，不继承其视觉骨干、类别数估计、聚类或成绩。未使用未知真实K或预置新类头。DANCE的[原论文](https://arxiv.org/abs/2002.07953)提供目标结构学习的背景，本轮不实现其邻域/熵拒识算法。

| 分支 | 初始化 | 新NN更新 | 实际目标数据 | 用途 |
| --- | --- | ---: | --- | --- |
| DA0 | 固定r03、原统计与两套源tau | 0 | 仅冻结测试 | 原冻结基线 |
| SOURCE | 独立复制同r03 | 444 | U0 | 匹配source-only续训控制 |
| TARGET_SSL | 独立复制同r03 | 444 | star U384、无TX标签 | 唯一目标适应学生 |

每分支报告known_support与density_ratio，六列固定 `DA0_known,DA0_ratio,SOURCE_known,SOURCE_ratio,SSL_known,SSL_ratio`，主索引5。比较三个神经状态不增加独立物理数据。所有训练先完成，再冻结全部模型并进行源统计/阈值处理，最后三类DURA预测完整后独立评分。

## 两个学生的实际训练

父r03/code dc01a402582f93e452dbaa3ef59a9fbb44151818为headless232033＋MLP102688＝334721参数，源L/OE训练2220步，特征继承原F011100步。两学生均从同一已冻结完整r03 state独立复制，不将SOURCE训练结果串接给TARGET_SSL。新增444各自单列，特征累计11100＋2220＋444＝13764、投影2220＋444＝2664；原F0仍E50/11100且232993参数。

source项严格保留r03的L跨RX且跨day的cross-set metric与OE topology KL，权重1＋1。每步L192/OE64，L真实旧六类标签合法，OE仅outside-six成员和公共RX/day，教师为已有原F0源OE单位特征。两臂使用同L_seed2026103017/OE_seed2026103018，TARGET逐步检查SOURCE已存索引相同；U抽样/增强使用独立RNG，不影响源序列。

SOURCE只计算source项，完成其全部444步并保存后才打开真实U。TARGET_SSL在相同source项上增加权重1的对称NT-Xent：同一U记录两view为正例，当前batch其它记录为负例，温度0.1、每batch64个不同opaque ID。U_rng2026103101，无放回抽取本batch；增强CPU_rng2026103102。

两view复用项目既有phase_gain_view：每条P1 packet独立global phase均匀[-pi,pi]、gain均匀[0.95,1.05]，两次独立draw。无新RMS/crop/noise/channel步骤，无clean逐条参考；增强不增加物理观测。不同record可能来自同TX，实例负例可能有副作用；不借真实U TX去除这些负例，是否改善拒识是待验证假设。

优化器全部重置为fresh AdamW/CENTER_UNITS：lr1e-4、wd1e-4、eps1e-8；Sinc坐标lr×1000、eps1e-11、wd0，沿既有物理clip5，444次clip/step。没有恢复父optimizer、EMA、硬伪标签、目标类别配额、U置信门控或查询早停。全部BN/Dropout保持eval、buffers不更新，autograd和独立学生参数更新。

首步实际F0/r03/DURA三CP的两条syntheticG10无query smoke PASS立即继续；另做无optimizer步的synthetic source/U梯度检查，确认U梯度到达Sinc、中层与投影，模型state保持。梯度smoke不用真实U、Q或TX真值。每个学生保存独立完整state而非另训身份头，严格回载，SOURCE source_only=true、TARGET_SSL source_only=false/U祖先显式保存；原F0/G/r03保持。

## 数据与角色

源数据使用已VALIDATED_ONCE的r03 `IQ_root`/feature_root：star P1 L6300（六类各1050）和OE6134（21辅助设备，TX真值隐藏），全部来自source5RX×3day。L是原源训练池L6300/U_s56700中的10%有标签，OE成员监督和V标签另披露。原U_s本轮不读；原L/OE既有观测用于重放，不生成新source IQ或new received。源RX索引1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19；日期2021-03-08/15/23已见。

target U为既有RX24-1/index26、2021-03-08的star discovery384条，capsule `p3dev-star-g10-6e1292b62ab09286ebf1a0df`、split `83c5cc1fe2f2a1512dbc1287`。仅SOURCE保存后打开public received，以同冻结DURA恢复一次384条并保存private U_restored；训练只消费其中restored，F0不对真实U新前向。算法不读U真实TX、known/unknown角色、K、配额或场景真值。适应行的未知是未注册设备，可能已无标签见于U，不冒称严格未见TX泛化；source-only控制的U曝光0单独保留。

源V27000（旧六类各4500）仅在NN和密度冻结后校准阈值，source novel3150（19-15/19-16/7-1各1050）只在全部tau冻结后推理、评分侧最后读取TX。所有source L/OE/V/NV物理互斥；两NNsource角色拟合和source V校准不开放novel真值。

两个target Q是同864物理记录、RX24-1、2021-03-15，与U及source物理互斥。旧六类共576/未知三类共288仅builder/scorer知，Q不提供角色或quota给训练/预测。clean仅冻结测试，所有训练/适应IQ均星地；TARGET_SSL一份状态同时用于clean/star，不能为clean另训练。日期已在source出现，单开发RX/model且sourceNV/Q历史曝光，不是独立盲测。正式RX24-18和未来保留query未访问。

全部星地沿G10 MIX01：600km/25MHz/2.45GHz、residual/post_sync/noEQ、refSNR30dB/reflect、郊区/城区各0.5及原六面积概率；receiver/scene/channel seeds2027/2026103001/2026104001。只有信道前一次input RMS，输出不再RMS，不重新抽信道。DURA固定R69 v1.0、E20/4940/266951参数、reuse/graph开native关、FP32/TF32off，目标端更新0。

## 冻结、密度和三条件测试

两NN全部固定444步后严格冻结，在source L/OE P1分别导出新z160，各用原source算法拟合known90/OE32（共享sourceL residual whitener、背景domain-balanced kmeans32/25次、seed2026103019）。不以V/U/Q拟合这些统计；TARGET_SSL的特征权重有U祖先，聚合元数据保留此身份，不能冒称完整bundle纯source。DA0原11数组与source tau精确复用best/code9fc23a4cb78111aa96cc72fde0d35c036a876e5d。

两个新density×两readout共四套六tau，仅源已知V27000用原exactFraction common-tail hard12（总体＋5RX＋6scene）确定；原两tau组不重拟合。sourceV不更新NN、whitener或密度。全部六组tau冻结后才打开sourceNV或任一Q；三个条件共享各方法自己的源star tau，无目标阈值/normalizer/跨Q统计。

source_DURA复用原source P1 IQ，原F0身份/DA0两列精确保留，新两NN提特征。clean_DURA与star_DURA都将各自原始public输入先过同G，再由原F0及DA0/SOURCE/TARGET_SSL同时接收这一G输出；原F0同输出闭集基线由实际本次前向计算，不以旧绕过G成绩替代。Q逐样本固定模型，batch仅工程分块。source失败不取消两个已登记Q。

六列×三条件全部预测后，唯一独立CUDAempty scorer先检查完整训练/target祖先、source统计/各tau/三组z的全部raw/margin，再分别连接sourceNV、cleanQ和starQ truth各一次。scorer不回载CP/做NN前向/训练/密度fit/阈值校准；U缓存仅可读ids做角色隔离，不读restored值或U truth。原预测/训练日志保持，不按评分回流、换主或选择性重跑。

## 验收与成本

主报逐真实TX宏A_K/R_U/H，伴随C_K、同G原F0闭集及损失pp、AUROC/OSCR、逐TX/RX/day/scene与覆盖。主SSL_ratio同时对比DA0_ratio与SOURCE_ratio的正确接收/接受/未知拒对gain/loss/net。适应成功需star A_K同时严格优于两个控制，并满足clean R_U95%、star整体及六主景R_U90%、两个target C_K90%/原F0损≤2pp/全部必需TX覆盖。单独胜DA0不能把source续训收益归于U。source unknown仅诊断，known12不是known42；小Q high缺19-15保持N/A，不能认证或事后补抽。

神经更新新SOURCE444/TARGET444，共888。每臂source exposures=L85248/OE28416；SOURCE学生113664真实前向，TARGET源同预算＋U两view56832＝170496，共284160。U实际抽取实例28416、unique384可用，实际覆盖由日志报告；训练G仅384、真实F0/r03前向0。旧F0/r03/G训练及cache成本继承，不称新444从零训练。

预测新增两学生source42584各一次、Q1728各一次，DA0仅Q1728一次，共90352 signal行；G/F0各1728。合训练新增signal374512、G2112、F01728，synthetic另计（训练parent2/student梯度16，预测3模型各2，G/F0各阶段2）。新source密度2、新source阈值4实际拟合，DA0旧统计/两tau不变。训练/sourcefit角色重复不增加独立记录。

主部署仍F0232993＋one open334721＋G266951＝834665参数；比较时三open同载是实验成本，不是主部署结构。完整三状态各11聚合与36个FP64tau共288B（原96＋新192），单主只用一份density11与六tau。private U IQ/逐行source特征不部署、不Git。分别报告训练、sourcefit/预测、CPUscore实际墙钟与显存离散采样，不冒充B1/峰值。

N607一个GPU train worker串行完成两臂，之后GPU预测、独立CPU评分；actualfree≥4096MiB、每卡本任务新增≤2，保留其它owner健康进程。一个actualcode commit、focused角色/梯度/匹配/冻结顺序负测和真实CP smoke、一直接P0/P1review、最小登记/不可覆盖root/单preflight/archive SHA比较/compile、PID/CWD/argv/GPU/log增长及终态。非法祖先/角色、nonfinite、输出碰撞、无法合法预测等保留失败，无partial评分/自动重启；UNKNOWN核原PID/root，科学低分不停止健康任务。

本轮无未知发现或匿名注册，A_N/H_OW为N/A。完整多RX/seed、未来匿名类A_N80%、旧类注册损≤2pp和后续未注册保护目标继续ACTIVE，不以小开发池/单项通过替代。
