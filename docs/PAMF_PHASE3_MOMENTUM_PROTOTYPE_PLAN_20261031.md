# Phase3 固定教师与动量教师的无标签开放表征适应

Run `20261031-phase3-momentum-prototype-s2026092701-r01`。当前 PCD_predicted_ratio 继续作为研究基线；本轮从其原始纯源 r03 独立初始化两个学生，比较相同无标签目标自蒸馏目标下固定教师与动量教师的差异。唯一主方法预先固定 `EMA_TEACHER_predicted_ratio`，不按源 novel 或目标 Q 选列、checkpoint、温度或轮次。

## 依据与方法

旧 PCD 的全局平衡分配与固定最近邻局部一致性可能传播跨设备邻接错误；已有源侧邻接与簇结果不足以保证这些关系是真实设备关系。本轮只用同一物理记录的原 P1 与两份既有 phase/gain 派生视图，不跨记录构造身份正/负标签。它不改变当前身份分类器，也不预先保证拒识或未知发现改善。

动量编码器、教师输出中心化与锐化的背景来自 [DINO 原论文](https://arxiv.org/abs/2104.14294)及[作者实现](https://github.com/facebookresearch/dino/blob/main/main_dino.py)。本轮只借鉴这些机制：不是完整 DINO 复现；没有 ViT、多裁剪、65536 输出头、官方优化器或温度/动量日程，不继承其视觉成绩。本文的16容量余弦头只用于训练，不是真实未知类数。

两臂为 FIXED_TEACHER 和 EMA_TEACHER，学生各从同一原 source-only r03（E_SIGNAL2220，code dc01a402582f93e452dbaa3ef59a9fbb44151818）独立复制。仅冻结 Sinc48 与全部 buffers，更新 body231985＋projection102688；原 F0/default6、原 r03、DURA 不改。两臂 AdamW/lr1e-4/wd1e-4/eps1e-8、普通 norm clip5，各444步。它是 Phase3 独立开放分支增量，不改名为原 Phase1 SSL 或普通 PL。

源项严格保持当前 PCD 的 crossset metric＋OE topology KL＋10×父 r03 源 z 锚定。每步 L192/OE64；随机流2026103017/18与已完成 SOURCE444 全部索引逐步匹配，原 SOURCE 必须已封存且纯源、U0。目标学生不从 SOURCE444 或任何 target-adapted 权重初始化。固定 source-only SOURCE444 与旧 fresh PCD444 只复用已冻结读出，不重复训练。

U576 的三视图直接复用原 fresh r01 训练缓存：原 P1 及两份固定 phase[-pi,pi]/gain[.95,1.05] 派生视图（seed2026103114）。原 parent_z 以 spherical kmeans++/seed2026103113/25轮初始化16×160单位 head，两臂完全相同；初始化和训练均不使用 TX/K/配额。旧缓存虽含 neighbors，本算法不消费该字段，不做局部一致性、Sinkhorn、硬伪标签或 U 置信门控。

每臂教师独立从原 r03 与同一初始16头复制，所有教师参数停止梯度；center16 初始0。每步教师对全部576弱 P1 前向，以归一化 z/head 的余弦 logits 生成 q=softmax((teacher_logits−center)/.04)，q停止梯度，使用更新前center。U_rng2026103112 抽64个不同 ID，学生两强视图以温度.1预测对应同记录q，两份 CE 平均。总目标为 source＋.1×U_CE。

随后 center←.9center＋.1mean(all-U teacher logits)，每步一次。学生 optimizer.step 后将学生临时 head 单位化。FIXED 的教师网络/head不更新；EMA 的教师在学生更新后用 .996teacher＋.004student 更新对应可训练参数和head，Sinc/buffers跳过且严格不变。教师head参数不额外归一化，计算余弦时归一化。两臂的数据、head初始化、center公式、预算与前向次数相同；差异是教师权重是否 EMA 更新。没有查询时教师更新。

冻结部署只使用各自的学生 SignalOpen334721；临时head2560、教师副本和center只留审计，不部署、不作匿名注册槽或预测类别。身份仍来自冻结 Tri-SIG-TIME SSL F0 E50/11100、232993参数的原默认六类头，closed与同DURA原始基线相同；DURA v1R69 266951参数/原E20/4940、reuse=True/graph=True/native=False、FP32/TF32off不变。单主完整神经部署仍834665参数。

## 数据与固定三条件

源合同392005：L6300、OE6134用于源重放与冻结后密度；原隐藏Us56700只有祖先，本轮新读0。源五RX为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23；旧六TX为14-10/14-7/20-15/20-19/6-15/8-20。源V27000只校准拒识阈值，源NV3150（19-15/19-16/7-1各1050）只在预测完成后由独立评分器连接真值。源V已参与校准，不能当独立已知测试。

目标开发RX3-2/index34：03-01星地U576，未来03-15/23 Q5525/4713，共10238条物理记录；已知4508，四未注册5730。U中的真实构成、三可发现新设备与后来专有7-11仅builder/scorer可见，预测/训练不消费这些类别角色或数量。Q中的5-5/10-7/2-19/7-11分别1454/1439/1427/1410；03-23缺旧20-19/8-20，该日完整AK/CK/H保持NA。U与Q、源与目标物理互斥；clean/star Q同canonical同顺序。

Phase1后学习输入全部是原固定G10星地received通过DURA的P1；不新建clean U，不重抽观测，不读clean参考。源scene/channel seed2026103001/2026104001，未来2026103133/34，receiver2027；600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect、一次input RMS/输出无RMS与原fresh契约一致。clean只作冻结后的配对测试，仍先过同一DURA；新适配状态在clean与星地测试中相同。

## 冻结、评分与门槛

两学生全部444步后冻结，各自仅源L/OE拟合known90/OE32及原sourceL共享残差白化、固定10%收缩/背景32容量25次Lloyd/seed2026103019。源V不能拟合神经或密度，仅按原known12硬保护规则校准两种分数各六阈值，其余交叉组完整报告。每学生两组×六=12个新τ，两学生共24个新标量；旧四组×六=24个复用，总48个FP64=384B。

八读出为SOURCE/OLD_PCD/FIXED_TEACHER/EMA_TEACHER各pooled_ratio与predicted_ratio，唯一主是最后一列。所有密度和τ先于sourceNV与未来Q冻结；完成八读出×源/clean/星地三条件预测后，独立CPU评分器才连接NV/cleanQ/starQ三份truth各一次。Utruth读取0；独立评分器重建固定分数与原控制，不加载/更新教师或模型，不调τ。

主报逐真实TX宏AK/RU/H，伴CK、同DURA原F0闭集−AK、AUROC/OSCR、逐RX/day/TX/六scene及配对纠错。保持clean跨RX RU≥95%、星地跨RX RU≥90%、CK≥90%、原同DURA F0闭集−AK≤2pp及主景完整覆盖；主须星地AK高于SOURCE和OLD_PCD且RU均不低才报告适应成功。另单列EMA−FIXED的变化，不将PCD−旧源的差归因EMA。无发现/注册，AN/HOW=NA，完整Phase3目标ACTIVE。

## 资源、成本与技术边界

N607，单模型seed2026092701，两臂在一个训练进程内顺序执行；每卡新增最多2个本任务GPU进程，启动实际free≥4096MiB，foreign健康任务不停止/修改。先真实原F0/r03/DURA合成无query smoke，PASS立即进入训练；另验证teacher stopgrad、student梯度、源/U采样及严格checkpoint加载。一个候选一次P0/P1审查、一次实际代码提交身份、最小登记、不可覆盖root、一个release单包比较/remote compile和独立进程/GPU/log观测，不新增设计审批或额外验收链。

每臂学生真实(192+64+2×64)×444=170496行、教师576×444=255744行；两臂学生340992、教师511488，源父z缓存新算12434行，U P1和parent_z精确复用祖先，不新增真实F0/G训练前向。新增优化888，合成烟测另计。评估新学生源(6300+6134+27000+3150)×2=85168、未来10238×2条件×2学生=40952，共126120 signal前向；原F0/G各20476，query两学生共用一次DURA输出。旧控制真实前向0，祖先F0/r03/PCD/G成本不隐去。

非有限值、输入角色/祖先或实际commit不符、冻结状态变化、输出冲突、阶段无法产出合法prediction为技术失败，保留产物；低性能不停止、不改主、不追加步数。开发Q/NV已曝光、单目标RX/seed、global packet历史UNKNOWN边界保持；正式RX24-18不访问，不称独立盲测。
