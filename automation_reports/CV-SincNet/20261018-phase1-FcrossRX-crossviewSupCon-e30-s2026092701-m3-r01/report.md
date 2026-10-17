# P41：既有跨 RX 同 TX 对比监督的真实星地视图覆盖

运行前固定设计。状态 PLANNED；尚无新训练成绩，不能宣称新 A_g 或联合目标完成。起点 commit 8105911d77983fda193b5af57841801409ae6bd4；实际发布代码 commit 在 release_commit.txt/verification.json 记录。

源代码确认：原四个 L 前向中，SupCon只读取普通 pair；混合 pair 内64个真实full practical IQ只参与默认头CE/RDrop。本轮唯一假设：保留完整clean监督组织和既有对比预算，让已有真实星地观测进入同TX不同物理RX的身份证据关系，能改善low/mid urban且保持clean。P40未发现默认W间隔梯度反向，故不加梯度修正；原IQmix、RX分组、RXMIX、谱/时序/正负噪声等既定失败路线不重开。

|组|原SupCon的输入|epoch/更新|
|---|---|---|
|ORD128|原128普通maps|30/6660|
|DUP192|原128普通maps＋对应64包的同普通maps重复|30/6660|
|SAT192|原128普通maps＋已有混合前向中真实64卫星maps|30/6660|

三组均重新scratch，完整F/SIG_TIME_X3＋统一Sinc CENTER_25K/CENTER_UNITS＋NoExplicitCFO＋joint160/defaultcos30，900722参数/206state；骨干与头原流程联合训练。没有新分类器、投影器或部署模块。新模型不继承P39/历史目标暴露权重；只从自身固定stage20严格恢复model/ownEMA，重置AdamW与cos10。总每模型E30，非额外E30；三组总90模型epoch/19980更新。全30轮L均训练，前13纯L，后17保留原U/门控/EMA。普通增强冻结实际E13强度，tailSupCon按累计E21..30，不误重新warmup。

每包原.5 RD(ordinary128)+.5 RD(mixed128)，mixed中64ordinary＋64fullAs(raw)，有效ordinary/As=.75/.25，所有真实标签As始终CE，不用学生置信度过滤。每pair复用相同已生成IQ做两次RDrop，四128前向，总损失CE/RDrop绝对权重不改。原SupCon温度.2、最大系数.1、warm5/ramp10、time_down和t2.pool、detach批内均值/std与L2 eps1e-4、两个depth/两个对应dropout前向平均均保留。新增的是已有maps的视图成员，不叠加额外loss。sameTX不同物理RX为正；sameRX（包括同一物理包跨视图）从正/分母排除；不同TX为负。DUP匹配额外anchor与描述子标准化，防止把数目/归一化改变当跨信道作用。实际loss尺度仍可能改变，必须报告。

关闭P39为U教学匹配而增加的零损失学生U前向和first-U资格读出；三组均无新U目标/教师。原U阶段全部保留，U TX始终隐藏。P39只作来源及配方参考，因前向成本不同不冒充本轮同预算对照。数据保持原L6300/U56700/V27000、源RX1/3/4/6/8、days1/2/3、6TX、split392005；无训练域留出、无L_dev，不把V用于反向/EMA/归一化更新。

真实L信道保留原 perpacket low_suburban/midurban/lowurban：fs25MHz/fc2.462GHz、IQ256、residual/post_sync、EQoff G1、ref30、reflect、质量/相位/IQ补偿开启及几何/遮挡/传播/噪声/质量依赖；硬件按RX/day/eq session稳定，非TX定义。源物理raw经center crop/pad后逐包RMS，原256记录crop为no-op，As(raw)无post-AsRMS。正式缓存clean＋6practical不改。冻结源1800/2draw/3scene读出沿用P39/P24既有同一IQ（fc2.45GHz），这是公开披露的源诊断代理，与训练实际2.462不同；不是新的盲测或更弱正式信道。新增噪声比不当总SNR。

预登记否证：SAT192相对ORD128和DUP192，在每一个2draw×mid/low源格都至少+1个百分点；fullV clean相对二者均下降不超过.25个百分点。任一失败则关闭这条固定视图覆盖方案，不改系数/温度/层位/RX/seed/预算救援。即使通过，也只说明源候选资格；不是A_g或low>60/clean双突破证明。不自动扩种子；全部三行固定终点均自动clean+6预测、逐行独立truth-last评分，结果不回流设计/选择/重跑。

预算与产物：每模型6660更新、LNN3409920；预计nativeNN6255724＋冻结源12600＋正式1176002＝7444326NN（实际计数为准），共享真实CP合成烟测2＋三组synthetic1536＝1538NN，0smoke optimizer更新。新两组192anchor只增加关系矩阵/反向开销，不加模型前向；额外CPU/GPU时间实测。父成本own20已包含E30，历史诊断/模型成本单列。

代码=E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_crossview_supcon41；唯一run=/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01；唯一owner=codex/root/crossviewSupCon41；GPU遵守计入其他owner的最多3进程、每slot至少12GB空闲；普通uid，无admin操作，不改健康任务。CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_crossviewSupCon_E30_20261018_r01。命令：

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_crossviewSupCon_E30_20261018_r01/experiments/cvs_crossview_supcon41/runner.py
```

发布首步实际P39固定B0真实checkpoint严格来源/角色检查及2包合成no-query烟测；随后三组真实F合成每组512NN验证默认W/Sinc联合CE梯度、只有SAT既有aux到达As maps、aux不直接到W、ORD与旧SupCon公式相同、原4L/64As/同IQ/零额外U/同初始化，0optimizer更新；通过立即启动真实行。

技术停止：仅角色/来源/配置不符、非有限loss/grad、输出冲突、不可启动/预测失败等系统问题。保留局部产物，不自动重试，不修改未完成的科学阈值；健康任务不停止或重启。一次独立P0/P1审查、直接P1最多一次定向复审；单归档一次SHA比较、remotecompile；启动后核实PID/startticks/CWD/cmdline/GPU/log增长。

预期保留逐epoch/逐step/物理曝光序列、实际模块/config、CE/KL/SupCon与正样本覆盖、硬件/信道统计、原头冻结后源/全7正式结果、F1/最差RX/固定ORD clean正确集合保持率、训练曲线/分层图、收益与负结果、墙钟/更新/NN；不提交IQ、权重、逐包ID/梯度或运行原日志。

## 启动实际证据

运行代码 ec4c5d7f46aedb5aaece77fdd72dbe6ed4c157f2 已推送并经独立GitHub compare确认identical；单归档SHA比较及remotecompile VERIFIED/PASS。真实P39固定B0 CP2NN与三组真实F合成1536NN烟测PASS，0optimizer更新；只有SAT192辅助目标到达真实As maps（该synthetic L1=6.477223），不直接到默认W；三组初始state相同，ORD原公式等价。此为接口验证，不是收益证据。独立P0/P1发现Path导入缺失，已修复并确认，余项PASS。

三条真实scratch行已启动：ORD128 PID849390/GPU0，DUP192 PID849536/GPU1，SAT192 PID849617/GPU2；dispatcher PID849362。独立/proc核实CWD/argv/CUDA，nvidia-smi核实实际GPU PID，source角色EXACT_MATCH、旧checkpoint继承为0，stderr为空。source/teacher/默认头均按原流程更新；尚无正式新成绩。详细原始运行证据仅存本地 E:/type10-7/local_artifacts/cvs_FcrossRX_crossviewSupCon_E30_20261018，聚合verification.json已记录。

三条真实行均已完成E1/30，各222更新、113664个L NN、14208个完整As观测，额外U NN=0、target_access=false；日志已增长到约11613字节，worker仍活跃，无技术失败。独立读取3份新run自己的initial_smoke.pt逐项比较206个state tensor，ORD对DUP/SAT均0不一致（0NN/0源IQ/0target），不把后续GPU轨迹小差异写成bitwise等价。当前只证明启动和首轮执行；不报告早期源V为性能收益，不据此调整方案。

## 预热后运行与独立分析准备

同一三条PID已核实完成累计E6/30、各1332更新、681984个L NN、85248个fullAs观测，extraU NN=0、target_access=false；stdout均增长至约37400字节，原SupCon已按计划激活，observer预算检查无失败。仅报告执行进度，不把训练/源V指标当新性能收益。

独立分析程序 analysis/cvs_crossviewSupCon_E30_results_20261018.py 已完成，8项针对性复算/判据/视图归属/truth-last边界测试PASS，0真实IQ/CP/NN。它将完整解析三组90个epoch、19980步L/native/Sinc记录及全部L/U曝光，复算CE/RDrop/SupCon绝对尺度、5550个激活步/模型的真实/重复描述子覆盖、两段成本、两个draw源分层、固定ORD clean正确集合及全部正式混淆矩阵。原运行release代码不修改。

尚未完成：E30训练、完整源判据与独立源分析、各row全7正式评分、最终收益/clean代价/跨seed证据；逐包TX×RX×信道状态sidecar未保存，不能冒充已有逐包观测。先核查能否由实际L物理顺序和命名信道流精确复算，并与已有聚合记录对齐；任何额外物理生成成本单独报告。当前仅完成RX/day/scene/state/lock聚合日志的输出安排，不预先宣称覆盖验收完成。

独立分析代码commit 4f804e560bed6e29b1bc17a086aede2f9522321e已推送并经GitHub compare确认identical。源模式已在独立分析目录启动，remotecompile PASS；PID854753/startticks151256374/CWD/argv/CUDA空已独立核实，stderr为空，目标truth路径未传入。该进程等待既有4个训练/队列句柄与全部source完成产物，源输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-crossviewSupCon-e30-s2026092701-m3-r01/source_analysis_r01 当前尚未创建，不会提前写完整结论；不是新训练或额外审批/哈希环节。最近源进度ORD/DUP为E8（各1776更新），SAT为E7（1554更新），全部仍存活。详细证据analysis_execution.json。

## 传播状态覆盖复算准备

实际首批128个L包通过原base_index与既有L6300角色ID精确映射TX，未读取U真值、真实IQ值、模型CP或target。原索引中的RX/day/sig不能单独分辨TX（存在6重歧义），已使用完整base_index解决。残余信道包内传播状态在原代码中冻结；命名geometry/states流可只复算其先验抽样，不渲染真实波形、不声称重现receiverlock/AGC/噪声或准确率。已通过64次synthetic residual状态逐包一致核对与7项映射/权限/实际seed配置测试。

首个辅助程序r01因引用不存在的CHANNEL_SEEDS退出，保存failure与已完成的L映射；训练未改变。修复读取实际reference_config.channel.seeds，commit 34a67169a6bb40bac5856ad1e5b8512c5c6120eb已独立GitHub确认identical，r02独立目录编译/进程/CWD/argv/CUDA空/烟测产物均VERIFIED，等待完整冻结源记录后再逐批核对与汇总TX×RX×day×scene×state覆盖。当前未取得完整覆盖结果。最近ORD E18、DUP/SAT E17；仍为单模型seed的三组对照，无新正式成绩、无新A_g、目标未完成。

## 自身 E20 完成与后 10 轮启动

三组均完成自身stage20：各4440更新、2273280个L前向样本、284160个实际As观测，extraU NN=0、target_access=false。三条原worker PID/startticks保持，source初始化均只严格继承对应自身stage20模型/EMA，数据角色EXACT_MATCH；AdamW状态0、cos10重置，未加载其他实验权重，总预算仍每模型30轮。此时训练尚未完成，未读取正式成绩。

首次直接比对实际已保存训练记录与命名流复算：step1的64个As观测场景为midurban20、low_suburban25、lowurban19，传播状态LOS30、BLOCKED22、PARTIAL12，两组计数分别完全一致。只证明该批与原引擎一致；完整6660步/模型及TX×RX×day×scene×state覆盖仍待冻结后复算。未消费真实IQ值、模型CP、U真值或target，0NN。

## 完整 E20 执行记录复算

完整读取每组4440个L步骤及全部L/U物理曝光，三组实际顺序完全一致；各L曝光568320、U曝光396900，唯一L6300/U56700（U只数元数据，不解码TX）。SupCon各激活3330步，两个depth×两pass形成13320条描述子记录；只有SAT存在852480个真实As map条目，此数包含depth/pass重复，不是独立IQ样本数。每组L前向2273280、实际As观测284160；0额外U前向，所有卫星L参与CE。

逐步复算.5(commonCE+commonKL)+.5(extraCE+extraKL)+w*SupCon，与实际total最大绝对差ORD2.98023e-7、DUP/SAT1.78814e-7；权重warm5/ramp10、四前向、同IQ配对、无ordinary重抽、sameRX正例0全部符合。E20每组native前向3661082（student_train2670180＋student_eval594002＋teacher_eval396900）；墙钟ORD1663.8696s、DUP1670.4930s、SAT1710.3920s。不同GPU上这些墙钟不作为关系矩阵的独立因果成本。E20是自身E30内的前20轮，后10轮成本另计后合并，不重复计算父训练。源训练内容及默认头没有修改，没有读取正式准确率。
