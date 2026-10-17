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
