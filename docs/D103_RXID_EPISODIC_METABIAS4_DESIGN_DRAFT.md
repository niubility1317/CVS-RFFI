# D103-RXID-Episodic-MetaBias4-qKNN设计草案

状态：`FEASIBILITY_REVIEW / REVISION_2_REJECTED / TARGET25_NO_GO`

日期：2026-07-24

## 1.不超过20行的可行性结论

1. D102解析实例关闭；其K1/K5/K10平均增益仅0.0358–0.0591pp，不能覆盖TX泄漏和尾部负迁移。
2. D102的TX probe为mean35.12%、max50.32%，class-LOCO有9/42个负fold。
3. source-only硬TX零空间诊断把TX probe降至mean20.87%、max32.34%，方向有效但仍未过max≤25%硬门。
4. D103只改Phase1资产学习；Phase2继续使用D102的4维类对称闭式求解和统一typed qKNN。
5. `U=roworth(WP⊥)`先消除source TX线性均值子空间，再以多尺度MMD抑制剩余非线性TX信息。
6. receiver表示使用跨day、跨TX同receiver正对；禁止把同day/cell记忆当成receiver证据。
7. `B∈R^(160×4)`在receiver-held episodic任务中学习，目标直接约束独立query上的全类qKNN代价。
8. 每个episode按类等权，K1/K5/K10等权；support/query物理ID互斥，数学view不增加K。
9. K1必须同时通过最小奇异值、条件数、先验占比、view消融、非恒零活动和独立OOF尾部门。
10. 7个receiver外层fold与42个receiver×class双留出LOCO fold均为真正nested，外层receiver和class都不参与任何选择。
11. 所有超参选择算法、候选网格、tie-break和失败规则在外层运行前冻结。
12. Phase2只封存INT8学习数组及其量化尺度；不封存TX/RX/day标签、`P⊥`、核样本、optimizer或source cache。
13. 独立线性和核TX probe逐held fold的mean与max均须≤25%，训练MMD不能自证通过。
14. matched M0、D102和D103必须同row报告old、seen-new、H、全类floor、遗忘和net correct。
15. K1/K5/K10每个receiver fold与42个LOCO fold均不得出现BA、floor或net correct负迁移。
16. `Ba`必须产生非平凡pre-ReLU mask、类别相关几何、邻居或argmax变化；共同变换抵消不算DA。
17. 完整nested流程总GPU时、单fold峰值显存和失败封口须由本地短探针实测后再冻结。
18. PS-NLTR不并入D103；只有共享同一表示base时才允许严格2×2factorial，否则只作`M_DA→M_JOINT`增量消融。
19. 任一协议、TX、tail、量化、资源或活动门失败即`PHASE1_HELD_REJECT`，不得打开Target。
20. 只有修订稿独立复审达到`P0=0、P1=0`并完成实现闭环后，才可重新讨论固定Target25。

## 2.证据起点与研究假设

### 2.1 D102已经证明和没有证明的内容

|证据|观测|D103解释|
|---|---:|---|
|K1平均ΔBA|+0.0591pp|4维pre-ReLU MetaBias并非严格失活|
|K5平均ΔBA|+0.0358pp|均值增益太小，不能支持晋级|
|K10平均ΔBA|+0.0540pp|均值掩盖1/7个receiver退化|
|TX probe mean/max|35.12%/50.32%|D102的解析`U`明显携带TX身份|
|class-LOCO|9/42个负fold|类间代价转移未被训练目标约束|
|硬零空间TX probe mean/max|20.87%/32.34%|线性TX均值子空间是根因之一，但不是全部根因|

D103的研究假设是：在不改变Phase2权限和qKNN分类规则的前提下，Phase1可通过“线性TX零空间+非线性分布对齐+跨day receiver对比+receiver-held元任务”学习一个更纯的类无关domain表示，并让4维MetaBias在独立query上产生尾部安全的真实决策变化。

硬零空间结果仅是source只读方向诊断，不是D103性能证据；它的max32.34%仍高于25%上限，且可能同时删除身份判别信息。

### 2.2 当前真实数据可行性

D102 r6的真实Phase1-held tap含8,400个互异物理样本、7个source receiver、4个day和6个TX。每个receiver均覆盖4个day、6个TX和24个day×TX cell；每个receiver的6个TX均存在跨day样本。因此“同receiver、不同day且不同TX”的正对构造在当前source-held面可执行，不需要复制物理样本或制造第二个K-shot观测。

## 3.方法定义

### 3.1 冻结主干与合法输入

冻结ADV3B02 checkpoint和既有功能tap。Phase1训练可读取source weak-IQ tap的`pre_relu`、`z_dom`、TX、receiver、day和物理ID；不得读取target、正式query或未来Target25结果。

Phase2只读取：

- checkpoint共同封存的D103只读INT8聚合资产及其量化尺度；
- 当前row固定received-IQ得到的`pre_relu u`和`z_dom`；
- 当前注册表和合法K-shot support标签；
- 状态封存后的逐样本query。

query不参与拟合、选择、回滚、阈值、温度、bank、MetaBias或qKNN状态更新。

### 3.2 线性TX零空间

仅在每个nested训练面的有TX标签source子集上，先按receiver×day配平，再计算各TX相对总均值的差向量。对其SVD得到固定rank-5行空间`V_TX`，定义：

`P⊥=I−V_TX^T V_TX`。

训练中的domain encoder为：

`r(x)=Norm(Uz_dom(x))`，

`U=roworth(WP⊥)∈R^(32×160)`。

`P⊥`只用于Phase1构造约束，不进入deployment bundle；Phase2只读取最终量化后的`U`。outer-held receiver或LOCO class不得参与`V_TX`、rank确认或任何量化尺度估计。

### 3.3 非线性TX抑制

在receiver×day配平的batch上，以固定核宽集合计算多尺度RBF MMD：

`L_TX=mean_(a<b) MMD²({r_i:y_i=a},{r_j:y_j=b})`。

训练目标中的MMD只负责优化，不能作为泄漏通过证据。通过与否由训练流程之外、容量锁定的线性probe和RBF-kernel probe共同判定。

每个probe fold必须在outer训练面内部按physical ID固定划分互斥的probe-train/probe-test，并对receiver、day和TX做同一配平。线性probe的正则、RBF probe的核宽/正则、容量、seed、tie-break和缺类失败规则在任何outer结果产生前冻结。outer probe结果只允许一次性接受或拒绝当前candidate；若据此修改loss、网格、阈值、probe或资产布局，必须创建新的candidate/version和`REENTRY_CARD`，不得在D103内部循环调参。

probe划分固定使用`SHA256(candidate_id|receiver|day|TX|physical_id|probe_v1)`排序，每个receiver×day×TX cell前60%进入probe-train、后40%进入probe-test，不足5个物理样本即该fold失败。容量集合固定为：

- 标准化32维`r`上的多项logistic regression，`C∈{0.1,1,10}`；
- RBF SVM，`C∈{1,10}`、`gamma∈{0.5/32,1/32,2/32}`；
- class weight=`balanced`，最大2,000次迭代，seed=`103713`；
- 对全部容量取最大test balanced accuracy，任一fold的mean或max超过25%即拒绝。

### 3.4 跨day receiver保持

receiver对比学习的正对必须满足：

- receiver相同；
- day不同；
- TX不同；
- physical sample ID不同。

负对要求receiver不同，并对TX和day频次配平。以监督式NCE形成`L_RX`，另加VICReg方差/协方差项`L_VIC`防止坍缩。禁止使用“同receiver且同day”的近邻作为主要正对，也禁止以day预测receiver作为成功证据。

必须同时执行：

- leave-one-receiver-out外层审计；
- leave-one-day-out内层审计；
- receiver probe与day probe；
- 跨dayreceiver检索和同dayreceiver检索的差异审计。

若receiver可识别性只在同day成立，D103实例拒绝。

### 3.5 MetaBias4元任务

适配位置保持：

`z(a)=Norm(ReLU(u+Ba))`，

其中`B∈R^(160×4)`。

每个receiver-held episode从训练receiver中构造pseudo-target，K在`{1,5,10}`中等权抽取；support/query物理ID严格互斥，各注册类逐类等权。Phase2的`a`仍由D102同一个类自由bank、`Λ0≻0`和4维闭式解得到，不增加optimizer step，不读取old/new角色。

独立query上的元任务损失为：

`L_meta=mean_c CE_c^DA+μτ logmeanexp_c((CE_c^DA−stopgrad(CE_c^M0))/τ)`。

第一项优化整体全类qKNN，第二项惩罚最差类别相对M0的代价转移。所有class使用相同公式；`μ,τ`只由nested source训练面选择。

总目标为：

`L=L_meta+λ_TX L_TX+λ_RX L_RX+λ_V L_VIC+λ_O||UU^T−I||²`。

所有系数、核宽、训练步数、学习率、量化方式和tie-break均按第4节选择；Target不得调节。

## 4.真正nested的Phase1训练和证伪

### 4.1 receiver外层

共7个outer receiver fold。对每个outer fold：

1. outer receiver完全隔离；
2. 剩余receiver内部执行固定leave-one-receiver与leave-one-day选择程序；
3. 选择面只能决定预注册网格中的超参；
4. outer receiver不得进入`U/B/bank/P⊥`、核宽、`μ/τ/λ`、epoch、量化尺度或门限选择；
5. 选择完成后只在outer receiver上评估M0、D102和D103的matched K1/K5/K10。

最终deployment asset使用全部source训练数据，按同一固定内部选择算法重新选参和训练；outer结果只决定是否拒绝整个方法，不决定最终超参。

### 4.2 固定训练常量

本candidate使用singleton配置，不从outer或Target性能选超参：

- seed=`103713`；
- Adam，learning rate=`1e-3`；
- 每fit固定20epoch×20meta step，共400step；
- 每step使用同一个K1/K5/K10三episode组合；
- balanced batch每个receiver×day×TX cell取2个互异物理样本；
- `μ=0.1`、`τ=0.1`；
- `λ_TX=λ_RX=λ_V=λ_O=1.0`；
- MMD核`gamma∈{0.5,1.0,2.0}`并取均值；
- qKNN训练温度固定为`0.2`；
- 不做基于loss或性能的early stopping。

leave-one-day只作预注册的稳定性审计，不进行网格或阈值选择。若任一day子fold不满足门，直接拒绝当前candidate；不得用其结果更换上述常量。

### 4.3 class-LOCO外层

共7×6=42个receiver×class双留出fold。每个fold同时留出一个receiver和一个class：

- 被留出的receiver不得进入`U/B/bank/P⊥`、probe、核宽、超参、量化尺度或门限选择；
- 被留出的class不得进入D103新增的`U/B/bank/P⊥`训练、probe、核宽、超参、量化尺度或门限选择；
- 评估只在该held receiver的held class上完成，并与同一物理ID和同一冻结基础checkpoint下的M0/D102 matched；
- 唯一例外是基础checkpoint已经包含的历史Phase1知识；它在全部臂中逐字节相同，不能被D103删除或重训。

LOCO的“留类”不声称从基础checkpoint中删除Phase1历史知识，只验证D103新增资产和更新是否对未参与其学习的class产生负迁移。

### 4.4 K1统计可识别性

K1的每类singleton只计一个物理样本；由同一received-IQ产生的任何数学view不得增加rank或样本数。每个K1 fold必须同时记录并通过：

- `rank(A_data)=4`；
- 预注册的最小奇异值下限；
- 预注册的最大condition number；
- `prior_fraction`上限，排除几乎完全由先验决定；
- `a`非恒零且跨独立episode方向稳定；
- 合法view开/关消融的预测稳定性；
- 独立query OOF的BA、floor、H、old/new net correct尾部；
- support自拟合指标不得作为保护或晋级证据。

数值门在任何outer结果前固定为：

- data information rank=`4`；
- `min_singular_value(A_data)≥0.05`；
- `condition(Λ0+A_data)≤10`；
- `prior_fraction≤0.80`；
- `||a||_2≥1e-4`且不超过冻结ellipsoid；
- 合法view开/关的top1 agreement≥99.5%、large-margin flip=0；
- 独立episode间系数方向余弦的中位数≥0.80。

K1不可辨识时输出协议完整的M0安全预测并标记`INACTIVE_NON_PROMOTABLE`；该fold不能计为D103成功，也不能用identity结果冲淡失败率。任一正式K1 fold inactive即拒绝整个D103实例。

## 5.资产、量化与资源

### 5.1 Phase2 deployment bundle

允许封存的学习数组只有：

- INT8`U`；
- INT8`B`；
- INT8类自由bank`g_m,t_m,Λ_m,σ_m`的量化编码；
- 上述数组逐张量或逐行量化尺度；
- 固定正标量`T,Λ0,a_max,R`的规范编码；
- 聚合、训练、nested和量化回执。

禁止封存：

- FP16/FP32学习数组或持久sidecar；
- `P⊥`、`V_TX`、TX/RX/day标签；
- source样本、核样本、exemplar、单样本feature；
- optimizer、训练episode、成员ID或可逆样本索引。

Phase2 bundle必须继续满足项目的不可变、多样本聚合、共同封存和不可独立替换要求。INT8解码只发生于既有合法推理路径，不产生可更新全精度状态。

### 5.2 资源预注册

最终全source fit的参数规模为`U5120+B640+t112+Λ112+σ28=6012`个Phase1学习参数；留一个receiver的微探针有24个cell，共5,976个参数。

本地三步微探针实测：

- 设备：RTX5070Ti，PyTorch2.10.0+cu128；
- 3次warmup+3次计时，代表step同时包含K1/K5/K10 episode、MMD、receiver contrastive、VICReg和反向传播；
- 平均0.276916秒/meta step；
- 峰值allocated21.74MiB、reserved26.00MiB；
- 288行balanced batch、24个receiver×day cell；
- loss全部finite。

完整流程按每个outer fold执行4个leave-one-day fit+1个outer fit，共`49×5=245`个fit，再加1个全source最终fit，总计246fit、98,400meta step。按本机实测为7.569GPUh；对N607采用3倍设备/实现安全因子并再加35%的probe、量化、M0/D102 matched评估、I/O和失败artifact开销，估算30.655GPUh，冻结总上限为36GPUh。

冻结峰值显存上限为4GiB/fit、完整run-root磁盘上限为20GiB。超出36GPUh、4GiB/fit、20GiB/artifact或Phase2 state/MAC任一上限时，停止本run并记`NO_PERFORMANCE_RESULT`；不得减少fold、epoch、K或LOCO覆盖来换取完成。

Phase2目标仍为0 trainable parameter、0 optimizer step、总state<80KiB且post-backbone MAC/query≤262,144；完整support编码、bank matching、两次统一重编码和4维求解均计入。

## 6.硬门和matched报告

### 6.1 协议与泄漏

- `protocol_schema=p2_min_v1`；
- source/target、support/query和物理ID互斥；
- 每物理样本一个固定weak观测；
- 标签置换等价；
- query重复推理state hash和prediction一致；
- query fit access=0；
- 线性TX probe和RBF-kernel TX probe逐held fold的mean、max均≤25%；
- receiver表示不得依赖同day/cell proxy；
- INT8 top1 agreement≥99.5%，large-margin flip=0；
- 资产闭包、state和MAC通过。

### 6.2 性能与活动

source-held每个matched row同时发布M0、D102和D103的：

- 匿名source-held lifecycle proxy的B-old、A-old、seen-new和H-old-new；
- 全注册类floor和逐类old accuracy；
- forgetting；
- wrong→correct、correct→wrong和old/new net correct；
- ReLU mask flip、pairwise angle、邻居贡献、margin和argmax变化；
- `a`、information rank、最小奇异值、condition和prior fraction。

这里的source-held 6个TX均已被基础checkpoint见过；LOCO只把class排除于D103新增资产。因此`seen-new/H/old-new`只能标为匿名source-held lifecycle proxy/diagnostic，不能形成真实`Y_new`证据、Stage2-C新类注册声明或Target25授权。

receiver-held K1/K5/K10的每个fold以及42个LOCO fold均要求BA、floor和net correct相对M0非负，且联合指标严格优于D102。`Ba`若只产生共同加性、归一化抵消或无类别相关决策变化，则标记`INACTIVE_NON_PROMOTABLE`，不把数值非零当作DA成功。

所有门只使用source-held结果。任一失败均关闭D103实例，不扫描Target门限，不从失败fold选择性回退，不删除负fold，不启动Target25。

## 7.PS-NLTR隔离与factorial规则

PS-NLTR当前不进入D103 DA资产，也不能掩盖D102的TX泄漏。只有某个冻结DA/base先独立通过同一TX前置门后，才允许单独设计：

|臂|表示DA|分类头|
|---|---|---|
|M0|共同冻结base|原typed qKNN|
|M_DA|共同冻结base+D103|原typed qKNN|
|M_HEAD|共同冻结base|PS-NLTR|
|M_JOINT|共同冻结base+D103|PS-NLTR|

严格factorial要求四臂共享同一base、checkpoint、capsule/split、support/query、seed、资源口径和矩阵。若PS-NLTR只能在D103表示上工作，则不定义异base的`M_HEAD`，只报告`M_DA→M_JOINT`的增量head消融，不能声称2×2factorial或独立head主效应。

K≥2的head约束必须按physical ID cross-fit；K1只能使用Phase1冻结cap和独立query OOF稳定性门，不能使用同一singleton support self-margin自证。若无可行非零缩放，仍须输出全query的M0安全预测并标记`INACTIVE_NON_PROMOTABLE`，不能把identity计为head成功。

## 8.进入下一状态的条件

本稿已获准进入`FEASIBILITY_REVIEW`，但不授权正式训练器、N607训练或Target25。唯一允许的代码是第9节限定的本地非性能可行性微探针。进入`DESIGN_FROZEN`前必须：

1. 独立复审确认P0=0、P1=0；
2. 通过真实tap只读检查确认跨day配对和nested fold可构造；
3. 独立复核微探针资源外推、36GPUh/4GiB/20GiB上限和失败封口；
4. 独立复核singleton训练常量、probe容量、全部数值门和tie-break；
5. 建立D103需求—代码—测试—artifact追踪表；
6. 明确实现文件、测试文件、runner和非覆盖run ID。

在此之前，D102保持`PHASE1_HELD_FALSIFIER_REJECT / TARGET25_BLOCKED`，D103保持`TARGET25_NO_GO`。

## 9.唯一获准的本地可行性微探针

`FEASIBILITY_PROBE_NON_PERFORMANCE`仅可：

- 读取冻结D102 Phase1 source-held tap；
- 固定一个inner训练fold和全部常量，不读取任何outer评估结果；
- 统计跨day配对是否可构造、检查tensor shape；
- 对K1`A_data`做rank/condition机械检查；
- 执行1–3个warmup和3个计时meta step，测forward/backward峰值显存、墙钟和临时磁盘；
- 只输出资源/可构造性JSON。

微探针不得接触target、capsule、正式query或Target25，不得计算BA、TX通过率、LOCO性能，不得保存`U/B/bank`或deployment bundle，也不得据微探针结果调loss、超参、阈值或候选。结果只能决定完整D103训练在资源上“可行/不可行”，不能作为方法或性能证据。

## 10.Revision2终审裁决

Revision2独立终审为`P0=0、P1=6 / NO_GO_TO_DESIGN_FROZEN`。本稿作为被拒绝的设计记录保留，后续修改必须使用新candidate/version和`REENTRY_CARD`，不得覆盖以下问题：

1. 微探针误把tap成员`z_id`绑定为`z_dom`，K1机械rank/condition证据无效；
2. 正式训练没有把`L_s=0.07`、`U_s=0.63`、source-val=`0.30`的梯度和证伪权限写死；
3. singleton常量、probe容量和K1数值门在首次微探针后才写入，不能由原candidate自证；
4. 资源外推未解决inner leave-one-receiver文字与246fit算式的矛盾，且未实测临时磁盘；
5. 双probe聚合轴、candidate ID和量化ABI仍未完全冻结；
6. 追踪表、正式文件清单、runner和非覆盖run ID尚未闭合。

首次微探针的0.276916秒/meta step和26.00MiB reserved只能作为160维代表计算的资源shape近似；其K1最小奇异值和condition不得进入D103可识别性结论。
