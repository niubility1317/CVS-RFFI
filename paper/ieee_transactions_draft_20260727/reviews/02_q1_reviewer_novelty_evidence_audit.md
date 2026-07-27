# CVS-RFFI匿名审稿：创新性、证据与Q1投稿风险审计

审稿角色：IEEE TIFS/TMC/IoTJ级别严格匿名审稿人
审查快照：Git提交`df68c515`，2026-07-28
审查对象：`manuscript.tex`、`claim_evidence_matrix.md`、`项目.md`、Phase1 ADV3B02审计和D92/RTB-IDR完整报告
建议决定：**Reject and resubmit after substantial new evidence**

## 1.总评

这篇稿件已经形成一个有价值的研究问题：地面端在接收机不重叠、TX标签有限的条件下学习表征，部署端只凭冻结bundle和目标接收机上的少量旧/新类support完成适配与注册。访问边界、单物理样本单接收观测、query不可达、全注册类独立决策和truth-side scorer均比常见RFFI论文更严格。稿件也如实披露了当前结果的负面部分，没有把Role-Oracle、不同权限基线或轻量分类头误写成部署成功。

这些优点目前不足以支持一区录用。核心障碍不是语言表达，而是场景尚未闭合、主方法的因果证据不足、Phase1结果与当前协议不一致、Phase2主候选被现有报告明确标记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。当前证据只能支持“提出一个严格协议并完成一个任务均衡协方差诊断”，不能支持“提出并验证了一个完整、有效、可星上部署的两阶段方法”。

如果现在投稿，我预计外审结论会是拒稿，而不是可通过文字修改解决的大修。最可能的理由依次为：主方法没有可晋级结果；两阶段模块缺少消融；外部基线不可比；Phase1存在选择偏差和协议错配；LEO与星上部署仅为代理；算法创新容易被视为已知组件拼接。

## 2.分项评分

评分采用1—5分，5分代表达到顶级Transactions标准。

|维度|评分|审稿判断|
|---|---:|---|
|问题重要性|4.0|跨接收机、标签受限、少样本注册和星上资源的交集值得研究|
|场景定义|2.0|访问协议清楚，但链路方向、发射机角色和星上任务尚未闭合|
|Phase1方法新颖性|2.0|任务特定组合有潜力，当前仍是多组件堆叠，缺少容量匹配消融|
|Phase2方法新颖性|2.5|统一support-only状态编译器有结构价值，但现有因果证据只覆盖0.5/0.5协方差|
|理论深度|1.5|没有命题、风险分析或类规模不平衡下的推导|
|实验完整性|1.5|执行日志完整，但缺当前协议确认、主方法基线和模块消融|
|统计可信度|1.0|无置信区间；开发与最终确认未隔离；125不能当作125个独立重复|
|基线公平性|1.5|CSIL/MoPC-HR权限和base质量不同；支持型matched基线尚未完成|
|外部有效性|1.0|WiSig/ManySig与模拟残余信道，未验证真实卫星或星载接收机|
|星上部署证据|1.0|仅有分类头存储/MAC解析，完整注册器仍依赖FP64和Python栈|
|可复现性|2.0|内部追溯较强，公开数据清单、代码、hash和环境仍缺失|
|安全与伦理叙事|1.0|没有威胁模型、攻击评估、错误认证代价或负责任使用边界|

## 3.当前稿件最强的部分

1. **协议边界可审计。**`p2_min_v1`明确禁止runtime source replay、query truth、old/new query role、类别配额和跨query重排。该边界比“few-shot adaptation”这一宽泛表述更有研究价值。
2. **两阶段任务不是简单的预训练加分类头。**Phase1负责在标签受限的source receivers上构造表征和封存知识；Phase2负责在目标接收机上完成旧类适配与新类注册。这个生命周期划分可以成为论文主线。
3. **D81/D92配对诊断是可信的局部证据。**相同receiver、seed、support/query和特征下，只改变任务均衡协方差，能够隔离Eq. (balanced)的影响。稿件同时报告旧类改善和新类下降，没有选择性披露。
4. **资源表述较克制。**16.11 KiB和7,488 MAC/query只归于26类编译头；11.15—11.74 GMAC注册上界、FP64工作区和FP32解码均被保留。
5. **限制陈述诚实。**稿件没有把WiSig/ManySig称作卫星数据，也没有把模拟LEO残余信道称作在轨验证。

这些优点应保留，但它们主要证明研究治理与报告质量，不能替代方法有效性。

## 4.主要问题

### Major 1：星上任务、链路方向和被识别对象没有形成唯一场景

稿件同时使用“spaceborne monitor”“satellite-to-ground channel”“satellite transmitter fingerprinting”和“onboard deployment”。这四个概念对应的系统并不相同：

- 若目标接收机部署在卫星上，地面终端或其他航天器是被识别发射机，传播方向应是ground-to-space uplink或inter-satellite link；
- 若目标是识别Starlink等卫星发射机，接收机通常位于地面，不能据此声称星上推理；
- PAST-AI和SatIQ证明的是从接收信号识别卫星发射机的可行性，不能自动证明“星载接收机识别地面终端”的场景；
- 当前目标receiver仍来自WiSig/ManySig，模拟器只叠加传播扰动，没有包含卫星payload receiver的RF/ADC/AGC链路。

这是P0级问题。作者必须在系统模型首段给出唯一的数据流：谁发射、谁接收、RFFI在哪里运行、旧类和新类各是什么实体、识别结果触发什么操作。图1应画出ground training、uplink/downlink、target receiver、support enrollment、query prediction和ground truth所在位置。

如果Introduction加入Starlink，应把它限定为大规模LEO星座带来的运维动机，不能暗示本文使用了Starlink信号、波形、频段、payload receiver或运营数据。当前2.462 GHz WiSig代理不能验证任何命名星座。可接受写法是“modern LEO constellations such as Starlink motivate scalable onboard or edge-side emitter attribution”；不可接受写法是“the proposed method secures Starlink”或“validated for Starlink”。

### Major 2：当前没有可作为论文主结果的Phase2方法

D92报告将RTB-IDR标记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。在\(K=10,C_{\mathrm n}=20\)时，注册后旧类准确率为71.333%，最低旧类准确率为42.667%，新类准确率为68.150%；\(K=1,C_{\mathrm n}=20\)时，区别性模块回退，配对增益为0。项目目标是否写入论文并不改变这一事实：当前绝对结果低，且K1没有方法增益。

严格配对结果只证明任务均衡协方差在大规模注册时提高旧类2.622 pp、最低旧类4.600 pp，同时降低新类0.653 pp。它不能证明完整RTB-IDR优于原始backbone、ProtoNet、qKNN、普通shrinkage LDA或其他support-only头。

作者必须在以下两条路线中选择一条：

1. 提出新的、冻结后可晋级的Phase2主候选，并在完整确认矩阵上证明相对matched support-only基线的优势；
2. 将论文重构为协议/benchmark论文，公开capsule、scorer和基线套件，不再把D92写成性能型主方法。

目前将D92作为方法主角、同时在Discussion承认它不可晋级，会使审稿人直接判断“论文尚未完成”。

### Major 3：Phase1虽然篇幅充足，但证据仍不与Phase2等权

Method对Phase1的描述已经扩展，但Results只有一张历史结果表。该表存在三项关键问题：

- 当前协议要求0.07/0.63/0.30，而ADV3B02实际使用0.10/0.70/0.20；
-ADV3B02是32个候选中依据同一验证面选出的候选，存在selection multiplicity和winner's curse；
-对照是“ADV2 average”，不是相同seed、相同split、相同训练预算、相同容量下的严格paired baseline。

89.18%、84.89%、75.55%和68.77%只能称为历史内部审计结果。它们不应在最终摘要中作为当前协议下Phase1已经确认的性能。Phase1若要与Phase2同等构成贡献，至少需要当前协议下的多seed冻结复验、参数量匹配基线和A/B/C/D四个模块级消融。

### Major 4：创新性尚未超过“已知组件的任务化拼接”

Phase1组合了SincNet、PA多项式、CosFace、DANN/GRL、正交约束、MixStyle、EMA伪标签、分位阈值、CVaR式尾部风险、prototype、source episode和LEO增强。Phase2组合了FFT/RF手工特征、Cauchy加权中心、Ledoit–Wolf收缩、0.5/0.5协方差、LDA、full/block融合、Fisher残差和INT8量化。

各组件均有明确先例。论文目前用“lifecycle and coupling”解释创新，但没有实验回答以下问题：

-双表征是否优于参数量匹配的单embedding，而不是仅增加容量；
-receiver-day阈值是否优于相同coverage的全局阈值；
-尾部角几何是否独立改善worst receiver，而不是依赖选出的checkpoint；
-身份保持课程的MixStyle、source episode和LEO CE是否互补；
-RTB-IDR的稳健中心、三视图、双几何、Fisher残差和量化各自贡献多少；
-固定0.5/0.5是否有任务级不变性、风险或条件数方面的理论依据。

在没有这些证据时，顶级审稿人会把主要算法贡献压缩成“对旧/新协方差取平均后做LDA”。该操作直观但较简单，难以单独支撑TIFS/TMC/IoTJ长文。

建议增加至少一个形式化结果：说明样本数加权协方差如何随\(C_{\mathrm n}/C_{\mathrm o}\)偏向新任务；证明固定任务权重对任务内类别复制或类别数变化的某种不变性；给出收缩、正定门和仿射编译的数值稳定条件。理论不必宏大，但必须解释为何该设计不是经验拼装。

### Major 5：缺少能够支撑完整方法主张的消融

Phase1技术报告已给出合理的A—D消融框架，但这些实验尚未出现在论文证据中。Phase2只消融了任务均衡协方差。当前稿件不能归因以下表述：“asymmetric identity--nuisance representation有效”“multimodal representation有效”“robust centers有效”“Fisher residual有效”“quantization不损失性能”。

必须补充的最小消融见第7节。所有消融应使用同一capsule、同一physical IDs、同一receiver/seed/scenario/K/\(C_{\mathrm n}\)行，并报告旧类、新类、\(H\)、floor和资源，而不是各取一个最优数字。

### Major 6：外部基线表不构成公平比较

Table external中CSIL和MoPC-HR的old-pre分别约42.8%和45.3%，而RTB-IDR为86.1%。这说明三种方法进入增量阶段时的base representation、训练权限和初始状态不同。虽然正文承认“不构成clean ranking”，把这种表留在主结果区仍会被视为用明显失配的基线强化主方法。

主表必须加入相同frozen backbone、相同support、相同query、相同seed、相同新类规模下的support-only基线：

-frozen cosine/linear head；
-nearest class mean和ProtoNet；
-identity-only单qKNN；
-288维joint-feature qKNN；
-普通pooled shrinkage LDA；
-class-balanced但不task-balanced的LDA；
-ridge或multinomial logistic support classifier；
-D81及完整RTB-IDR。

CSIL、MoPC-HR、DPDS、FSCIL-SEI等原论文流程可以保留在单独的“different lifecycle”表或附录中，但不能与matched基线共享“best method”视觉排序。

Phase1同样需要匹配基线：CE-only CV-SincNet、parameter-matched single embedding、DANN、MixStyle、接收机解耦方法和channel-robust方法。标签率、训练epoch、选择规则和source receivers必须一致。

### Major 7：125个job不能替代统计设计，且存在研究者层面的query过拟合风险

每个slice只有5个receiver×5个seed的25个matched row。三个scenario、嵌套新类集和同一frozen backbone造成强相关，不能把125个job或375个scenario cell视作独立样本。当前没有置信区间、配对分布、receiver-level方差或多重比较控制。

更严重的是，predictor/query隔离只阻止算法在一次运行中读取query，不会阻止研究者在多个D版本、权重和候选之间反复查看同一query结果。现有Stage2路线经过大量探索，D92矩阵属于开发诊断，不是未触碰的最终确认。

最终统计方案应满足：

1. 以receiver为主要cluster，seed为receiver内重复，scenario和class为更低层单位；
2. 使用receiver-clustered bootstrap或分层bootstrap报告95%置信区间；
3. 对主要配对效应报告逐receiver、逐seed差值和效应分布；
4. 预先固定主要终点，例如\(H\)、\(A_{\mathrm o}^{post}\)、min-old和\(A_{\mathrm n}\)，并用Holm或FDR控制多重比较；
5. 将开发receiver/seed/capsule与最终确认receiver/seed/capsule隔离；
6. 新方法的最终确认只运行一次，失败后不得在同一确认集继续调参。

### Major 8：LEO模拟器可作为压力代理，但不能支撑Named-constellation或轨道外推

当前模拟器作用于已经包含地面传播与接收机响应的WiSig/ManySig IQ，再叠加一个后同步残余链路。它在25 MS/s、256个complex samples、2.462 GHz下运行；绝对FSPL不进入IQ，显式大气衰减和附加IQ imbalance关闭，bulk Doppler被视为已补偿，“rainy-link”只由post-AGC SNR、衰落、相位噪声和残余CFO表示。

这个实现适合做受控stress test，不是完整LEO link、轨道时变信道或某星座波形仿真。主要缺口包括：

-参数范围与实测LEO trace之间没有校准；
-没有展示clean、单一impairment和full channel的分解效应；
-没有频道相关、时变Doppler、同步失败或burst/contact-window动态；
-没有证明三种scenario的physical IDs、支持/查询和样本数完全匹配；
-没有跨载频、跨波形或跨硬件前端证据；
-“rainy-link”没有显式rain attenuation，名称可能造成物理含义误读。

如无法增加真实卫星或hardware-in-the-loop证据，标题、摘要和结论应统一使用“spaceborne deployment proxy”或“LEO-inspired residual-channel stress test”，并删除任何“validated for Starlink/on orbit”的暗示。

### Major 9：星上部署必要性有动机，但星上可执行性没有证据

26类编译头的16.11 KiB和7,488 MAC/query是有用结果，但完整系统还包括冻结encoder、FFT96、RF32、support特征提取、88次component fit、FP64协方差、特征分解、Fisher候选和Python对象。D92报告给出的11.15—11.74 GMAC只是审计上界，不是目标处理器实测。

要保留“onboard”作为贡献，至少应在一个代表性ARM CPU、嵌入式GPU、DSP或FPGA平台报告：

-单IQ encoder/FFT/RF延迟；
-K1/K5/K10注册总延迟和WCET；
-query端端到端延迟，而不仅是affine head；
-峰值RSS、常驻state、临时workspace；
-单次注册能耗、query能耗和持续功耗；
-FP32/FP16/INT8数值闭合、失败回退和状态原子切换；
-掉电/异常更新下的state完整性。

如果无法完成，应把“onboard deployment”降级为“designed for a compact persistent decision state”，不能声称飞行适用或星载实时。

### Major 10：可复现性仍停留在内部仓库

当前稿件含多个`[AUTHOR ACTION]`，缺少data release identifier、split物理样本数、manifest hash、公开代码URL、环境版本和artifact DOI。Phase1总目标中的\(\mathcal L_{\mathrm{risk}}\)把多个实际非零损失折叠为一项，系数和warm-up没有在稿件或算法框中完整给出。Phase2的cross-fitting、Fisher候选、安全门和量化编译主要靠文字描述，无法仅凭论文复现。

提交前应公开或随supplement提供：

-完整算法伪代码和所有active coefficients；
-receiver-day/domain定义、14个domain与7个receiver的映射；
-old/new TX清单、每个split/scenario/support/query的physical counts；
-capsule ID、split ID、channel seed和manifest SHA-256；
-软件版本、随机性控制、硬件和运行命令；
-每个主表的row-level CSV、prediction receipt和独立scorer；
-Phase1 checkpoint/bundle hash、量化格式和bundle license；
-开发/验证/确认集合的不可混用声明。

### Major 11：若目标是TIFS，必须增加威胁模型和认证指标

稿件把RFFI描述为physical-layer authentication signal，但实验只有闭集身份准确率、旧/新调和均值和遗忘。没有攻击者、非法发射机、replay、waveform regeneration、relay、support poisoning、receiver compromise或模型提取测试。闭集identification不能证明authentication。

作者必须二选一：

1. 删除“authentication/security guarantee”式表述，将RFFI定位为受控注册集合内的辅助归属信号；
2. 增加正式威胁模型和攻击实验，报告FAR、FRR、EER、ROC/DET、固定FAR下TPR、校准误差、拒识性能和恶意注册鲁棒性。

还应增加负责任使用段落：RF指纹可能被用于未经授权的设备跟踪和归属；数据采集应说明授权、许可和是否存在主动发射；RFFI不能替代密码学身份；误识别可能触发错误封禁；bundle和support更新应签名、审计并防回滚。若提及Starlink，应避免暗示作者对运营网络实施了未经授权的监听、逆向或接入。

## 5.两阶段是否都构成方法贡献

|对象|当前能成立的主张|当前不能成立的主张|审稿结论|
|---|---|---|---|
|Phase1|实现了一个标签受限、source-only的双表征候选；历史候选在旧split上有较好source DG指标|当前协议下已确认优于强基线；每个模块均有效；真正receiver-invariant或disentangled|方法描述完整，贡献证据不完整|
|Phase2|实现了support-only统一仿射状态编译器；任务均衡协方差有局部paired效应|完整RTB-IDR优于简单support头；K1有效；大规模注册达到部署要求|完整实现成立，性能贡献尚未成立|
|两阶段整体|形成严格ground-to-deployment访问合同|Phase1与Phase2分别且联合带来端到端提升|缺少交叉因子实验，尚未证明整体协同|

要证明“两阶段同等重要”，建议完成一个最小2×2因子实验：

|Phase1|Phase2|目的|
|---|---|---|
|匹配基础encoder|简单support head|全系统基础线|
|CVS Phase1|简单support head|隔离Phase1表征贡献|
|匹配基础encoder|最终Phase2 head|隔离Phase2状态编译贡献|
|CVS Phase1|最终Phase2 head|证明联合收益及交互|

四组必须共享目标capsule、support/query和统计方案。否则“两阶段等权”只是论文结构，而不是实验结论。

## 6.拒稿风险登记

|风险|概率|影响|典型审稿意见|
|---|---|---|---|
|主方法当前为negative/not-promotable diagnostic|极高|致命|“The paper does not demonstrate an effective final method.”|
|Phase1结果使用历史split和候选选择面|极高|致命|“The reported numbers do not follow the claimed protocol.”|
|创新被视为known components的复杂集成|高|致命|“The technical novelty beyond engineering integration is limited.”|
|缺matched baselines和组件消融|极高|致命|“It is impossible to attribute the reported gain.”|
|125/375被误当独立重复、无区间|高|严重|“The statistical evidence is inadequate.”|
|星上场景与satellite-to-ground方向矛盾|高|严重|“The system model is internally inconsistent.”|
|Starlink命名但无对应数据或物理模型|高|严重|“The named use case is unsupported.”|
|仅有Terrestrial proxy和残余信道|高|严重|“External validity to satellite operation is unproven.”|
|只测head，不测完整onboard pipeline|高|严重|“The deployment claim is not substantiated.”|
|TIFS缺威胁模型和认证指标|极高|致命|“This is closed-set classification, not a security evaluation.”|
|TMC缺移动系统、调度和端侧实测|高|严重|“The work lacks a mobile computing system contribution.”|
|正文保留AUTHOR ACTION与内部代号|极高|形式性拒稿|“The manuscript is not submission ready.”|

## 7.必须补充的实验

### 7.1 P0：任何Q1投稿前必须完成

|编号|实验|最小设计|必须报告|通过后才能解锁的主张|
|---|---|---|---|---|
|P0-E1|Phase1当前协议冻结确认|0.07/0.63/0.30；预先固定选择规则；至少3—5个training seeds；未用于32候选选择的确认receiver/seed|overall、strict UDU、receiver floor、min class、stress floor、95%cluster CI|当前协议下的Phase1性能|
|P0-E2|Phase1模块级消融|A0参数量匹配单embedding、B0无SSL、C0无角尾风险、D0无反事实课程；通过后再拆内部模块|同row差值、domain leakage probe、pseudo precision/coverage、Q90/Q95角半径|双表征、SSL、尾风险、课程各自有效|
|P0-E3|Phase1强matched baselines|CE-only、DANN、MixStyle、receiver-disentanglement、channel-robust；相同数据、epoch、容量和选择规则|均值、floor、参数量、FLOPs、CI|超越现有cross-receiver方法|
|P0-E4|Phase2 support-only matched baselines|frozen cosine、NCM/ProtoNet、identity qKNN、joint qKNN、pooled shrinkage LDA、class-balanced LDA、ridge/logistic、D81|全部5 receiver×seed×K×\(C_{\mathrm n}\)×scenario同row结果|完整Phase2优于简单头|
|P0-E5|Phase2模块消融|identity-only；+FFT/RF；+robust center；+task balance；+full/block fusion；+Fisher；+quantization|old-pre、old-post、min-old、new、min-new、H、F、存储和注册成本|RTB-IDR各模块贡献|
|P0-E6|两阶段2×2因子实验|基础/最终Phase1×基础/最终Phase2；共享确认capsule|四组same-row性能与交互|两阶段同等且互补|
|P0-E7|最终未触碰确认|开发完成后冻结commit、bundle、配置、primary endpoints；使用新receiver/seed/capsule一次确认|row-level结果、receiver-clustered CI、全部失败行|最终性能和泛化主张|
|P0-E8|数据与实现闭合|公开manifest、counts、hash、代码、环境、scorer和算法伪代码|复现receipt和独立运行说明|可复现性|
|P0-E9|可晋级Phase2候选|不能只沿用D92负结果；方法须在主要old/new/H/floor上达到预注册标准且没有隐藏tradeoff|完整矩阵、负结果、资源|“proposed final method”|

若目标期刊为TIFS，威胁模型、非法/未知发射机、重放/伪造或恶意注册实验也是P0；若不做，应删除authentication主张并优先考虑IoTJ或信号处理方向。

### 7.2 P1：支撑卫星与星上部署主张

|编号|实验|建议设计|
|---|---|---|
|P1-E1|信道机制消融|无LEO、仅SNR、仅CFO、仅phase noise、仅multipath、完整profile；保持同物理样本和seed|
|P1-E2|信道参数敏感度|elevation、SNR、CFO、K-factor、delay和phase-noise的低/中/高档；报告性能曲线而非三个均值|
|P1-E3|LEO trace或hardware-in-the-loop|将模拟参数与公开/授权卫星接收trace或SDR信道仿真器对齐；至少验证统计量和性能排序|
|P1-E4|真实卫星外部验证|使用合法获得的real-satellite数据完成外部test；无法完成则永久保留proxy-only标题与结论|
|P1-E5|完整端侧profile|代表性ARM/嵌入式GPU/DSP/FPGA上测encoder、registration、query、RAM、energy和WCET|
|P1-E6|量化闭合|FP32、FP16、INT8状态逐row性能差、数值误差、最小特征值和故障回退|
|P1-E7|标注与注册规模曲线|\(\rho_{\mathrm label}\in\{0.005,0.01,0.02,0.05,0.1\}\)，\(K\in\{1,2,5,10,20\}\)，多种old/new比例|
|P1-E8|跨数据集/波形验证|训练与测试跨WiSig/ManySig或另一合法RFFI数据集，证明不是特定WiFi采集流程的结果|
|P1-E9|长期漂移|receiver-day、温度、时间和重注册间隔；报告旧类漂移、rollback和更新稳定性|

### 7.3 P2：增强论文而非当前录用前提

-Phase3 unknown rejection、FAR/FPR95和开放集校准；
-support标签噪声、恶意enrollment和class-injection鲁棒性；
-多次增量session，而不只是一次同时注册；
-受限通信窗口下的注册调度、state传输和断点恢复；
-掉电、bit flip、bundle损坏、CRC/hash失败和原子回滚；
-辐射容错、热稳态和飞行软件认证相关评估。

## 8.推荐的Phase1与Phase2消融顺序

### Phase1

1. `A0_parameter_matched_single`：先排除双backbone只是容量增加。
2. `B0_no_ssl`和`B1_global_equal_coverage`：区分“用了无标签数据”与“receiver-day门控有效”。
3. `C0_no_angular_risk`、`C1_mean_geometry_only`、`C2_plus_tail`：从均值几何逐步增加尾风险。
4. `D0_no_counterfactual`和receiver challenge×LEO challenge的2×2实验：证明两类域外挑战是否互补。
5. 只对通过上述模块级检验的组件继续拆PA、DAC/RCN、GRL/orth/cons、temporal gate和softmix。

Phase1每个表应同时报告准确率、最差receiver、最差class、receiver leakage和伪标签质量。只提高overall而降低floor的模块不能写成稳健性贡献。

### Phase2

1. 从`identity-only NCM`和`identity-only shrinkage LDA`开始。
2. 增加FFT96/RF32，并单独测试固定权重4的敏感度；权重不能通过最终query选择。
3. 增加ground aggregate扰动谱和Cauchy中心，报告K1/K2 fallback。
4. 比较sample-pooled、class-balanced和task-balanced covariance，说明0.5/0.5解决的具体偏差。
5. 比较full、block和cross-fitted fusion。
6. 增加Fisher residual和safety gates，报告gate激活率、回退率和失败原因。
7. 最后比较FP32 head、单层INT8和双残差INT8。

每一步都应保留同一row的old/new/H/floor。若某模块只改善旧类而损伤新类，应将其定义为trade-off mechanism，而不是整体提升。

## 9.统计报告模板

建议把receiver视为部署域的基本泛化单位。主表报告：

-每个receiver上的paired mean和95% CI；
-跨receiver的cluster bootstrap CI；
-每个seed的差值分布；
-old-post、new、\(H\)、min-old和forgetting的same-row tuple；
-scenario×receiver交互，而不只给三个scenario均值；
-模块消融的效应量和Holm-adjusted显著性；
-失败率、fallback率和gate激活率。

query样本数可以用于估计单个receiver内的二项不确定性，但不能把数万个query当作跨receiver泛化的独立重复。嵌套new-class集合也不能当作相互独立的数据集。

## 10.建议立即降级的措辞

|当前或潜在强表述|现阶段允许表述|解锁强表述所需证据|
|---|---|---|
|“for Starlink”|“motivated by large LEO constellations such as Starlink; no Starlink signal is evaluated”|Starlink授权数据或对应硬件/波形验证|
|“spaceborne RFFI system”|“terrestrial proxy study of a spaceborne-deployment lifecycle”|真实星载receiver或HIL|
|“satellite-to-ground channel”且声称星上receiver|先明确uplink/downlink和节点角色|唯一系统模型及对应物理参数|
|“receiver-invariant representation”|“representation trained to reduce receiver dependence”|receiver probe、跨receiver基线和容量匹配消融|
|“disentangles identity and nuisance”|“encourages identity–nuisance allocation”|独立性、泄漏和干预证据|
|“RTB-IDR improves registration”|“task-balanced covariance improves old-class retention in an internal matched diagnostic, with a new-class tradeoff”|完整方法对matched baselines的确认|
|“few-shot adaptation”覆盖K1|“support-only registration; distinctive adaptation modules activate only when support variation is identifiable”|K1有效机制和确认|
|“compact onboard inference”|“compact compiled classifier state”|端到端目标硬件profile|
|“INT8 acceleration”|“INT8 state storage with FP32 decode in the current implementation”|整数kernel和实测加速|
|“rain attenuation model”|“post-AGC rainy-link stress profile without explicit atmospheric attenuation”|完整link budget和rain model|
|“authentication”|“closed-set registered-device attribution signal”|攻击者、unknown、FAR/FRR和安全评估|
|“statistically significant”|“descriptive paired difference”|预注册确认和clustered interval|
|“no prior work”|“among the studies surveyed, we found no common evaluation enforcing all listed constraints”|系统检索协议和更新日期|
|“state of the art”|不得使用|公平matched基线和外部确认|

## 11.次要问题

1. CosFace公式被称为“frozen logit”，但margin项需要训练标签；应明确这是training logit，部署embedding不读取query label。
2. \(\mathcal L_{\mathrm{risk}}\)折叠过多active losses，不利于复现，也使创新边界不清。
3. 7个source receivers、14个receiver-day domains和84个domain-class centers之间的映射没有解释。
4. “WiSig/ManySig corpus”缺少ManySig的公开来源、版本、许可和与WiSig的关系。
5. 只有6个old classes，可能使old covariance和floor结论对类别选择敏感。应增加old-class规模或多组class split。
6. nested 5/10/20 new classes会产生相关性，应披露具体类表和嵌套方式。
7. 三个scenario的physical IDs必须给出交集为零的receipt，不能只写协议要求。
8. `ADV2 avg.`没有说明由多少run构成、是否同seed、是否同预算。若无法matched，应从主表移至历史参考。
9. D81/D92是内部编号，最终论文应使用可读的ablation名称。
10. 外部比较表应增加“permission/lifecycle”列，或移到附录。
11. 论文只有流程框图，没有可读的网络架构图、状态编译图和误差分析图。
12. 应给出parameter count、encoder FLOPs和训练成本；仅报告head MAC会低估系统资源。
13. “one fixed observation”需要说明Phase1 LEO增强是否每epoch重采样，以及它与Phase2 fixed observation的区别。
14. role-blind query与role-aware registration应分开说明。类在注册时的old/new状态是已知的，query的真实role未知。
15. “class-permutation-symmetric”最多对old集合内部和new集合内部成立；ground aggregate只覆盖old类，任务协方差也区分old/new。不要声称跨角色完全置换对称。
16. Data and Code Availability、Acknowledgment、作者信息和所有`AUTHOR ACTION`必须在投稿前清除。
17. “Evidence Lock for the Submission Version”适合作为内部审计，不适合最终论文主表；最终稿应改为标准Limitations与Reproducibility段落。
18. 摘要中的Phase1数字来自历史split。当前协议复验完成前应删除或明确标为historical internal audit。

## 12.期刊适配判断

### IEEE Internet of Things Journal

三个候选中最匹配，但仍需把RFFI放入明确的satellite-IoT/IoE运维流程，完成matched baselines、两阶段消融、统计确认和至少HIL/端侧profile。IoTJ的正式scope覆盖IoT enabling technologies、communications、services、security和社会影响；当前稿件的protocol-governed edge identification可以进入该范围，但proxy-only negative diagnostic仍不足以录用。

### IEEE Transactions on Information Forensics and Security

当前不建议直接投稿。TIFS需要信息安全、authentication或forensics贡献。本文缺少威胁模型、攻击者、非法设备、FAR/FRR、伪造/重放和恶意enrollment。若不增加安全实验，应减少“authentication”并选择更偏IoT或signal processing的期刊。

### IEEE Transactions on Mobile Computing

当前适配度最低。TMC强调mobile environments、algorithm/protocol design、mobility、limited bandwidth、intermittent connectivity、power management和系统运行。本文没有contact-window调度、星上任务编排、移动链路系统实验、端侧实现或通信/计算权衡。只有RFFI算法和代理信道很难形成TMC系统贡献。

官方scope参考：

-IEEE IoT Journal Purpose and Scope：<https://ieee-iotj.org/>
-IEEE Transactions on Mobile Computing Scope：<https://www.computer.org/digital-library/journals/tm/cfp-ieee-transactions-mobile-computing>
-IEEE Information Forensics主题与TIFS范围：<https://technav.ieee.org/topic/ieee-transactions-on-information-forensics-and-security/>

## 13.P0/P1/P2修改路线

### P0：先使论文科学上完整

1. 冻结唯一星上/地面节点和链路方向，明确Starlink只是动机还是实际对象。
2. 选择论文类型：性能型两阶段方法，或协议/benchmark型论文。
3. 按当前0.07/0.63/0.30协议重跑Phase1确认和A/B/C/D消融。
4. 完成Phase2 matched support-only baselines、模块消融和可晋级最终候选。
5. 完成Phase1×Phase2的2×2因子实验。
6. 隔离开发集与最终确认集，补clustered CI和多重比较控制。
7. 补齐manifest、physical counts、hash、代码、scorer和算法细节。
8. 删除全部AUTHOR ACTION、内部代号和不对应当前协议的摘要数字。

### P1：使卫星与星上主张可信

1. 校准LEO残余信道，完成impairment/sensitivity实验。
2. 增加真实卫星或hardware-in-the-loop外部验证；做不到则永久降级为proxy。
3. 在代表性端侧硬件上测完整pipeline，而不是只算head。
4. 增加label-rate、K、old/new规模、receiver和跨数据集稳健性。
5. 若投TIFS，加入威胁模型、攻击和认证指标；若投TMC，加入移动任务调度、通信/计算权衡和系统实现。

### P2：形成更强长文

1. unknown rejection和开放集校准；
2. 多session连续注册与长期漂移；
3. 恶意support、state完整性和安全更新；
4. 能量、热、故障恢复和辐射相关工程评估。

## 14.最终审稿建议

当前版本不应以“Q1-ready manuscript”提交。最合理的定位是高质量内部初稿和实验路线图。作者已经完成最难的一部分：把访问权限、负结果和证据边界说清楚。下一步不应继续增加方法术语或Introduction篇幅，而应集中完成三个闭环：

1. 当前协议下Phase1的冻结确认与模块消融；
2. Phase2最终候选相对matched support-only基线的完整因果证据；
3. 唯一星上场景下的信道/HIL/端侧证据，或彻底降级为proxy研究。

三个闭环完成后，IoTJ级别投稿具有现实可能。若再增加系统实现，可考虑TMC；若增加真实威胁模型、攻击和认证评估，才具备TIFS方向的说服力。
