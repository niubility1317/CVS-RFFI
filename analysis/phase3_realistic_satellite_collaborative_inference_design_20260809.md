# 面向真实部署的卫星协同推理算法设计

版本：2026-08-09

设计标识：`EV-CARE-Track v1`

状态：`DESIGN_REVIEW_MERGE / P0=0 / P1=0 / CURRENT_PHASE3_CAPABILITY_NOT_OVERRIDDEN`

本文面向CVS-RFFI/CV-SincNet的Phase3部署阶段。它把现有`CARE-PoE`技术原型扩展为可部署的“事件核验—相关性感知融合—匿名轨迹—可信确权”系统，但不把设计稿写成已经实现、已经达到unknown FAR目标或已经完成真实在轨多星验证。当前ManySig与LEO弱信道叠加仍是地面代理证据；严格same-event协同必须等待采集前生成的`emission_event_id`、逐节点`satellite_reception_id`和物理绑定receipt。

## 1.任务定义

系统接收来自动态卫星节点集合的射频观测。每个节点先独立形成冻结本地证据，协同层再对一个物理发射事件输出三态决策：

```text
registered(identity)
unknown
defer
```

协同层同时维护跨过境的`anonymous_entity_id`，但匿名实体只表示多个事件可能来自同一物理射频链，不是语义身份。只有外部可信确权产生`registration_authorized=true`后，系统才重新采集K个独立物理事件并交给Stage2-C注册；历史unknown观测永不转成support。

设计目标不是让多节点“投票更自信”，而是在现实条件下解决六个具体问题：

1.不同接收机链路响应、SNR、遮挡和多普勒条件导致的证据异质性；
2.节点缺失、迟到、断链和算力差异形成的动态节点集合；
3.同波束、同中继、相邻链路或同一前端产生的相关证据重复计权；
4.本地registered、unknown和defer之间的冲突；
5.同一物理事件的多节点接收与跨时间匿名轨迹关联；
6.未知拒识、运营确权和新类注册之间的职责隔离。

## 2.现实部署架构

### 2.1三级计算位置

|层级|部署位置|主要职责|禁止职责|
|---|---|---|---|
|L0本地证据层|卫星载荷或接收站边缘设备|IQ预处理、冻结Phase1/合法Phase2状态推理、本地质量估计、不可变证据封存|读取query真值、跨query类别配额、在线重训Phase2|
|L1事件融合层|星间协同节点、星座边缘节点或地面网关|事件完整性核验、相关组构建、协同三态决策、贡献解释|把不同时刻观测拼成same-event、根据性能挑节点|
|L2轨迹与确权层|地面网关或受控运营中心|匿名轨迹维护、外部credential核验、授权状态机、fresh-K任务编排|把anonymous ID直接当真实身份、把历史unknown改为support|

星间链路充足时，L1可在星座边缘节点完成低时延事件融合；带宽不足或节点无法互联时，L0先输出单节点结果，L1在地面网关完成延迟融合。L2需要跨过境持久状态、运营凭证和审计能力，默认放在地面。

### 2.2运行模式

系统必须显式报告当前运行模式：

- `SINGLE_NODE`：只有一个有效接收节点，逐字节复用本地决策；
- `VERIFIED_SAME_EVENT`：物理绑定证明多个接收记录来自同一发射事件，可进行严格事件级融合；
- `PROXY_MULTI_RECEIVER`：仅以采集元数据在truth不可见时构造代理分组，先封存预测再由独立scorer连接真值；可研究融合机制，但不能声明真实同步多星或真实unknown能力；
- `ASYNC_TRACK_ONLY`：不同发射事件只进入匿名轨迹关联，不进行same-event概率融合；
- `DEGRADED_INTEGRITY`：绑定、时间、哈希或节点身份不完整，输出defer并保留故障原因。

## 3.输入与不可变证据

### 3.1本地证据

节点`m`对单个`satellite_reception_id`输出：

```text
z_id^(m)       身份表征
z_dom^(m)      接收机／信道域表征
q^(m)          观测质量与可靠性
d_class^(m)    到全部注册身份的距离
e_unknown^(m)  连续未知证据
p_local^(m)    全部注册身份加unknown的本地分布
local_decision registered/unknown/defer
reason_code
```

证据还必须携带`node_id`、共同`base_bundle_id`、`feature_space_id`、`class_registry_id`、`calibration_contract_id`、节点合法Phase2`state_hash`、时间戳、频率、波束、轨道／位置可见性、延迟和canonical evidence hash。不得包含truth、真实old/new/unknown角色、scorer输出、raw IQ、样本成员清单、节点自报的`correlation_group_id`或`registration_authorized`。

接收节点可以具有receiver-specific的合法Phase2适配状态，不能要求所有`state_hash`相同。任务roster必须在query到达前封存每个节点允许的`state_hash`及其共同base bundle、类表和校准合同。只有`class_registry_id`、类别顺序、unknown维度和概率校准语义一致的节点才能进入同一融合；不同校准空间的分数不得直接PoE。

### 3.2事件绑定

严格协同的主键为：

```text
一个emission_event_id
→多个satellite_reception_id
→仍计一个shot
```

`emission_event_id`必须由采集系统在标签可见前生成。系统优先使用同步时间、频率占用、波束可见性、轨道几何和采集触发receipt核验绑定。`z_id`、预测类别、unknown分数或后验真值不能生成事件ID。

当无法证明same-event时，算法只能建立`proxy_group_id`或进入`ASYNC_TRACK_ONLY`。软关联概率不能升级为`verified_physical`，也不能增加K-shot计数。

### 3.3完整性门

进入融合前逐项检查：

1.证据schema、共同base bundle、feature space、class registry和calibration contract一致；每个节点的receiver-specific`state_hash`分别命中预封存roster；
2.`satellite_reception_id`全局唯一，同一event内`node_id`唯一；
3.到达时间不晚于冻结deadline；
4.节点在签名roster中，时间、频率、波束和可见性不矛盾；相关组由roster的前端／拓扑／中继registry派生，节点不能自报或改写；
5.物理binding sidecar与冻结binding root一致。

同一event出现ID、hash、bundle或binding冲突时，正式路径输出`defer/EVENT_INTEGRITY_FAILURE`。不能静默删除坏节点后把剩余结果包装成完整正式协同；可另行输出降级诊断，但必须保留模式标记。

## 4.EV-CARE事件级融合

### 4.1预查询可靠性

对deadline前有效节点定义：

\[
r_m=I_m\cdot\operatorname{clip}\left(q_m\pi_m v_m
\exp(-\lambda_t\Delta t_m),0,1\right)
\]

其中`I_m`是完整性指示，`q_m`来自受bundle合同约束的本地冻结质量输出，`pi_m`是任务开始前封存的节点健康先验，`v_m`由可见性、波束和频率一致性计算，`Δt_m`是相对事件deadline的延迟。融合器把`q_m`裁剪到roster规定范围，并核对质量receipt；`pi_m`不能根据当前query真值、当前批次性能或预测类别更新。节点失陷时仍可能伪造合法范围内的`q_m`，因此正式鲁棒性不能只依赖质量权重。

`z_dom`只用于质量与异常解释，不执行节点间表示对齐。这样避免把真实传播差异强行拉成同一表征，也符合当前项目减少对齐操作的研究方向。

### 4.2相关组

节点按query前签名的拓扑registry派生`correlation_group_id`。共享射频前端、同一波束形成链、同频强相关路径、共同中继、相同预处理缓存或无法证明独立的节点进入同组。不确定独立性时宁可合组，避免伪造独立证据。证据包只携带registry版本和查找键，最终group ID由融合器生成。

进入组内融合前，先按`satellite_reception_id`和canonical evidence hash拒绝重放；同一签名`evidence_origin_id`的重复记录只保留registry预定代表，不允许用更换node ID制造副本。随后每个`p_local`必须finite、非负、长度为同一`class_registry_id`的`C+1`，且和为1；超出固定数值容差时拒绝该event的正式融合。组内和跨组产生的`P_g/P_fuse`也逐次执行相同simplex检查。

组内权重为：

\[
\alpha_{m|g}=\frac{r_m}{\sum_{j\in g}r_j},\qquad \sum_{j\in g}r_j>0
\]

总质量为0的组不进入概率融合，并记录`ZERO_GROUP_QUALITY`；若因此没有有效组，结果为defer。`epsilon`不进入权重归一化，只用于后续log数值保护。组内分布采用线性池而不是重复PoE：

\[
P_g(k)=\sum_{m\in g}\alpha_{m|g}P_m(k)
\]

组内冲突用加权Jensen–Shannon散度表示：

\[
\chi_g=\sum_{m\in g}\alpha_{m|g}D_{KL}(P_m\Vert P_g)
\]

组可靠度取：

\[
\gamma_g=\max_{m\in g}r_m\cdot\exp(-\beta\chi_g)
\]

精确归一化保证`P_g`仍是概率分布。重放或同一`evidence_origin_id`副本在加权前已去重，所以精确复制不改变结果；新增的真实独立reception可以改变组内`P_g`，但不能让该相关组的`γ_g`超过最可靠成员。组内冲突过大时，系统降低该组影响并设置冲突标志；若冲突足以改变最终决策，则输出defer，而不是只依靠衰减掩盖问题。

### 4.3跨组融合

对全部注册类和unknown使用标签对称先验`π_k=1/(C+1)`。跨独立相关组采用带保护项的对数意见池：

\[
L_k=\log\pi_k+\sum_g\gamma_g
\log\left((1-\eta)P_g(k)+\frac{\eta}{C+1}\right)
\]

\[
P_{fuse}(k)=\operatorname{softmax}(L)_k
\]

`η`只防止零概率导致数值失效，不作为调节拒识性能的自由阈值。所有超参数由source-only、合法support或query前deployment prior冻结，不读取query真值或scorer结果。实现必须以类别置换、节点排列、同组精确复制和`N_sat=1`属性测试证明这些性质，而不是只在文档中声明。

### 4.4鲁棒性审计

每个event同时计算三类不改变主预测的诊断量：

- `leave_one_group_out_sensitivity`：逐组移除后决策是否翻转；
- `group_conflict_max`：最大组内或组间分布冲突；
- `effective_independent_group_count`：达到最低质量的独立相关组数。

如果单个相关组的移除使registered类别、unknown状态或margin跨越决策边界，系统输出`defer/DOMINANT_GROUP_SENSITIVITY`。这比“选最高置信节点”更稳健，也避免一个节点或一个相关组垄断正式决策。

### 4.5三态决策

记注册类最高后验为`c*`，第二竞争项包含其他注册类和unknown。

`registered(c*)`需要同时满足：

1.`P_fuse(c*)≥τ_reg`；
2.`P_fuse(U)≤τ_u_low`；
3.top-2 margin不低于`τ_margin`；
4.没有完整性或主导组敏感性故障；
5.多节点模式下，高质量相关组之间不存在未解决冲突。

多节点正式registered还要求至少两个独立高质量组支持同一`c*`。若到deadline只有一个有效相关组，系统不宣称协同结果：由matrix预注册leader存在时返回其逐字节单节点决策并标记`SINGLE_GROUP_FALLBACK`，否则defer。leader由roster顺序冻结，不能按当前`q`或预测选择。

`unknown`需要同时满足：

1.`P_fuse(U)≥τ_unknown`；
2.至少两个独立高质量相关组支持unknown；
3.独立组总质量不低于`τ_group_quality`；
4.没有registered类别获得相互独立的强支持。

其余情况输出`defer`。单节点模式不伪造协同增益，逐字节返回本地`p_local/local_decision/reason_code`。registered query被unknown或defer时，在已注册身份准确率中按错误计数；unknown的defer只计unresolved，不能记为安全拒绝。

威胁模型限定为：节点身份和registry签名不可伪造，但允许一个独立相关组产生任意有限概率分布。两个及以上组串谋、共同base bundle被攻陷或签名根失守不在首版保证范围内，报告不得宣称Byzantine完备防御。两组场景要求组间decision与margin一致；三个及以上组额外要求leave-one-group-out不翻转，否则defer。

## 5.匿名轨迹关联

事件融合回答“这次发射是什么”，轨迹层回答“不同事件是否可能来自同一匿名射频链”。两者不能混为一次K-shot。

### 5.1轨迹状态

每个`anonymous_entity_id`维护：

```text
track_id
event_count
robust common-base z_track centroid and dispersion
frequency/Doppler envelope
time/visibility/beam history
supporting independent node groups
association confidence
split/merge conflict flags
expiry
```

轨迹不存raw IQ，不修改Phase1/Phase2模型，不把类别分布写回本地预测器。`z_track`不是receiver-specific适配状态的`z_id`；它由所有节点共同的只读Phase1 base runtime额外输出，带相同`feature_space_id`，不经过节点适配、在线学习或跨receiver对齐。

### 5.2关联代价

同一event先按相关组各选取组内球面medoid，再对各组medoid取等组权球面medoid，得到`z_track,e`。该构造不学习映射、不给节点数量更多的相关组更大权重，并在同组复制时保持不变。若共同base`z_track`缺失、feature space不一致或组间离散度超过冻结门，身份表征项不可用，关联只能依靠物理元数据并倾向defer。

medoid距离并列时按canonical evidence hash选择，禁止依赖输入顺序。Tier-1区间solver、量化舍入方向和simplex数值容差必须在G0实现前冻结并加入属性测试。

新事件`e`与历史轨迹`t`的代价为：

\[
C(e,t)=w_z\theta(z_{track,e},\mu_t)+w_fD_f(e,t)+w_vD_v(e,t)+w_tD_t(e,t)
\]

其中`θ`是身份表征角距离，`D_f`是频率／多普勒一致性，`D_v`是轨道可见性与波束约束，`D_t`是时间可达性。权重在query前冻结，并对所有匿名实体使用同一公式。

关联采用门控多假设策略：

- 单一轨迹明显最优且物理约束一致时更新该轨迹；
- 两个候选接近时保留多假设或输出`association_defer`；
- 同一时间窗出现物理上不可能属于同一发射链的事件时强制分轨；
- 证据长期冲突时触发split，不为减少轨迹数而强行merge；
- 轨迹过期后只能新建或由外部确权重新连接。

一对一事件—轨迹约束来自物理发射过程，不是类别配额，也不得改变event级registered/unknown决定。轨迹更新单独写不可变`track_update_artifact`和deadline；association、多假设、split或merge均不得回写已封存event decision、shot、fusion threshold、credential状态或fresh-K资格。

## 6.可信确权与Stage2-C交接

unknown轨迹进入运营确权流程后，外部credential至少包含：

```text
candidate_physical_identity
evidence_sources
evidence_independence
conflict_flags
label_confidence
valid_from / expires_at
issuer / signature
registration_authorized
```

系统遵循以下状态机：

```text
LOCAL_EVIDENCE_SEALED
→EVENT_DECISION
→ANONYMOUS_TRACK
→CREDENTIAL_PENDING
→AUTHORIZED或REJECTED或EXPIRED
→FRESH_K_COLLECTION
→P2_MIN_V1_VALIDATED
→STAGE2_C
```

只有`AUTHORIZED`才能创建fresh-K采集任务。K个support必须来自K个新的独立`emission_event_id`；同一event的多节点接收仍只算一个shot。新support生成新的`split_id`并按`p2_min_v1`验证后，Stage2-C完成旧类适应、新类注册、全部注册类统一竞争和遗忘／floor评价。

## 7.通信与实时性设计

### 7.1分级证据包

为适应星间链路和下行带宽，证据分为三个固定层级：

|层级|内容|使用条件|
|---|---|---|
|Tier-0|ID/hash、q、local decision、unknown分数、top-1/margin、时空频波束元数据|快速完整性检查与单节点降级|
|Tier-1|全类压缩`p_local/d_class`、量化`z_id/z_dom`、相关组字段|默认协同|
|Tier-2|更高精度全类证据与完整解释字段|冲突、低margin或运营审计时按预注册规则请求|

Tier-1不能只发送top-L类别后把遗漏类别当零。类别数较小时发送完整量化向量；类别数较大时，每个遗漏类携带统一下界0和由量化协议证明的上界`u_m`，并另发遗漏总质量。融合器对每个class和unknown传播区间：对所有遗漏质量在合法类间的最坏分配分别计算`P_fuse`下界／上界。只有同一`c*`在全部合法分配下仍满足registered门，或unknown在全部合法分配下仍满足unknown门，才能不请求Tier-2；否则必须请求Tier-2或defer。系统不能为节省带宽破坏全部注册类统一竞争。

### 7.2deadline与anytime行为

每个event在任务开始前冻结`soft_deadline`和`hard_deadline`：

- soft deadline前证据足够时可输出`provisional`结果；
- hard deadline到达后封存最终结果，迟到证据只进入延迟审计，不改写预测；
- 紧急业务可优先返回单节点结果，随后另发不可覆盖的协同补充结果；
- 任何补充结果必须带新artifact hash，不能覆盖原决策。

## 8.故障、安全与降级

|现实故障|检测依据|系统行为|
|---|---|---|
|节点失联或迟到|roster、deadline、heartbeat|在实际节点子集上运行；报告缺失率，不从分母删除|
|时钟漂移|同步receipt、时间残差|不能核验same-event时降为track-only或defer|
|重复或相关证据|correlation group、链路／中继元数据|组内线性池，组强度封顶|
|节点预测冲突|JS散度、leave-one-group-out|降低组权重并在翻转风险时defer|
|重放或篡改|reception ID、nonce、hash、签名|拒绝重复或不一致artifact|
|单节点被攻陷|节点先验、跨组冲突、历史健康|隔离该节点；不得由当前query真值更新信任|
|模型或bundle不一致|共同bundle/class/calibration合同，节点state是否命中各自roster|共同合同不一致时正式融合失败；单节点state未命中roster时拒绝该节点并按quorum规则降级|
|unknown全拒绝取巧|registered reject/defer计错|known指标立即暴露退化|
|协同结果不稳定|节点子集敏感性|输出defer并保留贡献解释|

节点信任只能由维护记录、硬件状态、无真值一致性和长期独立审计更新；不能使用当前query标签或scorer结果在线修正。

## 9.泛化性原则

EV-CARE-Track不依赖具体TX、receiver或卫星编号，冻结以下不变性：

1.类别标签置换等价：交换注册类handle只交换对应输出维度；
2.节点排列不变：改变输入顺序不改变融合结果；
3.相关复制不增益：同组复制相同证据不提高组强度；
4.单节点恒等：`N_sat=1`时协同入口与本地入口逐字节一致；
5.可变节点数：相同代码支持1到任务roster上限的节点集合；
6.缺失显式化：缺失节点不被当作负证据，也不从部署分母消失；
7.事件与轨迹分离：跨事件关联不增加shot、不改event预测；
8.模型与协同解耦：可在相同协同器下替换Phase1 bundle，并通过A/B/C/D分解贡献。

该方法不做接收机特征对齐、clean↔LEO表示拉齐、teacher匹配或基于query的阈值更新。跨节点泛化来自事件核验、质量建模、相关性去重和物理约束，而不是把所有节点表征强行映射到同一分布。

## 10.训练、校准与在线更新边界

协同器默认无在线梯度训练。允许冻结的量来自三类来源：

- source-only开发事件或合法代理事件；
- Stage2合法support产生的只读状态；
- query到达前封存的节点健康、链路和任务先验。

阈值、质量映射、相关组规则、deadline和通信层级一旦进入正式矩阵便不可根据query性能调整。匿名轨迹可以更新轨迹状态，但不能更新身份分类器、unknown阈值或Stage2注册状态。

## 11.评价矩阵

### 11.1因果拆分

使用同一输入口径评价：

|Arm|Phase1基座|部署推理|
|---|---|---|
|A|原Phase1|单节点|
|B|新Phase1|单节点|
|C|原Phase1|EV-CARE|
|D|新Phase1|EV-CARE|

报告`B-A`为底层表征贡献，`C-A`和`D-B`分别为原／新Phase1下的协同贡献，`D-B-C+A`为交互贡献。`N_sat=1`必须满足A=C、B=D。

Phase2相关结果使用四个明确状态：`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`。`REG0`的`seen_new_acc`和`H_old_new`为N/A。

A/B/C/D必须与四状态形成交叉矩阵，而不是分开选择有利状态。每个交叉cell复用同一event、node roster、received IQ、old/new support、query physical ID、deadline和truth sidecar；唯一变化是Phase1基座与是否启用协同。分别报告两个DA effect、两个registration effect、`C-A`、`D-B`和`D-B-C+A`，不能用某个状态的协同收益补偿另一状态的失败。

### 11.2节点与故障矩阵

至少覆盖：

- `N_sat∈{1,2,3,4,5}`及预注册节点子集；
- clear、low-elevation、rain等场景；
- 节点随机缺失、固定节点失联、迟到和乱序；
- 相关组复制、同源证据比例上升；
- 单个低质量或异常节点；
- 时钟漂移、event binding缺失和bundle不一致；
- 同时存在registered、unknown和defer冲突的事件。

节点子集在结果产生前冻结，不能按性能选择“最佳卫星组合”。

### 11.3指标

事件级指标：

- 已注册身份overall、min-class、min-receiver和min-scenario准确率；
- seen-new accuracy与`H_old_new`，仅在REG1定义；
- unknown FAR、safe rejection、defer/unresolved；
- calibration error、Brier score和决策margin；
- 节点缺失率、迟到率、有效独立组数和决策翻转敏感性。

轨迹与运营指标：

- anonymous association precision/recall、IDF1、track fragmentation和false merge；
- credential conflict率、授权precision、授权到fresh-K完成时延；
- 历史unknown转support违规数，必须为0。

资源指标：

- 每event通信字节、Tier-2请求率、星间／下行占用；
- 本地推理、融合和轨迹更新的p50/p95时延；
- 峰值内存、功耗估计、CPU/GPU占用和deadline完成率。

## 12.数据采集需求

当前N607库存没有可验证的真实event/reception binding，因此下一份真实协同数据至少需要：

1.两到五个同步接收节点记录同一物理发射事件；
2.采集前生成`emission_event_id`，每个节点生成唯一`satellite_reception_id`；
3.GNSS/PTP或等价时间基准、中心频率、采样率、波束、节点位置／轨道和可见性；
4.节点前端、共享中继和相关链路元数据，用于构建correlation group；
5.原始接收数据封存后再连接truth sidecar，防止标签参与事件构造；
6.registered、未注册但可外部确权、长期无法确权三类运营样本；
7.跨多个过境的重复匿名实体，用于评价track split/merge；
8.独立fresh-K重新采集流程，而不是复用历史unknown观测。

G2采集在启动前还必须冻结一份acceptance spec：

- GNSS/PTP时钟审计目标不高于1微秒；传播时延校正后的event residual门由脉冲／burst持续时间预注册；
- 通过合成碰撞注入和相邻事件回放测量false binding率，目标不高于`10^-4`；
- 每个正式event至少有2个可证明独立的前端／相关组；共享中继或共享前端必须入同组；
- 每个正式unknown评价cell至少60个独立emission event，以便在零误接收时给出约5%的95%单侧上界；注册类样本量按min-class/floor置信区间预先做功效计算；
- unknown来源至少包含完全排除于训练、注册表和阈值冻结的授权留出发射机，以及独立运营unknown；二者分表报告；
- truth sidecar只在received IQ、binding、预测和artifact hash全部封存后连接，并带独立签名与有效期；
- 采集清单记录节点前端序列号、时钟源、轨道／位置、波束、中继链和故障注入，保证相关组可审计。

在完成上述采集前，WiSig/ManySig只能用于`PROXY_MULTI_RECEIVER`技术验证。报告必须使用“多接收节点代理协同”，不能使用“真实同步多星”或“真实在轨协同”。

## 13.分阶段实现路线

### G0：接口与不变性

目标：证明本地证据、物理binding、相关组、N=1恒等、重复证据不增益、三态决策和生命周期状态机可运行。只使用合成fixture，不报告性能。

当前状态：现有CARE-PoE G0与binding v3已完成技术闭环；EV-CARE只需在其上补充冲突衰减、组移除敏感性、分级通信和track状态。

### G1：代理多接收节点

目标：使用truth-blind采集元数据先构造代理group并封存预测，再由独立scorer连接truth，比较单节点、平均、质量加权、普通PoE、CARE-PoE和EV-CARE。source-proxy unknown只作研发诊断，不能充当真实unknown门；结论限定为代理协同，不宣称same-event。

### G2：物理same-event离线回放

目标：只有数据通过同步误差、event collision、独立前端／相关组、样本量、unknown来源和签名truth sidecar acceptance spec后，才使用真实binding的同步多节点数据。冻结完整矩阵后一次性评分，报告known、unknown、defer、关联和资源指标；未通过时只能停留在G0/G1。

### G3：闭环运营验证

目标：接入外部credential、授权、fresh-K和Stage2-C，验证unknown→anonymous→authorized→new registration全链路，同时证明历史unknown零转support。

### G4：在轨或硬件在环

目标：在真实星间链路、动态可见性、节点失联和deadline约束下验证实时性、通信和故障降级。只有此阶段才能讨论真实在轨协同主张。

## 14.首个可证伪实验

首个正式实验不应堆叠完整系统，而应冻结一个窄问题：在相同Phase1 bundle、相同verified event、相同节点子集和相同阈值下，比较`leader单节点`、`q加权平均`、`普通PoE`、`CARE-PoE`与`EV-CARE`。

主假设：相关组复制或单节点异常时，EV-CARE能降低unknown误接收和registered错误接受，同时不通过大规模defer规避责任。

非补偿门：

1.全部registered类准确率和floor不低于leader单节点2pp以上；
2.unknown FAR不高于5%，safe rejection不低于95%；
3.registered reject/defer全部计错后仍通过known门；
4.相关复制前后结果近似不变；
5.`N_sat=1`逐字节等于本地基线；
6.任何一个节点／相关组删除后出现主导性翻转，正式结果记defer并单独报告；
7.通信和p95时延满足任务预算；
8.任一event binding或truth隔离失败，整run无性能结果。

如果现有数据仍缺真实binding，只运行G0或代理G1，不把技术闭环写成性能成功。该限制不是保守措辞，而是算法可验证性的必要条件。

## 15.设计结论

`EV-CARE-Track`把卫星协同拆成三个互不越权的决策层：事件级融合负责registered/unknown/defer，匿名轨迹负责跨事件实体连续性，外部确权负责语义身份与注册授权。算法对节点数、节点顺序、类别标签和相关复制具有明确不变性；对缺失、延迟、冲突、重复、时钟漂移和节点异常提供可观察的降级路径。

它不依赖新的表示对齐或teacher结构，也不让unknown观测回流Phase2。真正制约下一步的不是再设计一个更复杂的融合公式，而是采集前事件绑定、独立节点相关性元数据和fresh-K运营链。完成这三类资产后，现有Phase1 bundle、Phase2状态和CARE-PoE技术底座可以自然升级为可验证的现实协同系统。

独立科学复审结论为`P0=0，P1=0，MERGE`。该结论允许进入分阶段实现，不构成当前Phase3性能达标、真实same-event数据可用或真实在轨验证声明。
