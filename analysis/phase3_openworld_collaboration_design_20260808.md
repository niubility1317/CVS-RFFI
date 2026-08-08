# Phase1开放世界表征与Phase3协同推理设计草案

版本：2026-08-08

执行模式：`GOAL_MODE=ACTIVE`

设计状态：`DEFERRED_FULL_PHASE3_DESIGN/REVISION_3`

候选标识：`P1-OWR-H__CARE-PoE_r3`

本文件整合Phase1方法作者、Phase3方法作者与独立监督者的只读审查。它不是`DESIGN_FROZEN`、实现完成或性能达标声明。自2026-08-08起，本文件只保留为后续完整Phase3设计，不再阻塞Phase1最小高泛化实验；Phase1快速通道以`phase1_geosat_lite_design_20260808.md`为唯一方法锁。

## 1.主代理裁决

1.接受`P1-OWR-H`作为唯一Phase1主候选，但保留其名称中的`H`为“feature-head-only”，不把它误写成只训练最终分类线性层。
2.接受`P1-OWR-0`作为接口回退，只用于验证v2 bundle、密封与资源；不得根据proxy结果自动切换候选，也不得声明unknown增益。
3.接受`CARE-PoE`作为唯一Phase3科学融合候选；均分、质量加权、多数投票、最高质量节点和普通PoE只作为基线。
4.拒绝把现有R8路径修补成Phase3预测器。其真值参与事件构造，只能保留为`NON_DEPLOYMENT_DIAGNOSTIC`历史负面证据。
5.正式Phase3实现新建独立预测入口、不可变本地证据artifact和真值侧scorer；旧评估器只允许复用经focused tests证明不读取真值的纯函数原语。
6.在缺少不依赖真值的物理事件绑定时，只允许“多接收节点代理协同”结论；不得声称same-emission、真实同步多星或真实在轨验证。
7.`P1-OWR-H`必须继承此前Phase1探索的已验证控制和负面证据，不重复post-hoc adapter、动态软门、局部球并集、reject-all或单纯放大open loss路线。

### 1.1既有Phase1探索继承矩阵

| 既有探索 | 已获得的同排证据或工程事实 | 本候选的继承方式 |
|---|---|---|
| ADV3B02 V31 feature-centered adapter | `strong_target1_pass=0/80`，最佳unknown FAR约0.542；hard receiver/TX floor未修复 | 拒绝继续post-hoc adapter扫参；把修改点前移到Phase1表示训练 |
| DualGuard16 | 16/16训练稳定，但source-episode overflow约0.987-0.989、legacy proxy约0.616-0.623、legacy bridge=1；open梯度份额不足且U_s open路由空转 | 继承稳定训练与closed保护；首轮不把U_s当unknown负样本，不用动态DM代替最终边界 |
| P0Closed8 | fixed p99可下降，但proxy、bridge、overflow、radius/inter无法同一候选联合改善；高open预算会收紧tail却恶化低密度/类间边界 | proxy下降只有在known覆盖和类间margin不坍缩时才可解释；不以单指标晋级 |
| CorePath8 | 低fixed p99常由known hard-core TPR坍缩到0.010-0.111造成；异类最小间隔可缩到1.4-2.8度 | 强制known hard-core TPR前置门；先保正覆盖，再解释unknown proxy；禁止reject-all伪改善 |
| Phase1 P0闭环 | 已有balanced TX×receiver/day采样、`endpoint_accept_v1`密封、三入口parity、terminal fail-closed和source-val-only选择 | 复用hash/seal/reason-code/parity与终态控制；旧local-component接收公式只作legacy基线，不继承为新决策 |
| Phase1 P1不变性 | 已有train/eval信道族隔离、TX条件receiver/day/channel泄漏probe、final-only checkpoint和local component审计 | 作为必开控制；`q/z_dom`只有在泄漏probe与独立信道评估通过后才能增加主张 |

由此冻结四条经验性硬约束：

1.动态batch soft gate只作训练遥测，正式证据只读取冻结source-val的`open_world_evidence_v2`与最终bundle同一公式。
2.known接收必须是`shared invariant core AND local density support`，禁止把多个receiver/day/channel局部球直接取并集形成宽接收域。
3.当clean或satellite known hard-core TPR低于0.85时，proxy/bridge下降不具有正向含义，候选直接判为正覆盖失败。
4.首轮`P1-OWR-H`只用有标签known几何训练`L_OW`；`U_s`继续用于既有domain/ADV/clean-sat一致性，不伪装为unknown负样本，也不启用历史上长期空转的U_s direct gate。

## 2.Phase1数据与proxy边界

### 2.1双层划分

每条source物理样本同时保留两个正交字段：

```text
source_role ∈ {L,U,V}
tx_partition ∈ {T_train,T_val,T_proxy}
T_train = source_known_train_tx
T_val = source_known_validation_tx
T_proxy = source_proxy_unknown_tx
```

`source_role`保持项目协议的`0.07/0.63/0.30`原始角色语义；`tx_partition`在其上实施TX级开放世界研发隔离。两者不得互相推断，特别是`V`不自动等于`T_val`。

| 条件 | 允许用途 |
|---|---|
| `T_train∩L` | 带TX标签训练 |
| `T_train∩U` | 无TX标签训练 |
| `T_train∩V` | source-known早停与checkpoint选择 |
| `T_val∩V` | 冻结公式的source-known外部验证，不参与梯度 |
| `T_val∩{L,U}` | 不进入本轮训练或选择；只保留来源记录 |
| `T_proxy∩{L,U,V}` | 候选、checkpoint、公式与阈值全部冻结后的一次性审计 |

builder必须对TX ID、physical ID和view lineage执行三重两两互斥。某TX一旦属于`T_proxy`，其全部物理样本及clean/LEO/数学视图均从训练、早停、校准和候选选择路径排除。任何交叠均fail closed，不做局部容错。

### 2.2一次性proxy审计

`T_proxy`只回答“冻结的source-only几何是否在held-TX上出现正向迹象”，不回答Phase3真实unknown性能。审计结果不得调节loss、epoch、候选、fallback、阈值或bundle公式，也不得回写训练状态。

为消除“proxy门”与“禁止模型选择回流”的冲突，本轮按以下口径执行：

- `T_proxy`不是优化或候选选择门；
- 审计通过只允许增加“source held-TX proxy正信号”这一受限主张；
- 审计失败则如实报告`NO_PROXY_OPEN_WORLD_GAIN`，本轮候选保持冻结且不自动重训、不切换fallback；
- 后续若另立新候选，必须使用新目标、新预注册和不复用本轮proxy反馈的设计依据。

## 3.Phase1候选`P1-OWR-H`

### 3.1训练机制

主候选从同一ADV3B02基座初始化：

\[
L=L_{\mathrm{ADV3B02}}+\lambda_{\mathrm{ow}}L_{\mathrm{OW}}(z_{id},y,d)
\]

`L_OW`只读取`T_train`的source received IQ、允许的source TX标签和source domain标签，使用现有known-only类内紧致、类间margin和同类跨域对齐公式。以下路线显式锁零：legacy batch轮换proxy loss、soft-mixup、source episode、direct metric、EVT/tail/vacuum以及任何confirmed unknown或target/query输入。

v2的共享身份核心预注册为`z_id=normalize(id_feat_cls)`，不使用混入DAC/PA defect分支与gate的`feat_joint`作为open-world核心。原ADV3B02分类logits和v1的`feat_joint`路径保持不变；该选择在任何候选结果前冻结，不允许事后在`feat_cls/feat_joint`之间切换。`L_OW`只作用于`id_feat_cls`，精确训练allowlist至少包含其生产者`id_backbone.cls_head.id_proj`，其它已有closed/domain参数是否更新由完整loss梯度审计逐项列入receipt。

训练继续复用已验证的balanced TX×receiver/day采样、clean-sat同物理样本对应、source-val-only/final-only控制、TX条件泄漏probe及artifact identity/parity框架。新增`L_OW`不得关闭或旁路这些控制，也不得重新启用已被负面结果否定的动态接收域作为promotion依据。`endpoint_accept_v1`保留为legacy对照；新候选的决策公式标识固定为`open_world_evidence_v2.0`。

训练allowlist不得继续使用模糊的字符串包含规则。冻结前必须从真实checkpoint解析`z_id_key=feat_joint`的可达参数，预期至少包括`id_backbone.cls_head`内产生`feat_joint`的projection/gate/joint projection及最终class head；是否训练`dom_backbone.cls_head`、`dom_head`和`adv_head`由真实梯度审计决定。allowlist外参数全部冻结。

P0可达性条件：单个有效batch反向传播后，至少一个允许的`z_id`生产参数收到finite、非零的`L_OW`梯度；一次优化步改变`z_id`几何；禁训参数逐字节不变。若失败，则`P1-OWR-H`退回`DESIGN_DRAFT`，不得用分类head梯度替代几何可达性证据。

### 3.2v2 sibling bundle

v1字节行为、成员allowlist和Stage2-C读取路径保持不变。新增独立schema：

```text
cvs.phase1.open_world_local_evidence_bundle.v2
```

v2与base-v1的runtime hash、content-root、class binding和checkpoint hash共同封存，包含full-dual TorchScript runtime、`id_geometry`、`dom_geometry`、`open_world_evidence_v2.0`公式版本、量化参数和独立seal。`id_geometry`复用已验证的聚合几何格式，但明确分为共享类核心与receiver/day/channel残差component；component只能提供局部密度支持，不能独立授予known acceptance。bundle不得包含raw IQ、sample/member ID、特征库、source cache或可逆样本索引。

| 输出 | 冻结定义 |
|---|---|
| `z_id[B,160]` | 当前IQ的`id_feat_cls`经unit normalization得到的共享身份核心；真实checkpoint已验证维度为160 |
| `z_dom[B,E_dom]` | 同一IQ得到的unit-normalized域扰动表征 |
| `d_class[B,C]` | 对全部已注册类统一计算到共享invariant class core的归一化距离 |
| `e_unknown[B]` | `1-max_c min(a_core[c],a_density[c])`，是共享核心与局部密度AND后的连续陌生度证据，不是unknown概率 |
| `q[B]` | `z_dom`相对sealed source-domain geometry的domain plausibility，经预注册单调映射裁剪到`[0,1]` |
| `p_local[B,C+1]` | 由`a_known[c]=min(a_core[c],a_density[c])`与`e_unknown`经冻结温度形成的类别加unknown证据分布 |

冻结构造如下，其中角距离使用弧度，`epsilon=1e-8`：

\[
\mu_c=\operatorname{normalize}\left(\frac{1}{N_c}\sum_{i:y_i=c}z_i\right),\qquad
R_c=Q_{0.95}\{\theta(z_i,\mu_c):i\in T_{train}\cap V,y_i=c\}
\]

每个类按预注册的receiver×day×channel stratum建立聚合component；每个component至少包含2个独立物理样本，中心为`nu_ch`，尺度为该stratum角距离的固定`Q_0.95`。令`H_c`为类`c`的有效component数，`H_c=0`时整类bundle构建失败：

\[
a_{core,c}=\exp\left[-\frac{1}{2}\left(\frac{\theta(z,\mu_c)}{\max(R_c,\epsilon)}\right)^2\right]
\]

\[
\log a_{density,c}=\operatorname{logsumexp}_{h=1}^{H_c}\left[-\frac{1}{2}\left(\frac{\theta(z,\nu_{ch})}{\max(\sigma_{ch},\epsilon)}\right)^2\right]-\log H_c
\]

`-log H_c`把component mixture归一为等stratum平均，防止component更多的类天然占优。随后：

\[
a_{known,c}=\min(a_{core,c},a_{density,c}),\qquad e_{unknown}=1-\max_c a_{known,c}
\]

\[
p_{local}=\operatorname{softmax}\left(\left\{\frac{\log\max(a_{known,c},\epsilon)}{T_k}\right\}_{c=1}^{C},\frac{\log\max(e_{unknown},\epsilon)}{T_u}\right)
\]

`T_k/T_u`和所有阈值只由预声明`known_validation_manifest`冻结，禁止读取`T_proxy`。对top class`c*`，本地决策固定为：

- `registered(c*)`：`q>=tau_q`、`a_core[c*]>=tau_core`、`a_density[c*]>=tau_density`、`p_local[c*]>=tau_accept`且top-2 margin不低于`tau_margin`；
- `unknown`：`q>=tau_q`且`e_unknown>=tau_unknown`；
- 其余、低质量、缺字段、非有限或阈值冲突均为`defer`。

artifact必须同时写`local_decision∈{registered,unknown,defer}`和固定reason code，不能只写`p_local`。class handle置换时`d_class/p_local/local_decision handle`相同置换，`e_unknown/q/reason family`不变。共同正交变换同时作用于query和sealed geometry时，距离与决策必须不变。`load_any(v1)`必须显式返回`phase3_local_evidence_supported=false`，不得伪造v2字段。

known hard-core TPR定义为`known_validation_manifest`中registered样本被正确`registered`的比例，reject/defer均为失败。它只使用注册TX的`T_train∩V`物理验证样本，禁止使用`T_proxy`：global TPR≥0.85、min-class≥0.80、每个receiver/day/三种`leo_*_weak`共同floor≥0.70，且每项相对原ADV3B02同切片下降不超过2pp。任一门失败时，proxy下降不允许解释为open-world改善。

## 4.Phase3候选`CARE-PoE`

### 4.1本地证据与双ID

每个接收节点只发布已封存的`LocalEvidenceV2`：

```text
schema_version
satellite_reception_id
linkage_mode
emission_event_id或proxy_group_id
node_id
bundle_id/capsule_id/split_id及hash
evidence_hash/sealed_at/deadline
z_id/z_dom/q/d_class/e_unknown/p_local
visibility/beam/frequency/track元数据
correlation_group_id/delay_ms
```

`linkage_mode=verified_physical`时，采集系统必须在标签可见前生成`emission_event_id`和唯一`satellite_reception_id`，并提供物理绑定receipt；一个event可对应多个reception，但K-shot始终按独立event计数。

`linkage_mode=proxy_unverified`时，不得填写或伪造`emission_event_id`，只允许使用预测前、无truth元数据生成的`proxy_group_id`；该模式不能产生same-event、同步多星或K-shot事件结论。对`role/true_label`派生键再次哈希仍属非法。当前R8 truth-ranked分组不满足任一模式。

重复reception、跨group/event拼接、hash/bundle不一致或deadline后到达均不得改变已封存预测。event内任一ID/hash/bundle冲突使整event fail closed并输出`defer/EVENT_INTEGRITY_FAILURE`，不能丢弃坏节点后继续给正式决策。

`LocalEvidenceV2`不得含truth、role、credential或`registration_authorized`。真值只存在于独立只读`ScorerTruthSidecar(reception_id→role,true_label)`；预测artifact封存后评分进程才可打开sidecar。sidecar缺失时预测仍必须完成且hash不变。

### 4.2相关性感知融合

对deadline前通过校验的接收证据`m`，定义预查询可靠性：

\[
r_m=\operatorname{clip}(q_m\pi_m\exp(-\lambda\Delta t_m),0,1)
\]

其中`pi_m`只来自预查询node prior。类别先验固定为`pi_k=1/(C+1)`，不得从query batch计数估计，因而对class handle置换保持对称。`correlation_group_id`只能由接收链、波束、频段、时间窗和共同中继等无真值元数据预先生成；不确定是否独立时必须并入同一相关组。

对每组`g`：

\[
\alpha_{m|g}=\frac{r_m}{\sum_{j\in g}r_j},\qquad
\gamma_g=\max_{m\in g}r_m
\]

\[
\log P_g(k)=\log\sum_{m\in g}\alpha_{m|g}P_m(k),\qquad
L_k=\log\pi_k+\sum_g\gamma_g\log P_g(k)
\]

\[
P(k\mid E)=\operatorname{softmax}_k(L),\quad k\in C\cup\{U\}
\]

所有进入log的概率先裁剪到`[1e-8,1]`并重新归一化。组内归一混合限制相关重复证据；`gamma_g`封顶该组强度；不同独立组才允许PoE累积。同组复制不得增加`gamma_g`，也不得改变结果。

单有效节点时不执行上式，直接逐字节返回该节点的`p_local/local_decision/reason code`；A/B单节点与C/D的`N=1`均调用这一identity分支，从定义上满足`C=A,D=B`。零节点、零权重或整event完整性失败统一`defer`。

### 4.3三态决策

- `accept(c*)`：`P(U)<tau_u`、`P(c*)-max(P(U),max_{c!=c*}P(c))>=tau_m`且`max_g JS(P_g,P)<=tau_conflict`。
- `unknown_reject`：`P(U)>=tau_reject`；当有效节点数大于1时，还必须有至少2个独立相关组且`sum_g gamma_g>=tau_group_quality`。同一相关组复制不能满足该门。
- 其余为`defer`。

温度、先验、质量映射、独立组质量门和阈值只允许由source-only、合法support或pre-query deployment prior冻结，并写入`calibration_receipt`。不得使用query真值、真实角色、真实batch构成、scorer输出、类别配额或全局重分配。

## 5.anonymous关联、确权与fresh-K桥

生命周期固定为：

```text
LOCAL_EVIDENCE_SEALED
→CARE_DECISION
→anonymous_entity_id关联
→外部credential验证与冲突检查
→registration_authorized
→重新采集K个独立emission_event
→生成并验证新split_id
→Stage2-C统一全类竞争
```

`anonymous_entity_id`不是语义身份，CARE-PoE不得自行生成授权。credential至少包含候选物理身份、证据来源、独立性、冲突、置信度、有效期和授权签名；缺失、过期或冲突均fail closed。历史unknown reception及其数学视图永不得转成support。

代理协同机制的第一轮本地实现可把外部credential适配器做成fail-closed接口和状态机fixture；但完整Phase3生命周期主张必须等待真实授权链与fresh-K lineage闭环。

## 6.冻结矩阵

### 6.1A/B/C/D

| Arm | Phase1 | 部署推理 |
|---|---|---|
| A | 原ADV3B02 bundle | 预注册leader单节点 |
| B | `P1-OWR-H` bundle | 同一leader单节点 |
| C | 原ADV3B02 bundle | CARE-PoE |
| D | `P1-OWR-H` bundle | CARE-PoE |

同输入以一个`base_manifest`和两个预封存supplement定义：`base_manifest`固定原始received-IQ/physical-ID、query、deadline、场景、K、seed和部署node roster；`old_support_supplement`只含旧类合法support；`new_support_supplement`只含授权后fresh-K新类support。不同Phase1或DA状态必须保留各自bundle/adaptation receipt，不能强行共用不同模型的feature artifact。

四状态的唯一干预冻结为：

| 状态因子 | 允许打开的状态 |
|---|---|
| `DA0` | 不打开`old_support_supplement`，使用Stage2-A恒等状态 |
| `DA1` | 只读同一`old_support_supplement`，运行已冻结的`P2-S2B-FULL`support-only旧类适配；query零fit/零update |
| `REG0` | 不打开`new_support_supplement`，新类未注册 |
| `REG1` | 只读同一`new_support_supplement`，运行`P2-FULL`的registration-only append子程序；对应DA状态的旧类prefix bytes/hash保持不变 |

`DA0_REG1`从DA0旧状态append新类，`DA1_REG1`从DA1旧状态append同一新类；REG1不能借新support反向适配旧状态。现有Stage2-B/C arm只作为冻结接口谱系，不把历史开发性能当作当前目标达标证据。

节点全集为5个预注册receiver时，使用全部31个非空节点子集，天然覆盖`N_sat_deployed∈{1,2,3,4,5}`，不根据结果挑子集。`node_order`与每个子集的leader在matrix manifest中按采集roster顺序冻结，leader为该子集中排序第一的节点，禁止按`q`、预测或结果选择。每个event同时报告`N_sat_deployed`和deadline前的`N_rx_observed`；缺失节点不得从事件分母删除。`N=1`必须逐字节满足`C=A`和`D=B`。

### 6.2四状态

每个arm固定报告：

```text
DA0_REG0
DA1_REG0
DA0_REG1
DA1_REG1
```

`REG0`的`seen_new_acc`与`H_old_new`为`N/A`，不得把未注册新类改记为unknown。按项目规则报告两个DA effect、两个registration effect及对四态均有定义指标的difference-in-differences。

每个同排结果至少包含old accuracy、每类old floor、`seen_new_acc`、`H_old_new`、unknown FAR、unknown safe rejection、defer/unresolved、coverage、缺失/迟到率、`N_rx_observed`直方图、通信字节、p50/p95时延和资源状态。registered的reject/defer均按身份错误；unknown的defer只计unresolved，不能充当safe rejection。

## 7.基线与归因

在相同artifact、节点子集、deadline和阈值来源上运行：leader单节点、等权均分、`q`加权均分、多数投票、最高`q`节点、无相关封顶的普通PoE和CARE-PoE。

主归因固定为：

```text
Phase1 effect = B-A
collaboration effect = C-A
joint interaction = D-B-C+A
```

任何单项最优不得脱离同一row的old/new/unknown/floor/defer/resource字段。实际unknown、proxy unknown和registered new必须分表，不得混成同一unknown指标。

## 8.从`DESIGN_DRAFT`进入`DESIGN_FROZEN`的P0门

1.生成可重复的TX/physical/view三重互斥manifest fixture，并证明所有非法交叠fail closed。
2.在`ssr-gpu`中完成真实模型反向传播可达性审计，列出允许更新的精确参数，证明`L_OW`改变`z_id`而禁训参数不变。
3.用真实checkpoint完成full-dual runtime smoke，验证`z_id/z_dom/logits`shape、finite、归一化及v1身份输出parity。
4.冻结v2 bundle schema、v1兼容行为、forbidden-content检查和seal/hash/class-order校验。
5.证明`z_id=id_feat_cls`的真实checkpoint shape/parity和精确allowlist；证明`open_world_evidence_v2.0`的component构建、等stratum mixture归一、shared core与local density AND语义及local三态/reason code，local component并集不能独立接收。
6.证明known hard-core TPR的global/min-class/receiver/day/三种LEO共同floor和相对基座门全部成立后才解释proxy指标；`endpoint_accept_v1`只作为legacy基线。
7.证明predictor schema拒绝`role/true_label`，sidecar缺失或truth置换不改变prediction bytes/hash。
8.证明`verified_physical/proxy_unverified`条件schema、采集前ID依赖、一个event多reception计1 shot及整event完整性fail closed。
9.证明节点排列不变、class handle置换等变、共同正交特征变换不改变距离决策、同相关组复制不增益。
10.证明单节点identity、无节点/零权重/冲突defer、registered reject/defer计错及unknown三态分离。
11.证明31个节点子集、预注册leader、A/B/C/D、四状态与base/supplement干预完整；`N=1`满足`C=A,D=B`。
12.证明anonymous不能直接注册，credential fail closed，历史unknown不能变support，fresh-K使用新event并生成新`split_id`。

独立监督者必须对上述证据给出`P0=0/P1处置明确`后，主代理才能把状态改为`DESIGN_FROZEN`并拆分非重叠实现任务。

### 8.1当前可行性证据

2026-08-08在`ssr-gpu`中完成只读/内存内审计，未写回checkpoint：

| 检查 | 结果 | 证据边界 |
|---|---|---|
| 现有loss与协同原语回归 | `python -m pytest -q code/tests/test_open_world_feature_space_loss.py code/tests/test_proxy_unknown_loss.py code/tests/test_collaborative_open_set_qknn_eval.py`通过 | 共89项，只证明现有原语未回归，不证明新候选 |
| 真实checkpoint身份 | SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`，CPU，`input_len=256` | 只读加载，不访问dataset、support或query |
| `L_OW`反向可达性 | loss=`0.234210`；8个非零梯度tensor | 全部位于`id_backbone.cls_head`的`id_proj/pa_proj/id_gate/joint_proj`权重与偏置 |
| 单步更新 | 8个允许tensor改变，禁训参数改变数为0；`z_id`最大绝对变化=`0.0444369` | 证明feature-head路线可达，不证明训练稳定或性能增益 |
| 旧allowlist审计 | 旧字符串规则共放开44个tensor，但本项loss只触达8个 | 正式实现仍须替换为精确allowlist并测试全部启用loss的梯度归属 |

该证据关闭“`L_OW`完全无法改变`z_id`”这一可行性疑问，但未关闭第8节其余P0门，也不把设计状态提升为`DESIGN_FROZEN`。

## 9.本轮明确拒绝项

- 拒绝legacy batch轮换label作为TX级proxy unknown。
- 拒绝对含`role/true_label`的旧event_id再次哈希后冒充opaque ID。
- 拒绝把R8的`receiver_domain_ranked_by_role_tx_scenario`称为物理same-event。
- 拒绝用全拒绝抬高unknown指标或把defer计作safe rejection。
- 拒绝用`T_proxy`选择epoch、候选、fallback、阈值或公式。
- 拒绝为不同节点子集重复运行backbone；本地证据只提取一次并按bundle/adaptation receipt缓存。
- 拒绝在P0门关闭前连接N607、同步方法文件或预注册正式性能实验。
