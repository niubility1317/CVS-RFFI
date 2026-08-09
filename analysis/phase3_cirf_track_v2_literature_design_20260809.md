# CIRF-Track v2：相关性—区间风险协同推理与匿名轨迹

状态：`DESIGN_FROZEN_INDEPENDENT_REVIEW_MERGE`

日期：2026-08-09

适用阶段：Phase3部署期多接收节点协同

证据边界：本文件是文献驱动的方法设计，不是当前性能结果、真实same-event数据证明或在轨验证声明。

## 1.结论先行

现有EV-CARE-Track已经正确解决事件绑定、证据去重、相关组、三态决策、匿名轨迹与Stage2-C交接边界，但其性能上限仍受四个因素约束：节点概率未证明具有共同先验与共同校准语义；硬相关组不能表达部分相关；静态线性池／对数池不能显式传播校准和量化误差；球面medoid轨迹没有出生、消亡、漏检、杂波与多假设概率。

本轮建议的新候选为`CIRF-Track v2`：Correlation-aware Interval Risk Fusion with Probabilistic Track。它不是更深的特征对齐网络，而是在冻结本地证据之上增加六个低对齐组件：标签置换等价的概率校准、共同先验修正、连续相关性有效证据量、组内鲁棒统计、固定有限样本风险外壳、固定滞后多假设轨迹。Tier-0/1/2按可证明的区间收缩逐级请求，hard deadline后不改写事件决定。

该路线比普通PoE、质量加权和EV-CARE更可能提高性能，因为它允许真正独立的节点累积证据，同时抑制相关副本、坏节点和过度自信。它也比端到端Set Transformer更现实：当前项目没有足够的真实same-event联合数据，直接学习跨节点融合器很容易记住receiver、节点ID或代理场景。

## 2.协议不变量

设计必须同时满足以下不变量：

1.一个`emission_event_id`的多节点reception仍只计一个shot。
2.每节点本地证据先形成不可变artifact，再进入融合。
3.融合器不读取query真值、old/new/unknown真实角色、类别配额、batch类别构成或独立scorer结果。
4.unknown和defer不能回流Phase2；只有外部确权为`registration_authorized=true`后，重新采集fresh-K并形成新`split_id`，才能进入Stage2-C。
5.registered样本被unknown或defer时按身份错误计数；unknown的defer只计unresolved。
6.`N_sat=1`逐字节返回冻结本地结果，不运行多节点校准、风险重判或轨迹反馈。
7.track只做匿名关联，不回写event decision、threshold、shot、credential或fresh-K资格。
8.WiSig/ManySig与LEO弱信道只能支持代理协同，不得称为真实同步多星或在轨协同。

## 3.文献综合与取舍

|机制族|文献启示|本方案吸收|本方案拒绝或限制|
|---|---|---|---|
|多分类概率校准|Kull等的Dirichlet calibration直接校准完整概率向量，而非只校准top-1置信度|采用受标签置换约束的Dirichlet校准器，把不同节点映射到同一registry和共同先验语义|不允许用正式query、scorer或当前event更新校准器|
|相关分类器Bayes融合|Trick与Rothkopf显式建模分类器偏差、方差与相关性，并说明相关越强，融合带来的不确定性下降越小|把“相关性决定有效证据量”作为核心，不再把每个节点当作独立专家|原模型需要联合标签数据和采样推断；当前星上版本不用Gibbs sampling|
|校准集成与分布偏移|Kumar等表明校准是集成在分布偏移下工作的关键条件之一|融合前强制校准合同和分层校准审计|不把ID校准结果直接宣称为真实unknown保证|
|未知相关下的保守融合|Covariance Intersection在未知交叉相关时通过保守信息组合保持一致性|相关性元数据不完整时使用预封存相关上界，降低有效证据量|不把CI直接套到类别概率，也不以过度保守替代性能评价|
|排列不变集合模型|Set Transformer能表达可变规模集合和元素交互|只把排列不变性作为测试合同和未来候选|当前不让attention直接学习类别融合、节点选择或receiver捷径|
|Conformal Risk Control|CRC把预测器外包成有界损失的风险控制器|在事件／实体互斥校准集上冻结scenario×有效组数的risk controller|不使用正式query在线调整；没有authorized unknown校准时不声明unknown风险保证|
|选择性分类|one-sided prediction强调高准确率区域的class-wise误接收控制|registered与unknown分别设置不可补偿的单侧风险门|不允许用大规模defer掩盖known退化|
|anytime与主动取证|nested prediction set和active feature acquisition说明逐级证据请求应保持决策一致并优化单位成本信息增益|Tier请求按预注册的最坏区间收缩／byte排序，决策集只能收窄|不训练query自适应RL，不用当前真值选择节点|
|Byzantine分布式检测|小规模节点下，单个恶意组即可显著破坏全局检测|明确威胁范围、组级异常包络、leave-one-group-out与quorum|两个独立组时不宣称容忍一个任意恶意组；冲突只能defer|
|多目标概率轨迹|trajectory PMBM/MHT显式表示轨迹存在、多帧数据关联和多假设，并用N-scan pruning控制复杂度|采用固定滞后MHT/PMBM-lite维护匿名轨迹|track posterior不得改写event身份或注册状态|

主要参考：

- Kull et al.,“Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration,”NeurIPS 2019。https://proceedings.neurips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html
- Trick and Rothkopf,“Bayesian Classifier Fusion with an Explicit Model of Correlation,”2021。https://arxiv.org/abs/2106.01770
- Kumar et al.,“Calibrated ensembles can mitigate accuracy tradeoffs under distribution shift,”UAI 2022。https://proceedings.mlr.press/v180/kumar22a.html
- Julier and Uhlmann,“Using covariance intersection for SLAM,”Robotics and Autonomous Systems,2007。https://doi.org/10.1016/j.robot.2006.06.011
- Angelopoulos et al.,“Conformal Risk Control,”2022。https://arxiv.org/abs/2208.02814
- Gangrade et al.,“Selective Classification via One-Sided Prediction,”AISTATS 2021。https://proceedings.mlr.press/v130/gangrade21a.html
- Lee et al.,“Set Transformer,”ICML 2019。https://proceedings.mlr.press/v97/lee19d.html
- Li and Oliva,“Active Feature Acquisition with Generative Surrogate Models,”ICML 2021。https://proceedings.mlr.press/v139/li21p.html
- Jazbec et al.,“Early-Exit Neural Networks with Nested Prediction Sets,”UAI 2024。https://proceedings.mlr.press/v244/jazbec24a.html
- Xia et al.,“Multi-Scan Implementation of the Trajectory Poisson Multi-Bernoulli Mixture Filter,”2019。https://arxiv.org/abs/1912.01748
- Vempaty et al.,“False Discovery Rate Based Distributed Detection in the Presence of Byzantines,”2012。https://arxiv.org/abs/1212.5654

## 4.输入合同

节点`m`对event`e`只发送：

```text
event_id / reception_id / evidence_origin_id
node_id / node_state_hash / base_bundle_id
class_registry_id / prior_id / calibration_id
p_local[registered classes + unknown]
quality_vector q_m
time/frequency/beam/position/visibility metadata
topology lookup key / nonce / signature / artifact hash
optional Tier-1/2 quantized evidence interval
```

所有节点必须命中预封存roster。不同节点可以有合法的receiver-specific state，但必须共享base bundle、feature contract、class registry、prior语义和校准协议。不同先验或不同校准语义的概率不能直接进入同一融合。

## 5.第一层：标签置换等价校准

原始本地概率记为`p_m∈Δ^(C+1)`。对概率作固定下限保护后，校准器为：

\[
\widetilde p_m=\operatorname{softmax}(A_{s_m}\log p_m+b_{s_m})
\]

`s_m`是预封存节点状态或质量桶。为保持registered类别置换等价，registered block只允许共享对角、共享非对角和共享bias结构；unknown维可以使用独立参数，但不能按具体registered类学习不同自由度。每个校准器必须绑定`prior_id`、`class_registry_id`、训练事件hash和有效期。

正式unknown门只有在存在authorized unknown fit与calibration资产时启用；两者与formal评价集在event、TX和anonymous entity上三方互斥。只有registered校准或source proxy时，unknown输出保留为研发诊断，不能生成正式FAR保证。这是CIRF v2的新方法合同，替代EV-CARE v1中不使用unknown冻结正式阈值的做法；两者结果必须分开报告，不能跨方法复用阈值。

校准输出必须finite、非负并满足simplex；否则该节点不进入正式融合。

## 6.第二层：先验修正证据

共同先验为`π_k>0`且`Σ_kπ_k=1`。节点对候选类`k`提供：

\[
\ell_m(k)=\log\widetilde p_m(k)-\log\pi_k
\]

这一步把posterior意见改写为相对共同先验的证据。它不是严格likelihood，除非校准合同和条件模型通过G2验证；因此G0/G1报告使用“prior-corrected evidence”，不写“Bayes-optimal likelihood”。

## 7.第三层：去重、连续相关性与鲁棒组内统计

### 7.1去重和拓扑相关核

先按`reception_id`、canonical artifact hash和`evidence_origin_id`去重。相关核`K`只能由query前签名topology registry生成，满足：

\[
K=K^T,\quad K\succeq0,\quad K_{mm}=1,\quad 0\le K_{mn}\le1
\]

共享前端、中继、时钟源或可证明同源路径提高`K_mn`。节点不能自报`K`。`K`不是运行时区间：每个正式run只使用一份query前冻结的`K*`。若依赖元数据不足，registry把无法证明独立的节点并入同一完全相关component，而不是给`K`添加乐观区间；只有可证明独立的component之间才允许`K_mn=0`。若registry原始矩阵不满足PSD，使用冻结的nearest-PSD投影，并把投影前后矩阵和特征值写入artifact。该修正只生成固定策略，不宣称覆盖所有未知相关结构。

### 7.2鲁棒位置

每个硬相关组只使用一个聚合器：带质量上限的加权Huber M估计。对类别`k`：

\[
\mu_g(k)=\arg\min_{\mu}\sum_{m\in g}\bar r_m\,\rho_c(\ell_m(k)-\mu)
\]

其中`rho_c`、迭代次数、收敛容差和并列规则在G0冻结；`bar r_m`先在`evidence_origin_id`层封顶，再在组内归一化。质量只能来自query前健康记录、物理质量和当前无真值完整性信息。不存在另一个“质量均值”分支。

组内离散度、Huber截断数和最大单点影响同时写入receipt。组内样本不足时退回预注册代表节点，不从当前置信度选择代表。

### 7.3有效证据量

组内有效证据量定义为：

\[
n_{\mathrm{eff},g}=\frac{(\sum_m r_m)^2}{r^T K_g r}
\]

组贡献为：

\[
L_g(k)=n_{\mathrm{eff},g}\,\mu_g(k)
\]

独立且等质量的节点使`n_eff`接近节点数；完全相关的复制使`n_eff=1`。精确复制在去重阶段被删除，即使进入属性测试，新增全相关行列和origin质量封顶也必须保持`L_g`不变。组划分由签名registry一次生成，节点不能通过自报group拆分同源证据；本方案只声明canonical grouping下复制不增益，不声明对任意人为拆组不变。

跨硬组使用同一公式，把组级质量、固定`K*`和`L_g`作为输入。G2有足够联合fit事件时，残余相关核由fit残差估计后向topology prior做PSD shrinkage，并在独立calibration split验证；数据不足时采用registry的完全相关component合并策略。相关核不根据正式query相似度更新，也不把`K*`称为未知相关性的置信区间。

## 8.第四层：区间证据与融合

每个`L_g(k)`同时携带加性误差区间：

\[
L_g(k)\in[\underline L_g(k),\overline L_g(k)]
\]

区间只覆盖三类可验证误差：校准有限样本误差、量化舍入和top-L遗漏质量。固定相关核`K*`属于方法策略，不放入区间，也不形成未知相关鲁棒证书。每类误差的校准集、置信水平、同时覆盖修正和适用context必须在run前封存。

固定相关核后，全局score区间为闭式线性求和：

\[
\underline S_k=\log\pi_k+\sum_g\underline L_g(k),\qquad
\overline S_k=\log\pi_k+\sum_g\overline L_g(k)
\]

softmax后验区间不得逐维独立归一化。实现使用区间log-sum-exp：

\[
\underline P_k=\frac{e^{\underline S_k}}{e^{\underline S_k}+\sum_{j\ne k}e^{\overline S_j}},\quad
\overline P_k=\frac{e^{\overline S_k}}{e^{\overline S_k}+\sum_{j\ne k}e^{\underline S_j}}
\]

该包络保持全部注册类和unknown竞争。若Tier-1只发送top-L，则遗漏类分别进入最坏分母，而不是当作0。

## 9.第五层：固定风险控制和三态决策

数据固定分成`fit/calibration/formal-test`三部分，并在event、TX和anonymous entity层互斥。Dirichlet校准器、固定`K*`残差修正、Huber参数以及有限样本／量化／top-L区间规则只能由fit split确定。calibration split只能运行这些冻结变换、计算nonconformity和阶统计阈值，不得重新估计`P_lower`、区间宽度、bucket或任何模型参数。

风险控制器唯一采用class-conditional split conformal prediction，不在同一run混用其它CRC调参。风险loss为`1[y不在Gamma(e)]`，每个类别和预注册bucket的anytime总体目标miscoverage固定为`alpha=0.05`。Tier`t`的nonconformity为：

\[
a_t(e,k)=1-\underline P_{t,k}(e)
\]

为控制完整Tier序列而不是单个Tier，calibration event计算`A(e,k)=max_t a_t(e,k)`。对类别`k`、bucket`b`，阈值`q_kb`取`A(e,k)`样本的`ceil((n_kb+1)(1-alpha))`阶统计量。运行时集合为：

\[
\Gamma_t(e)=\{k:\max_{s\le t}a_s(e,k)\le q_{kb}\}
\]

同一阈值直接保证集合随Tier嵌套，且miscoverage针对整个预注册Tier序列，不需要把3个单Tier的5%错误率累加。bucket只由query前可观测的scenario和effective-group-count确定；bucket表在采集前冻结。

fit、calibration和formal-test必须在event、TX、anonymous entity上三方互斥。authorized unknown的fit TX、calibration TX和formal评价TX也必须身份级三方互斥。每个calibration`class×bucket`至少60个独立event；不足时不根据结果合并，而是按预注册树逐级回退到更粗bucket；根bucket仍不足则该类风险门fail-closed。

split conformal提供每个固定cell的边际miscoverage目标，不宣称任意分布偏移下的条件保证。正式结果另用Clopper-Pearson上界审计；同时报告全部`(C+1)×B`cells时，置信度使用Bonferroni校正`delta_cell=0.05/((C+1)B)`。该审计不回流阈值。

registered输出`c*`需要：

1.`c*`在全部score区间合法分配下仍为唯一winner；
2.固定split conformal prediction set为单元素`{c*}`；
3.registered单侧误接收上界、unknown上界和margin门同时通过；
4.至少两个独立高质量组同向支持；
5.leave-one-group-out不改变三态或winner；
6.没有完整性、校准、roster、deadline或dominant-group故障。

unknown输出需要：

1.存在authorized unknown校准集并通过预注册样本量门；
2.全部合法score区间下unknown仍为唯一winner；
3.至少两个独立高质量组同向支持unknown；
4.known FAR上界和unknown safe-rejection下界同时通过；
5.没有registered类获得独立强支持。

其余情况为defer。该方法不声称容忍“一独立组任意恶意”并仍作出确定身份；在只有两个组且冲突时只能defer。leave-one-group-out是敏感性门，不是Byzantine证明。

## 10.渐进通信与anytime决策

证据分三层：

|层级|内容|用途|
|---|---|---|
|Tier-0|签名、ID、完整性、local三态、top-1、margin、质量和物理元数据|快速检查与单节点降级|
|Tier-1|量化全类证据，或top-L＋遗漏总质量及逐类上界|默认协同|
|Tier-2|高精度全类证据、完整区间分解和解释字段|冲突或边界事件|

下一份证据的请求顺序按预注册的“最大最坏区间收缩／字节”规则计算。它可以使用当前区间宽度、节点可见性、预计时延和链路成本，但不能使用query真值或scorer。没有合法生成模型时，不使用学习式信息增益，只用解析上界。

正式anytime集合直接使用第9节的累计最大nonconformity公式。它等价于对各Tier候选集合做显式交，但风险阈值由完整Tier序列一次校准，避免单Tier误覆盖累积。类别一旦排除不能重新进入；集合为空立即defer。soft deadline可产生带版本号的provisional artifact；hard deadline封存final，迟到证据只进入延迟审计。

## 11.匿名轨迹：固定滞后MHT

事件级三态先封存。轨迹层再维护：

```text
track existence probability
birth/death/expiry model
frequency/Doppler/visibility/beam state
common-base z_track robust centroid and dispersion
missed-detection and clutter likelihood
top-H association hypotheses
fixed-lag history and N-scan pruning receipt
```

每个event先按相关组求common-base球面medoid，再按组等权生成`z_track`。缺少共同feature contract或离散度过大时禁用表示项，只用物理元数据。

实现合同唯一冻结为MHT：每个新event最多保留3个门内track关联分支和1个birth分支；全局最多保留`H_max=32`个假设；fixed lag为最近5个event opportunity或120秒中先到者；`N-scan=3`后封存更早关联；相对最优log mass低20以上的假设剪枝。birth rate、missed-detection probability和clutter rate只由G2 fit split按visibility/scenario估计，并在calibration/test前冻结；缺少合法估计时使用G0预注册保守值且只作技术验证。

轨迹更新保留有限个多假设，而不是每次贪心合并。固定滞后窗口后封存历史关联；后续证据可以产生新的纠错artifact，但不能覆盖已封存event decision，也不能改变shot、credential和fresh-K资格。

## 12.计算与通信预算

设节点数`M≤5`、相关组数`G≤M`、注册类数`C`：

- 校准和prior修正：`O(MC)`；
- Huber组内统计：固定迭代数时`O(MC)`；
- 相关核和有效证据量：`O(M²+GC)`；
- score区间与三态门：`O(GC)`；
- leave-one-group-out：朴素`O(G²C)`，可用前缀和降至`O(GC)`；
- fixed-lag track：由`H`个假设和窗口`W`控制，必须预注册`H_max/W/N-scan`。

默认Tier-1发送全类int8／int16区间证据。类别很多时采用top-L＋遗漏质量上界，但只有最坏区间仍能证明同一结论时才省略Tier-2。

## 13.为什么预期比EV-CARE更强

1.EV-CARE依赖原始`p_local`的可比性；CIRF先做共同先验和多分类校准。
2.EV-CARE只有硬相关组；CIRF用连续PSD相关核计算有效证据量，既利用部分独立性，又抑制残余相关。
3.EV-CARE组内线性池仍受异常概率牵引；CIRF使用质量封顶的Huber位置。
4.EV-CARE的leave-one-group-out主要是事后故障门；CIRF把校准、量化和top-L误差直接传播到决策区间。
5.EV-CARE的门是固定点阈值；CIRF增加事件级、分层且完全query前冻结的风险控制。
6.EV-CARE轨迹是medoid＋代价；CIRF显式表示存在概率、漏检、杂波和多假设，降低false merge和fragmentation。
7.CIRF不增加新的RF特征对齐、teacher或本地模型反传，工程上可以直接复用现有Phase1/Phase2本地artifact。

“预期更强”是机制推断，不是性能事实。只有G2真实same-event heldout矩阵才能确认增益。

## 14.失败模式和降级

|失败|检测|降级|
|---|---|---|
|校准合同不一致|prior/class/calibration hash|拒绝该节点；不足quorum则leader或defer|
|相关核缺失|topology registry miss|把无法证明独立的节点并入完全相关component|
|相关核非PSD|eigenvalue audit|冻结nearest-PSD修正并记录；不宣称未知相关鲁棒性|
|组内异常|Huber截断、离散度、LOO|降低有效证据；翻转则defer|
|unknown校准不足|unknown calibration gate|unknown只诊断，不作正式unknown决定|
|Tier-1区间太宽|最坏决策不唯一|请求Tier-2；超时则defer|
|只有一个有效组|quorum|逐字节leader fallback并标记非协同|
|event binding失败|签名时间频率物理约束|不得融合，可转track-only审计|
|track证据冲突|hypothesis mass、物理不可达|保留多假设或split，不回写event|

## 15.实验路线

### G0：纯技术性质

不报告性能。验证：

- simplex、finite和共同先验合同；
- 类别置换、节点排列、N=1逐字节恒等；
- reception/origin复制不增益；
- `K*`对称、PSD、nearest-PSD修正、canonical group和精确复制性质；
- score区间外包络、区间softmax和top-L遗漏最坏分配；
- registered/unknown/defer三态和unknown校准缺失降级；
- Tier集合嵌套、hard deadline不可覆盖；
- track零回写和shot不增加。

### G1：代理多接收节点

先truth-blind构造group并封存预测，再由独立scorer连接truth。比较：leader、质量平均、普通PoE、CARE-PoE、EV-CARE、CIRF。数据标记为`PROXY_MULTI_RECEIVER`；source-proxy unknown只作诊断。

所有多节点主基线必须消费同一份`ptilde`、共同prior和相同Tier字节预算，以免把校准收益混入融合收益。另设`raw local→calibrated local`校准消融；`N_sat=1`正式恒等测试仍绕过CIRF校准并逐字节返回原本地artifact。

G1必须拆分贡献：

1.校准＋prior修正；
2.连续相关折扣；
3.Huber鲁棒组内统计；
4.区间风险控制；
5.anytime Tier策略。

任何提升若只来自defer增加，视为失败。

### G2：真实same-event离线回放

只有采集通过EV-CARE设计中的同步误差、false-binding、独立前端、unknown来源、样本量和truth-sidecar acceptance spec后才进入G2。按`emission_event_id`和anonymous entity做fit/calibration/test三级互斥，所有方法使用同一event、roster、deadline和证据字节预算。

### 因果矩阵

保持A/B/C/D与`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`全交叉。分别报告`B-A`、`C-A`、`D-B`和`D-B-C+A`。`REG0`的`seen_new_acc/H_old_new`为N/A。

## 16.非补偿晋级门

1.P0协议、event binding、truth隔离、artifact不可变和N=1恒等全部通过。
2.所有registered类accuracy、min-class、min-receiver、min-scenario不低于leader基线2pp以上；defer按错计。
3.正式unknown FAR≤5%、safe rejection≥95%，且unknown calibration acceptance通过。
4.每个scenario、有效组数和节点子集分别过门，不能用均值补偿失败cell。
5.复制、缺失、迟到、单异常组和预注册`K*`变体矩阵逐项通过；不靠defer淹没故障。
6.Brier、NLL和校准检验不得显著恶化；风险证书的经验违约率在冻结置信上界内。
7.Tier-2请求率、平均字节、p95时延、峰值内存和deadline完成率分别满足预算。
8.track precision/recall、IDF1、fragmentation和false merge分别过门；track不能补偿event级失败。
9.任何门失败即`REJECT_CIRF_TRACK_V2`，不根据query重调相关核、阈值、校准器或Tier策略。

## 17.当前最需要完善的资产

算法之外，真正决定性能能否被识别的是四类资产：

1.真实same-event联合reception和可验证时间／传播校正；
2.共享前端、中继、时钟和链路相关性registry；
3.与formal query互斥的registered及authorized unknown校准事件；
4.跨过境anonymous track truth、漏检和杂波标注。

没有这四类资产时，可以完成G0和代理G1，但不能可靠估计相关核、正式unknown风险或轨迹false merge。继续堆叠跨节点深度网络不会绕过这个识别问题。

## 18.最终建议

`CIRF-Track v2`应作为EV-CARE-Track之后唯一的新协同候选进入分阶段验证。它保留EV-CARE已经正确的事件、证据、生命周期和降级边界，把主要创新集中在概率可比性、部分相关、有限样本风险、通信渐进性和概率轨迹五个实际瓶颈。

第一步不是大规模训练，而是完成G0性质测试和G1同event格式代理执行器。只有CIRF在不增加defer的前提下稳定优于EV-CARE，并且真实G2采集资产通过acceptance spec，才值得拟合残余相关核和启动正式性能矩阵。

独立科学复审结论：`P0=0，P1=0，MERGE`。该结论只允许进入实现与分阶段实验，不构成性能提高、真实same-event数据可用或在轨验证声明。
