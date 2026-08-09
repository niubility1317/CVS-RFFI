# CIRF-Track v3：事件账本约束的因子化相关风险协同推理

状态：`DESIGN_FROZEN_INDEPENDENT_REVIEW_MERGE`

日期：2026-08-09

适用阶段：Phase3部署期多接收节点协同

证据边界：本文是对CIRF-Track v2的第二轮文献驱动修订。它不构成性能提高、真实same-event数据可用、在轨验证或Phase3能力已经实现的声明。

## 1.结论先行

CIRF-Track v2已经闭合概率校准、固定相关核、区间融合、anytime conformal和固定滞后MHT，但进一步审查发现，决定真实部署成败的瓶颈仍位于融合公式之外：同一事件是否真的成立、本地unknown分数能否被当作一个普通类别概率、运行先验是否漂移、证据请求策略是否破坏校准、网络缺失是否与样本难度相关，以及轨迹漏检是否只是卫星不可见。

v3不增加端到端跨节点网络。它把系统拆成四个相互隔离的平面：

1.事件平面：先用物理时间、传播、频率、波束和签名账本证明same-event；
2.决策平面：分别校准known条件分类与unknown门，再做相关性约束的证据融合；
3.网络平面：把消息顺序、缺失、量化和停止时刻作为完整transcript校准；
4.轨迹平面：只处理匿名实体关联、乱序量测和可见机会，不回写事件身份。

默认部署拓扑冻结为`receiver nodes→唯一gateway fusion→immutable event decision`。星间链路或地面网关只改变传输位置，不产生多个相互竞争的最终裁决者。网关不可达时，各节点只保留原始`N_sat=1`本地结果，不运行临时多数投票。

## 2.v2仍存在的关键不足

|编号|不足|现实后果|v3处理|
|---|---|---|---|
|V3-01|same-event只有原则性绑定，没有时钟、传播和碰撞的可计算证书|异步发射可能被错误叠成一个shot|新增event-opportunity ledger与唯一可行事件假设门|
|V3-02|把registered类别和unknown直接放入同一`C+1`校准向量|能量型unknown分数被伪装成普通类别posterior|将unknown门与known条件分类因子化校准，再组合成联合意见|
|V3-03|只绑定共同运行先验，没有绑定各节点训练参考先验|不同节点可能重复加入或错误移除先验|同时封存`reference_prior_id`与query前operational prior set|
|V3-04|`n_eff×Huber mean`缺少唯一的硬组降维和跨组合并算子|改变组切分可能改变winner|删除合成专家，直接在去重证据单元上使用共享、非负的相关QP|
|V3-05|nearest-PSD可能破坏对角为1和完全相关component|矩阵修复可能凭空制造独立性|改为受约束PSD completion；不可行则完全相关合并|
|V3-06|conformal集合覆盖被用于辅助身份与unknown风险门|集合coverage合格时，unknown FAR仍可能超标|把集合覆盖与决策级风险账本分开|
|V3-07|固定每格60事件与多重置信门不自洽|小样本cell可偶然通过|按目标风险和familywise置信度反推样本量|
|V3-08|Tier请求由当前证据和网络状态决定，但校准只描述静态Tier|选择和停止策略会产生feedback shift|对冻结scheduler的完整网络transcript重放校准|
|V3-09|MHT把不可见期和漏检混合|计划遮挡会造成虚假死亡和fragmentation|只在至少一个独立组可见的opportunity上计算miss|
|V3-10|签名证明来源但不证明诚实|合法密钥被攻陷的节点仍可发送伪证据|明确fail-silent、数值故障和Byzantine三种不同威胁范围|

## 3.不可改变的协议边界

1.一个`emission_event_id`对应的多节点reception仍只计一个shot。
2.本地证据先封存，融合器不读取query真值、真实角色、类别配额、batch构成或scorer。
3.registered样本输出unknown或defer均按身份错误计数；unknown的defer只计unresolved，不计safe rejection。
4.`N_sat=1`逐字节返回节点冻结结果，绕过v3的校准、先验、融合和轨迹反馈。
5.unknown或anonymous track不更新Phase2。只有外部确权、fresh-K独立事件和新`split_id`才能交给Stage2-C。
6.WiSig/ManySig和LEO弱信道只能形成`PROXY_MULTI_RECEIVER`证据，不能声明真实同步多星或在轨结果。
7.track不能改变已封存event decision、shot、threshold、credential、registration authorization或fresh-K资格。

## 4.事件平面：Event-Opportunity Ledger

每个节点必须在当前query之前命中同一签名roster epoch，并提交：

```text
capture_time_interval
clock_state_id / clock_error_bound / drift_bound
receiver_ephemeris_interval
propagation_delay_interval
carrier_frequency_interval / Doppler_residual_interval
beam / visibility / transmission-opportunity ID
waveform-content digest / nonce / monotonic node counter
key epoch / revocation epoch / evidence origin
```

`transmission-opportunity_id`必须由query前的物理调度面生成，唯一允许的字段为`roster_epoch、time-slot、band、beam、visibility-cell、schedule-epoch`；其hash输入不得含TX／class、registered／unknown角色、credential身份、模型预测或历史track身份。所谓`waveform-content digest`也不是任意payload摘要：它只能取协议公开且与设备身份、地址和payload无关的同步／训练序列，或内容中性的匹配滤波时序摘要；不得读取`z_id`、RFFI learned feature、MAC／设备地址或分类输出。若协议不存在这种内容中性字段，则该项固定为`NULL`并失去绑定作用，不能以身份特征替代。对TX身份做任意置换必须保持event ledger候选与same-event证书逐字节不变。

对候选物理事件`h`，节点`m`的传播校正发射时刻区间记为`I_m(h)`。只有同时满足以下条件才生成same-event certificate：

1.`intersection_m I_m(h)`非空；
2.频率与Doppler残差区间存在共同物理解；
3.波束、可见性和发射机会没有物理冲突；
4.内容摘要、计数器、nonce、key epoch和撤销状态闭合；
5.在全部误差边界内只剩一个可行事件假设；
6.heldout collision set上的false-binding置信上界通过预注册门。

若存在两个及以上可行事件假设，reception不得进入same-event身份融合，只能分别保留并交给track层做多假设关联。账本允许历史物理轨迹提供查询前的可达性约束，但禁止读取历史身份posterior来帮助当前身份分类。

时钟同步必须作为测量不确定性而不是布尔健康位处理。LEO同步研究表明clock offset、drift和链路异常会直接影响TOA／FOA融合，因此G2采集必须同时封存时钟误差状态，而不能只记录时间戳。参考：Leonardi et al.,“Robust and Resilient GNSS Synchronization of LEO Satellites for Space-based Aircraft Multilateration,”IEEE MetroAeroSpace 2025，https://doi.org/10.1109/MetroAeroSpace64938.2025.11114554 。

## 5.决策平面第一步：因子化本地意见

节点不再被要求直接提供一个未经定义的`C+1`posterior。正式v3接口不接收可能含0／1端点的概率，而是提供两个finite raw-score：

```text
g_m                 unknown gate raw logit
z_m[1:C]            registered条件分类raw logits
reference_prior_id  本地模型训练／校准参考先验
```

每条证据还必须绑定`base checkpoint、class registry／order、unknown converter、gate calibrator、registered temperature、reference prior、Tier codec`的canonical hash，以及节点在roster中预封存的合法receiver-specific state hash。任一hash缺失、路径漂移、class-order不一致或state未命中roster都只淘汰该origin；不得临时兼容或重校准。

输入先在float64中执行固定单调数值适配：`g_m=clip(g_m,-30,30)`；`z_m`先减去其最大值，再逐维clip到`[-30,0]`。任何NaN／Inf、维度漂移或无法追溯的energy／distance输入均fail-closed。只有绑定fit资产的冻结单调转换器才能把energy／distance变成`g_m`；否则正式unknown门禁用。`g_m`的正方向必须在fit资产中固定为“更像unknown”，不允许运行时翻转。对`g_m`使用带固定`L2=1e-4`的正斜率Platt scaling，对`z_m`只使用正逆温度`tau_s=1/T_s`；参数域固定为`a_s,tau_s in [1e-3,1e3]、b_s in [-30,30]`，二者均以fit NLL加固定L2的唯一凸最优解冻结，优化容差和样本hash写入artifact：

\[
\log\widetilde u_m=\operatorname{logsigmoid}(a_s g_m+b_s),\quad a_s>0,\qquad
\log\widetilde q_m=\operatorname{logsoftmax}(z_m/T_s),\quad T_s>0.
\]

其中`s`只能是预注册的scenario／quality context，必须具有唯一pooled fallback；未命中时defer。registered分支禁止逐类温度或逐类bias，确保标签置换等价。随后形成联合意见：

\[
\log\widetilde p_m(U)=\log\widetilde u_m,\qquad
\log\widetilde p_m(k)=\log(1-\widetilde u_m)+\log\widetilde q_m(k).
\]

全部运算使用稳定`logsigmoid／log1m-sigmoid／logsoftmax／logsumexp`，不先生成0或1再取log。最终log-probability必须finite且`logsumexp=0±1e-12`。

该分解避免把energy、distance或reject score直接当成一个普通类别logit。双分支open-set方法也通常把闭集分类和open-set能量分开建模；这里只吸收接口分解，不增加新的训练网络。参考：Wang et al.,“Glocal Energy-Based Learning for Few-Shot Open-Set Recognition,”CVPR 2023，https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Glocal_Energy-Based_Learning_for_Few-Shot_Open-Set_Recognition_CVPR_2023_paper.html 。

unknown校准资产必须按未知来源族报告，例如未注册同协议TX、异协议发射、干扰／非目标信号和近边界硬unknown。正式FAR声明只覆盖预注册并被采样的operational unknown mixture；不得从有限authorized unknown集合外推到“所有未知发射机”。

数据角色必须区分registered identity与unknown identity，并拆成`fit、interval-calibration、conformal-calibration、formal-test`四层。registered四层共享同一份注册表TX集合，只在`emission_event_id、physical_sample_id、risk_cluster_id`和采集机会层互斥；否则校准阶段将没有对应的registered类别。unknown四层的TX与anonymous entity必须身份级四方互斥，并且全部不在registered registry中。anonymous entity互斥只适用于unknown／defer生命周期，不作用于registered类别定义。acquisition-pilot另与四层全互斥。

## 6.参考先验与运行先验不确定性

每个节点的参考先验`pi_ref,m`必须随校准artifact封存。只有参考先验语义一致，或能使用冻结变换还原到共同reference prior的节点，才能进入同一证据池。

对全部registered类和被授权建模的unknown维，要求`pi_ref,m(k)>0`，并先构造参考先验修正证据：

\[
e_m(k)=\log\widetilde p_m(k)-\log\pi_{ref,m}(k).
\]

它在G0／G1中只称为prior-corrected evidence，不宣称严格likelihood。后续相关QP只能作用于`e_m`的相对log-odds，不能再次对posterior直接去先验。

运行先验不从当前query batch、预测直方图或track身份历史估计。运营方在query之前依据注册表和身份盲的`transmission-opportunity_id`映射给出有限先验场景集`Pi_b={pi^(1),...,pi^(J)}`，每个向量严格正且归一化；映射输入只能含计划窗口、band／beam、visibility cell和任务日历，不得含TX／class、known／unknown角色、credential、当前预测或track身份。v3不接受未离散化的连续先验区间。当业务没有可信先验时，集合退化为单个预注册uniform prior，而不是用当前观测自适应。

对每个`pi in Pi_b`传播融合结果；只有所有合法先验下winner和三态一致时才能作出确定决策。label shift会损害普通概率校准与marginal coverage；本设计的class-conditional split conformal在`P(score|Y,context)`保持不变时不因纯先验变化而失效，但conditional／covariate／network shift仍会失效。因此`Pi_b`只解决决策先验稳健性，不能替代条件分布监控。参考：[Podkopaev and Ramdas, UAI 2021](https://proceedings.mlr.press/v161/podkopaev21a.html)。

## 7.唯一相关性融合算子

### 7.1受约束相关核

拓扑registry先生成query前冻结的0／1先验核`K_top`：同一evidence origin、共享模拟前端、共享中继或共享时钟误差源为1，只有具有独立前端与独立误差源证明的单元之间为0。v3分别维护registered轴`K_R`和unknown轴`K_U`，但registered的全部类别共享同一个`K_R`，禁止逐类相关权重。若经验残差不足，两个轴都直接使用保守`K_top`。

`K_top`必须来自evidence-origin equivalence partition，因此天然是由全1对角block组成的PSD矩阵；任意不满足等价关系传递性的手工0／1图都拒绝。G2存在合法联合fit事件时，`R`轴只用registered事件，`U`轴使用预注册registered类与unknown来源族的门控误差，残差唯一为：

\[
r^R_{m,e}=\left[d^R_m(y_e)-\max_{k\ne y_e}d^R_m(k)\right]-\operatorname{median}_{fit,context}(margin_m),
\]

\[
r^U_{m,e}=d^U_{m,e}-\operatorname{median}_{fit,context,gate\ family(e)}d^U_m.
\]

其中`d^U=e(U)-e(r)`与formal融合量逐字一致，因此unknown核同时吸收gate与参考registered logit的共同扰动。`gate family`只能是采集前冻结的`registered`或具体authorized unknown来源族，不能使用当前预测角色。context只含query前可观测的scenario／quality bucket，并为每个bucket预注册唯一pooled fallback；formal时未命中bucket与fallback均defer。每轴每origin定义`sigma_m^R=max(1e-3,1.4826 MAD(r_m^R))`、`sigma_m^U=max(1e-3,1.4826 MAD(r_m^U))`。

缺失模式可能与轨道几何、信号难度和节点负载相关，因此不能从全节点complete-case相关矩阵直接抽取子核。对每个预注册`availability set A×context c×axis a`单独建立fit cell，只使用该cell内`A`中全部origin都具有同一sealed event的listwise-complete blocks。对每轴残差按origin做cell内中心化和unit-standard-deviation标准化得到`x^a_{b,m}`，计算普通样本相关`K^A_emp,a`；MAD尺度只供QP的`D^A_a`使用，不混入相关系数。若联合block数未达到预注册下限，该cell直接使用子拓扑核`K^A_top`和fit中全部合格cell的逐origin保守最大MAD尺度；若连保守尺度也不可得则defer。只有经独立validation证明残差与availability在给定context下条件独立时，才允许多个availability cell共享同一核，并在artifact中封存检验、功效和hash。否则禁止从全节点核简单取子矩阵。满足样本量时使用唯一的固定目标解析收缩：

\[
\widehat V^{a,A}_{ij}=\frac{1}{B(B-1)}\sum_b\left(x^{a,A}_{b,i}x^{a,A}_{b,j}-K^{a,A}_{emp,ij}\right)^2,
\]

\[
\lambda_{a,A}=\operatorname{clip}_{[0,1]}\frac{\sum_{i\ne j}\widehat V^{a,A}_{ij}}{\lVert K^{a,A}_{emp}-K^A_{top}\rVert_F^2},\qquad
K^A_a=(1-\lambda_{a,A})K^{a,A}_{emp}+\lambda_{a,A}K^A_{top}.
\]

若分母为0，定义`lambda_{a,A}=1`。估计公式、availability／context、联合block ID、block数、`lambda_{a,A}`和双核hash全部写入fit artifact。该构造只把方差过大的经验相关收缩到物理拓扑先验，不使用validation调系数。独立validation只检查区间覆盖和PSD合同；任一轴失败则该cell该轴退回`K^A_top`，不得调参。两轴相关方向相反时必须保留双核，不能平均成单核。替换unknown来源标签、context、availability或任一fit block必须改变相应轴hash。

若任一轴矩阵不满足对称、PSD、单位对角和`[0,1]`边界，分别求解受约束PSD completion：

\[
\min_{K'}\lVert K'-K_{axis}\rVert_F^2
\quad\text{s.t.}\quad K'\succeq0,\ K'_{gg}=1,\ 0\le K'_{gh}\le1,
\]

并强制已知完全相关component的对应子矩阵保持全1。若问题不可行或输入约束不足，相关component直接合并，绝不能通过投影制造独立性。

v3不再把节点先变成Huber“合成专家”，因此也不需要把节点核非线性降维为组核。按`reception_id`、canonical artifact hash和`evidence_origin_id`去重后，每个唯一证据origin直接占据`K_R／K_U`的一行／列。完全相关component仍用于quorum和威胁边界，但不改变证据向量维度。任何残差核、质量尺度、影响上限或QP约束的选择都只能使用fit split，不能读取interval-calibration或conformal-calibration。替换后二者的标签、分数或顺序不得改变双核及其hash。

### 7.2双轴相对证据

每个唯一证据origin产生两条不可互换的参考先验修正证据。参考类`r`按canonical registry固定：

\[
d^R_m(k)=e_m(k)-e_m(r),\quad k\in\mathcal C_R,
\]

\[
d^U_m=e_m(U)-e_m(r).
\]

registered的全部类别共享同一个`d^R`权重向量；unknown只使用`d^U`权重向量。类别置换测试必须同时置换registry与参考映射。v3不做逐类trim、逐类节点选择或第二个质量平均分支，避免不同类别由不同隐含节点子集拼成一个不存在的专家。

### 7.3证据单元级双轴非负相关QP

每个证据单元的边际误差尺度只能来自fit split中相同`availability A×context c`合同的残差，并在formal query前冻结。对当前活跃cell，registered轴定义`D_R^{A,c}=diag(sigma^R_m)`、`Sigma_R^{A,c}=D_R^{A,c}K_R^{A,c}D_R^{A,c}`；unknown轴同理。以下QP分别以`a in {R,U}`独立求解；`R`轴的权重对全部registered类别共享，`U`轴只服务unknown门，二者不得平均或按当前query选择。为简洁，下面省略`A,c`上标；全roster只是其中一个availability cell。

先冻结可行域：

\[
\mathcal F=\{\beta\ge0:\mathbf1^T\beta=1,\ \beta_m\le c_m,\ \sum_{m\in component_j}\beta_m\le c_j\}.
\]

普通完整性模式使用fit冻结的origin上限`c_m`且`c_j=1`；只有声明`f=1`的模式才要求至少3个独立component并设置所有`c_j=0.5`。若`F`不可行，自动撤销Byzantine确定决策声明，回到普通模式；不得放松上限。定义`v^a_m=(sigma^a_m)^{-2}/sum_i(sigma^a_i)^{-2}`，二级参考权重唯一为欧氏投影：

\[
\beta^a_0=\arg\min_{\beta\in\mathcal F}\lVert\beta-v^a\rVert_2^2.
\]

随后主QP为：

\[
\beta_a^*=\arg\min_{\beta\in\mathcal F}\beta^T\Sigma_a\beta,
\qquad
\nu_a=\min\left(M,\frac{1}{\beta_a^{*T}K_a^*\beta_a^*}\right).
\]

QP采用字典序唯一化：先最小化`beta^T Sigma_a beta`；若存在多个主最优解，再在主最优集合中最小化`||beta-beta^a_0||_2^2`。严格凸二级目标给出唯一解，不再用hash选择连续权重。双精度`feasibility_tol=1e-10`、`primary_optimality_tol=1e-10`和求解器版本写入artifact；任一轴残差超界即defer。

双轴融合证据为：

\[
L_R(k)=\nu_R\sum_m\beta^*_{R,m}d^R_m(k),\qquad
L_U=\nu_U\sum_m\beta^*_{U,m}d^U_m.
\]

独立等质量证据单元在各轴给出`beta=1/M,nu=M`，因此恢复证据求和；完全相关component给出`nu=1`，因此整个component只贡献一份证据。等误差时二级目标给出均匀意见，异误差时偏向fit误差较小的origin，但证据数量仍为1。每个origin另有query前质量／影响上限，不能通过当前高置信度抢占权重。

质量bucket与`c_m`只能来自网关可验证的传感器健康、量化档位、链路状态和fit误差表；节点自报SNR／quality只能降低自身上限，不能提高权重。任何随当前类别置信度增加的“质量分”都禁止进入QP，以免错误节点用过度自信抢占融合。

对每个实际transcript`T`，先定义已通过完整性与deadline门的活跃origin集合`A(T)`。若`|A|=1`，逐字节返回该origin的冻结本地artifact并标记`DEGRADED_N1_NONCOLLABORATIVE`，不运行QP，也不声明协同风险保证。若`|A|>=2`，显式重建：

\[
\mathcal F_A=\{\beta_A\ge0:\mathbf1^T\beta_A=1,\ \beta_m\le c_m,\ \sum_{m\in component_j\cap A}\beta_m\le c_j\},
\]

\[
v^a_{A,m}=\frac{(\sigma^a_m)^{-2}}{\sum_{i\in A}(\sigma^a_i)^{-2}},\qquad
\beta^a_{0,A}=\operatorname{Proj}_{\mathcal F_A}(v^a_A).
\]

随后使用该`availability A×context c`预先冻结的`K^{R,A,c}／K^{U,A,c}`和`Sigma^{R,A,c}／Sigma^{U,A,c}`求解同一字典序QP；非活跃origin权重严格为0。任意模式下`F_A`不可行、cell未注册或任一轴求解失败均立即defer，不得放松origin或component cap。不得从全节点权重或核简单删行后重新归一化。

需要声明一个authenticated Byzantine component仍可确定决策时，必须有至少3个独立component、冻结“任一component总权重不超过0.5”的约束，并要求删除任一component后，区间winner、三态、风险门和conformal singleton全部不变。否则只能使用冲突检测与defer，不声称容错。

对每个query前operational prior`pi in Pi_b`，最终相对score为：

\[
S(r;\pi)=\log\pi_r,\qquad S(k;\pi)=\log\pi_k+L_R(k),\qquad S(U;\pi)=\log\pi_U+L_U.
\]

区间与三态门必须对`Pi_b`中的全部合法先验同时成立。这是一套固定双轴组合，不允许根据当前样本选择相关核、尺度或权重。

该算子仍是相关性策略，不是未知依赖结构的概率证书。必须分别压力测试`K*`低估、过估、split和merge；任何性能结论都绑定具体registry与核版本。

## 8.区间、conformal集合与决策风险必须分离

### 8.1事件级同时区间

区间只给真正随机的校准有限样本误差分配置信预算。所有生成`P_lower`的calibrator、区间参数、context bin、trace族和error envelope先由fit确定形式，再只在独立`interval-calibration`上估计并冻结；不得读取`conformal-calibration`或formal-test标签。统计原子唯一为`origin×class×context×stochastic-error-source`；令其总数为`N_atomic`，每项使用Bonferroni预算`delta_atomic=delta_event/N_atomic`。量化、top-L遗漏、有限`Pi_b`和冻结transcript都是已知集合上的确定性区间传播，只取逐类max-envelope，不再次切分`delta`，避免把同一校准误差重复计数。`delta_event、N_atomic、每项预算、确定性包络版本和区间hash`写入同一账本。无法进入统计原子或确定性有限包络的误差不得被包装为`P_lower`证书。

v3首版配置上限固定为：`M<=5、C_R+1<=32、context buckets<=12、stochastic error sources<=4、|Pi_b|<=4`，因此`N_atomic<=7680`；三个interval摘要各最多4个bin，delay最多4个bucket，quantization最多3档，每条origin最多1次重传。冻结scheduler裁剪物理不可能状态后，每个roster／context的全部可达transcript前缀不得超过`N_transcript_max=65536`。这些是设计容量，不是可调超参。

G0启动前必须精确枚举并写出`N_atomic、各风险cell的n_min、reachable transcript count、QP solve count、primitive-operation upper bound、peak-memory upper bound`。默认实现门为`N_atomic<=7680、N_transcript<=65536、peak memory<=8GiB、primitive operations<=2e9／fold`。任一项超限、已有独立block不足`n_min`或枚举器不能闭合时，该配置在启动前拒绝；不得事后合并bucket、删除prior、抽样trace或缩减unknown来源。扩大规模必须另立版本并重新审查。

### 8.2anytime prediction set

conformal只证明预注册数据与网络条件下的集合覆盖。它不直接证明singleton后的known误识别率、unknown FAR或safe rejection。v3保留嵌套prediction set作为必要门，但最终裁决另由决策级风险账本约束。

`conformal-calibration`只读取已经冻结并hash的`P_lower`函数，重放与formal完全相同的冻结scheduler，并只计算本节的`A_b／q_{k,c}`。默认G1／G2证书模式不把一条随机network trace附着到事件上，而是对每个校准事件枚举采集合同允许的全部missingness、arrival-order、delay bucket、重传和量化组合；物理不可能组合在query前由ledger排除。令`T(e)`为事件`e`在这个有限联合trace族下所有可达的Tier-1及以上transcript前缀，令`a_t(e,k)=1-P_{lower,t}(k)`。只有未来另立候选、并用独立数据验证条件trace生成器后，才可用抽样trace替代全枚举。置换conformal-calibration标签只能改变`A／q`及其hash，必须保持`P_lower`、interval参数和interval hash逐字节不变。block级class-conditional nonconformity唯一为：

\[
A_b(k,c)=\max_{e\in b:\,y_e=k,\ context(e)=c}\ \max_{t\in T(e)}a_t(e,k).
\]

若该cell有`n_{k,c}`个“至少包含一个真类`k`事件”的独立校准block，冻结split-conformal order statistic：

\[
j_{k,c}=\left\lceil(n_{k,c}+1)(1-\alpha_{k,c})\right\rceil,\qquad
q_{k,c}=A_{(j_{k,c})}(k,c).
\]

空cell或`j_{k,c}>n_{k,c}`时没有有限证书并defer，不用插值、邻近bucket或结果驱动合并。运行时的嵌套集合定义为：

\[
\Gamma_t(e)=\left\{k:\max_{s\le t}a_s(e,k)\le q_{k,c(e)}\right\}.
\]

由于累计最大值单调不减，`Gamma_t`只能缩小。registered／unknown确定裁决都要求当前集合为singleton，并同时通过独立决策风险门。context bucket、trace生成器、scheduler和`alpha_{k,c}`都在查看calibration结果前冻结；formal transcript未命中完整support时只能defer或进入预注册的静态full-Tier降级。此证书只在独立`event_opportunity_block`的预注册交换性范围内成立，不声称覆盖任意label／covariate／feedback shift。

### 8.3非补偿风险向量

每个事件生成以下有界loss：

```text
R_known_id       registered事件未输出正确registered ID；unknown/defer也计1
R_unknown_FA     unknown事件输出任一registered ID
R_unknown_safe   unknown事件输出unknown的补事件
R_false_binding  不同物理事件被融合
R_false_nonopportunity  实际可见机会被错误标为不可见
R_deadline       hard deadline前未形成要求的artifact
```

track的false merge、fragmentation和IDF1保持独立，不得补偿事件级风险。v3不在两种风险控制方法之间运行后择优：prediction set仅使用8.2的split conformal；上述每个正式decision risk都在独立formal-test block上把block内event loss取最大值，再使用预注册置信度的一侧精确Clopper-Pearson二项上界。所有规则与阈值在fit／interval-calibration／conformal-calibration结束后冻结，formal-test只审计、绝不选参；本版不使用Conformal Risk Control作决策门。proxy unknown不得进入正式unknown风险证书。

### 8.4样本量由门反推

删除固定“每格至少60事件”的常数。若某cell要求真实风险不超过`alpha`，familywise失败概率为`delta_cell`，即使观察到0次失败，也至少需要：

\[
n_{min}=\left\lceil\frac{\log\delta_{cell}}{\log(1-\alpha)}\right\rceil.
\]

当观察到失败时，使用精确二项分布上界重新判定。`delta_cell`由全部正式cell数量在采集前分配；样本不足只能扩大采集或fail-closed，不能依据结果合并bucket。多源和分布偏移下的conformal有效性需要额外假设，不能由来源数量自动获得，参考[Liu et al., ICML 2024](https://proceedings.mlr.press/v235/liu24ag.html)。

独立样本单位不是reception条数。same-event多节点仍只算一个event；同一过境、同一连续发射窗口或同一anonymous entity内存在时间相关时，风险审计以预注册`event_opportunity_block／mission_pass_id`为cluster。fit、interval-calibration、conformal-calibration和formal-test的block四方互斥。正式精确门把每个block的loss定义为该block内事件loss的最大值，再对独立block使用一侧精确二项上界；cluster bootstrap只作附加诊断。不得把一个过境中的大量reception当成大量独立样本。上述`n_min`按独立block计数。

`event_opportunity_block_id`也必须身份盲：只由mission pass、schedule epoch、visibility cell、band／beam和固定时间窗生成，不得含TX、class、known／unknown角色、预测或track身份。分块规则在采集前冻结；若独立性诊断失败，只能扩大block或撤销精确证书，不能为增加`n`而拆分相关事件。

G2采集acceptance spec必须在calibration前封存独立采样框：不同block至少跨不同`mission_pass_id`和连续发射窗口，并满足采集pilot预先估计后冻结的最小时间／轨道间隔`Delta_ind`。另外，独立scorer在truth sidecar内生成不回传预测器的`risk_cluster_id`，把同一registered TX或同一anonymous entity在该间隔内的全部事件保守合并；该字段不得进入event ledger、scheduler、prior或预测。`Delta_ind`只能由独立acquisition-pilot的自相关上界确定，pilot不得进入fit／calibration／formal。正式前须用另一份validation检查cluster间残差相关上界；失败时扩大cluster或撤销精确Clopper-Pearson／conformal证书。诊断通过不是数学独立性的证明，因此正式声明必须明确绑定这个采样框，不能外推到连续同轨过境。

## 9.网络平面：Transcript-Calibrated Anytime Policy

Tier策略不是三个静态数据包，而是一个确定性有限状态机：

```text
state=(active origins,independent components,arrival order,delay bucket,dropout mask,quantization level,interval-bin)
action=(request next fixed message | seal registered | seal unknown | defer)
```

所有tie-break、重传次数、deadline、节点顺序和每字节代价在run前冻结。scheduler只能读取三个冻结区间摘要：最大逐类宽度、第一／第二候选最坏margin、unknown最坏margin；三者的有限bin边界只能由fit split确定。有限transcript key必须包含roster epoch、活跃origin集合、arrival order、delay bucket、dropout mask、重传次数、量化级、三个interval-bin、Tier路径和停止原因。边界值使用左闭右开区间及最后一桶右闭的唯一规则。conformal-calibration必须在与formal-test互斥的事件上，以8.2的有限联合trace族重放完整scheduler；nonconformity取该策略所有可达transcript和停止时刻的最大值。这样证书不依赖随机trace与事件难度独立。formal transcript、interval-bin或missingness pattern未命中枚举support时，风险证书失效并进入defer或预注册全Tier静态降级。

scheduler的唯一fit目标是：在冻结event风险约束下，最小化`bytes+lambda_E*energy+lambda_T*deadline_cost`。`lambda_E／lambda_T`由运营约束在fit前给定。请求候选先按“尚未贡献的独立component优先”过滤，再使用fit lookup table中该动作对最坏区间宽度的保守收缩下界除以增量字节和时延进行排序；并列按canonical origin hash。lookup table只读query前物理状态、当前三个interval-bin和transcript，不读当前winner身份。若所有动作的保守收缩下界不为正，系统只能用当前已过门结果封存或defer，不能为追求置信度无界请求。

每次请求之前还必须用冻结的最大消息字节、最大重传次数、能耗上界和最坏链路时延检查剩余预算；只有`bytes_remaining、energy_remaining、deadline_slack`三项都能容纳该动作及封存回执时才允许发送。否则立即以当前已过门结果seal或defer。预算失败不能先传输、再仅记一条`R_deadline`。

dropout不假设missing-at-random。network trace的context必须显式包含轨道几何／可见性、链路拥塞、节点负载和scenario；这些字段只作query前状态，不含TX身份或prediction。任一context或联合missingness pattern超出校准support时撤销证书。缺失节点不做概率插补，也不把未返回证据当作中性票；只对实际活跃集合重建双轴QP。

Tier-0只携带完整性、事件账本、local三态、top-1、margin和质量摘要，不能给全部类别构造合法`P_lower`。因此它不得产生多节点registered或unknown最终裁决；它只能执行完整性检查、请求Tier-1或defer。唯一例外是`N_sat=1`逐字节返回原本地artifact。正式融合conformal序列从具有全类向量或所有遗漏类合法区间的Tier-1开始；Tier-2提供全精度证据。calibration与formal必须使用完全相同的请求、missing、乱序和deadline控制流。

只校准Tier0、Tier1、Tier2三个静态点不足以覆盖由当前证据触发的选择偏差。反馈依赖会破坏普通exchangeability；这一风险与反馈协变量偏移文献讨论的问题一致，参考[Prinster et al., ICML 2023](https://proceedings.mlr.press/v202/prinster23a.html)。

运行期允许无标签shift monitor检查网络trace、质量向量、区间宽度和node-state分布是否离开calibration support。monitor只能撤销风险证书并降级，不能在线改阈值、先验、相关核或校准器。

## 10.信任与故障模型

v3明确区分三类故障：

1.`fail-silent/crash`：丢包、延迟、节点不可用，由transcript和quorum处理；
2.`bounded numeric fault`：量化、有限精度、已知校准误差，由事件级区间处理；
3.`authenticated Byzantine`：密钥合法但内容被蓄意伪造，只能检测冲突并defer。

签名只证明消息来自某个密钥，不证明该节点诚实。要声明容忍`f=1`并仍给出确定融合结论，至少需要3个可证明独立的组、预注册威胁模型和逐组证据；只有2组时，一致可以融合，冲突必须defer。v3不声明抵抗两个组串谋、gateway被攻陷或拓扑registry被攻陷。分布式检测研究表明，Byzantine比例达到临界值后fusion center可以完全失明，因此小节点系统不能用“Huber稳健”替代明确攻击边界，参考[Kailkhura et al., 2013](https://arxiv.org/abs/1307.3544)。

节点counter、nonce、key epoch、roster epoch和撤销列表用于拒绝重放、分叉和过期状态。节点隔离只能由签名、计数器、物理不可能性或明确数值合同失败触发，不能因为它经常与多数身份结论不一致就自动拉黑。

## 11.轨迹平面：可见机会与乱序量测

anonymous MHT只接收已封存的unknown或defer事件；registered事件不得创建、合并或延长anonymous track。MHT只在event decision封存后运行，但可以读取事件前已经封存的物理track状态作为可达性约束。它不读取或累计历史registered身份posterior。

`missed detection`只在至少一个独立组对该匿名实体的预测可见区间成立时计入；只有整个sealed ephemeris、beam、clock和传播误差包络都证明所有节点不可见时，该时段才是`non-opportunity`。可见性不确定时按可能存在opportunity处理，不能乐观免计miss。另设`R_false_nonopportunity`：真实存在可见机会却被标为non-opportunity；其正式上界独立过门。所有节点均被地球遮挡、波束未覆盖或链路不可达时，该时段不降低track existence probability。

在fixed lag内到达的out-of-sequence measurement可以形成新的track revision artifact；超过lag或N-scan封存点的迟到证据只进入审计。连续时间多目标跟踪文献表明，乱序量测需要retrodiction和再更新，不能简单按到达顺序当作当前量测，参考“Continuous-Discrete Multiple Target Tracking With Out-of-Sequence Measurements,”IEEE TSP 2021，https://doi.org/10.1109/TSP.2021.3100999 。

v3初版禁止`z_track`，关联likelihood只使用frequency／Doppler、visibility／beam、position／time物理项。每项分布、门、权重、并列hash、event-time排序、scan clock、miss、death和expiry转移在fit前注册。按arrival time重排同一组event必须得到相同N-scan封存结果。未来若启用表示项，必须另立候选，限定为同一sealed artifact中的冻结、身份盲、无跨节点训练／对齐／在线更新特征，并重新独立复审。

首版MHT状态机固定为：每个event对每条存活track最多保留门内log-likelihood最高的3个association branch，外加1个birth branch；全局hypothesis上限`H_max=32`；fixed lag取`min(5个event opportunity,120s)`；`N_scan=3`；相对最佳hypothesis log-mass低20以上的分支剪除；所有并列用canonical event／track hash破平。科学death只由经历4个“至少一个独立组可能可见”的observable opportunity仍未关联触发；已证明的non-opportunity不计入4次。最后一次关联后24h只是冷存储／索引TTL：可以把track从在线内存归档，但不得降低existence probability或计为death；归档后重新出现的事件按新birth生成新track，历史归档只供审计，不执行身份re-identification。birth／death／clutter／`P_D`只能由G2 fit资产估计并冻结，formal阶段不得更新。缺少合法资产时，MHT只能运行G0性质测试，不能输出operational anonymous track。

registered事件必须产生0条anonymous birth／association／延寿记录。anonymous track identifier由第一条sealed unknown／defer event的canonical hash和branch rank生成，不含任何预测TX或credential。高clutter、交叉轨迹、临近birth、迟到量测和容量达到`H_max`都必须进入G0压力测试；容量溢出时按冻结log-mass规则剪枝或defer，不能扩大上限。

每次lag内OOS更新都新建不可覆盖revision，至少包含`event_time、arrival_time、processing_watermark、parent_revision_hash、included_event_hashes、sealed_at`。在线下游在时刻`t`只能读取`watermark<=t`的`online_as_of`版本；`lag_final`版本仅供延迟审计，不能回写过去的event decision，也不能用于抬高当时可见的轨迹指标。G2分别报告online-as-of和lag-final的precision／recall／IDF1／fragmentation，主运营门只用online-as-of。任何后续event可达性约束也只能读取其到达前已经封存的revision。

轨迹参数`P_D`、clutter、birth、death按visibility和scenario估计。缺少合法G2 fit资产时，track只能执行G0技术性质，不输出性能或身份增益结论。

## 12.计算位置、资源与通信

默认gateway fusion避免在M≤5的小规模系统里引入分布式共识。节点侧只执行现有本地模型、因子化校准和Tier编码；网关执行事件账本、QP、风险门和MHT。

必须封存：

- 节点侧额外CPU时间、能耗、峰值内存和消息字节；
- gateway p50／p95／p99时延与峰值内存；
- ISL／下行可见窗口、重传预算和hard deadline完成率；
- 每次Tier升级带来的条件风险下降、正确决策增量和字节成本。

卫星边缘计算研究把通信窗口、时延和能耗作为联合约束，而不是只报告模型FLOPs；CIRF也必须用真实链路trace或经验证的trace回放评价，而不能假定全节点即时在线。参考“Computation Offloading in Delay-Sensitive Multisatellite Cooperative Edge Computing Systems,”IEEE Internet of Things Journal 2026，https://doi.org/10.1109/JIOT.2025.3580504 。

## 13.G0性质测试

G0不读性能，至少覆盖：

1.事件账本：偏钟、漂移、传播边界、事件碰撞、乱序、重放、撤销、counter分叉、身份置换不变，以及不含身份字段的opportunity／waveform digest；
2.因子化意见：known／unknown组合simplex、标签置换、极端finite logits、0／1端点不出现、NaN／Inf拒绝、unknown score非概率输入拒绝，以及checkpoint／class-order／calibrator／prior／codec任一hash漂移拒绝；
3.参考先验：同一likelihood但不同训练先验经还原后结论一致；无法还原则拒绝融合；
4.相关核：`K_top`等价关系、R／U残差轴、联合fit block、MAD下限、解析收缩、受约束PSD completion、完全相关component保持、split／merge和复制不增益；
5.QP：枚举全部非空活跃origin子集，验证单origin逐字节降级、双轴`F_A／v^a_A／beta^a_0,A`、cap不足defer、唯一字典序解、独立求和、完全相关单份证据和类别排列不变；
6.区间：fit／interval-calibration／conformal-calibration／formal四层互斥、所有误差源事件级同时外包络、top-L最坏遗漏、prior set全传播；置换conformal标签只改`q`而不改`P_lower／interval hash`；
7.风险：构造conformal coverage通过但unknown FAR失败的反例，确认两门不可互换；
8.transcript：冻结scheduler的全部合法prefix与联合missingness枚举、class-conditional block-max nonconformity、嵌套集合、证据依赖请求、请求前bytes／energy／deadline硬预算、区间bin边界抖动、乱序、缺失、重传、未知网络state和hard deadline；
9.故障：fail-silent、bounded numeric、一个authenticated Byzantine、两个冲突组；
10.track：计划不可见不计miss、24h归档不作death、registered零anonymous birth、非法`z_track`输入拒绝、`H_max／N_scan／fixed-lag／expiry`边界、容量溢出、高clutter、lag内OOS as-of revision、lag外只审计、event零回写；
11.`N_sat=1`原始artifact逐字节一致；
12.所有registered defer按错计，禁止reject-all通过。
13.容量预检：枚举最大合法配置，核对`N_atomic／n_min／transcript／QP／operation／memory`上界；任一超限在启动前拒绝且不得抽样或合并bucket。

## 14.G1代理矩阵与消融

G1仍使用`PROXY_MULTI_RECEIVER`。预测先封存，独立scorer后接truth。所有融合基线消费相同本地artifact、因子化校准、参考先验语义、event ledger和字节预算。

主基线：

```text
raw leader
calibrated leader
quality average
ordinary PoE
EV-CARE
CIRF v2
CIRF v3
```

v3消融：

```text
-factorized unknown interface
-operational prior set
-constrained correlation QP
-decision-level risk ledger
-transcript calibration
-event ledger
```

压力轴必须包括prior shift、unknown-family shift、相关核低估／高估、clock bias、false binding、节点dropout、证据依赖延迟、量化、单个合法密钥伪证据、轨迹交叉和clutter。任何提升若只来自defer增加均判失败。

## 15.G2采集与非补偿门

真实same-event G2在采集前冻结：

- clock／ephemeris／propagation error budget；
- false-binding collision set和目标上界；
- 至少两个独立前端组，Byzantine结论需要至少三个；
- registered和operational unknown来源族；
- registered四split共享同一registry TX但event／physical sample／risk cluster互斥；unknown TX与anonymous entity在fit／interval-calibration／conformal-calibration／formal-test间身份级四方互斥；
- `event_opportunity_block／mission_pass_id`跨split互斥，样本量按独立block而非reception行数计算；
- acquisition-pilot冻结的`Delta_ind`、scorer-only `risk_cluster_id`和独立validation撤证条件；
- 按风险门和cell数量反推的样本量；
- 有限联合network trace族、可见机会、clutter、track truth和online-as-of revision truth；
- immutable truth sidecar与独立scorer。

非补偿门：

1.event binding、truth隔离、artifact不可变、`N_sat=1`恒等零失败；
2.registered overall、min-class、min-receiver、min-scenario和worst-node-subset逐项不低于leader超过2pp，defer计错；
3.正式unknown FAR≤5%、safe rejection≥95%，且按unknown来源族分别报告；
4.`R_known_id`、`R_unknown_FA`、`R_false_binding`、`R_false_nonopportunity`和`R_deadline`的冻结置信上界分别过门；
5.每个正式cell满足采集前推导的样本量，不用均值补偿；
6.校准、prior shift、相关核变体、dropout、clock bias和量化压力逐项过门；
7.平均字节、Tier-2率、p95／p99时延、能耗、峰值内存和deadline完成率分别过门；
8.online-as-of track precision／recall、IDF1、false merge和fragmentation分别过门；lag-final只作配对延迟诊断，不得补偿；
9.A／B／C／D与`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`保持同event、node、support和query口径；
10.任一门失败即`REJECT_CIRF_TRACK_V3`，不得根据formal query调整阈值、先验、K、scheduler或unknown mixture。

## 16.为什么v3更可能提升性能

v3的性能收益不是来自更深网络，而来自减少四类系统误差：

1.因子化校准防止unknown能量扭曲known类间概率；
2.参考先验还原和operational prior set降低先验漂移造成的过度自信；
3.受约束QP在独立组上恢复证据累积，在相关组上抑制重复计权；
4.transcript校准让通信升级只在真正能收窄风险时发生，避免静态全量传输和证据依赖选择偏差。

事件账本与轨迹可见机会主要提高结果可信度和false-binding／false-merge表现；它们不能被写成RFFI分类精度本身的提升。所有“更可能”均为机制推断，只有完整G2矩阵可以形成性能结论。

## 17.最小实现顺序

1.先实现G0的event ledger、因子化意见、先验还原和受约束QP纯函数；
2.补事件级区间、风险账本和样本量计算器；
3.实现冻结scheduler有限状态机及network trace replay；
4.复用现有EV-CARE／CIRF v2 artifact做G1，不重训Phase1；
5.G1通过后才实现MHT的visibility opportunity和OOS update；
6.只有G2采集acceptance spec满足后，才启用正式unknown风险与track性能评价。

第一轮发布不需要大型训练。每张GPU至多两个实验的资源规则只在后续真实矩阵中使用；G0和G1融合器优先CPU执行并复用冻结本地预测。

## 18.文献证据边界

- 标签／先验偏移会破坏普通校准和coverage：[Podkopaev and Ramdas, 2021](https://proceedings.mlr.press/v161/podkopaev21a.html)。
- 多来源并不自动带来有效不确定性，需要处理来源偏差：[Liu et al., 2024](https://proceedings.mlr.press/v235/liu24ag.html)。
- 证据驱动的采集／停止会形成feedback covariate shift：[Prinster et al., 2023](https://proceedings.mlr.press/v202/prinster23a.html)。
- 小节点分布式检测在Byzantine比例超过临界值时可被致盲：[Kailkhura et al., 2013](https://arxiv.org/abs/1307.3544)。
- 乱序多目标量测需要retrodiction而非按到达顺序更新：[Garcia-Fernandez and Yi, 2021](https://doi.org/10.1109/TSP.2021.3100999)。
- 跨接收机RFFI存在显著receiver-induced shift；协同层不能假定节点概率天然可比：[Yang et al., 2024](https://arxiv.org/abs/2404.08566)、[Zhang et al., 2024](https://arxiv.org/abs/2411.03636)。

## 19.当前裁决

相比v2，v3把same-event证明、unknown语义、先验漂移、相关核修复、动态通信选择和轨迹可见性从“实现注意事项”提升为正式方法合同。2026-08-09两条独立只读复核均已关闭阻塞：科学监督结论`P0=0、P1=0、MERGE`，系统／算法作者复核结论`P0=0、P1=0、可冻结G0`。因此本文冻结为G0实现依据；它仍不是G1／G2性能结果，也不授权跳过四层数据隔离、容量预检或真实same-event采集门。
