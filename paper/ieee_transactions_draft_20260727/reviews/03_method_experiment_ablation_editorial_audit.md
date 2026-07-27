# CVS-RFFI方法、实验、消融与IEEE编辑联合审稿

日期：2026-07-28

审查对象：`manuscript.tex`、`README.md`、`claim_evidence_matrix.md`、`项目.md`、`docs/CVS_ADV3B02_QKNNV42_TECHNICAL_REPORT_20260709.md`、`docs/D92_METHOD_COMPLETE_REPORT_20260727.md`、`code/training_controls.py`、`code/sat_channel.py`、`code/scripts/build_cvs_leo_weak_iq_cache.py`

审查方式：只读核对。本文件不改正文，不把待做实验写成已完成结果。

## 一、总体审稿结论

当前稿件已经具备清楚的两阶段生命周期、严格的Phase2访问协议和一项可信的配对组件诊断，但仍属于`MAJOR REVISION / NOT SUBMISSION READY`。阻碍一区Transactions投稿的不是英文格式，而是方法定义与证据闭合之间仍有四个断点：

1. Phase1正文主要描述并报告历史`0.10/0.70/0.20`实现，正式协议却是`0.07/0.63/0.30`。新协议下尚无冻结独立复验，也没有用新Phase1 bundle重跑Phase2，因此当前两阶段结果不是同一正式版本的端到端证据。
2. Phase2的125-row结果严格隔离了“任务均衡协方差”相对D81的作用，却没有隔离整套RTB-IDR相对同权限ProtoNet、qKNN、普通LDA等基线的作用。CSIL和MoPC-HR权限不同，不能填补这一证据缺口。
3. LEO模型是对已被地面传播和接收机处理过的WiSig/ManySig记录再施加残余基带算子。绝对路径损耗、大气衰减和额外IQ不平衡关闭；高度和载频在当前配置下只进入metadata，不改变IQ。该代理可用于压力测试，但不足以直接支撑Starlink链路或真实星上有效性。
4. “增量注册”和“星上轻量”仍缺生命周期证据：当前实验一次性提供全部旧/新类support并重新编译状态，尚未证明后续新类到达时能否只依赖已保存状态更新；16.11 KiB只覆盖最终分类核心数组，不覆盖编码器、特征提取、support保留、注册工作区和11.15–11.74 GMAC等价注册开销。

### 优先级定义

|等级|含义|处理要求|
|---|---|---|
|P0|会造成任务、公式或证据归属错误|进入新实验前修正|
|P1|会阻止主要方法主张或公平比较成立|投稿前必须补齐|
|P2|影响复现性、解释力或IEEE表达质量|正式定稿前补齐|
|P3|增强展示，但不改变证据合法性|版面允许时补充|

## 二、P0：符号、公式与实现闭合

### 2.1同一符号承担多个不相容含义

当前正文至少有以下冲突：

|符号|当前含义一|当前含义二或三|风险|建议|
|---|---|---|---|---|
|\(d_i\)|receiver/domain标签|卫星斜距|读者无法判断下标对象|receiver用\(r_i\)，receiver-day group用\(g_i\)，斜距用\(\rho_i\)|
|\(d_{i,\ell}\)|多径tap delay|\(d_i\)又表示域和斜距|公式可读性差|tap delay统一为\(\delta_{i,\ell}\)|
|\(K\)|每类shot数|Rician \(K\)-factor|Phase2公式与信道公式冲突|shot用\(K_{\mathrm{s}}\)，Rician因子用\(K_{\mathrm{R},i}\)|
|\(c\)|传播场景|分类类别索引|跨章节引用易混淆|场景用\(\chi\)，类别用\(c\)|
|\(g\)|残余AGC增益|hard-domain group索引|同一方法节内冲突|增益用\(a_i\)，group用\(g\)|
|\(\epsilon\)|信道功率保护常数\(10^{-12}\)|特征归一化常数\(10^{-8}\)|复现时可能误用|分别定义\(\epsilon_{\mathrm{ch}}\)和\(\epsilon_{\mathrm{norm}}\)|

应新增一张“Nomenclature and Dimensions”表，并在全文只保留一套符号。最小推荐集合为：transmitter \(y\)、receiver \(r\)、receiver-day group \(g\)、channel profile \(\chi\)、class \(c\)、shot count \(K_{\mathrm{s}}\)、Rician factor \(K_{\mathrm{R}}\)、slant range \(\rho\)、tap delay \(\delta_\ell\)。

### 2.2真实任务模型与仿真生成顺序不同

正文式(1)表达真实接收链：

\[
\mathbf{x}^{\mathrm{orb}}_{y,r,\chi}
=\mathcal R_r\!\left[\mathcal H_\chi\!\left(\mathcal T_y(\mathbf s)\right)\right]+\mathbf n.
\]

实际数据不是在\(\mathcal T_y(\mathbf s)\)上直接施加LEO信道。当前实现先读取已经包含地面传播、地面接收机和原采集预处理的记录\(\mathbf{x}^{\mathrm{terr}}_{y,r}\)，再执行：

\[
\widetilde{\mathbf x}_i
=\mathcal A_{\chi_i}\!\left(\mathbf{x}^{\mathrm{terr}}_{y_i,r_i};\boldsymbol\xi_i\right).
\]

这两个算子顺序不可交换。正文虽写了“simulator is applied to a captured record”，但没有把该构造差异放入正式公式。必须同时给出“目标物理任务模型”和“本研究代理生成模型”，并明确：

- \(\mathcal A_\chi\)是post-capture residual overlay，不是真实\(\mathcal H_\chi\)的数值复现；
- WiSig/ManySig原记录中的地面信道和接收机响应不会被移除；
- 当前结果证明的是在该复合代理扰动上的鲁棒性，不证明真实星地信道可迁移性。

若这一差异不显式建模，Starlink动机越强，审稿人越会追问“为什么Starlink链路被施加在Wi-Fi地面接收记录之后”。

### 2.3论文中的逐样本随机变量与当前缓存生成器不完全一致

`manuscript.tex`将第二tap delay写为\(\delta_{i,1}\)，将随机状态写为每个样本的\(\boldsymbol\xi_i\)。当前实现有两个需要闭合的细节：

1. `sat_channel.py::_apply_weak_multipath`在一次batch内只采样一组`L`和`delays`，然后让batch内所有样本共享该delay pattern；tap复系数仍按样本随机。
2. `build_cvs_leo_weak_iq_cache.py`按dataset role建立一个顺序RNG，并把同一个`role_seed`记录到该role的所有样本。不同样本得到顺序随机抽样，但不能仅凭`physical_sample_id+recorded seed`独立重建该样本；改变batch size或此前样本顺序可能改变输出。

这不推翻已经封存的固定IQ字节，但会影响“逐样本seed可复现”和论文公式的实现一致性。处理方式只能二选一：

- 修正生成器：使用`hash(base_seed,scenario,physical_sample_id)`构造逐样本counter-based seed，并逐样本采样delay；这会改变IQ字节，必须生成新capsule并按`p2_min_v1`重新验证；
- 保留现有capsule：正文准确写成“role-seeded ordered RNG stream”，并说明batch共享delay实现，不再声称每条记录有独立可回放seed。

一区稿更推荐第一种，因为它能使单条样本独立复现，也消除batch边界成为隐含信道因素的问题。

### 2.4Phase1标签率的分母存在实质差异

协议定义：

\[
\rho_{\mathrm{label}}
=\frac{|\mathcal L_s|}{|\mathcal L_s|+|\mathcal U_s|}\leq0.10.
\]

当前正式划分`0.07/0.63/0.30`按source全池计数，因此：

\[
\rho_{\mathrm{label}}=\frac{0.07}{0.07+0.63}=0.10.
\]

历史ADV3B02的实际样本数为`8400/58800/16800`，即`0.10/0.70/0.20`。按同一个定义：

\[
\rho_{\mathrm{label}}=\frac{8400}{8400+58800}=0.125,
\]

而不是0.10。历史报告中“\(\rho_{\mathrm{label}}=0.1\)”与其实际计数不一致。正文目前写“10% of the source pool carries labels”在全池口径上成立，但不能同时把该运行描述成“训练池内至多10%标注”。必须在摘要、引言、Method和Results中统一分母：

- 当前协议：source全池7%有TX标签，训练池内10%有TX标签；
- 历史证据：source全池10%有TX标签，训练池内12.5%有TX标签；
- 历史结果只能是旧协议内部审计。

### 2.5Phase1方法公式仍不足以复现

正文将大量非零项合并为\(\mathcal L_{\mathrm{risk}}\)，但该项同时包含hard-domain、FishR、prototype、角半径、CVaR、core-tail和soft-boundary机制。当前问题包括：

- \(\mathcal L_{\mathrm{risk}}\)没有完整展开，也没有逐项权重和warm-up；
- receiver标签与receiver-day domain标签共用\(d_i\)，没有定义\(g_i=(r_i,t_i)\)；
- `TopK`式中的\(K_g\)同时像“hard group数量”和“group索引”，建议改为\(K_{\mathrm{hard}}\)；
- 伪标签阈值的统计窗口未说明：当前batch、epoch缓存还是全部receiver-day无标签池；
- temporal gate的“adjacent time-window”没有给出窗口构造和边界；
- `a_s`和`a_{\mathrm{LEO}}`没有给出抽样概率、场景分布和seed策略；
- CosFace式被称为“frozen logit”，但它发生在训练期，应称为training logit。

建议把主文保留四个科学模块，把所有非零损失、权重、起始轮次、warm-up、输入集合、是否反传到`z_id/z_dom`放入一张附录表。不得用“实现中已冻结”替代数学定义。

### 2.6Phase2核心公式缺少从support到最终头的闭合映射

当前Method对RTB-IDR的叙述比Phase1简洁，但简化掉了决定可复现性的关键步骤：

1. \(\mathbf G\)没有定义。需要给出84个domain-class中心如何按类去中心、如何等权汇总，以及量化噪声\(\sigma_q^2\)如何估计。
2. \(\mathbf G_+\)的名称暗示PSD，但当前式只做对称化和减噪；减噪后可有负特征值。应写出正谱投影或把它重命名为\(\mathbf G_{\mathrm{corr}}\)。
3. Cauchy权重只写“\(\propto\)”，没有写\(\sum_k\omega_{c,k}=1\)、\(\tau_c=0\)和无正特征值时的回退。
4. “weighted identity center defines a translation”没有公式。至少应写：

\[
\boldsymbol\mu^{\mathrm{rob}}_c
=\sum_k\bar\omega_{c,k}\mathbf z^{\mathrm{id}}_{c,k},\quad
\Delta_c=\boldsymbol\mu^{\mathrm{rob}}_c-\bar{\mathbf z}^{\mathrm{id}}_c,
\]

\[
\widetilde{\mathbf z}_{c,k}
=\mathbf z_{c,k}
+[\Delta_c;\mathbf 0_{96};\mathbf 0_{32}].
\]

5. 当前正文的`ShrinkCov({all old supports})`容易被理解为对old pool直接估计一次Ledoit–Wolf协方差。实现报告实际描述的是“逐类标准化Ledoit–Wolf协方差，再在任务内按类等权平均”。应明确：

\[
\widehat{\boldsymbol\Sigma}_{t}
=\frac{1}{|\mathcal Y_t|}
\sum_{c\in\mathcal Y_t}
\widehat{\boldsymbol\Sigma}^{\mathrm{LW}}_c,\qquad
t\in\{\mathrm{o},\mathrm{n}\}.
\]

6. 式(25)中的\(\widehat{\boldsymbol\Sigma}\)没有说明是full、block、balanced还是融合后的对象。类级full/block融合和Fisher安全门后，最终\(\mathbf w_c^\star,b_c^\star\)通常不再能由一个公共\(\widehat{\boldsymbol\Sigma}^{-1}\boldsymbol\mu_c\)完整表示。应区分“基础LDA分支公式”和“最终编译头公式”。
7. Fisher残差、support-cross-fitted可靠性和原子安全门都是贡献描述，却只有文字，没有公式或Algorithm。至少需要一段可复现伪代码，列出fit集合、held-support集合、接受条件和最终状态。
8. 两层残差INT8只给存储公式，没有量化式。应给出\(\mathbf W\approx s_1\mathbf Q_1+s_2\mathbf Q_2\)、码值范围、scale粒度，以及FP32与量化头之间的logit误差和argmax flip率。

## 三、P0/P1：两阶段输入输出和训练—推理边界

### 3.1当前两阶段结果不是同一个正式bundle的闭环

正式论文要回答的不仅是“Phase1和Phase2各有结果”，还要回答“Phase2使用的是否正是正文正式Phase1产生的bundle”。当前答案是否定的：

- Phase1结果来自历史ADV3B02和历史split；
- D92也依赖该历史checkpoint及84个历史domain-class聚合中心；
- 当前`0.07/0.63/0.30`下的新Phase1 bundle尚未产生；
- 一旦Phase1 checkpoint或84个中心改变，D92的身份特征、扰动谱和最终结果均可能改变。

因此，不能只补一张新Phase1表后沿用旧D92表。正式闭环应按顺序完成：

1. 冻结当前Phase1 split、训练seed和checkpoint selection；
2. 产生新bundle ID及其聚合中心；
3. 在任何target结果不可见的前提下冻结Phase2方法锁；
4. 用新bundle重跑全部Phase2确认矩阵；
5. 在论文主表同时记录`bundle_id/capsule_id/split_id/method_lock_id`。

### 3.2Stage2-B、Stage2-C和后续增量session尚未形成完整状态机

当前正文把“注册前旧类状态”和“一次性加入5/10/20个新类”定义清楚，但没有说明下一批新类到达时：

- 是否仍保存所有旧类target support；
- 是否只保存16.11 KiB仿射核心；
- 是否能从旧状态\(\Theta_t\)和新类support得到\(\Theta_{t+1}\)；
- 是否必须重新读取先前全部old/new support并执行约88次fit；
- 新类成为历史类后，其support、均值、协方差或量化行保存在哪里。

如果下一次注册必须重读全部历史target support，方法仍可合法，但“compiled state”不是完整持久状态，资源表必须把support archive算入。如果不保存support，就要证明仅凭\(\Theta_t\)可更新。否则当前工作更准确的名称是“few-shot joint enrollment”而不是完整的streaming class-incremental learning。

### 3.3Stage2-A参考、Stage2-B适配和Stage2-C注册的指标来源要隔离

正文当前只定义了\(A_{\mathrm{o}}^{\mathrm{pre}}\)、\(A_{\mathrm{o}}^{\mathrm{post}}\)和\(A_{\mathrm{n}}\)，这是合理的。后续补实验时必须避免：

- 把不同seed的Stage2-A参考与D92 row作paired差；
- 把Stage2-B旧类方法的`old_acc`与Stage2-C的\(H_{\mathrm{old,new}}\)排名；
- 用Role-Oracle替代合法全类预测；
- 用不同新类数或不同class set的最高值拼一行。

Role-Oracle若保留在Discussion，必须写出独立run ID、seed范围和“NON-PROMOTABLE”标签。当前主表\(H=69.555\%\)与Oracle fresh run所引用的non-oracle \(H\approx69.7\%\)来自不同artifact，不应被写成一组严格paired主结果。

## 四、LEO/Starlink场景与参数审计

### 4.1当前参数中哪些真正改变IQ

|参数或机制|正文设置|当前是否改变IQ|审稿判断|
|---|---:|---|---|
|记录长度\(N\)|256 samples|是|对应10.24 \(\mu\)s；不能代表完整过境动态|
|采样率\(f_s\)|25 MS/s|是|用于CFO相位和数字特征|
|载频\(f_c\)|2.462 GHz|否，当前仅用于FSPL metadata|不能称为Starlink载频模型|
|高度\(h\)|500–2000 km|否，当前仅影响斜距/FSPL metadata|高度变化不产生性能压力|
|仰角\(\theta\)|按场景均匀采样|是|通过Rician \(K\)-factor改变fading|
|绝对FSPL|计算但不施加|否|不能声称完成link budget|
|bulk orbital Doppler|预补偿|否|只保留50/70/90 Hz残余CFO|
|残余CFO|Gaussian|是|在10.24 \(\mu\)s记录内总相位变化很小，需敏感性验证|
|Wiener phase noise|按场景范围采样|是|需要报告实际累计相位分布|
|SNR|14–32 dB范围|是|当前最直接的场景强度来源|
|Rician/shadowed Rician|按场景|是|low-elevation只shadow LOS分量|
|两tap multipath|1–3 samples|是|当前delay pattern按batch共享|
|显式大气衰减|关闭|否|rainy-link不是雨衰物理模型|
|额外IQ imbalance|关闭|否|不能把该项列为仿真压力|
|residual AGC|±0.2/0.3 dB|是|发生在加噪前|

正文已有部分限定，但建议在channel table新增一列“Applied to IQ / Metadata only / Disabled”。这比仅在段落末尾解释更能防止误读。

### 4.2Starlink只能作为系统动机，不能作为当前数据或信道等价物

在Introduction加入Starlink是合理的，但必须把三种方向分开：

1. 地面站接收卫星下行并识别卫星发射机：PAST-AI/SatIQ一类工作属于这个方向。
2. 卫星在轨接收上行并识别地面终端或其他发射机：CVS-RFFI的主场景属于这个方向。
3. 星座规模、动态拓扑和自治运维带来的身份管理需求：Starlink可作为这一系统背景的实例。

不得写成“本文在Starlink上验证”或“SatIQ已经验证了本方法的上行星载场景”。当前数据载频为Wi-Fi 2.462 GHz，波形来自WiSig/ManySig，方向和载频均不等同于Starlink业务链路。Introduction建议使用以下证据边界：

- RFFI是密码身份、协议标识和链路监测的补充信号，可支持终端真实性复核、异常发射源归因、快速注册和干扰调查；
- RFFI本身不是不可伪造的认证机制，不能在缺少抗重放、抗仿冒和密钥协议时写成独立安全保证；
- 星上本地判决可能减少原始IQ回传和响应时延，但当前论文尚未实测这些收益；
- Starlink相关星座数量、轨道和业务事实必须引用截至投稿日期的官方或权威来源，不能用新闻数字或无日期网页；
- 若要宣称Starlink-specific relevance，需新增与其实际频段、轨道范围、Doppler和帧结构一致的独立profile；否则保持“LEO constellation motivation”。

### 4.3场景描述仍缺少物理充分性验证

当前三种profile主要由人为参数范围区分，尚未展示参数取值为何对应clear-sky、low-elevation和rainy-link。至少需要：

- 参数来源表：每个范围对应的标准、测量论文或工程假设；
- 仿真输出分布：SNR、CFO、累计phase drift、\(K_{\mathrm R}\)、tap delay、EVM和PSD；
- “active-on-IQ”消融：关闭一个机制后输出变化；
- 与至少一组真实卫星IQ或硬件信道模拟器输出的低阶统计对比；
- 对10.24 \(\mu\)s片段适用范围的说明，以及跨packet Doppler rate、遮挡状态和AGC动态为何未建模。

若不能补真实或硬件证据，应把“satellite-to-ground channel model”统一收敛为“physics-inspired residual LEO stress operator”。

## 五、现有证据能支持什么、不能支持什么

|主张面|现有证据|当前强度|缺口|
|---|---|---|---|
|Phase1历史checkpoint可运行|32个历史候选、完整训练trace和source-only指标|内部审计|受候选选择偏差影响；无当前split独立复验|
|Phase1双表征、SSL、尾风险、反事实课程有效|只有完整历史候选与ADV2 aggregate比较|不足|没有参数量匹配和模块消融|
|当前10%训练池标签协议有效|仅协议定义|无性能证据|`0.07/0.63/0.30`尚未运行|
|任务均衡协方差降低大规模注册旧类遗忘|D81/D92同row paired 125矩阵|较强组件证据|需区间、class-set重复和新bundle确认|
|完整RTB-IDR优于普通支持型方法|无同权限同capsule主表|无|必须补ProtoNet/qKNN/LDA等|
|RTB-IDR解决Stage2-C|绝对floor低、新类有退化、K1无增益|负证据|需要新的promotable候选或收窄主张|
|星上状态紧凑|26类核心数组16.11 KiB、7488 MAC/query|只证明头部|不含encoder、FFT/RF、support和workspace|
|星上可部署|解析式资源上界|无硬件证据|缺WCET、RAM、energy、thermal、numerical parity|
|Starlink相关|当前文献和LEO系统背景可建立动机|动机级|没有Starlink数据、波形、频段或onboard实验|
|真实卫星迁移|无|无|真实卫星或硬件在环|
|严格class-incremental|一次性联合注册|不足|缺连续session、到达顺序和持久状态实验|

## 六、Phase1必须补的实验与消融

### 6.1正式确认实验

冻结`0.07/0.63/0.30`后至少运行5个独立训练seed。checkpoint selection只能读取source validation。所有target receiver、LEO target support和query在选择完成前保持不可见。

主表每个seed必须报告：

- overall source validation accuracy；
- strict unseen-domain/unseen-split accuracy；
- receiver floor、worst receiver、min-class accuracy；
- satellite-stress mean/floor；
- `z_id→receiver-day`线性probe准确率；
- `z_dom→receiver-day`准确率和`z_dom→TX`泄漏probe；
- pseudo-label precision、coverage、每receiver-day coverage和最差domain precision；
- Q90/Q95类内角半径、tail overflow；
- checkpoint epoch、bundle ID和参数量。

报告均值、标准差和95%区间，但不把同一训练run内的样本当成独立训练重复。

### 6.2第一层模块消融

|Arm|改动|主要因果问题|必须成对观察的指标|
|---|---|---|---|
|P1-FULL|完整Phase1|参考|全部指标|
|P1-A0|参数量匹配单embedding|双表征是否超越容量增加|UDU、floor、两个leakage probe、参数量|
|P1-B0|关闭\(\mathcal L_u,\mathcal L_{\mathrm{ent}}\)|无标签闭环是否有效|UDU、pseudo precision/coverage、floor|
|P1-C0|关闭全部角尾风险组|尾风险是否改善最差类而非只改善均值|min-class、floor、Q90/Q95、overall|
|P1-D0|关闭MixStyle、source episode和LEO CE|身份保持外推课程是否有效|UDU、receiver floor、satellite-stress floor|

只有第一层整体消融显示稳定作用，才进入子组件消融。不得从32个历史候选中挑最有利差值替代冻结消融。

### 6.3第二层机制消融

|模块|最小子消融|必要对照|
|---|---|---|
|双表征|no-PA、no-frequency、no-DAC/RCN、no-GRL/orth/cons|参数量和训练轮数匹配|
|SSL|global threshold with equal coverage、no-domain gate、no-temporal gate、no-strong agreement、student teacher|必须同时报告precision与coverage|
|尾风险|mean geometry only、+tail、no-softmix、core quantile、CVaR fraction|不得只报overall|
|外推课程|MixStyle only、episode only、LEO CE only、receiver×LEO 2×2、no annealing|clean与stress指标同row|

### 6.4标签率敏感性

协议允许：

\[
\rho_{\mathrm{label}}\in\{0.005,0.01,0.02,0.05,0.10\}.
\]

固定validation为source全池30%时，训练池划分应写为：

\[
f_L=0.70\rho_{\mathrm{label}},\quad
f_U=0.70(1-\rho_{\mathrm{label}}),\quad
f_V=0.30.
\]

每个标签率至少3个训练seed，报告性能—标签成本曲线和最差receiver曲线。这样才能证明“label-limited”不是只在单个10%点上的命名。

### 6.5Phase1公平基线

至少加入：

- 相同CV-SincNet backbone的supervised-only CosFace；
- 单embedding DANN/GRL；
- MixStyle-only；
- Mean Teacher或FixMatch式全局阈值SSL；
- 已下载跨接收机方法中能够在source-only权限下运行的同split版本。

`ADV2 avg.`不能作为唯一主基线。必须说明它平均了哪些run、是否同seed、是否同split、是否同参数量；若无法严格配对，应移到历史开发附录。

## 七、Phase2必须补的基线、消融和敏感性

### 7.1同权限主基线

所有方法使用同一个新Phase1 bundle、同一capsule、相同physical IDs、support/query、class set、seed和全类逐样本argmax：

|基线|作用|
|---|---|
|Cosine nearest centroid / ProtoNet|最小支持型注册基线|
|Euclidean ProtoNet|检验归一化余弦先验|
|Single qKNN|检验保留局部support是否优于参数化头|
|Diagonal LDA|检验仅使用逐维尺度|
|Ledoit–Wolf pooled LDA|检验task balancing以外的收缩收益|
|Full/block shrinkage LDA without robust center|RTB-IDR几何主干对照|
|冻结轻量adapter+统一head|检验少量支持训练是否必要|

CSIL和MoPC-HR保留在“不同权限外部比较”表，不能和上述同权限主表混成单一排名。

### 7.2RTB-IDR组件消融

|Arm|关闭或替换内容|隔离对象|
|---|---|---|
|P2-A0|identity160 only|FFT96/RF32联合特征整体贡献|
|P2-A1|identity+FFT、identity+RF、identity+FFT+RF|两个辅助块各自贡献|
|P2-A2|\(\beta_{\mathrm{aux}}\in\{0,1,2,4,8\}\)|固定权重4的敏感性；只在development matrix选择一次|
|P2-B0|关闭ground perturbation basis，普通均值中心|INT8地面聚合与稳健中心整体贡献|
|P2-B1|Cauchy改普通均值/Huber|稳健权重形式|
|P2-B2|不减quantization noise floor|量化噪声校正|
|P2-C0|D81，即关闭0.5/0.5 task balancing|已有严格paired组件效应|
|P2-C1|\(\lambda_{\mathrm{old}}\in\{0.25,0.5,0.75\}\)|敏感性；选择后必须独立确认|
|P2-D0|full only、block only、support-cross-fitted fusion|双几何融合|
|P2-E0|关闭Fisher residual|残差贡献|
|P2-E1|关闭per-class gate或atomic gate|安全门作用；只作诊断，不直接部署|
|P2-F0|FP32、single-residual INT8、dual-residual INT8|量化精度—存储权衡|

每个消融必须同时报告\(A_{\mathrm{o}}^{\mathrm{pre}}\)、\(A_{\mathrm{o}}^{\mathrm{post}}\)、\(A_{\mathrm n}\)、\(H\)、forgetting、min-old、min-new和fallback/accept counts，不能只报组件最有利的一侧。

### 7.3完整因素矩阵

当前五个slice不是完整factorial。最低可接受确认矩阵：

- \(K_{\mathrm s}\in\{1,2,5,10\}\)；
- \(C_{\mathrm n}\in\{5,10,20\}\)；
- 5个target receivers；
- 至少5个support/query seeds，优选10个；
- 3个LEO profiles；
- 至少3个独立new-class set draws，而不是只使用一组nested class identities。

K2是必要点，因为正文把\(K\leq2\)定义为fallback，却只实测K1。new-class set draw必须和support seed分开记录，否则无法区分类别难度与support抽样波动。

资源允许时，主方法与最强两个同权限基线执行完整矩阵；其余消融可先在预登记代表slice筛选，再对保留arm作独立完整确认。筛选和确认seed不得重合。

### 7.4连续注册和顺序敏感性

建议构造三次注册session，例如每次加入5类，至少比较：

- 一次性加入15类；
- 5+5+5顺序注册；
- 三种new-class到达顺序；
- 保存全部target support；
- 只保存声明的持久状态。

每个session报告旧类、新近类、历史增量类的accuracy、min-class、forgetting、状态大小、更新时间和是否读取历史support。若“只保存核心头”无法继续注册，正文必须把该限制写入方法和资源结论。

### 7.5类别规模和query先验

当前6个旧类、最多20个新类不足以支撑大星座规模结论。至少增加：

- 更多old-class数量；
- \(C_{\mathrm n}>20\)的容量曲线；
- balanced与自然不平衡query分布；
- macro accuracy、balanced accuracy和per-class floor。

方法仍使用equal prior，不得读取真实query比例；不平衡query只用于测试该设计的稳健性。

## 八、跨接收机划分与数据说明

正文列出类似`1-1`、`20-1`的receiver标识，但没有解释编码规则。投稿前的数据表至少包含：

- receiver ID、capture day、hardware family；
- old/new TX ID或匿名稳定编号；
- 每个TX×receiver×day的physical record数；
- labeled/unlabeled/validation/support/query计数；
- support pool与query pool的构造顺序；
- 三个scenario的physical-ID互斥计数；
- target receiver为何不出现在任何Phase1 normalization、threshold或selection中；
- new class set是否nested、如何抽样；
- 每个artifact的`bundle_id/capsule_id/split_id`和SHA-256。

还需要明确`WiSig/ManySig`的关系。正文引用的是WiSig的公开规模，却把实际文件称为`ManySig.pkl`。必须说明ManySig是WiSig子集、内部重打包、扩展数据还是另一数据集，并给出公开release ID和可复现映射。

## 九、统计检验与报告单位

### 9.1独立性单位

- 样本级packet不能充当方法重复；
- support seed只反映support/query抽样；
- receiver是跨接收机泛化的主要外推单位，但当前只有5个；
- scenario physical IDs互斥，因此场景间不是同物理样本paired比较；
- D81/D92在同receiver、seed、slice、scenario内可作paired比较。

### 9.2推荐统计输出

1. 对每个预登记slice先计算row-level paired delta，再报告receiver×seed层级分布。
2. 给出receiver-clustered或two-way receiver/seed bootstrap区间；由于receiver只有5个，同时展示5个receiver各自delta，不能只报p值。
3. 对主要结论预先指定一个primary endpoint，例如\(H_{\mathrm{old,new}}\)，old/new/floor作为共同安全指标。
4. 多slice、多arm检验使用Holm或FDR修正；探索性消融明确标为exploratory。
5. 报告效应量和区间，不使用“statistically significant”替代实际幅度。
6. Phase1使用独立训练seed区间，并公开32-candidate selection过程；不能把候选间波动当作重复区间。

### 9.3指标定义还需补齐

- accuracy是micro、macro还是每类等权；
-三场景汇总是样本数加权还是场景等权；
- min-old是在每个row先取min再平均，还是聚合后取min；
- \(H\)是先逐row计算再平均，还是用聚合old/new计算；
- confidence interval在哪个层级计算；
- fallback rate、Fisher gate accept rate、full/block选择率；
- quantization logit误差、argmax flip rate和数值闭合率；
- ECE/NLL或Brier score，用于评价old/new跨角色校准。

## 十、星上部署与资源实验

### 10.1资源账本必须分层

|阶段|必须测量|
|---|---|
|Phase1地面训练|训练GPU时长、参数量、checkpoint大小；仅用于完整性|
|星上特征提取|encoder latency、FFT96/RF32 latency、峰值RAM、模型大小|
|注册状态构造|wall-clock、WCET/p95/p99、峰值RSS、workspace、能耗、88次fit实际次数|
|query推理|端到端latency、吞吐、energy/query，而非只报7488 MAC|
|持久状态|bundle、84个INT8 aggregate、affine head、registry、scales、可能保留的support|
|数值可靠性|FP64研究实现与目标精度之间的预测一致率、异常/fallback计数|

### 10.2目标平台

应先冻结一个真实目标处理平台，再测量。桌面CPU、N607 GPU或Jetson式开发板可作工程代理，但不能自动等同于radiation-tolerant flight processor。若没有飞行级平台，结论写为“embedded/onboard-oriented prototype”，并保留辐射、热和容错限制。

### 10.3通信收益不能只靠推断

若Introduction声称星上RFFI减少回传，应报告：

- raw IQ回传字节；
- 本地prediction/metadata回传字节；
- bundle上传与注册support存储字节；
- 在一个明确任务窗口内的带宽节省和时延；
- 错误识别造成的安全代价不在本稿验证范围内。

## 十一、卫星与硬件验证路线

按证据强度由低到高：

1. 仿真输出审计：当前参数分布、EVM/PSD/CFO/phase drift和机制消融。
2. SDR硬件在环：固定TX经信道模拟器或数字回放进入至少两种异构接收机，验证receiver shift和数值闭合。
3. 真实卫星信号辅助验证：用于检查仿真低阶统计是否落在真实范围，必须说明它与“星上接收地面TX”方向不同。
4. 真实上行或等效payload receiver：才可支持接近目标场景的外部有效性。

如果只能完成第1–2层，标题和结论仍可保留“spaceborne-oriented”或“spaceborne deployment proxy”，不能写“validated for Starlink”。

## 十二、必须新增的图和表

### 12.1主文图

|图|内容|回答的问题|
|---|---|---|
|Fig. 1应用场景图|地面训练、星上receiver、old/new TX support、独立query、可选Starlink式LEO系统背景|为什么需要星上RFFI，本文识别方向是什么|
|Fig. 2物理任务与代理生成|真实\(\mathcal T\to\mathcal H_{\mathrm{LEO}}\to\mathcal R_t\)和当前\(\mathbf x^{terr}\to\mathcal A_{\chi}\)并列|仿真没有冒充真实链路|
|Fig. 3Phase1架构|Sinc/HF、identity多视图、domain nuisance、四类训练模块、bundle输出|Phase1不再只是一个总loss|
|Fig. 4Phase2算法|三块特征、稳健中心、task covariance、full/block、Fisher gate、量化头|support如何变成最终状态|
|Fig. 5性能曲面|按\(K\)和\(C_n\)画old/new/H/floor|方法在哪些条件有效或失效|
|Fig. 6receiver×scenario热图|主方法相对matched baseline的paired delta|改善是否由少数receiver或场景驱动|
|Fig. 7资源分解|encoder、FFT/RF、registration、head、storage|“轻量”发生在哪一段|

不要用仅展示好看聚类的t-SNE代替定量leakage probe或receiver-wise结果。

### 12.2主文表

- 数据集、receiver/day/TX和physical record split表；
- channel参数及“Applied/Metadata/Disabled”表；
- Phase1同split公平基线表；
- Phase1四模块消融表；
- Phase2同权限基线表；
- RTB-IDR组件消融表；
- 完整\(K\times C_n\)确认表；
- 不同权限外部方法表；
- 端到端资源表；
- evidence/claim boundary表。

## 十三、IEEE英文与结构编辑意见

### 13.1必须改

1. 摘要中的89.18%、84.89%和75.55%来自历史split。要么明确写“historical 10/70/20 audit”，要么等当前split复验后替换；不能让读者默认这些数字来自正式`0.07/0.63/0.30`。
2. “complete 125-row Phase2 diagnostic”应改为“complete prespecified five-slice diagnostic”。当前不是完整\(K\times C_n\)factorial。
3. Results中的`ADV2 avg.`需要定义样本单位、run数和可比性。若不是paired baseline，列名和解释都要降级。
4. “soft-unknown baseline's 69.55%”“source-overflow”“bridge-accept”没有定义，且会把闭集Phase1结果带向open-set支线。没有完整定义和表格时应移入附录或删除。
5. “All adaptations are constructed from support only”只对主方法成立，与随后允许source/base访问的CSIL/MoPC-HR冲突。改为“Within the proposed CVS pipeline...”并单列外部方法权限。
6. 主文所有`[AUTHOR ACTION: ...]`必须在投稿前清零；当前至少涉及data release、同权限baseline、statistics、code availability和acknowledgment。
7. Introduction的Starlink段落必须说明识别方向：SatIQ/PAST-AI多为地面识别卫星downlink，本文是星上receiver识别ground/up-link transmitters的代理研究。
8. “The appeal is clear”等修辞可收紧为直接的系统功能陈述；避免把RFFI写成不可仿冒的单一认证根。

### 13.2结构建议

推荐主文顺序：

1. Introduction：Starlink/LEO系统背景→星上RFFI作用→四重空白→研究问题→贡献。
2. Related Work：satellite fingerprinting方向差异→cross-receiver DG→label-limited SSL→few-shot incremental→本稿交叉点。
3. System Model and Protocol：真实任务模型、代理模型、访问边界、指标。
4. CVS-RFFI Method：Phase1和Phase2等权；各自以“输入—机制—输出—资源”为统一模板。
5. Experimental Setup：data manifest、channel、baselines、statistics、hardware。
6. Results：Phase1主表与消融→Phase2主表与消融→channel/HIL→resource。
7. Discussion：Starlink适用边界、失败模式、威胁。

正文已经做到Phase1/Phase2方法篇幅基本对等，但证据仍不对等：Phase1只有历史内部表，Phase2只有负诊断和不同权限比较。最终篇幅应由新实验填充，而不是继续增加方法形容词。

## 十四、投稿前最小闭合路线

### Gate A：先修定义，不启动性能实验

- 统一全部符号；
- 写出真实任务模型与代理生成模型；
- 决定是否修复逐样本seed和batch共享delay；
- 明确\(\rho_{\mathrm{label}}\)分母；
- 补全RTB-IDR最终状态公式与Algorithm；
- 明确连续注册时support保留策略。

### Gate B：冻结Phase1

- 当前split下完成P1-FULL和A0/B0/C0/D0，至少5个训练seed；
- 产生唯一bundle ID；
- 证明target receiver未进入选择；
- 完成label-rate敏感性。

### Gate C：冻结Phase2

- 使用新bundle；
- 先完成同权限ProtoNet/qKNN/LDA基线；
- 完成RTB-IDR核心消融；
- 独立class-set draws和完整\(K\times C_n\)矩阵；
- 报告paired区间、receiver-wise和scenario-wise结果。

### Gate D：补星上证据

- 至少完成端到端嵌入式硬件测量；
- 最好完成SDR硬件在环；
- 若无真实卫星证据，全文保留proxy限定。

只有Gate B和Gate C同时闭合，才能把当前“协议+组件诊断”提升为“完整两阶段方法性能证据”；只有Gate D闭合，才能把“onboard-oriented”提升为较强的星上部署主张。

## 十五、可直接转为实验登记表的字段

每个正式row至少保存：

```text
phase1_bundle_id
protocol_schema
capsule_id
split_id
method_lock_id
receiver_id
receiver_hardware_family
class_set_draw_id
support_seed
channel_profile
K_shot
C_old
C_new
support_physical_id_root
query_physical_id_root
old_acc_pre
old_acc_post
new_acc
H_old_new
forgetting
min_old_class_acc
min_new_class_acc
macro_balanced_acc
fallback_counts
full_block_selection_counts
fisher_gate_accept_counts
quantization_flip_rate
registration_latency_ms
query_latency_ms
peak_ram_bytes
persistent_state_bytes
prediction_artifact_sha256
score_artifact_sha256
```

这些字段能直接支撑same-row结果、统计区间、资源表和truth-isolated复核。

## 十六、最终审稿判断

当前最强、最可信的论文贡献是：严格定义了一个跨接收机、标签受限、固定LEO代理观测、support-only旧类适应与新类注册的两阶段协议，并用完整运行证明任务均衡协方差在特定大规模注册slice上减少旧类遗忘，同时公开其新类代价和K1失效。

当前尚未被证据支持的主张是：Phase1各机制分别有效、当前正式split有效、完整RTB-IDR优于同权限基线、持续class-incremental注册成立、Starlink链路有效、以及端到端星上轻量部署成立。

建议后续修改优先级为：

1. 修正公式和版本归属；
2. 完成当前split的Phase1独立复验与模块消融；
3. 用新bundle完成Phase2同权限基线、完整矩阵和连续注册；
4. 补统计区间、class-set重复和receiver级证据；
5. 补硬件在环与端到端资源；
6. 最后再强化Starlink叙事。

这样可以避免“引言系统意义很强，但实验仍停留在历史split、代理信道和单组件诊断”的审稿落差。
