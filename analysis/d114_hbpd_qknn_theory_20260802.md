# D114-HBPD-qKNN轻型异方差预测域适应理论草案

状态：`DESIGN_FROZEN_R2 / FEASIBILITY_REVIEW_MERGE / IMPLEMENTATION_NOT_STARTED / NO_NEW_PERFORMANCE_RESULT`

日期：2026-08-02

## 0.结论

D114不再移动feature或拟合全矩阵。D113已经证明，共同加性平移可以让588/588个feature、score和margin发生变化，却在K1、K10均没有跨过任何argmax边界；继续放大平移或改收缩系数只是调强度。D114改为直接建模“一个类的K-shot support对单条query提供多大预测不确定度”，候选名为`HBPD-qKNN`（Heteroscedastic Bayesian Predictive Density qKNN）。

核心机制是：Phase1为旧类封存按类条件残差方差，同时封存一个对未见类使用的全局方差；Phase2用同一个公式将K-shot类内离散度与先验合并，得到每类predictive bandwidth。该带宽替换原经验带宽并直接改变类间密度边界；它不是old logit bias，也不依赖query选择。已有D112 ground单位质量head只作为独立HEAD因素，不能把其正收益写成HBPD的DA收益。

## 1.候选比较

|路线|K1信息|边界作用|历史冲突|裁决|
|---|---|---|---|---|
|共同平移放大／改收缩|六旧类ground-support|已证实score变而K1/K10不翻转|D113直接否定当前机制强度|拒绝，不调参|
|sealed低秩非正交warp|六旧类ground-support|可改变邻序|D93/D94的低coverage ground→target transport在K1/K10均负；提高rank或强度会重复旧路线|拒绝首发|
|pairwise margin／混淆转移|support自margin|直接改logit|D108证明support margin不能代理独立query错误，且new数量会改变pair graph|拒绝|
|局部Fisher／协方差metric|K1只能用Phase1 prior|改变距离方向|D110真实G1中old/new/H/floor均下降；低方差轴放大有负迁移|拒绝|
|HBPD异方差预测密度|旧类按类sealed先验；新类全局sealed先验|直接改变每类核宽、峰值与尾部|不移动ground、不估计混淆、不使用query|主线草案|

## 2.生成模型

在固定单位表征空间中，对注册类`c`写成：

\[
z_{ci}=\mu_c+\epsilon_{ci},
\qquad
E[\epsilon_{ci}]=0,
\qquad
E\lVert\epsilon_{ci}\rVert^2/p=\sigma_c^2.
\]

这里`σ_c²`不是任意类别难度标签，而是同一feature extractor、同一接收后计算链下的每坐标条件残差。receiver、LEO弱信道和类×域交互可使不同TX的有效残差不同；忽略这种异方差会让同一个qKNN带宽把高噪类变成过尖密度，或把低噪类过度摊平。

HBPD不声称Gaussian是真实feature分布。它只用Gaussian共轭矩估计确定predictive variance，再继续采用已有Student-t核承担重尾密度，因此称为Bayesian predictive proxy，而不是精确后验。

## 3.Phase1封存资产

对旧类`c`的source receiver×day cell中心`μ_dc`和cell内多物理样本单位feature`z_dci`，定义：

\[
\sigma_{0,c}^2=\epsilon_0+
\operatorname{Mean}_{d,i\mid d,c}
\frac{\lVert z_{dci}-\mu_{dc}\rVert^2}{p}.
\]

未见类没有按类Phase1资产，使用对旧类等权、与标签置换无关的全局先验：

\[
\sigma_*^2=\frac1{|Y_{old}|}\sum_{c\in Y_{old}}\sigma_{0,c}^2.
\]

Phase2对任一已注册类定义：

\[
\sigma_{prior,c}^2=
\begin{cases}
\sigma_{0,c}^2,&c\text{在Phase1旧类registry中},\\
\sigma_*^2,&\text{否则}.
\end{cases}
\]

这是模型知识可用性分支，不读取query的old/new真实角色。任意旧类标签置换会同步置换`σ0`；所有无旧类资产的注册类使用完全相同的`σ*`。资产以int8正值code和量化尺度与checkpoint、feature schema和三个K-specific qKNN lock共同封存，不保留source行、样本feature、ID、路径或成员清单。

## 4.Phase2 support-only后验预测方差

令`z_ck`为当前row类`c`的K个合法support，`~z_c=N(Σ_k z_ck)`。固定每类样本方差proxy：

\[
\widehat\sigma_{c,K}^2=
\begin{cases}
0,&K=1,\\
\dfrac{1}{(K-1)p}\sum_{k=1}^{K}
\lVert z_{ck}-\widetilde z_c\rVert^2,&K>1.
\end{cases}
\]

把Phase1先验视为一个固定pseudo-degree，与`K-1`个target自由度合并：

\[
\bar\sigma_{c,K}^2=
\frac{\sigma_{prior,c}^2+(K-1)\widehat\sigma_{c,K}^2}{K}.
\]

HBPD仍对每个原始support核做qKNN混合，因此需要的是“单条query与一条独立support之差”的方差，而不是query相对K-shot中心的方差。两者在同类条件下独立，故每坐标pairwise预测方差为：

\[
v_{pair,c}=2\bar\sigma_{c,K}^2.
\]

K1没有target类内自由度，但旧类有按类sealed`σ0,c²`，新类有共同`σ*²`，因此不是结构性identity；同时不把单support距离伪装成方差。K增加时，target类内离散度以明确自由度进入，不用query调节强度。首轮监督否决了把`σ²(1+1/K)`类中心不确定度塞入逐support核的原式；R2已按核对象改为pairwise方差。

## 5.HBPD-qKNN分数

原qKNN对类`c`使用经验带宽`h_c`、Student-t自由度`ν`、有效维数`d_eff`和单位质量：

\[
L_c^{0}(q)=\operatorname{LSE}_{k\le K}\ell(q,z_{ck};h_c)-\log K.
\]

原`h_c²`在K>1本身已经由support pairwise chord distance收缩得到；再次加上预测方差会重复计量。HBPD因此不把两个尺度相加，而是用共轭矩估计得到的pairwise预测尺度替换经验带宽：

\[
\bar h_c^2=p\,v_{pair,c}=2p\bar\sigma_{c,K}^2.
\]

量纲是固定的：核分子`2(1-q^Tz)=||q-z||²`是160维总chord squared；`σ²`按每坐标chord-MSE定义，所以`E||ε_q-ε_s||²=2pσ²`，乘`p=160`不是可调温度。`d_eff=12`只控制Student-t尾指数和体积近似，不改变距离的物理单位。实现不得再乘经验系数、加原`h_c²`或按G0结果裁剪范围。

HBPD分数为：

\[
L_c^{H}(q)=\operatorname{LSE}_{k\le K}
\left[
-d_{eff}\log\bar h_c
-\frac{\nu+d_{eff}}2
\log\left(1+\frac{2(1-q^Tz_{ck})}{\nu\bar h_c^2}\right)
\right]-\log K.
\]

`kernel_volume_gamma=1`保持类密度积分质量的同一近似归一；较大方差同时降低峰值、扩展尾部，不能被解释成对困难类单向加分。Gaussian矩只确定尺度，最终Student-t核仍是重尾proxy，不宣称精确后验族。

## 6.严格四臂

|臂|support密度|ground expert|因果含义|
|---|---|---|---|
|`M0`|原经验带宽qKNN`L0`|无|共同基线|
|`M_DA`|HBPD`LH`|无|异方差predictive DA独立效应|
|`M_HEAD`|原经验带宽qKNN`L0`|D112固定ground单位质量head|已有head独立效应|
|`M_JOINT`|HBPD`LH`|同一个ground单位质量head，anchor核使用该类`~h_c`|DA在head存在时的效应|

对旧类，`M_HEAD/M_JOINT`都只在该类内部把support密度的一部分质量分配给ground anchor；禁止增加第`K+1`票、old logit bias或不同logit原点。两臂使用同一个由support不确定度、ground不确定度和support-ground discrepancy确定的`ρ_c`，不得由HBPD结果重新选择rho。

对所有没有ground资产的注册类：

\[
L_{n}^{HEAD}=L_n^0,
\qquad
L_n^{JOINT}=L_n^H.
\]

因此new列在HEAD因素上严格恒等，但DA对old/new使用同一predictive公式。G1必须报告`DA_AT_BASE=M_DA-M0`、`HEAD_AT_BASE=M_HEAD-M0`、`DA_AT_HEAD=M_JOINT-M_HEAD`和factorial interaction。

## 7.协议与跨new规模不变性

- fit只读取sealed Phase1 variance、当前row support、注册表和K；函数签名无query。
- query只进入逐样本Student-t距离；fit/update/selection均为0。
- new5/new10/new20只append各自类列；相同old support下，全部旧类`σprior/hatσ/barσ/vpair/hbar/rho`必须逐值不变。
- 不读取query truth、query old/new role、batch类数、quota、Hungarian或跨query重排。
- 类别分数均为`LSE-logK`且`kernel_volume_gamma=1`；无类ID白名单或专属超参数。

## 8.资源

|项目|额外开销|
|---|---:|
|Phase1|已有cell内残差的一次等权归约|
|部署资产|6个旧类int8方差code＋scale，1个全局方差；与checkpoint共同封存|
|row enrollment|`O(CKp)`计算离散度；每类4个标量状态|
|单query|与原qKNN相同数量的点积和核；只替换每类一个带宽标量|
|query依赖状态|0B|
|训练/optimizer|0|

## 9.证伪与停止

1.真实G0中任一K的`M_DA` argmax变化为0：按目标文档立即关闭，不调variance、pseudo-degree或带宽系数；
2.G1中`DA_AT_BASE`与`DA_AT_HEAD`均无独立正收益：关闭HBPD，不以ground-head收益保留DA；
3.K1因旧类按类方差造成new、H或总正确数下降，即使old提升也不能晋级；
4.高方差weak old类因峰值惩罚进一步失去floor，说明predictive density与真实尾部不匹配；
5.new5/new10/new20改变任何旧类状态receipt，判为实现错误；
6.HBPD只改变score/margin却不改变三Kprediction，与D113同样止损。

## 10.证据边界

D114目前只有理论草案，没有实现、真实G0、source-held或Target性能。D112 ground-head的source-held小正收益只支持`M_HEAD`作为对照因素；D113的零功能结果只支持停止共同平移。HBPD是否有效必须由新的三K真实G0和未开封四臂G1决定，不能用公式完整度或方差非退化替代。

首轮独立监督给出`P0=2/P1=2/REVISE`：原稿把类中心预测方差用于逐support核，并在未证明尺度关系时与经验`h²`相加。R2改为逐support pairwise方差`2barsigma²`，用`2p barsigma²`直接替换经验带宽，并明确总chord squared与每坐标MSE的单位换算。增量复审结论为`P0=0/P1=0/MERGE`；这只允许进入一个最小实现波次和真实G0，不构成性能结论。
