# D110-SCPM轻型快速域适应理论研究与候选收敛

状态：`G0_PASS_PROCEED_G1 / NO_PERFORMANCE_RESULT`

日期：2026-08-02

## 0.结论

本轮不启动D109，也不做参数扫描。现有Target证据中没有可称为有效正收益的版本：D108的M_DA相对M0仅有H约+0.01pp，属于数值等价；M_HEAD与M_JOINT则显著为负。理论审计后，主线改为`D110-SCPM`：Sealed Conditional Predictive Metric。

1.复用D106与checkpoint共同封存的rank3 receiver/day正交基\(U\)，但不复用其K1固定衰减或`tanh`强度；
2.在Phase1的168个TX×receiver×day cell内部估计“同域、同类、单观测”的条件残差方差，压缩为rank3方向＋正交补共4个int8聚合量；
3.Phase2只用support估计同一4参数结构协方差，以class-block Ledoit–Wolf式矩估计连续收缩到sealed先验；K1自动退回sealed条件方差，而不是identity；
4.query使用显式预测Mahalanobis距离，不把可逆共同变换再交给可能将其代数抵消的完整LDA。

WP-SQR已移除：当前256点片段没有固定preamble／重复字段对齐。CV-GSA/RGM也已移除：现有sealed runtime asset没有构造cross-receiver等权old-TX Gram所需的信息。旧EBSS公式同样被否决：它错误地把只随support中心缩小的\(\nu^2/K\)当成全部预测噪声，并把可能在同row距离中抵消的receiver/day共同漂移直接加入分母。SCPM的4方差asset、runtime核心和真实G0已经完成；G0只证明真实决策非恒等，尚无held或Target性能结果。

## 1.问题为什么不能靠盲目对齐解决

固定received IQ可写为：

\[
x=R_d\!\left(H_d*T_y(s)\right)+n,
\]

其中\(T_y\)是TX硬件指纹，\(H_d\)是传播信道，\(R_d\)是接收机响应。若允许\(H_d,R_d\)为任意可逆算子，则TX变化可以被吸收到信道或接收机算子中；不存在对任意域算子完全不变、又完整保留TX身份的通用变换。

统计学习理论也给出同一边界：仅让源域与目标域边缘分布更相似，并不能在conditional shift下保证目标风险下降；无额外结构时，两个对训练数据不可区分的适配任务可以分别受益或严重受损。[Ben-David等](https://proceedings.mlr.press/v9/david10a.html)

因此，本项目中的合法轻型DA必须满足：

- 明确写出被消去的物理对象及成立条件；
- 只估计低维、共享、可由support和sealed aggregate识别的量；
- 不把query批次用于CORAL、OT、伪标签、熵最小化或模型选择；
- 不把所有频偏、IQ不平衡、PA谱形一律当成receiver nuisance，因为这些量也可能携带TX指纹；
- 证据不足时收缩到恒等映射，而不是增强删除。

## 2.历史性能与可复用教训

### 2.1完整125同口径结果

|方法|B-old|A-old|Min-old|seen-new|H|遗忘|结论|
|---|---:|---:|---:|---:|---:|---:|---|
|D62|81.51%|64.39%|35.15%|59.11%|61.09%|17.11pp|351/375个scene状态回退或零接纳；K1为75/75恒等|
|D81|81.55%|64.40%|35.20%|59.11%|61.09%|17.15pp|ground稳健中心真实激活，但K1无作用|
|D92|81.55%|65.56%|36.81%|58.93%|61.57%|15.99pp|旧类／floor改善，同时交换性损害new；K1恒等|
|D107 M_JOINT|74.77%|47.26%|17.87%|30.11%|35.95%|27.51pp|完整125；比D92少41,008个post正确预测，已淘汰|
|SVRN-qKNN-BCRR/r4.2|73.10%|43.03%|11.21%|23.46%|29.25%|30.07pp|125/125行seen-new和H均低于D62|

D92的K10/new20相对D81为：A-old+2.622pp、Min-old+4.600pp、seen-new−0.653pp、H+0.964pp；这说明共享协方差能减遗忘，但旧／新竞争仍未被共同解决。K1/new20的D62、D81和D92均为A-old44.03%、Min-old14.20%、seen-new27.15%、H33.41%，没有任何K1适配收益。

D91只有固定K10/new5开发单元：B-old92.78%、A-old82.22%、Min-old53.33%、seen-new84.67%、H82.62%、遗忘10.56pp；其15/15个outer预测与D62相同，没有完整125，不能外推。

D106只有完整source-held证据：DA_AT_BASE相对基线为BA+0.2604pp、seen-new+0.3632pp、H+0.4447pp、floor+0.2824pp，但存在负尾；多次Target发布均未形成完整性能结果，不能把source-held小正值称为Target正收益。D105同样没有完整Target性能。二者不再继续修发布器或重复启动。

### 2.2最近Target25证据

D108完整25个outer、75个scene row、600个prediction surface。总体同row结果：

|arm|B-old|A-old|Min-old|seen-new|H|遗忘|
|---|---:|---:|---:|---:|---:|---:|
|M0|82.81%|66.46%|37.60%|59.73%|62.43%|16.36pp|
|M_DA|82.86%|66.47%|37.67%|59.74%|62.44%|16.39pp|
|M_HEAD|70.21%|49.86%|30.40%|44.43%|46.31%|20.36pp|
|M_JOINT|70.09%|49.83%|30.27%|44.45%|46.32%|20.26pp|

M_JOINT在25/25个outer上均劣于M0。K1/new20的ΔH=−8.01pp、ΔA-old=−8.39pp、Δseen-new=−6.87pp，正确数1821对2384。D108证明：support训练内margin或混淆不能当作独立query误差估计，按类整列迁移logit会把低K噪声扩散到全部query。

### 2.3方法设计必须吸收的结论

|历史路线|已经确认的问题|D110必须怎样不同|
|---|---|---|
|D62/D88安全门|support-safe不等于query-safe；hard gate造成大面积恒等|不靠support自正确率决定方法强度|
|D73/D83/D89|连续分数、中心或metric变化可能被后续头吸收，argmax不变|预先写明改变邻序／方向metric的因果路径|
|D74/D78-D80|直接删ground方向或旧类保护会伤害new|只做类无关共享非扩张变换，并保护全部注册类|
|D92|共享协方差减遗忘，但K1不可辨识且损害new|K1只估计跨类共享量；不分old/new协方差池|
|D93/D94|ground→target全坐标transport在低coverage下负迁移|禁止全坐标搬运；只处理sealed低秩因果方向|
|D108/D109|support自margin／混淆不等于query转移|不估计整类logit偏置或C×C混淆转移|

## 3.候选空间与裁决

|候选|理论对象|K1|适配计算|裁决|
|---|---|---|---:|---|
|标准CORAL／Gaussian OT|边缘协方差或经验分布|query不可用时目标统计不足|矩阵平方根／OT|拒绝作为首发；query-batch版本协议不合法|
|SO-BOT|support-only OAS-Bures类内协方差|K1类内残差为0，必然恒等|O(CKr²+r³)|后备K5/K10诊断|
|旧EBSS|target跨类信号／sealed域噪声比|可计算，但query噪声项错误|O(CKd+Crd)|拒绝原公式；support中心噪声与单query残差不能混写|
|SCPM|Phase1条件残差先验＋target support结构协方差|sealed先验给出各向异性metric|O(CKd+Crd)|**主线；需在D106 bundle内新增4个int8聚合方差**|
|CV-GSA/RGM|target与ground TX类Gram的全局alignment|可计算但最多5个类空间自由度|O(C²r)|当前sealed asset缺class＋receiver/day联合Gram，拒绝实施|
|WP-SQR|包内准静态乘性响应|单条IQ内可计算，不增加K|约18–22kFLOPs／样本|输入无固定重复字段对齐，降为探索视图|
|UCB-LS-qKNN|K>1类内尺度与局部密度|严格退化原型|O(CKd)／query|暂缓；首轮不与DA混合|

## 4.主模块：SCPM条件预测度量

### 4.1成立的表示层模型

在L2归一化后的\(z_{id}\)层，对同一个Phase2 row假设：

\[
z_{ci}=\mu_c+\delta_{row}+\varepsilon_{ci},\qquad
z_q=\mu_y+\delta_{row}+\varepsilon_q.
\]

\(\delta_{row}\)是该receiver、场景与固定处理链在这一row内共享的加性漂移。对任意类中心\(\bar z_c\)，\(z_q-\bar z_c\)中的\(\delta_{row}\)精确抵消；因此不能再把D106的receiver/day共同漂移谱直接当作query噪声。该假设只在指定表示层、同receiver、同scene的row内成立；L2归一化、类相关receiver交互或样本级信道变化都可能破坏它，这些是方法的直接证伪条件。

### 4.2Phase1封存的4参数条件方差

D106已封存行正交基\(U=[u_1^\top;u_2^\top;u_3^\top]\in\mathbb R^{3\times d}\)，满足\(UU^\top=I_3\)。令\(P_\perp=I_d-U^\top U\)，Phase1共有\(H=6\times7\times4=168\)个TX×receiver×day cell；每个cell有\(n_h=2\!-\!4\)个互异物理样本，每个样本仍只有一份固定LEO weak观测。对cell中心\(\bar z_h\)：

\[
s_{hj}^2=\frac{\sum_{i=1}^{n_h}[u_j^\top(z_{hi}-\bar z_h)]^2}{n_h-1},
\qquad
v_{0j}=\frac1H\sum_{h=1}^{H}s_{hj}^2,
\]

\[
s_{h\perp}^2=
\frac{\sum_i\|P_\perp(z_{hi}-\bar z_h)\|_2^2}
{(n_h-1)(d-3)},
\qquad
v_{0\perp}=\frac1H\sum_hs_{h\perp}^2.
\]

等权cell平均估计的是平均条件方差，不冒充普通pooled variance；正交补按\(d-3\)维归一，保证四个量同为“每维方差”。cell内中心化去除Phase1 receiver/day共同加性漂移，同时保留物理样本差异和一次LEO弱信道造成的样本级残差。四个正值以`int8[4]＋FP16 scale[4]`与\(U\)、checkpoint、method lock共同封存；这是4B code＋8B scale的12B联合量化态，数值精度主要由逐值FP16 scale决定，不宣称4B纯int8精度。不保留source row、单样本feature、成员ID或独立sidecar。

### 4.3Phase2 support-only矩估计

当前row有\(C\)个已注册类，每类同为\(K\in\{1,5,10\}\)个support。记\(\bar z_c=K^{-1}\sum_i z_{ci}\)。K>1时，每类无偏条件方差明确为：

\[
t_{cj}=\frac{\sum_i[u_j^\top(z_{ci}-\bar z_c)]^2}{K-1},
\qquad
t_{c\perp}=
\frac{\sum_i\|P_\perp(z_{ci}-\bar z_c)\|_2^2}{(K-1)(d-3)}.
\]

对四个group \(g\in\{1,2,3,\perp\}\)令：

\[
v_{Tg}=\frac1C\sum_ct_{cg},\qquad
\widehat V_g=\frac1{C(C-1)}\sum_c(t_{cg}-v_{Tg})^2.
\]

以维数权重\(d_g=(1,1,1,d-3)\)构造一个全group共享、无Target性能参数的class-block Ledoit–Wolf式矩收缩：

\[
\alpha=
\operatorname{clip}_{[0,1]}
\frac{\sum_gd_g\widehat V_g}
{\sum_gd_g(v_{Tg}-v_{0g})^2}.
\]

若分母为0，定义\(\alpha=1\)；最终\(v_g=\alpha v_{0g}+(1-\alpha)v_{Tg}\)。K1没有类内自由度，直接定义\(\alpha=1,v_g=v_{0g}\)。这不是标准OAS：单位球特征、类内去均值和异质class block不满足标准Wishart假设；它只能称为Ledoit–Wolf式矩估计。[Ledoit与Wolf](https://www.sciencedirect.com/science/article/pii/S0047259X03000964)

### 4.4显式预测距离

对类中心\(\bar z_c\)和单条query，记\(\Delta_c=z_q-\bar z_c\)。在support与query残差同分布的对角结构模型下，query到K-shot中心的预测方差为\(v_g(1+1/K)\)，而不是\(v_g/K\)：

\[
D_c(q)=
\sum_{j=1}^{3}\frac{(u_j^\top\Delta_c)^2}{v_j(1+1/K)}+
\frac{\|P_\perp\Delta_c\|_2^2}{v_\perp(1+1/K)}.
\]

本协议同row各类K相同，公共因子\(1+1/K\)可从argmin中删除；保留它只为表达正确的预测不确定性。若未来允许不等K，必须按类保留\(1+1/K_c\)并加入相应log-determinant，当前版本不支持这种扩展。

为防止方差倒数爆炸，先令\(v_{max}=\max_gv_g\)。若\(v_{max}=0\)，四个相对方差统一置为1并回退欧氏距离；否则令\(r_g=v_g/v_{max}\)，再固定\(r_g^{safe}=\max(r_g,1/20)\)，用\(r_g^{safe}\)替换距离分母中的\(v_g\)。条件数20沿用D106的\(m_{min}=0.05\)数值边界，不由Target性能选择。每个封存方差采用独立FP16 scale和固定正int8 code，builder必须记录解码相对误差、拒绝零或非有限scale。运行时只使用解码后的正值和上述条件数界。

### 4.5K1为何不是恒等、与D92/D106有何不同

K1时公共因子2虽然可删，但\(v_{01},v_{02},v_{03},v_{0\perp}\)的相对权重仍形成各向异性Mahalanobis距离；只有四个方差相等、被条件数裁剪拉平或量化后相等时，才退化为欧氏排序。必须在本地真实checkpoint smoke中记录四方差、条件数和相对M0的邻序／prediction hash；“非退化”不等于“保证提升”。

- 相对D92：D92在K1没有可估计的target类内协方差而回退；SCPM用合法Phase1条件方差先验定义K1度量；
- 相对D106：D106使用固定K1衰减和`tanh(scatter/tau)`，SCPM估计明确的共享类条件协方差，并显式进入预测距离；
- 相对D62：SCPM没有TP/FP安全门、匿名行替换、交叉折选行或完整LDA训练，只拟合4个置换等价方差；
- 相对旧EBSS：不估计噪声污染严重的K1类间信号，不把单query噪声错误除以K，也不重复加入同row公共shift。

### 4.6复杂度

- Phase1一次性构造：O(588dr)，新增4个int8 code＋4个FP16 scale；
- support构建：O(CKdr)，r=3；
- 单query：投影与回写约960MAC，再加O(Cr)类距离；
- 运行时状态：4个方差、C个类中心；0epoch、0optimizer step、0query fit。

## 5.被拒绝的首发路线A：有界对称频谱商残差

### 5.1原理

把256点复IQ按固定窗切为相邻短帧。若相邻帧在同一频点满足：

\[
Y_{t+1}(f)=H(f)X_{t+1}(f),\qquad
Y_t(f)=H(f)X_t(f),
\]

且\(H_{t+1}(f)\approx H_t(f)\)，则公共乘性幅频响应可在帧间比值中消去。为避免普通商在谱零点爆炸，使用有界对称形式：

\[
r_{t,f}=\frac{|Y_{t+1}(f)|-|Y_t(f)|}
{|Y_{t+1}(f)|+|Y_t(f)|+\epsilon P},
\qquad -1<r_{t,f}<1,
\]

其中\(P\)是当前IQ的包内能量尺度，\(\epsilon\)只取数值精度下界。固定频带内提取均值、RMS和MAD，形成单块L2归一化的`SQR`视图。

这不是“完全接收机不变”。它只针对相邻短窗共享的慢变线性幅频响应；快速衰落、时序错位、接收机非线性以及TX／RX混合CFO都不会自动消失。已有channel-robust RFFI研究支持频谱商／信道弱相关表征的方向，但其LoRa或其他波形结论不能直接外推到本项目256点WiSig片段。[Shen等](https://arxiv.org/abs/2107.02867)

2026年的cross-receiver工作把denoised spectral quotient与另一个receiver calibration网络分开，也从侧面说明：频谱商主要处理channel，不能独自解决全部receiver shift。[He等](https://arxiv.org/abs/2603.08402)

### 5.2为什么不直接拼入分类特征

直接把SQR与现有288D拼接会引入未解释的block权重，并可能与FFT96/RF32重复。第一版只允许SQR形成support类中心Gram，作为后续receiver metric的弱相关物理证据；不得按query置信度选择视图，也不得把包内短窗冒充多个物理K-shot。

### 5.3只读审计结论

- 当前输入是`WiSig equalized=1 record→中心裁剪／补零到256点→RMS归一化→一次LEO weak叠加`；
- 256点内部连续，当前模拟LEO信道参数在整段共享，但loader不读取packet start，不检测STF/LTF，不保存裁剪offset；
- 固定协议字段对齐为`UNSUPPORTED`，相邻窗重复／同构内容为`UNKNOWN`；
- 因此128/64或64/32窗／hop都不能冻结为物理有效channel-independent配置。

因此WP-SQR只保留为`EXPLORATORY_WITHIN_IQ_SPECTRAL_RATIO_VIEW`，不进入D110首发，也不为保住它增加数据工程。

## 6.被拒绝的首发路线B：全局可靠度ground metric

### 6.1为什么不能逐方向拟合K1 alignment

K1时6个旧类虽然产生15个成对距离，但这些距离只由6个节点生成；中心化类空间最多5个自由方向，任一异常类同时污染5条边。不能把15条边当15个独立样本，也不能用同一5维类空间分别自由选择多个\(\rho_j\)。

原始CV-GSA还把低alignment直接解释为“可以最大衰减”，方向相反：低alignment也可能只是单样本噪声。正确规则必须是“不可靠→恒等”。

### 6.2理论公式及资产缺口

sealed ground必须提供：

- 与当前checkpoint坐标一致的正交receiver/day basis\(U\in\mathbb R^{d\times r}\)；
- 每方向receiver/day可归因强度\(\pi_j\in[0,1]\)，不能只用未分解的高方差特征值；
- 按receiver/day等权聚合的old-TX类中心Gram\(G_g\)，用于保护跨域稳定身份几何。

当前support类中心矩阵为\(Z\in\mathbb R^{C\times d}\)，中心化矩阵为\(H=I-\mathbf1\mathbf1^\top/C\)。仅估计一个全局alignment：

\[
G_u=HZUU^\top Z^\top H,
\qquad
\rho=\frac{\langle G_u,G_g\rangle_F}
{\|G_u\|_F\|G_g\|_F+\epsilon}.
\]

记full CKA为\(a=\rho\)，720个类置换值为\(a_\pi\)，机会均值为\(\rho_0=720^{-1}\sum_\pi a_\pi\)，6个leave-one-class-out值为\(\ell_c\)。定义连续、无可调阈值的可靠度：

\[
e_{\mathrm{perm}}=
\left[\frac1{720}\sum_{\pi\in S_6}(a-a_\pi)_+^2\right]^{1/2},
\]

\[
g_{\mathrm{loo}}=\left(\prod_{c=1}^{6}\ell_c\right)^{1/6},
\qquad
d_{\mathrm{loo}}=
\left[\frac16\sum_{c=1}^{6}(\ell_c-a)^2\right]^{1/2},
\]

\[
r_{\mathrm{stab}}=
e_{\mathrm{perm}}g_{\mathrm{loo}}(1-d_{\mathrm{loo}}).
\]

三因子均在\([0,1]\)，所以\(r_{\mathrm{stab}}\in[0,1]\)；零Gram、未定义CKA或任一LOO退化时统一取0。类标签同步置换只会重排置换集合和LOO集合，故严格类置换等价。它只是support几何可靠度，不是p值、置信度或错误率上界。

再定义：

\[
\hat\rho=\operatorname{clip}
\left(\frac{\rho-\rho_0}{1-\rho_0},0,1\right),
\]

\[
m_j=1-r_{\mathrm{stab}}\pi_j(1-\hat\rho),
\qquad
T=I-U\operatorname{diag}(1-\sqrt{m_j})U^\top.
\]

其中\(U^\top U=I\)、\(0<m_j\le1\)。当Gram退化、ground binding不匹配或留一类不稳定时，\(r_{\mathrm{stab}}=0\)，严格回到\(T=I\)。同一个冻结\(T\)作用于全部old/new support和每条query；无role、class quota、query truth或批次更新。

该可靠度不会根据Target性能调节，也不以单个support正确率决定开关；它只连续缩小低可信更新。720个置换和6个LOO高度相关，不能被解释为726个独立样本。

### 6.3K1可辨识性边界

K1能计算全局跨类几何，但当前资产审计裁决更直接：D106 wire有合法U和可归因spectrum，却明确不保留source类中心和名称；D19有domain×class中心，却没有domain→receiver/day映射。不能把D19整数domain等权Gram冒充cross-receiver等权Gram，也不能混合两个未联合封存的资产恢复被D106主动丢弃的信息。因此RGM当前为`ASSET_INCOMPLETE / REJECTED_FOR_D110`。

## 7.主模块B：统一类平衡局部尺度qKNN

DA变换先冻结，再在变换后的support中构造头。距离为：

\[
d_{ci}(q)=1-\cos(Tz_q,Tz_{ci}).
\]

K1时严格使用单原型：

\[
s_c(q)=-d_{c1}(q).
\]

K≥3时，每类尺度为平均成对距离：

\[
v_c=\frac{2}{K(K-1)}\sum_{i<j}
\left[1-\cos(Tz_{ci},Tz_{cj})\right].
\]

用leave-one-support-out jackknife估计\(\widehat{\mathrm{se}}_c^2\)，并作经验Bayes收缩：

\[
v_0=\frac1C\sum_c v_c,
\quad
\hat\tau^2=\left[\operatorname{Var}_c(v_c)
-\frac1C\sum_c\widehat{\mathrm{se}}_c^2\right]_+,
\]

\[
\lambda_c=\frac{\widehat{\mathrm{se}}_c^2}
{\widehat{\mathrm{se}}_c^2+\hat\tau^2+\epsilon},
\qquad
\tilde v_c=(1-\lambda_c)v_c+\lambda_cv_0.
\]

固定\(m_K=\max(1,\lceil\sqrt K\rceil)\)，类平衡局部证据为：

\[
\ell_c(q)=-\log\bar v_c+
\log\left[
\frac1{m_K}\sum_{i\in\mathcal N_{m_K}(c,q)}
\exp\left(-\frac{d_{ci}(q)}{\bar v_c}\right)
\right],
\]

其中\(\bar v_c=\max(\tilde v_c,\epsilon_q)\)，\(\epsilon_q\)由int8反量化误差界确定，不由Target性能选择。`−log v`项避免宽核类别因覆盖面积大而天然占优。

UCB只处理受限的类条件尺度／多模态，不是完整DA。它与现有qKNN同族，若所有\(\tilde v_c\approx v_0\)或top-m邻居不变，可能再次被吸收。因此它只保留为K>1独立头部臂，不能承担K1收益主张。

## 8.预期因果链与可证伪点

\[
\text{D106封存receiver/day子空间}
\rightarrow
\text{Phase1 cell内条件残差方差}
\rightarrow
\text{support-only class-block矩收缩}
\rightarrow
\text{显式结构化预测Mahalanobis距离}
\rightarrow
\text{统一old/new竞争}
\]

该因果链并不保证性能。以下任一现象都会直接否定对应模块，而不是触发调参：

|模块|理论证伪条件|
|---|---|
|Phase1条件模型|cell内方差主要由类相关receiver非线性而非可交换残差产生；同row公共shift在L2归一化后不抵消|
|SCPM估计|四方差经量化／条件数约束后相等；K1邻序与prediction逐值等价；K5/K10的\(\alpha\)恒为0或1且无可解释统计差异|
|SCPM决策|K1的A-old、floor、seen-new或H任一相对D92下降；旧类改善继续以新类损失交换；弱类floor系统下降|

### 8.1D92／D106路线重入卡

|字段|D92|D106|D110-SCPM改变|
|---|---|---|---|
|已有证据|完整125中H=61.57%；K1与D62逐值相同|source-held有小正均值但负尾；Target无完整性能|尚无性能结果|
|原机制限制|K1类内协方差不可识别而回退；K5/K10改善旧类但损害new|K1固定衰减；K>1用手工常数与`tanh`|不沿用二者强度公式|
|统计对象|target support共享协方差|receiver/day共同漂移方向及总体类内scatter|同TX×receiver×day cell内的单观测条件残差方差|
|决策机制|完整LDA／协方差头|先统一变换，再交给后续头|4参数结构协方差直接定义预测距离|
|K1差异|identity|固定常数非恒等|sealed条件方差形成各向异性metric|
|最小差异证据|—|—|真实G0三种K已非恒等；下一步为完整63行source-held四臂|
|证伪／停止|—|—|G1简单效应／交互方向错误或出现不可接受negative tail即关闭组合，不调rank、clip或带宽|
|额外成本|高维协方差|约960MAC/query|约960MAC/query＋4标量|

## 9.协议、量化与资源

### 9.1协议

- `p2_min_v1`；复用匹配`VALIDATED_ONCE`的数据，不重验received-IQ；
- 每个Phase2物理IQ只产生一个固定LEO weak观测；SCPM不调用信道模拟器、不生成第二观测；
- Phase1的4方差仅由既有588条\(L_s\)一次LEO weak特征的168个cell聚合，不需要多次增强；若未来改用成对增强，必须生成新bundle，不能作为sidecar；
- support拟合后冻结；query逐条面对全部注册类，零fit、零update、零truth、零role、零quota；
- before／after分别按实际注册类构造状态，不跨臂共享可变状态；
- 所有公式类置换等价，不按TX ID、receiver、scene或old/new角色分支。

### 9.2量化顺序

冻结顺序必须是：

\[
z_{id}\rightarrow L2\text{归一化}
\rightarrow\text{D106 INT8 basis解码与正交闭合}
\rightarrow\text{Phase1方差／Phase2 support统计}
\rightarrow\text{同一表示上的SCPM query距离}.
\]

Phase1与Phase2必须使用同一个closed \(U\)、同一种L2顺序和同一个正交补定义。不得用FP32原basis构造方差、再用INT8闭合basis推理；不得用query更新\(\alpha\)或\(v_g\)。

### 9.3初步资源上界

|模块|support构建|单query|新增持久态|
|---|---:|---:|---:|
|SCPM|O(CKdr)，r=3|O(dr+Cr)，约960MAC投影回写＋类距离|4个方差；新增4B int8 code＋8B FP16 scale|

资源仍需以真实实现计时和state wire审计闭合；当前数值是理论估算，不是实测性能数据。

## 10.G1冻结四臂：同一qKNN公式上的2×2因果分解

真实588条tap的G0已完成：K1/K5/K10分别有23/40/96个argmax改变，三种K均非零，允许进入G1；该证据不含性能指标。当前目标要求先做尚未打开的source-held四臂，不再沿用旧版“两臂Target5优先”安排。

令输入先按现有路径L2归一化；support仍沿用当前qKNN的逐向量INT8量化／解码。DA因子只改变如下平方距离：

\[
d_I(x,y)=\|x-y\|_2^2,
\qquad
d_S(x,y)=\sum_{j=1}^3\frac{[u_j^\top(x-y)]^2}{r_j}
+\frac{\|P_\perp(x-y)\|_2^2}{r_\perp},
\]

其中\(r_g\)是D110 runtime已经冻结的safe relative variance。SCPM预测方差中的公共\((1+1/K)\)不进入qKNN距离，因为它在固定K的类别决策中是公共尺度；去除它使四臂只比较SCPM各向异性，不引入额外K相关温度。G0最近中心决策对该公共因子本来就不敏感，因此G1使用的是同一个已通过G0的各向异性对象。

HEAD因子只改变带宽是否按类独立。对每类\(c\)在K>1时计算其全部无序support pair的均值距离\(e_c\)。class-specific头严格复用现有公式：

\[
h_c^2=\operatorname{clip}\!\left(
\frac{e_c+\lambda h_0^2}{1+\lambda},
h_0^2r_{min}^2,h_0^2r_{max}^2
\right).
\]

shared头不引入新的\(n_{eff}\)或\(\kappa\)，只把等K注册类的\(e_c\)先求算术均值\(\bar e=C^{-1}\sum_c e_c\)，再经过完全相同的Phase1锁定先验、分母和clip：

\[
h_{shared}^2=\operatorname{clip}\!\left(
\frac{\bar e+\lambda h_0^2}{1+\lambda},
h_0^2r_{min}^2,h_0^2r_{max}^2
\right).
\]

K1没有类内自由度，两种头都严格取\(h_0\)。因此K1的`M_HEAD=M0`、`M_JOINT=M_DA`是预期的可辨识边界，不得把HEAD在K1恒等当作失败。四臂统一使用同一Student-t核、\(\nu\)、\(d_{eff}\)、volume项、logsumexp-minus-log-K、INT8解码和全注册类竞争：

\[
s_c(q)=\operatorname{LSE}_{i\in c}\left[
-\gamma d_{eff}\log h_c
-\frac{\nu+d_{eff}}2\log\!\left(1+\frac{d_M(q,z_{ci})}{\nu h_c^2}\right)
\right]-\log K.
\]

|arm|距离因子|HEAD因子|
|---|---|---|
|M0|\(d_I\)|现有class-specific尺度|
|M_DA|\(d_S\)|同一class-specific尺度公式|
|M_HEAD|\(d_I\)|全注册类shared尺度|
|M_JOINT|\(d_S\)|同一shared尺度公式|

G1必须使用在D110公式冻结后生成、预测提交前不打开truth的新source-held split；已经被D106 G1评分并用于D110设计复盘的`d104_source_seed104713_v2`只能做机械回归，不能再次冒充held证据。新split从既有8400条单观测source feature pool中同时排除2478条D103历史query、2520条D104 held和588条D110 Phase1 tap；三组有部分交集，排除union后168个receiver×TX×day cell的最小剩余容量为7，因此不扫描容量，固定每cell按冻结salt取7条，共1176条。这样42个receiver×class组各有28条且四天等权；K1/K5/K10分别保留27/23/18条query/class。G1保持21个一般行＋42个held-class K1行的63行结构。完整运行后只看简单效应、交互、old/new平衡、floor和negative tail；不扫描rank、cap、带宽、温度或融合权重。方向正确才进入固定Target25四臂；方向错误则关闭该机制并研发下一revision。

## 11.交叉复审结论与研发顺序

三组研究与交叉复审得到：

- 物理侧：raw FFT/RF不是独立身份证据；WP-SQR因缺固定重复字段对齐，从主线移除；
- 资产侧：D106有合法U但没有cell内条件方差；SCPM只需从同一588条Phase1 tap新增4个共同封存聚合量，不读取clean/source运行时状态；
- 协议侧：Phase1可使用source clean和卫星增强视图，但SCPM当前甚至不需要新增增强；Phase2一次观测规则不变；
- 数学侧：独立终审为`P0=0 / P1=0 / P2=0 / DESIGN_FROZEN`；原SCPM草案的问题已通过自由度、正交补归一、class-block矩收缩、零方差回退、条件数界和唯一M0比较口径关闭；
- 头部侧：UCB与REPP暂缓；G1只用全注册类共享尺度，且与SCPM构成同一qKNN公式上的冻结2×2分解。

量化顺序与历史重入差异已在本报告关闭；剩余研发顺序为：

1.独立监督终审已完成：P0=0、P1=0、P2=0；
2.4方差asset、SCPM score及真实G0已经完成；三种K均改变决策；
3.实现同一Student-t qKNN公式上的四臂，不改Phase1锁定量；
4.完成一次窄验证和一次独立复审后，立即发布完整63行source-held G1；
5.G1方向正确即进入固定四臂Target25；性能弱则关闭当前组合并研发下一统计对象。

这不是增加发布流程gate，而是把实验矩阵按“先回答最关键未知量”缩到最小。SCPM组合若G1方向错误，直接关闭并研发下一机制；不会恢复WP-SQR、RGM、旧EBSS、D109或参数扫描。

### 11.1本地核心实现状态

最小功能核心与真实G0已经闭合；尚未实现的是G1四臂source-held入口：

|文件|已实现边界|当前验证|
|---|---|---|
|`code/cvsrffi/stage2_d110_scpm_asset.py`|168个TX×receiver×day cell等权条件方差、正交补157维归一、4×INT8＋4×FP16封存、D106 lineage／checkpoint绑定、loader-only formal authority、正式asset到runtime入口|零方差单cell允许但全局零方差拒绝；伪造formal状态拒绝；错误D106绑定拒绝|
|`code/cvsrffi/stage2_d110_scpm_runtime.py`|K∈{1,5,10}的class-block矩收缩、K1封存先验、相对方差cap=20、全类独立预测、零query更新|d=160、正交补权重157、K1非欧氏恒等、K2拒绝、置换等价与确定性|
|两份对应测试|只检查公式、资产边界和运行时行为，不读取accuracy选择配置|`ssr-gpu`中12项通过；两模块`py_compile`通过；`git diff --check`通过|

修复后独立Terra Max复审结论为`P0=0 / P1=0 / LOCAL_CORE_VERIFIED`；前序零方差cell、formal authority、K集合、d=160／perp=157、K1非恒等及正式asset→runtime入口问题均已闭合。随后真实G0确认K1/K5/K10均改变argmax。

G0通过仍不表示性能正收益。下一项唯一必要功能工作是把上述冻结四臂接入63行predict／independent score入口，并从既有source feature pool生成一份未打开、与历史held和D110 tap均不重叠的新split；完成一次窄验证与一次独立复审后立即发布G1。

## 12.主要参考

- [Impossibility Theorems for Domain Adaptation](https://proceedings.mlr.press/v9/david10a.html)：无额外结构时，边缘对齐不能保证目标风险下降。
- [Correlation Alignment for Unsupervised Domain Adaptation](https://arxiv.org/abs/1612.01939)：二阶统计对齐的经典轻量路线；本项目不能使用query批次统计。
- [Algorithms for Learning Kernels Based on Centered Alignment](https://www.jmlr.org/papers/v13/cortes12a.html)：centered alignment的定义与稳定性背景；本文将其用于ground身份几何是新设计推导。
- [A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices](https://www.sciencedirect.com/science/article/pii/S0047259X03000964)：线性协方差收缩的理论来源；SCPM使用的是适配class block的矩估计，不冒充原论文的标准设定。
- [Towards Scalable and Channel-Robust RFFI for LoRa](https://arxiv.org/abs/2107.02867)：频谱商／channel-independent feature方向的物理依据，不能直接当作WiSig Target证据。
- [Deep Learning based Cross-Receiver RFFI Under Varying Channels](https://arxiv.org/abs/2603.08402)：把spectral quotient与receiver calibration分开，支持两类干扰需要分别处理。
- [Self-Tuning Spectral Clustering](https://proceedings.neurips.cc/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html)：局部尺度思想来源；不构成本项目RFFI性能证明。
- [Prototypical Networks](https://proceedings.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)：低样本均值原型的归纳偏置。

## 13.证据路径

- `analysis/stage2_method_comprehensive_comparison_20260724.md`
- `automation_reports/CV-SincNet/d81_comprehensive_125_20260720/report.md`
- `automation_reports/CV-SincNet/d92_registration_balanced_125_20260720/report.md`
- `automation_reports/CV-SincNet/d91_crossfit_consensus_sigma_margin_20260720/report.md`
- `E:/type10-7/automation_reports/CV-SincNet/svrn_qknn_bcrr_125_r4_retry2_20260724/report.md`
- `automation_reports/CV-SincNet/d106_g1_sourceheld_b442472b_20260801_r2/report.md`
- `automation_reports/CV-SincNet/d107_scmkrr_target125_20260801_r1/report.md`
- `automation_reports/CV-SincNet/d108_cbrrc_smme_target25_s713102_20260801_r1/report.md`

本轮已在N607完成D110真实G0，但G0不读取truth且没有生成性能结果；下一步为冻结四臂source-held G1。
